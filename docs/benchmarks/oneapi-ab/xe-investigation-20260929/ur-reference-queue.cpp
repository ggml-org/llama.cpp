//===--------- queue.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <cstdint>
#include <optional>
#include <string.h>
#include <vector>

#include "adapter.hpp"
#include "common.hpp"
#include "event.hpp"
#include "queue.hpp"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"
#include "ur_util.hpp"
#include "ze_api.h"

// Hard limit for the event completion batches.
static const uint64_t CompletionBatchesMax = [] {
  // Default value chosen empirically to maximize the number of asynchronous
  // in-flight operations and avoid excessive synchronous waits.

  return getenv_to_unsigned("UR_L0_IMMEDIATE_COMMANDLISTS_BATCH_MAX")
      .value_or(10);
}();

static const uint64_t CompletionEventsPerBatch = [] {
  // The number of events to accumulate in each batch prior to waiting for
  // completion.
  return getenv_to_unsigned("UR_L0_IMMEDIATE_COMMANDLISTS_EVENTS_PER_BATCH")
      .value_or(256);
}();

ur_completion_batch::ur_completion_batch()
    : barrierEvent(nullptr), st(EMPTY), numEvents(0) {}

ur_completion_batch::~ur_completion_batch() {
  if (barrierEvent)
    urEventReleaseInternal(barrierEvent);
}

bool ur_completion_batch::isFull() {
  assert(st == ACCUMULATING);

  return numEvents >= CompletionEventsPerBatch;
}

void ur_completion_batch::append() {
  assert(st == ACCUMULATING);
  numEvents++;
}

ur_result_t ur_completion_batch::reset() {
  st = EMPTY;
  numEvents = 0;

  // we reuse the UR event handle but reset the internal level-zero one
  if (barrierEvent)
    ZE2UR_CALL(zeEventHostReset, (barrierEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

void ur_completion_batch::use() {
  assert(st == EMPTY);
  st = ACCUMULATING;
}

ur_completion_batch::state ur_completion_batch::getState() { return st; }

ur_completion_batch::state ur_completion_batch::queryState() {
  if (st == SEALED) {
    checkComplete();
  }

  return st;
}

bool ur_completion_batch::checkComplete() {
  assert(st == COMPLETED || st == SEALED);

  if (st == COMPLETED)
    return true;

  auto zeResult = ZE_CALL_NOCHECK(zeEventQueryStatus, (barrierEvent->ZeEvent));
  if (zeResult == ZE_RESULT_SUCCESS) {
    st = COMPLETED;
  }

  return st == COMPLETED;
}

ur_result_t ur_completion_batch::seal(ur_queue_handle_t queue,
                                      ze_command_list_handle_t cmdlist) {
  assert(st == ACCUMULATING);

  if (!barrierEvent) {
    UR_CALL(EventCreate(
        queue->Context, queue, false /*IsMultiDevice*/, true /*HostVisible*/,
        &barrierEvent, false /*CounterBasedEventEnabled*/,
        false /*ForceDisableProfiling*/, false /*InterruptBasedEventEnabled*/));
  }

  // Instead of collecting all the batched events, we simply issue a global
  // barrier for all prior events on the command list. This is simpler and
  // showed to be faster in practice.
  ZE2UR_CALL(zeCommandListAppendBarrier,
             (cmdlist, barrierEvent->ZeEvent, 0, nullptr));

  st = SEALED;

  return UR_RESULT_SUCCESS;
}

void ur_completion_batches::append(ur_event_handle_t event) {
  active->append();
  event->completionBatch = active;
}

void ur_completion_batches::moveCompletedEvents(
    ur_completion_batch_it it, std::vector<ur_event_handle_t> &events,
    std::vector<ur_event_handle_t> &EventListToCleanup) {
  // This works by tagging all events belonging to a batch, and then removing
  // all events in a vector with the tag (iterator) of the active batch.
  // This could be optimized to remove a specific range of entries if we had a
  // guarantee that all the appended events in the vector remain there in the
  // same order. Unfortunately that is not simple to enforce.
  // TODO: An even better approach would be to split the EventList vector into
  // smaller batch-sized ones, but that would require a significant refactor.

  auto end = std::remove_if(events.begin(), events.end(), [&](auto &event) {
    if (event->completionBatch == it) {
      EventListToCleanup.push_back(event);
      return true;
    } else {
      return false;
    }
  });
  events.erase(end, events.end());
}

ur_result_t ur_completion_batches::cleanup(
    std::vector<ur_event_handle_t> &events,
    std::vector<ur_event_handle_t> &EventListToCleanup) {
  bool cleaned = false;
  while (!sealed.empty()) {
    auto oldest_sealed = sealed.front();
    if (oldest_sealed->queryState() == ur_completion_batch::COMPLETED) {
      sealed.pop();
      moveCompletedEvents(oldest_sealed, events, EventListToCleanup);
      UR_CALL(oldest_sealed->reset());
      cleaned = true;
    } else {
      break;
    }
  }

  return cleaned ? UR_RESULT_SUCCESS : UR_RESULT_ERROR_OUT_OF_RESOURCES;
}

std::optional<ur_completion_batch_it>
ur_completion_batches::findFirstEmptyBatchOrCreate() {
  for (auto it = batches.begin(); it != batches.end(); ++it) {
    if (it->getState() == ur_completion_batch::EMPTY) {
      return it;
    }
  }

  // try creating a new batch if allowed by the limit.
  if (batches.size() < CompletionBatchesMax) {
    return batches.emplace(batches.end());
  }

  return std::nullopt;
}

ur_completion_batches::ur_completion_batches() {
  // Batches are created lazily on-demand. Start with just one.
  active = batches.emplace(batches.begin());
  active->use();
}

ur_result_t ur_completion_batches::tryCleanup(
    ur_queue_handle_t queue, ze_command_list_handle_t cmdlist,
    std::vector<ur_event_handle_t> &events,
    std::vector<ur_event_handle_t> &EventListToCleanup) {
  cleanup(events, EventListToCleanup);

  if (active->isFull()) {
    auto next_batch = findFirstEmptyBatchOrCreate();
    if (!next_batch) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES; // EWOULDBLOCK
    }

    UR_CALL(active->seal(queue, cmdlist));
    sealed.push(active);
    active = *next_batch;
    active->use();
  }

  return UR_RESULT_SUCCESS;
}

void ur_completion_batches::forceReset() {
  for (auto &b : batches) {
    b.reset();
  }
  while (!sealed.empty()) {
    sealed.pop();
  }

  active = batches.begin();
  active->use();
}

/// @brief Cleanup events in the immediate lists of the queue.
/// @param Queue Queue where events need to be cleaned up.
/// @param QueueLocked Indicates if the queue mutex is locked by caller.
/// @param QueueSynced 'true' if queue was synchronized before the
/// call and no other commands were submitted after synchronization, 'false'
/// otherwise.
/// @param CompletedEvent Hint providing an event which was synchronized before
/// the call, in case of in-order queue it allows to cleanup all preceding
/// events.
/// @return PI_SUCCESS if successful, PI error code otherwise.
ur_result_t CleanupEventsInImmCmdLists(ur_queue_handle_t UrQueue,
                                       bool QueueLocked, bool QueueSynced,
                                       ur_event_handle_t CompletedEvent) {
  // Handle only immediate command lists here.
  if (!UrQueue || !UrQueue->UsingImmCmdLists)
    return UR_RESULT_SUCCESS;

  ur_event_handle_t_ *UrCompletedEvent =
      reinterpret_cast<ur_event_handle_t_ *>(CompletedEvent);

  std::vector<ur_event_handle_t> EventListToCleanup;
  {
    std::unique_lock<ur_shared_mutex> QueueLock(UrQueue->Mutex,
                                                std::defer_lock);
    if (!QueueLocked)
      QueueLock.lock();
    // If queue is locked and fully synchronized then cleanup all events.
    // If queue is not locked then by this time there may be new submitted
    // commands so we can't do full cleanup.
    if (QueueLocked &&
        (QueueSynced || (UrQueue->isInOrderQueue() &&
                         (reinterpret_cast<ur_event_handle_t>(
                              UrCompletedEvent) == UrQueue->LastCommandEvent ||
                          !UrQueue->LastCommandEvent)))) {
      UrQueue->LastCommandEvent = nullptr;
      for (auto &&It = UrQueue->CommandListMap.begin();
           It != UrQueue->CommandListMap.end(); ++It) {
        UR_CALL(UrQueue->resetCommandList(It, true, EventListToCleanup,
                                          false /* CheckStatus */));
      }
    } else if (UrQueue->isInOrderQueue() && UrCompletedEvent) {
      // If the queue is in-order and we have information about completed event
      // then cleanup all events in the command list preceding to CompletedEvent
      // including itself.

      // Check that the comleted event has associated command list.
      if (!(UrCompletedEvent->CommandList &&
            UrCompletedEvent->CommandList.value() !=
                UrQueue->CommandListMap.end()))
        return UR_RESULT_SUCCESS;

      auto &CmdListEvents =
          UrCompletedEvent->CommandList.value()->second.EventList;
      auto CompletedEventIt = std::find(CmdListEvents.begin(),
                                        CmdListEvents.end(), UrCompletedEvent);
      if (CompletedEventIt != CmdListEvents.end()) {
        // We can cleanup all events prior to the completed event in this
        // command list and completed event itself.
        // TODO: we can potentially cleanup more events here by finding
        // completed events on another command lists, but it is currently not
        // implemented.
        std::move(std::begin(CmdListEvents), CompletedEventIt + 1,
                  std::back_inserter(EventListToCleanup));
        CmdListEvents.erase(CmdListEvents.begin(), CompletedEventIt + 1);
      }
    } else {
      // Fallback to resetCommandList over all command lists.
      for (auto &&It = UrQueue->CommandListMap.begin();
           It != UrQueue->CommandListMap.end(); ++It) {
        UR_CALL(UrQueue->resetCommandList(It, true, EventListToCleanup,
                                          true /* CheckStatus */));
      }
    }
  }
  UR_CALL(CleanupEventListFromResetCmdList(EventListToCleanup, QueueLocked));
  return UR_RESULT_SUCCESS;
}

/// @brief Reset signalled command lists in the queue and put them to the cache
/// of command lists. Also cleanup events associated with signalled command
/// lists. Queue must be locked by the caller for modification.
/// @param Queue Queue where we look for signalled command lists and cleanup
/// events.
/// @return PI_SUCCESS if successful, PI error code otherwise.
ur_result_t resetCommandLists(ur_queue_handle_t Queue) {
  // Handle immediate command lists here, they don't need to be reset and we
  // only need to cleanup events.
  if (Queue->UsingImmCmdLists) {
    UR_CALL(CleanupEventsInImmCmdLists(Queue, true /*QueueLocked*/,
                                       false /*QueueSynced*/,
                                       nullptr /*CompletedEvent*/));
    return UR_RESULT_SUCCESS;
  }

  // We need events to be cleaned up out of scope where queue is locked to avoid
  // nested locks, because event cleanup requires event to be locked. Nested
  // locks are hard to control and can cause deadlocks if mutexes are locked in
  // different order.
  std::vector<ur_event_handle_t> EventListToCleanup;

  // We check for command lists that have been already signalled, but have not
  // been added to the available list yet. Each command list has a fence
  // associated which tracks if a command list has completed dispatch of its
  // commands and is ready for reuse. If a command list is found to have been
  // signalled, then the command list & fence are reset and command list is
  // returned to the command list cache. All events associated with command
  // list are cleaned up if command list was reset.
  for (auto &&it = Queue->CommandListMap.begin();
       it != Queue->CommandListMap.end(); ++it) {
    // Immediate commandlists don't use a fence and are handled separately
    // above.
    assert(it->second.ZeFence != nullptr);
    // It is possible that the fence was already noted as signalled and
    // reset. In that case the ZeFenceInUse flag will be false.
    if (it->second.ZeFenceInUse) {
      ze_result_t ZeResult =
          ZE_CALL_NOCHECK(zeFenceQueryStatus, (it->second.ZeFence));
      if (ZeResult == ZE_RESULT_SUCCESS)
        UR_CALL(Queue->resetCommandList(it, true, EventListToCleanup));
    }
  }
  CleanupEventListFromResetCmdList(EventListToCleanup, true /*locked*/);
  return UR_RESULT_SUCCESS;
}

namespace ur::level_zero {

ur_result_t urQueueGetInfo(
    /// [in] handle of the queue object
    ur_queue_handle_t Queue,
    /// [in] name of the queue property to query
    ur_queue_info_t ParamName,
    /// [in] size in bytes of the queue property value provided
    size_t ParamValueSize,
    /// [out] value of the queue property
    void *ParamValue,
    /// [out] size in bytes returned in queue property value
    size_t *ParamValueSizeRet) {
  std::shared_lock<ur_shared_mutex> Lock(Queue->Mutex);
  UrReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  // TODO: consider support for queue properties and size
  switch ((uint32_t)ParamName) { // cast to avoid warnings on EXT enum values
  case UR_QUEUE_INFO_CONTEXT:
    return ReturnValue(Queue->Context);
  case UR_QUEUE_INFO_DEVICE:
    return ReturnValue(Queue->Device);
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{Queue->RefCount.load()});
  case UR_QUEUE_INFO_FLAGS:
    return ReturnValue(Queue->Properties);
  case UR_QUEUE_INFO_SIZE:
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_QUEUE_INFO_EMPTY: {
    // We can exit early if we have in-order queue.
    if (Queue->isInOrderQueue()) {
      if (!Queue->LastCommandEvent)
        return ReturnValue(true);

      // We can check status of the event only if it isn't discarded otherwise
      // it may be reset (because we are free to reuse such events) and
      // zeEventQueryStatus will hang.
      // TODO: use more robust way to check that ZeEvent is not owned by
      // LastCommandEvent.
      if (!Queue->LastCommandEvent->IsDiscarded) {
        ze_result_t ZeResult = ZE_CALL_NOCHECK(
            zeEventQueryStatus, (Queue->LastCommandEvent->ZeEvent));
        if (ZeResult == ZE_RESULT_NOT_READY) {
          return ReturnValue(false);
        } else if (ZeResult != ZE_RESULT_SUCCESS) {
          return ze2urResult(ZeResult);
        }
        return ReturnValue(true);
      }
      // For immediate command lists we have to check status of the event
      // because immediate command lists are not associated with level zero
      // queue. Conservatively return false in this case because last event is
      // discarded and we can't check its status.
      if (Queue->UsingImmCmdLists)
        return ReturnValue(false);
    }

    // If we have any open command list which is not empty then return false
    // because it means that there are commands which are not even submitted for
    // execution yet.
    using IsCopy = bool;
    if (Queue->hasOpenCommandList(IsCopy{true}) ||
        Queue->hasOpenCommandList(IsCopy{false}))
      return ReturnValue(false);

    for (const auto &QueueMap :
         {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID}) {
      for (const auto &QueueGroup : QueueMap) {
        if (Queue->UsingImmCmdLists) {
          // Immediate command lists are not associated with any Level Zero
          // queue, that's why we have to check status of events in each
          // immediate command list. Start checking from the end and exit early
          // if some event is not completed.
          for (const auto &ImmCmdList : QueueGroup.second.ImmCmdLists) {
            if (ImmCmdList == Queue->CommandListMap.end())
              continue;

            const auto &EventList = ImmCmdList->second.EventList;
            for (auto It = EventList.crbegin(); It != EventList.crend(); It++) {
              ze_result_t ZeResult =
                  ZE_CALL_NOCHECK(zeEventQueryStatus, ((*It)->ZeEvent));
              if (ZeResult == ZE_RESULT_NOT_READY) {
                return ReturnValue(false);
              } else if (ZeResult != ZE_RESULT_SUCCESS) {
                return ze2urResult(ZeResult);
              }
            }
          }
        } else {
          for (const auto &ZeQueue : QueueGroup.second.ZeQueues) {
            if (!ZeQueue)
              continue;
            // Provide 0 as the timeout parameter to immediately get the status
            // of the Level Zero queue.
            ze_result_t ZeResult = ZE_CALL_NOCHECK(zeCommandQueueSynchronize,
                                                   (ZeQueue, /* timeout */ 0));
            if (ZeResult == ZE_RESULT_NOT_READY) {
              return ReturnValue(false);
            } else if (ZeResult != ZE_RESULT_SUCCESS) {
              return ze2urResult(ZeResult);
            }
          }
        }
      }
    }
    return ReturnValue(true);
  }
  default:
    logger::error(
        "Unsupported ParamName in urQueueGetInfo: ParamName=ParamName={}(0x{})",
        ParamName, logger::toHex(ParamName));
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

// Controls if we should choose doing eager initialization
// to make it happen on warmup paths and have the reportable
// paths be less likely affected.
//
static bool doEagerInit = [] {
  const char *UrRet = std::getenv("UR_L0_EAGER_INIT");
  const char *PiRet = std::getenv("SYCL_EAGER_INIT");
  const char *EagerInit = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  return EagerInit ? std::atoi(EagerInit) != 0 : false;
}();

ur_result_t urQueueCreate(
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in] handle of the device object
    ur_device_handle_t Device,
    /// [in] specifies a list of queue properties and their corresponding
    /// values. Each property name is immediately followed by the
    /// corresponding desired value. The list is terminated with a 0. If a
    /// property value is not specified, then its default value will be
    /// used.
    const ur_queue_properties_t *Props,
    /// [out] pointer to handle of queue object created
    ur_queue_handle_t *Queue) {
  ur_queue_flags_t Flags{};
  if (Props) {
    Flags = Props->flags;
  }

  int ForceComputeIndex = -1; // Use default/round-robin.
  if (Props) {
    if (Props->pNext) {
      const ur_base_properties_t *extendedDesc =
          reinterpret_cast<const ur_base_properties_t *>(Props->pNext);
      if (extendedDesc->stype == UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES) {
        const ur_queue_index_properties_t *IndexProperties =
            reinterpret_cast<const ur_queue_index_properties_t *>(extendedDesc);
        ForceComputeIndex = IndexProperties->computeIndex;
      }
    }
  }

  UR_ASSERT(Context->isValidDevice(Device), UR_RESULT_ERROR_INVALID_DEVICE);

  // Create placeholder queues in the compute queue group.
  // Actual L0 queues will be created at first use.
  std::vector<ze_command_queue_handle_t> ZeComputeCommandQueues(
      Device->QueueGroup[ur_queue_handle_t_::queue_type::Compute]
          .ZeProperties.numQueues,
      nullptr);

  // Create placeholder queues in the copy queue group (main and link
  // native groups are combined into one group).
  // Actual L0 queues will be created at first use.
  size_t NumCopyGroups = 0;
  if (Device->hasMainCopyEngine()) {
    NumCopyGroups +=
        Device->QueueGroup[ur_queue_handle_t_::queue_type::MainCopy]
            .ZeProperties.numQueues;
  }
  if (Device->hasLinkCopyEngine()) {
    NumCopyGroups +=
        Device->QueueGroup[ur_queue_handle_t_::queue_type::LinkCopy]
            .ZeProperties.numQueues;
  }
  std::vector<ze_command_queue_handle_t> ZeCopyCommandQueues(NumCopyGroups,
                                                             nullptr);

  try {
    *Queue =
        new ur_queue_handle_t_(ZeComputeCommandQueues, ZeCopyCommandQueues,
                               Context, Device, true, Flags, ForceComputeIndex);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  // Do eager initialization of Level Zero handles on request.
  if (doEagerInit) {
    auto Q = *Queue;
    // Creates said number of command-lists.
    auto warmupQueueGroup = [Q](bool UseCopyEngine,
                                uint32_t RepeatCount) -> ur_result_t {
      ur_command_list_ptr_t CommandList;
      while (RepeatCount--) {
        if (Q->UsingImmCmdLists) {
          CommandList = Q->getQueueGroup(UseCopyEngine).getImmCmdList();
        } else {
          // Heuristically create some number of regular command-list to reuse.
          for (int I = 0; I < 10; ++I) {
            UR_CALL(Q->createCommandList(UseCopyEngine, CommandList));
            // Immediately return them to the cache of available command-lists.
            std::vector<ur_event_handle_t> EventsUnused;
            UR_CALL(Q->resetCommandList(CommandList, true /* MakeAvailable */,
                                        EventsUnused));
          }
        }
      }
      return UR_RESULT_SUCCESS;
    };
    // Create as many command-lists as there are queues in the group.
    // With this the underlying round-robin logic would initialize all
    // native queues, and create command-lists and their fences.
    // At this point only the thread creating the queue will have associated
    // command-lists. Other threads have not accessed the queue yet. So we can
    // only warmup the initial thread's command-lists.
    const auto &QueueGroup = Q->ComputeQueueGroupsByTID.get();
    UR_CALL(warmupQueueGroup(false, QueueGroup.UpperIndex -
                                        QueueGroup.LowerIndex + 1));
    if (Q->useCopyEngine()) {
      const auto &QueueGroup = Q->CopyQueueGroupsByTID.get();
      UR_CALL(warmupQueueGroup(true, QueueGroup.UpperIndex -
                                         QueueGroup.LowerIndex + 1));
    }
    // TODO: warmup event pools. Both host-visible and device-only.
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urQueueRetain(
    /// [in] handle of the queue object to get access
    ur_queue_handle_t Queue) {
  {
    std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);
    Queue->RefCountExternal++;
  }
  Queue->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t urQueueRelease(
    /// [in] handle of the queue object to release
    ur_queue_handle_t Queue) {
  std::vector<ur_event_handle_t> EventListToCleanup;
  {
    std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);

    if ((--Queue->RefCountExternal) != 0) {
      // When an External Reference exists one still needs to decrement the
      // internal reference count. When the External Reference count == 0, then
      // cleanup of the queue begins and the final decrement of the internal
      // reference count is completed.
      static_cast<void>(Queue->RefCount.decrementAndTest());
      return UR_RESULT_SUCCESS;
    }

    // When external reference count goes to zero it is still possible
    // that internal references still exists, e.g. command-lists that
    // are not yet completed. So do full queue synchronization here
    // and perform proper cleanup.
    //
    // It is possible to get to here and still have an open command list
    // if no wait or finish ever occurred for this queue.
    auto Res = Queue->executeAllOpenCommandLists();

    // Make sure all commands get executed.
    if (Res == UR_RESULT_SUCCESS)
      UR_CALL(Queue->synchronize());

    // Destroy all the fences created associated with this queue.
    for (auto it = Queue->CommandListMap.begin();
         it != Queue->CommandListMap.end(); ++it) {
      // This fence wasn't yet signalled when we polled it for recycling
      // the command-list, so need to release the command-list too.
      // For immediate commandlists we don't need to do an L0 reset of the
      // commandlist but do need to do event cleanup which is also in the
      // resetCommandList function.
      // If the fence is a nullptr we are using immediate commandlists,
      // otherwise regular commandlists which use a fence.
      if (it->second.ZeFence == nullptr || it->second.ZeFenceInUse) {
        // Destroy completions batches if they are being used. This needs
        // to happen prior to resetCommandList so that all events are
        // checked.
        it->second.completions.reset();
        Queue->resetCommandList(it, true, EventListToCleanup);
      }
      // TODO: remove "if" when the problem is fixed in the level zero
      // runtime. Destroy only if a queue is healthy. Destroying a fence may
      // cause a hang otherwise.
      // If the fence is a nullptr we are using immediate commandlists.
      if (Queue->Healthy && it->second.ZeFence != nullptr) {
        auto ZeResult = ZE_CALL_NOCHECK(zeFenceDestroy, (it->second.ZeFence));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return ze2urResult(ZeResult);
      }
      if (Queue->UsingImmCmdLists && Queue->OwnZeCommandQueue) {
        std::scoped_lock<ur_mutex> Lock(
            Queue->Context->ZeCommandListCacheMutex);
        const ur_command_list_info_t &MapEntry = it->second;
        if (MapEntry.CanReuse) {
          // Add commandlist to the cache for future use.
          // It will be deleted when the context is destroyed.
          auto &ZeCommandListCache =
              MapEntry.isCopy(Queue)
                  ? Queue->Context
                        ->ZeCopyCommandListCache[Queue->Device->ZeDevice]
                  : Queue->Context
                        ->ZeComputeCommandListCache[Queue->Device->ZeDevice];
          struct l0_command_list_cache_info ListInfo;
          ListInfo.ZeQueueDesc = it->second.ZeQueueDesc;
          ListInfo.InOrderList = it->second.IsInOrderList;
          ListInfo.IsImmediate = it->second.IsImmediate;
          ZeCommandListCache.push_back({it->first, ListInfo});
        } else {
          // A non-reusable comamnd list that came from a make_queue call is
          // destroyed since it cannot be recycled.
          ze_command_list_handle_t ZeCommandList = it->first;
          if (ZeCommandList) {
            ZE2UR_CALL(zeCommandListDestroy, (ZeCommandList));
          }
        }
      }
    }
    Queue->CommandListMap.clear();
  }

  for (auto &Event : EventListToCleanup) {
    // We don't need to synchronize the events since the queue
    // synchronized above already does that.
    {
      std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
      Event->Completed = true;
    }
    UR_CALL(CleanupCompletedEvent(Event, false /*QueueLocked*/,
                                  false /*SetEventCompleted*/));
    // This event was removed from the command list, so decrement ref count
    // (it was incremented when they were added to the command list).
    UR_CALL(urEventReleaseInternal(reinterpret_cast<ur_event_handle_t>(Event)));
  }
  UR_CALL(urQueueReleaseInternal(Queue));
  return UR_RESULT_SUCCESS;
}

ur_result_t urQueueGetNativeHandle(
    /// [in] handle of the queue.
    ur_queue_handle_t Queue, ur_queue_native_desc_t *Desc,
    /// [out] a pointer to the native handle of the queue.
    ur_native_handle_t *NativeQueue) {
  // Lock automatically releases when this goes out of scope.
  std::shared_lock<ur_shared_mutex> lock(Queue->Mutex);

  int32_t NativeHandleDesc{};

  // Get handle to this thread's queue group.
  auto &QueueGroup = Queue->getQueueGroup(false /*compute*/);

  if (Queue->UsingImmCmdLists) {
    auto ZeCmdList = ur_cast<ze_command_list_handle_t *>(NativeQueue);
    // Extract the Level Zero command list handle from the given PI queue
    *ZeCmdList = QueueGroup.getImmCmdList()->first;
    // TODO: How to pass this up in the urQueueGetNativeHandle interface?
    NativeHandleDesc = true;
  } else {
    auto ZeQueue = ur_cast<ze_command_queue_handle_t *>(NativeQueue);

    // Extract a Level Zero compute queue handle from the given PI queue
    auto &QueueGroup = Queue->getQueueGroup(false /*compute*/);
    uint32_t QueueGroupOrdinalUnused;
    *ZeQueue = QueueGroup.getZeQueue(&QueueGroupOrdinalUnused);
    // TODO: How to pass this up in the urQueueGetNativeHandle interface?
    NativeHandleDesc = false;
  }

  if (Desc && Desc->pNativeData)
    *(reinterpret_cast<int32_t *>((Desc->pNativeData))) = NativeHandleDesc;

  return UR_RESULT_SUCCESS;
}

ur_result_t urQueueCreateWithNativeHandle(
    /// [in] the native handle of the queue.
    ur_native_handle_t NativeQueue,
    /// [in] handle of the context object
    ur_context_handle_t Context,
    ur_device_handle_t Device, ///
    const ur_queue_native_properties_t
        *NativeProperties, ///
                           /// [out] pointer to the handle of the queue object
                           /// created.
    ur_queue_handle_t *RetQueue) {
  bool OwnNativeHandle = false;
  ur_queue_flags_t Flags{};
  int32_t NativeHandleDesc{};

  if (NativeProperties) {
    OwnNativeHandle = NativeProperties->isNativeHandleOwned;
    void *pNext = NativeProperties->pNext;
    while (pNext) {
      const ur_base_properties_t *extendedProperties =
          reinterpret_cast<const ur_base_properties_t *>(pNext);
      if (extendedProperties->stype == UR_STRUCTURE_TYPE_QUEUE_PROPERTIES) {
        const ur_queue_properties_t *UrProperties =
            reinterpret_cast<const ur_queue_properties_t *>(extendedProperties);
        Flags = UrProperties->flags;
      } else if (extendedProperties->stype ==
                 UR_STRUCTURE_TYPE_QUEUE_NATIVE_DESC) {
        const ur_queue_native_desc_t *UrNativeDesc =
            reinterpret_cast<const ur_queue_native_desc_t *>(
                extendedProperties);
        if (UrNativeDesc->pNativeData)
          NativeHandleDesc =
              *(reinterpret_cast<int32_t *>((UrNativeDesc->pNativeData)));
      }
      pNext = extendedProperties->pNext;
    }
  }

  // Get the device handle from first device in the platform
  // Maybe this is not completely correct.
  uint32_t NumEntries = 1;
  ur_platform_handle_t Platform{};
  ur_adapter_handle_t AdapterHandle = GlobalAdapter;
  UR_CALL(ur::level_zero::urPlatformGet(&AdapterHandle, 1, NumEntries,
                                        &Platform, nullptr));

  ur_device_handle_t UrDevice = Device;
  if (UrDevice == nullptr) {
    UR_CALL(ur::level_zero::urDeviceGet(Platform, UR_DEVICE_TYPE_GPU,
                                        NumEntries, &UrDevice, nullptr));
  }

  // The NativeHandleDesc has value if if the native handle is an immediate
  // command list.
  if (NativeHandleDesc == 1) {
    std::vector<ze_command_queue_handle_t> ComputeQueues{nullptr};
    std::vector<ze_command_queue_handle_t> CopyQueues;

    try {
      ur_queue_handle_t_ *Queue = new ur_queue_handle_t_(
          ComputeQueues, CopyQueues, Context, UrDevice, OwnNativeHandle, Flags);
      *RetQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
    auto &InitialGroup = (*RetQueue)->ComputeQueueGroupsByTID.begin()->second;
    InitialGroup.setImmCmdList(*RetQueue,
                               ur_cast<ze_command_list_handle_t>(NativeQueue));
  } else {
    auto ZeQueue = ur_cast<ze_command_queue_handle_t>(NativeQueue);
    // Assume this is the "0" index queue in the compute command-group.
    std::vector<ze_command_queue_handle_t> ZeQueues{ZeQueue};

    // TODO: see what we can do to correctly initialize PI queue for
    // compute vs. copy Level-Zero queue. Currently we will send
    // all commands to the "ZeQueue".
    std::vector<ze_command_queue_handle_t> ZeroCopyQueues;

    try {
      ur_queue_handle_t_ *Queue = new ur_queue_handle_t_(
          ZeQueues, ZeroCopyQueues, Context, UrDevice, OwnNativeHandle, Flags);
      *RetQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }
  (*RetQueue)->UsingImmCmdLists = (NativeHandleDesc == 1);

  return UR_RESULT_SUCCESS;
}

ur_result_t urQueueFinish(
    /// [in] handle of the queue to be finished.
    ur_queue_handle_t Queue) {
  if (Queue->UsingImmCmdLists) {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);

    UR_CALL(Queue->synchronize());
  } else {
    std::unique_lock<ur_shared_mutex> Lock(Queue->Mutex);
    std::vector<ze_command_queue_handle_t> ZeQueues;

    // execute any command list that may still be open.
    UR_CALL(Queue->executeAllOpenCommandLists());

    // Make a copy of queues to sync and release the lock.
    for (auto &QueueMap :
         {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
      for (auto &QueueGroup : QueueMap)
        std::copy(QueueGroup.second.ZeQueues.begin(),
                  QueueGroup.second.ZeQueues.end(),
                  std::back_inserter(ZeQueues));

    // Remember the last command's event.
    auto LastCommandEvent = Queue->LastCommandEvent;

    // Don't hold a lock to the queue's mutex while waiting.
    // This allows continue working with the queue from other threads.
    // TODO: this currently exhibits some issues in the driver, so
    // we control this with an env var. Remove this control when
    // we settle one way or the other.
    const char *UrRet = std::getenv("UR_L0_QUEUE_FINISH_HOLD_LOCK");
    const char *PiRet =
        std::getenv("SYCL_PI_LEVEL_ZERO_QUEUE_FINISH_HOLD_LOCK");
    static bool HoldLock =
        UrRet ? std::stoi(UrRet) : (PiRet ? std::stoi(PiRet) : 0);
    if (!HoldLock) {
      Lock.unlock();
    }

    for (auto &ZeQueue : ZeQueues) {
      if (ZeQueue)
        ZE2UR_CALL(zeHostSynchronize, (ZeQueue));
    }

    // Prevent unneeded already finished events to show up in the wait list.
    // We can only do so if nothing else was submitted to the queue
    // while we were synchronizing it.
    if (!HoldLock) {
      std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);
      if (LastCommandEvent == Queue->LastCommandEvent) {
        Queue->LastCommandEvent = nullptr;
      }
    } else {
      Queue->LastCommandEvent = nullptr;
    }
  }
  // Reset signalled command lists and return them back to the cache of
  // available command lists. Events in the immediate command lists are cleaned
  // up in synchronize().
  if (!Queue->UsingImmCmdLists) {
    std::unique_lock<ur_shared_mutex> Lock(Queue->Mutex);
    resetCommandLists(Queue);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urQueueFlush(
    /// [in] handle of the queue to be flushed.
    ur_queue_handle_t Queue) {
  std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);
  return Queue->executeAllOpenCommandLists();
}

ur_result_t urEnqueueKernelLaunchCustomExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_exp_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hKernel;
  std::ignore = workDim;
  std::ignore = pGlobalWorkOffset;
  std::ignore = pGlobalWorkSize;
  std::ignore = pLocalWorkSize;
  std::ignore = numPropsInLaunchPropList;
  std::ignore = launchPropList;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  logger::error("[UR][L0] {} function not implemented!",
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero

// Configuration of the command-list batching.
struct zeCommandListBatchConfig {
  // Default value of 0. This specifies to use dynamic batch size adjustment.
  // Other values will try to collect specified amount of commands.
  uint32_t Size{0};

  // If doing dynamic batching, specifies start batch size.
  uint32_t DynamicSizeStart{4};

  // The maximum size for dynamic batch.
  uint32_t DynamicSizeMax{64};

  // The step size for dynamic batch increases.
  uint32_t DynamicSizeStep{1};

  // Thresholds for when increase batch size (number of closed early is small
  // and number of closed full is high).
  uint32_t NumTimesClosedEarlyThreshold{3};
  uint32_t NumTimesClosedFullThreshold{8};

  // Tells the starting size of a batch.
  uint32_t startSize() const { return Size > 0 ? Size : DynamicSizeStart; }
  // Tells is we are doing dynamic batch size adjustment.
  bool dynamic() const { return Size == 0; }
};

// Helper function to initialize static variables that holds batch config info
// for compute and copy command batching.
static const zeCommandListBatchConfig ZeCommandListBatchConfig(bool IsCopy) {
  zeCommandListBatchConfig Config{}; // default initialize

  // Default value of 0. This specifies to use dynamic batch size adjustment.
  const char *UrRet = nullptr;
  const char *PiRet = nullptr;
  if (IsCopy) {
    UrRet = std::getenv("UR_L0_COPY_BATCH_SIZE");
    PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE");
  } else {
    UrRet = std::getenv("UR_L0_BATCH_SIZE");
    PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_BATCH_SIZE");
  }
  const char *BatchSizeStr = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  if (BatchSizeStr) {
    int32_t BatchSizeStrVal = std::atoi(BatchSizeStr);
    // Level Zero may only support a limted number of commands per command
    // list.  The actual upper limit is not specified by the Level Zero
    // Specification.  For now we allow an arbitrary upper limit.
    if (BatchSizeStrVal > 0) {
      Config.Size = BatchSizeStrVal;
    } else if (BatchSizeStrVal == 0) {
      Config.Size = 0;
      // We are requested to do dynamic batching. Collect specifics, if any.
      // The extended format supported is ":" separated values.
      //
      // NOTE: these extra settings are experimental and are intended to
      // be used only for finding a better default heuristic.
      //
      std::string BatchConfig(BatchSizeStr);
      size_t Ord = 0;
      size_t Pos = 0;
      while (true) {
        if (++Ord > 5)
          break;

        Pos = BatchConfig.find(":", Pos);
        if (Pos == std::string::npos)
          break;
        ++Pos; // past the ":"

        uint32_t Val;
        try {
          Val = std::stoi(BatchConfig.substr(Pos));
        } catch (...) {
          if (IsCopy)
            logger::error("UR_L0_COPY_BATCH_SIZE: failed to parse value");
          else
            logger::error("UR_L0_BATCH_SIZE: failed to parse value");
          break;
        }
        switch (Ord) {
        case 1:
          Config.DynamicSizeStart = Val;
          break;
        case 2:
          Config.DynamicSizeMax = Val;
          break;
        case 3:
          Config.DynamicSizeStep = Val;
          break;
        case 4:
          Config.NumTimesClosedEarlyThreshold = Val;
          break;
        case 5:
          Config.NumTimesClosedFullThreshold = Val;
          break;
        default:
          die("Unexpected batch config");
        }
        if (IsCopy)
          logger::error("UR_L0_COPY_BATCH_SIZE: dynamic batch param "
                        "#{}: {}",
                        (int)Ord, (int)Val);
        else
          logger::error("UR_L0_BATCH_SIZE: dynamic batch param #{}: {}",
                        (int)Ord, (int)Val);
      };

    } else {
      // Negative batch sizes are silently ignored.
      if (IsCopy)
        logger::warning("UR_L0_COPY_BATCH_SIZE: ignored negative value");
      else
        logger::warning("UR_L0_BATCH_SIZE: ignored negative value");
    }
  }
  return Config;
}

// UR_L0_LEVEL_ZERO_USE_COMPUTE_ENGINE can be set to an integer (>=0) in
// which case all compute commands will be submitted to the command-queue
// with the given index in the compute command group. If it is instead set
// to negative then all available compute engines may be used.
//
// The default value is "0".
//
static const std::pair<int, int> getRangeOfAllowedComputeEngines() {
  const char *UrRet = std::getenv("UR_L0_USE_COMPUTE_ENGINE");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE");
  const char *EnvVar = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  // If the environment variable is not set only use "0" CCS for now.
  // TODO: allow all CCSs when HW support is complete.
  if (!EnvVar)
    return std::pair<int, int>(0, 0);

  auto EnvVarValue = std::atoi(EnvVar);
  if (EnvVarValue >= 0) {
    return std::pair<int, int>(EnvVarValue, EnvVarValue);
  }

  return std::pair<int, int>(0, INT_MAX);
}

// Static variable that holds batch config info for compute command batching.
static const zeCommandListBatchConfig ZeCommandListBatchComputeConfig = [] {
  using IsCopy = bool;
  return ZeCommandListBatchConfig(IsCopy{false});
}();

// Static variable that holds batch config info for copy command batching.
static const zeCommandListBatchConfig ZeCommandListBatchCopyConfig = [] {
  using IsCopy = bool;
  return ZeCommandListBatchConfig(IsCopy{true});
}();

ur_queue_handle_t_::ur_queue_handle_t_(
    std::vector<ze_command_queue_handle_t> &ComputeQueues,
    std::vector<ze_command_queue_handle_t> &CopyQueues,
    ur_context_handle_t Context, ur_device_handle_t Device,
    bool OwnZeCommandQueue, ur_queue_flags_t Properties, int ForceComputeIndex)
    : Context{Context}, Device{Device}, OwnZeCommandQueue{OwnZeCommandQueue},
      Properties(Properties) {
  // Set the type of commandlists the queue will use when user-selected
  // submission mode. Otherwise use env var setting and if unset, use default.
  if (isBatchedSubmission())
    UsingImmCmdLists = false;
  else if (isImmediateSubmission())
    UsingImmCmdLists = true;
  else
    UsingImmCmdLists = Device->useImmediateCommandLists();

  // Set events scope for this queue. Non-immediate can be controlled by env
  // var. Immediate always uses AllHostVisible.
  if (!UsingImmCmdLists) {
    ZeEventsScope = DeviceEventsSetting;
  }

  // Compute group initialization.
  // First, see if the queue's device allows for round-robin or it is
  // fixed to one particular compute CCS (it is so for sub-sub-devices).
  auto &ComputeQueueGroupInfo = Device->QueueGroup[queue_type::Compute];
  ur_queue_group_t ComputeQueueGroup{reinterpret_cast<ur_queue_handle_t>(this),
                                     queue_type::Compute};
  ComputeQueueGroup.ZeQueues = ComputeQueues;
  // Create space to hold immediate commandlists corresponding to the
  // ZeQueues
  if (UsingImmCmdLists) {
    ComputeQueueGroup.ImmCmdLists = std::vector<ur_command_list_ptr_t>(
        ComputeQueueGroup.ZeQueues.size(), CommandListMap.end());
  }
  if (ComputeQueueGroupInfo.ZeIndex >= 0) {
    // Sub-sub-device

    // sycl::ext::intel::property::queue::compute_index works with any
    // backend/device by allowing single zero index if multiple compute CCSes
    // are not supported. Sub-sub-device falls into the same bucket.
    assert(ForceComputeIndex <= 0);
    ComputeQueueGroup.LowerIndex = ComputeQueueGroupInfo.ZeIndex;
    ComputeQueueGroup.UpperIndex = ComputeQueueGroupInfo.ZeIndex;
    ComputeQueueGroup.NextIndex = ComputeQueueGroupInfo.ZeIndex;
  } else if (ForceComputeIndex >= 0) {
    ComputeQueueGroup.LowerIndex = ForceComputeIndex;
    ComputeQueueGroup.UpperIndex = ForceComputeIndex;
    ComputeQueueGroup.NextIndex = ForceComputeIndex;
  } else {
    // Set-up to round-robin across allowed range of engines.
    uint32_t FilterLowerIndex = getRangeOfAllowedComputeEngines().first;
    uint32_t FilterUpperIndex = getRangeOfAllowedComputeEngines().second;
    FilterUpperIndex = (std::min)((size_t)FilterUpperIndex,
                                  FilterLowerIndex + ComputeQueues.size() - 1);
    if (FilterLowerIndex <= FilterUpperIndex) {
      ComputeQueueGroup.LowerIndex = FilterLowerIndex;
      ComputeQueueGroup.UpperIndex = FilterUpperIndex;
      ComputeQueueGroup.NextIndex = ComputeQueueGroup.LowerIndex;
    } else {
      die("No compute queue available/allowed.");
    }
  }
  if (UsingImmCmdLists) {
    // Create space to hold immediate commandlists corresponding to the
    // ZeQueues
    ComputeQueueGroup.ImmCmdLists = std::vector<ur_command_list_ptr_t>(
        ComputeQueueGroup.ZeQueues.size(), CommandListMap.end());
  }

  ComputeQueueGroupsByTID.set(ComputeQueueGroup);

  // Copy group initialization.
  ur_queue_group_t CopyQueueGroup{reinterpret_cast<ur_queue_handle_t>(this),
                                  queue_type::MainCopy};
  const auto &Range = getRangeOfAllowedCopyEngines((ur_device_handle_t)Device);
  if (Range.first < 0 || Range.second < 0) {
    // We are asked not to use copy engines, just do nothing.
    // Leave CopyQueueGroup.ZeQueues empty, and it won't be used.
  } else {
    uint32_t FilterLowerIndex = Range.first;
    uint32_t FilterUpperIndex = Range.second;
    FilterUpperIndex = (std::min)((size_t)FilterUpperIndex,
                                  FilterLowerIndex + CopyQueues.size() - 1);
    if (FilterLowerIndex <= FilterUpperIndex) {
      CopyQueueGroup.ZeQueues = CopyQueues;
      CopyQueueGroup.LowerIndex = FilterLowerIndex;
      CopyQueueGroup.UpperIndex = FilterUpperIndex;
      CopyQueueGroup.NextIndex = CopyQueueGroup.LowerIndex;
      // Create space to hold immediate commandlists corresponding to the
      // ZeQueues
      if (UsingImmCmdLists) {
        CopyQueueGroup.ImmCmdLists = std::vector<ur_command_list_ptr_t>(
            CopyQueueGroup.ZeQueues.size(), CommandListMap.end());
      }
    }
  }
  CopyQueueGroupsByTID.set(CopyQueueGroup);

  // Initialize compute/copy command batches.
  ComputeCommandBatch.OpenCommandList = CommandListMap.end();
  CopyCommandBatch.OpenCommandList = CommandListMap.end();
  ComputeCommandBatch.QueueBatchSize =
      ZeCommandListBatchComputeConfig.startSize();
  CopyCommandBatch.QueueBatchSize = ZeCommandListBatchCopyConfig.startSize();

  this->CounterBasedEventsEnabled =
      UsingImmCmdLists && isInOrderQueue() && Device->useDriverInOrderLists() &&
      Device->useDriverCounterBasedEvents() &&
      Device->Platform->ZeDriverEventPoolCountingEventsExtensionFound;
  this->InterruptBasedEventsEnabled =
      isLowPowerEvents() && isInOrderQueue() && Device->useDriverInOrderLists();
}

void ur_queue_handle_t_::adjustBatchSizeForFullBatch(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  auto &ZeCommandListBatchConfig =
      IsCopy ? ZeCommandListBatchCopyConfig : ZeCommandListBatchComputeConfig;
  uint32_t &QueueBatchSize = CommandBatch.QueueBatchSize;
  // QueueBatchSize of 0 means never allow batching.
  if (QueueBatchSize == 0 || !ZeCommandListBatchConfig.dynamic())
    return;
  CommandBatch.NumTimesClosedFull += 1;

  // If the number of times the list has been closed early is low, and
  // the number of times it has been closed full is high, then raise
  // the batching size slowly. Don't raise it if it is already pretty
  // high.
  if (CommandBatch.NumTimesClosedEarly <=
          ZeCommandListBatchConfig.NumTimesClosedEarlyThreshold &&
      CommandBatch.NumTimesClosedFull >
          ZeCommandListBatchConfig.NumTimesClosedFullThreshold) {
    if (QueueBatchSize < ZeCommandListBatchConfig.DynamicSizeMax) {
      QueueBatchSize += ZeCommandListBatchConfig.DynamicSizeStep;
      logger::debug("Raising QueueBatchSize to {}", QueueBatchSize);
    }
    CommandBatch.NumTimesClosedEarly = 0;
    CommandBatch.NumTimesClosedFull = 0;
  }
}

void ur_queue_handle_t_::adjustBatchSizeForPartialBatch(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  auto &ZeCommandListBatchConfig =
      IsCopy ? ZeCommandListBatchCopyConfig : ZeCommandListBatchComputeConfig;
  uint32_t &QueueBatchSize = CommandBatch.QueueBatchSize;
  // QueueBatchSize of 0 means never allow batching.
  if (QueueBatchSize == 0 || !ZeCommandListBatchConfig.dynamic())
    return;
  CommandBatch.NumTimesClosedEarly += 1;

  // If we are closing early more than about 3x the number of times
  // it is closing full, lower the batch size to the value of the
  // current open command list. This is trying to quickly get to a
  // batch size that will be able to be closed full at least once
  // in a while.
  if (CommandBatch.NumTimesClosedEarly >
      (CommandBatch.NumTimesClosedFull + 1) * 3) {
    QueueBatchSize = CommandBatch.OpenCommandList->second.size() - 1;
    if (QueueBatchSize < 1)
      QueueBatchSize = 1;
    logger::debug("Lowering QueueBatchSize to {}", QueueBatchSize);
    CommandBatch.NumTimesClosedEarly = 0;
    CommandBatch.NumTimesClosedFull = 0;
  }
}

ur_result_t
ur_queue_handle_t_::executeCommandList(ur_command_list_ptr_t CommandList,
                                       bool IsBlocking, bool OKToBatchCommand) {
  // Do nothing if command list is already closed.
  if (CommandList->second.IsClosed)
    return UR_RESULT_SUCCESS;

  bool UseCopyEngine =
      CommandList->second.isCopy(reinterpret_cast<ur_queue_handle_t>(this));

  // If the current LastCommandEvent is the nullptr, then it means
  // either that no command has ever been issued to the queue
  // or it means that the LastCommandEvent has been signalled and
  // therefore that this Queue is idle.
  //
  // NOTE: this behavior adds some flakyness to the batching
  // since last command's event may or may not be completed by the
  // time we get here depending on timings and system/gpu load.
  // So, disable it for modes where we print PI traces. Printing
  // traces incurs much different timings than real execution
  // ansyway, and many regression tests use it.
  //
  bool CurrentlyEmpty = !PrintTrace && this->LastCommandEvent == nullptr;

  // The list can be empty if command-list only contains signals of proxy
  // events. It is possible that executeCommandList is called twice for the same
  // command list without new appended command. We don't to want process the
  // same last command event twice that's why additionally check that new
  // command was appended to the command list.
  if (!CommandList->second.EventList.empty() &&
      this->LastCommandEvent != CommandList->second.EventList.back()) {
    this->LastCommandEvent = CommandList->second.EventList.back();
    if (doReuseDiscardedEvents()) {
      UR_CALL(resetDiscardedEvent(CommandList));
    }
  }

  this->LastUsedCommandList = CommandList;

  if (!UsingImmCmdLists) {
    // Batch if allowed to, but don't batch if we know there are no kernels
    // from this queue that are currently executing.  This is intended to get
    // kernels started as soon as possible when there are no kernels from this
    // queue awaiting execution, while allowing batching to occur when there
    // are kernels already executing. Also, if we are using fixed size batching,
    // as indicated by !ZeCommandListBatch.dynamic(), then just ignore
    // CurrentlyEmpty as we want to strictly follow the batching the user
    // specified.
    auto &CommandBatch = UseCopyEngine ? CopyCommandBatch : ComputeCommandBatch;
    auto &ZeCommandListBatchConfig = UseCopyEngine
                                         ? ZeCommandListBatchCopyConfig
                                         : ZeCommandListBatchComputeConfig;
    if (OKToBatchCommand && this->isBatchingAllowed(UseCopyEngine) &&
        (!ZeCommandListBatchConfig.dynamic() || !CurrentlyEmpty)) {

      if (hasOpenCommandList(UseCopyEngine) &&
          CommandBatch.OpenCommandList != CommandList)
        die("executeCommandList: OpenCommandList should be equal to"
            "null or CommandList");

      if (CommandList->second.size() < CommandBatch.QueueBatchSize) {
        CommandBatch.OpenCommandList = CommandList;
        return UR_RESULT_SUCCESS;
      }

      adjustBatchSizeForFullBatch(UseCopyEngine);
      CommandBatch.OpenCommandList = CommandListMap.end();
    }
  }

  auto &ZeCommandQueue = CommandList->second.ZeQueue;
  // Scope of the lock must be till the end of the function, otherwise new mem
  // allocs can be created between the moment when we made a snapshot and the
  // moment when command list is closed and executed. But mutex is locked only
  // if indirect access tracking enabled, because std::defer_lock is used.
  // unique_lock destructor at the end of the function will unlock the mutex
  // if it was locked (which happens only if IndirectAccessTrackingEnabled is
  // true).
  std::unique_lock<ur_shared_mutex> ContextsLock(
      Device->Platform->ContextsMutex, std::defer_lock);

  if (IndirectAccessTrackingEnabled) {
    // We are going to submit kernels for execution. If indirect access flag is
    // set for a kernel then we need to make a snapshot of existing memory
    // allocations in all contexts in the platform. We need to lock the mutex
    // guarding the list of contexts in the platform to prevent creation of new
    // memory alocations in any context before we submit the kernel for
    // execution.
    ContextsLock.lock();
    CaptureIndirectAccesses();
  }

  if (!UsingImmCmdLists) {
    // In this mode all inner-batch events have device visibility only,
    // and we want the last command in the batch to signal a host-visible
    // event that anybody waiting for any event in the batch will
    // really be using.
    // We need to create a proxy host-visible event only if the list of events
    // in the command list is not empty, otherwise we are going to just create
    // and remove proxy event right away and dereference deleted object
    // afterwards.
    bool AppendBarrierNeeded = true;
    if (ZeEventsScope == LastCommandInBatchHostVisible &&
        !CommandList->second.EventList.empty()) {
      // If there are only internal events in the command list then we don't
      // need to create host proxy event.
      auto Result = std::find_if(
          CommandList->second.EventList.begin(),
          CommandList->second.EventList.end(),
          [](ur_event_handle_t E) { return E->hasExternalRefs(); });
      if (Result != CommandList->second.EventList.end()) {
        // Create a "proxy" host-visible event.
        //
        ur_event_handle_t HostVisibleEvent;
        auto Res = createEventAndAssociateQueue(
            reinterpret_cast<ur_queue_handle_t>(this), &HostVisibleEvent,
            UR_EXT_COMMAND_TYPE_USER, CommandList,
            /* IsInternal */ false, /* IsMultiDevice */ true,
            /* HostVisible */ true);
        if (Res)
          return Res;

        // Update each command's event in the command-list to "see" this
        // proxy event as a host-visible counterpart.
        for (auto &Event : CommandList->second.EventList) {
          std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
          // Internal event doesn't need host-visible proxy.
          if (!Event->hasExternalRefs())
            continue;

          if (!Event->HostVisibleEvent) {
            Event->HostVisibleEvent =
                reinterpret_cast<ur_event_handle_t>(HostVisibleEvent);
            HostVisibleEvent->RefCount.increment();
          }
        }

        // Decrement the reference count of the event such that all the
        // remaining references are from the other commands in this batch and
        // from the command-list itself. This host-visible event will not be
        // waited/released by SYCL RT, so it must be destroyed after all events
        // in the batch are gone. We know that refcount is more than 2 because
        // we check that EventList of the command list is not empty above, i.e.
        // after createEventAndAssociateQueue ref count is 2 and then +1 for
        // each event in the EventList.
        UR_CALL(urEventReleaseInternal(HostVisibleEvent));

        if (doReuseDiscardedEvents()) {
          // If we have in-order queue with discarded events then we want to
          // treat this event as regular event. We insert a barrier in the next
          // command list to wait for this event.
          LastCommandEvent = HostVisibleEvent;
        } else {
          // For all other queues treat this as a special event and indicate no
          // cleanup is needed.
          // TODO: always treat this host event as a regular event.
          UR_CALL(urEventReleaseInternal(HostVisibleEvent));
          HostVisibleEvent->CleanedUp = true;
        }

        // Finally set to signal the host-visible event at the end of the
        // command-list after a barrier that waits for all commands
        // completion.
        if (doReuseDiscardedEvents() && LastCommandEvent &&
            LastCommandEvent->IsDiscarded) {
          // If we the last event is discarded then we already have a barrier
          // inserted, so just signal the event.
          ZE2UR_CALL(zeCommandListAppendSignalEvent,
                     (CommandList->first, HostVisibleEvent->ZeEvent));
        } else {
          AppendBarrierNeeded = false;
          ZE2UR_CALL(
              zeCommandListAppendBarrier,
              (CommandList->first, HostVisibleEvent->ZeEvent, 0, nullptr));
        }
      } else {
        // If we don't have host visible proxy then signal event if needed.
        this->signalEventFromCmdListIfLastEventDiscarded(CommandList);
      }
    } else {
      // If we don't have host visible proxy then signal event if needed.
      this->signalEventFromCmdListIfLastEventDiscarded(CommandList);
    }
    // Append Signalling of the inner events at the end of the batch if this is
    // an integrated gpu and out of order signal events are not allowed.
    if (!UrL0OutOfOrderIntegratedSignalEvent && this->Device->isIntegrated()) {
      for (auto &Event : CommandList->second.EventList) {
        // If the events scope does not apply a barrier already above, then we
        // need to apply a barrier to wait on all the previous commands without
        // signal events to complete before we can signal the batched events as
        // completed. This functionality is only used if this command list is
        // out of order and there are events created that were not used as
        // signal events.
        if (Event->IsInnerBatchedEvent) {
          if (AppendBarrierNeeded) {
            ZE2UR_CALL(zeCommandListAppendBarrier,
                       (CommandList->first, nullptr, 0, nullptr));
            AppendBarrierNeeded = false;
          }
          ZE2UR_CALL(zeCommandListAppendSignalEvent,
                     (CommandList->first, Event->ZeEvent));
        }
      }
    }

    // Close the command list and have it ready for dispatch.
    ZE2UR_CALL(zeCommandListClose, (CommandList->first));
    // Mark this command list as closed.
    CommandList->second.IsClosed = true;
    this->LastUsedCommandList = CommandListMap.end();
    // Offload command list to the GPU for asynchronous execution
    auto ZeCommandList = CommandList->first;
    auto ZeResult = ZE_CALL_NOCHECK(
        zeCommandQueueExecuteCommandLists,
        (ZeCommandQueue, 1, &ZeCommandList, CommandList->second.ZeFence));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      this->Healthy = false;
      // Reset Command List and erase the Fence forcing the user to resubmit
      // their commands.
      std::vector<ur_event_handle_t> EventListToCleanup;
      resetCommandList(CommandList, true, EventListToCleanup, false);
      CleanupEventListFromResetCmdList(EventListToCleanup,
                                       true /* QueueLocked */);
      return ze2urResult(ZeResult);
    }
  }

  // Check global control to make every command blocking for debugging.
  if (IsBlocking || (UrL0Serialize & UrL0SerializeBlock) != 0) {
    if (UsingImmCmdLists) {
      UR_CALL(synchronize());
    } else {
      // Wait until command lists attached to the command queue are executed.
      ZE2UR_CALL(zeHostSynchronize, (ZeCommandQueue));
    }
  }
  return UR_RESULT_SUCCESS;
}

bool ur_queue_handle_t_::doReuseDiscardedEvents() {
  return ReuseDiscardedEvents && isInOrderQueue() && isDiscardEvents();
}

ur_result_t
ur_queue_handle_t_::resetDiscardedEvent(ur_command_list_ptr_t CommandList) {
  if (LastCommandEvent && LastCommandEvent->IsDiscarded) {
    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CommandList->first, nullptr, 1, &(LastCommandEvent->ZeEvent)));
    if (!CounterBasedEventsEnabled) {
      ZE2UR_CALL(zeCommandListAppendEventReset,
                 (CommandList->first, LastCommandEvent->ZeEvent));
    }

    // Create new ur_event_handle_t but with the same ze_event_handle_t. We are
    // going to use this ur_event_handle_t for the next command with discarded
    // event.
    ur_event_handle_t_ *UREvent;
    try {
      UREvent = new ur_event_handle_t_(
          LastCommandEvent->ZeEvent, LastCommandEvent->ZeEventPool,
          reinterpret_cast<ur_context_handle_t>(Context),
          UR_EXT_COMMAND_TYPE_USER, true);
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }

    if (LastCommandEvent->isHostVisible())
      UREvent->HostVisibleEvent = reinterpret_cast<ur_event_handle_t>(UREvent);

    UR_CALL(addEventToQueueCache(reinterpret_cast<ur_event_handle_t>(UREvent)));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_handle_t_::addEventToQueueCache(ur_event_handle_t Event) {
  if (!Event->IsMultiDevice) {
    auto EventCachesMap = Event->isHostVisible() ? &EventCachesDeviceMap[0]
                                                 : &EventCachesDeviceMap[1];
    if (EventCachesMap->find(Device) == EventCachesMap->end()) {
      EventCaches.emplace_back();
      EventCachesMap->insert(std::make_pair(Device, EventCaches.size() - 1));
    }
    EventCaches[EventCachesMap->at(Device)].emplace_back(Event);
  } else {
    auto Cache = Event->isHostVisible() ? &EventCaches[0] : &EventCaches[1];
    Cache->emplace_back(Event);
  }
  return UR_RESULT_SUCCESS;
}

void ur_queue_handle_t_::active_barriers::add(ur_event_handle_t &Event) {
  Event->RefCount.increment();
  Events.push_back(Event);
}

ur_result_t ur_queue_handle_t_::active_barriers::clear() {
  for (const auto &Event : Events)
    UR_CALL(urEventReleaseInternal(Event));
  Events.clear();
  return UR_RESULT_SUCCESS;
}

void ur_queue_handle_t_::clearEndTimeRecordings() {
  uint64_t ZeTimerResolution = Device->ZeDeviceProperties->timerResolution;
  const uint64_t TimestampMaxValue = Device->getTimestampMask();

  for (auto Entry : EndTimeRecordings) {
    auto &Event = Entry.first;
    auto &EndTimeRecording = Entry.second;

    // Write the result back to the event if it is not dead.
    uint64_t ContextEndTime =
        (EndTimeRecording & TimestampMaxValue) * ZeTimerResolution;

    // Handle a possible wrap-around (the underlying HW counter is < 64-bit).
    // Note, it will not report correct time if there were multiple wrap
    // arounds, and the longer term plan is to enlarge the capacity of the
    // HW timestamps.
    if (ContextEndTime < Event->RecordEventStartTimestamp)
      ContextEndTime += TimestampMaxValue * ZeTimerResolution;

    // Store it in the event.
    Event->RecordEventEndTimestamp = ContextEndTime;
  }
  EndTimeRecordings.clear();
  EvictedEndTimeRecordings.clear();
}

ur_result_t urQueueReleaseInternal(ur_queue_handle_t Queue) {
  if (!Queue->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  for (auto &Cache : Queue->EventCaches) {
    for (auto &Event : Cache)
      UR_CALL(urEventReleaseInternal(Event));
    Cache.clear();
  }

  if (Queue->OwnZeCommandQueue) {
    for (auto &QueueMap :
         {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
      for (auto &QueueGroup : QueueMap)
        for (auto &ZeQueue : QueueGroup.second.ZeQueues)
          if (ZeQueue) {
            auto ZeResult = ZE_CALL_NOCHECK(zeCommandQueueDestroy, (ZeQueue));
            // Gracefully handle the case that L0 was already unloaded.
            if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
              return ze2urResult(ZeResult);
          }
  }

  Queue->clearEndTimeRecordings();

  logger::debug("urQueueRelease(compute) NumTimesClosedFull {}, "
                "NumTimesClosedEarly {}",
                Queue->ComputeCommandBatch.NumTimesClosedFull,
                Queue->ComputeCommandBatch.NumTimesClosedEarly);
  logger::debug(
      "urQueueRelease(copy) NumTimesClosedFull {}, NumTimesClosedEarly {}",
      Queue->CopyCommandBatch.NumTimesClosedFull,
      Queue->CopyCommandBatch.NumTimesClosedEarly);

  delete Queue;

  return UR_RESULT_SUCCESS;
}

bool ur_queue_handle_t_::isBatchingAllowed(bool IsCopy) const {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  return (CommandBatch.QueueBatchSize > 0 &&
          ((UrL0Serialize & UrL0SerializeBlock) == 0));
}

bool ur_queue_handle_t_::isDiscardEvents() const {
  return ((this->Properties & UR_QUEUE_FLAG_DISCARD_EVENTS) != 0);
}

bool ur_queue_handle_t_::isPriorityLow() const {
  return ((this->Properties & UR_QUEUE_FLAG_PRIORITY_LOW) != 0);
}

bool ur_queue_handle_t_::isPriorityHigh() const {
  return ((this->Properties & UR_QUEUE_FLAG_PRIORITY_HIGH) != 0);
}

bool ur_queue_handle_t_::isBatchedSubmission() const {
  return ((this->Properties & UR_QUEUE_FLAG_SUBMISSION_BATCHED) != 0);
}

bool ur_queue_handle_t_::isImmediateSubmission() const {
  return ((this->Properties & UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE) != 0);
}

bool ur_queue_handle_t_::isInOrderQueue() const {
  // If out-of-order queue property is not set, then this is a in-order queue.
  return ((this->Properties & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) ==
          0);
}

bool ur_queue_handle_t_::isLowPowerEvents() const {
  return ((this->Properties & UR_QUEUE_FLAG_LOW_POWER_EVENTS_EXP) != 0);
}

// Helper function to perform the necessary cleanup of the events from reset cmd
// list.
ur_result_t CleanupEventListFromResetCmdList(
    std::vector<ur_event_handle_t> &EventListToCleanup, bool QueueLocked) {
  for (auto &Event : EventListToCleanup) {
    // We don't need to synchronize the events since the fence associated with
    // the command list was synchronized.
    UR_CALL(
        CleanupCompletedEvent(Event, QueueLocked, true /*SetEventCompleted*/));
    // This event was removed from the command list, so decrement ref count
    // (it was incremented when they were added to the command list).
    UR_CALL(urEventReleaseInternal(Event));
  }
  return UR_RESULT_SUCCESS;
}

// Wait on all operations in flight on this Queue.
// The caller is expected to hold a lock on the Queue.
// For standard commandlists sync the L0 queues directly.
// For immediate commandlists add barriers to all commandlists associated
// with the Queue. An alternative approach would be to wait on all Events
// associated with the in-flight operations.
// TODO: Event release in immediate commandlist mode is driven by the SYCL
// runtime. Need to investigate whether relase can be done earlier, at sync
// points such as this, to reduce total number of active Events.
ur_result_t ur_queue_handle_t_::synchronize() {
  if (!Healthy)
    return UR_RESULT_SUCCESS;

  auto syncImmCmdList = [](ur_queue_handle_t_ *Queue,
                           ur_command_list_ptr_t ImmCmdList) {
    if (ImmCmdList == Queue->CommandListMap.end())
      return UR_RESULT_SUCCESS;

    // wait for all commands previously submitted to this immediate command list
    if (UrL0QueueSyncNonBlocking) {
      Queue->Mutex.unlock();
      ZE2UR_CALL(zeCommandListHostSynchronize, (ImmCmdList->first, UINT64_MAX));
      Queue->Mutex.lock();
    } else {
      ZE2UR_CALL(zeCommandListHostSynchronize, (ImmCmdList->first, UINT64_MAX));
    }

    // Cleanup all events from the synced command list.
    CleanupEventListFromResetCmdList(ImmCmdList->second.EventList, true);
    ImmCmdList->second.EventList.clear();
    if (auto &completions = ImmCmdList->second.completions; completions) {
      completions->forceReset();
    }
    return UR_RESULT_SUCCESS;
  };

  if (LastCommandEvent) {
    // For in-order queue just wait for the last command.
    // If event is discarded then it can be in reset state or underlying level
    // zero handle can have device scope, so we can't synchronize the last
    // event.
    auto savedLastCommandEvent = LastCommandEvent;
    if (isInOrderQueue() && !LastCommandEvent->IsDiscarded) {
      ZE2UR_CALL(zeHostSynchronize, (LastCommandEvent->ZeEvent));

      // clean up all events known to have been completed as well,
      // so they can be reused later
      for (auto &QueueMap : {ComputeQueueGroupsByTID, CopyQueueGroupsByTID}) {
        for (auto &QueueGroup : QueueMap) {
          if (UsingImmCmdLists) {
            for (auto &ImmCmdList : QueueGroup.second.ImmCmdLists) {
              if (ImmCmdList == this->CommandListMap.end())
                continue;
              // Cleanup all events from the synced command list.
              CleanupEventListFromResetCmdList(ImmCmdList->second.EventList,
                                               true);
              ImmCmdList->second.EventList.clear();
            }
          }
        }
      }
    } else {
      // Otherwise sync all L0 queues/immediate command-lists.
      for (auto &QueueMap : {ComputeQueueGroupsByTID, CopyQueueGroupsByTID}) {
        for (auto &QueueGroup : QueueMap) {
          if (UsingImmCmdLists) {
            for (auto &ImmCmdList : QueueGroup.second.ImmCmdLists)
              UR_CALL(syncImmCmdList(this, ImmCmdList));
          } else {
            for (auto &ZeQueue : QueueGroup.second.ZeQueues)
              if (ZeQueue) {
                if (UrL0QueueSyncNonBlocking) {
                  this->Mutex.unlock();
                  ZE2UR_CALL(zeHostSynchronize, (ZeQueue));
                  this->Mutex.lock();
                } else {
                  ZE2UR_CALL(zeHostSynchronize, (ZeQueue));
                }
              }
          }
        }
      }
    }
    // If the current version of the LastCommandEvent == savedLastCommandEvent,
    // then LastCommandEvent = nullptr; Otherwise, if LastCommandEvent !=
    // savedLastCommandEvent, then LastCommandEvent is unchanged.
    if (LastCommandEvent == savedLastCommandEvent) {
      LastCommandEvent = nullptr;
    }
  }

  // Since all timestamp recordings should have finished with the
  // synchronizations, we can clear the map and write the results to the owning
  // events.
  clearEndTimeRecordings();

  // With the entire queue synchronized, the active barriers must be done so we
  // can remove them.
  if (auto Res = ActiveBarriers.clear())
    return Res;

  return UR_RESULT_SUCCESS;
}

ur_event_handle_t ur_queue_handle_t_::getEventFromQueueCache(bool IsMultiDevice,
                                                             bool HostVisible) {
  std::list<ur_event_handle_t> *Cache;

  if (!IsMultiDevice) {
    auto Device = this->Device;
    Cache = HostVisible ? &EventCaches[EventCachesDeviceMap[0][Device]]
                        : &EventCaches[EventCachesDeviceMap[1][Device]];
    if (!Cache) {
      return nullptr;
    }
  } else {
    Cache = HostVisible ? &EventCaches[0] : &EventCaches[1];
  }

  // If we don't have any events, return nullptr.
  // If we have only a single event then it was used by the last command and we
  // can't use it now because we have to enforce round robin between two events.
  if (Cache->size() < 2)
    return nullptr;

  // If there are two events then return an event from the beginning of the list
  // since event of the last command is added to the end of the list.
  auto It = Cache->begin();
  ur_event_handle_t RetEvent = *It;
  Cache->erase(It);
  return RetEvent;
}

// This helper function checks to see if an event for a command can be included
// at the end of a command list batch. This will only be true if the event does
// not have dependencies or the dependencies are not for events which exist in
// this batch.
bool eventCanBeBatched(ur_queue_handle_t Queue, bool UseCopyEngine,
                       uint32_t NumEventsInWaitList,
                       const ur_event_handle_t *EventWaitList) {
  auto &CommandBatch =
      UseCopyEngine ? Queue->CopyCommandBatch : Queue->ComputeCommandBatch;
  // First see if there is an command-list open for batching commands
  // for this queue.
  if (Queue->hasOpenCommandList(UseCopyEngine)) {
    // If this command should be batched, but the command has a dependency on a
    // command in the current batch, then the command needs to have an event
    // to track its completion so this event cannot be batched to the end of the
    // command list.
    if (NumEventsInWaitList > 0) {
      for (auto &Event : CommandBatch.OpenCommandList->second.EventList) {
        for (uint32_t i = 0; i < NumEventsInWaitList; i++) {
          if (Event == EventWaitList[i]) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// This helper function checks to see if a signal event at the end of a command
// should be set. If the Queue is out of order and the command has no
// dependencies, then this command can be enqueued without a signal event set in
// a command list batch. The signal event will be appended at the end of the
// batch to be signalled at the end of the command list.
ur_result_t setSignalEvent(ur_queue_handle_t Queue, bool UseCopyEngine,
                           ze_event_handle_t *ZeEvent, ur_event_handle_t *Event,
                           uint32_t NumEventsInWaitList,
                           const ur_event_handle_t *EventWaitList,
                           ze_command_queue_handle_t ZeQueue) {
  if (!UrL0OutOfOrderIntegratedSignalEvent && Queue->Device->isIntegrated() &&
      eventCanBeBatched(Queue, UseCopyEngine, NumEventsInWaitList,
                        EventWaitList) &&
      !Queue->isInOrderQueue() && !Queue->UsingImmCmdLists) {
    ZeEvent = nullptr;
    (*Event)->IsInnerBatchedEvent = true;
    (*Event)->ZeBatchedQueue = ZeQueue;
  } else {
    (*ZeEvent) = (*Event)->ZeEvent;
  }
  return UR_RESULT_SUCCESS;
}

// This helper function creates a ur_event_handle_t and associate a
// ur_queue_handle_t. Note that the caller of this function must have acquired
// lock on the Queue that is passed in.
// \param Queue ur_queue_handle_t to associate with a new event.
// \param Event a pointer to hold the newly created ur_event_handle_t
// \param CommandType various command type determined by the caller
// \param CommandList is the command list where the event is added
// \param IsInternal tells if the event is internal, i.e. visible in the L0
//        plugin only.
// \param IsMultiDevice tells if the event must be created in the multi-device
//        visible pool.
// \param HostVisible tells if the event must be created in the
//        host-visible pool. If not set then this function will decide.
ur_result_t createEventAndAssociateQueue(ur_queue_handle_t Queue,
                                         ur_event_handle_t *Event,
                                         ur_command_t CommandType,
                                         ur_command_list_ptr_t CommandList,
                                         bool IsInternal, bool IsMultiDevice,
                                         std::optional<bool> HostVisible) {

  if (!HostVisible.has_value()) {
    // Internal/discarded events do not need host-scope visibility.
    HostVisible = IsInternal ? false : Queue->ZeEventsScope == AllHostVisible;
  }

  // If event is discarded then try to get event from the queue cache.
  *Event = IsInternal ? Queue->getEventFromQueueCache(IsMultiDevice,
                                                      HostVisible.value())
                      : nullptr;

  if (*Event == nullptr)
    UR_CALL(EventCreate(
        Queue->Context, Queue, IsMultiDevice, HostVisible.value(), Event,
        Queue->CounterBasedEventsEnabled, false /*ForceDisableProfiling*/,
        Queue->InterruptBasedEventsEnabled));

  (*Event)->UrQueue = Queue;
  (*Event)->CommandType = CommandType;
  (*Event)->IsDiscarded = IsInternal;
  (*Event)->IsMultiDevice = IsMultiDevice;
  (*Event)->CommandList = CommandList;
  // Discarded event doesn't own ze_event, it is used by multiple
  // ur_event_handle_t objects. We destroy corresponding ze_event by releasing
  // events from the events cache at queue destruction. Event in the cache owns
  // the Level Zero event.
  if (IsInternal)
    (*Event)->OwnNativeHandle = false;

  // Append this Event to the CommandList, if any
  if (CommandList != Queue->CommandListMap.end()) {
    CommandList->second.append(*Event);
    (*Event)->RefCount.increment();
  }

  // We need to increment the reference counter here to avoid ur_queue_handle_t
  // being released before the associated ur_event_handle_t is released because
  // urEventRelease requires access to the associated ur_queue_handle_t.
  // In urEventRelease, the reference counter of the Queue is decremented
  // to release it.
  Queue->RefCount.increment();

  // SYCL RT does not track completion of the events, so it could
  // release a PI event as soon as that's not being waited in the app.
  // But we have to ensure that the event is not destroyed before
  // it is really signalled, so retain it explicitly here and
  // release in CleanupCompletedEvent(Event).
  // If the event is internal then don't increment the reference count as this
  // event will not be waited/released by SYCL RT, so it must be destroyed by
  // EventRelease in resetCommandList.
  if (!IsInternal)
    UR_CALL(ur::level_zero::urEventRetain(*Event));

  return UR_RESULT_SUCCESS;
}

void ur_queue_handle_t_::CaptureIndirectAccesses() {
  for (auto &Kernel : KernelsToBeSubmitted) {
    if (!Kernel->hasIndirectAccess())
      continue;

    auto &Contexts = Device->Platform->Contexts;
    for (auto &Ctx : Contexts) {
      for (auto &Elem : Ctx->MemAllocs) {
        const auto &Pair = Kernel->MemAllocs.insert(&Elem);
        // Kernel is referencing this memory allocation from now.
        // If this memory allocation was already captured for this kernel, it
        // means that kernel is submitted several times. Increase reference
        // count only once because we release all allocations only when
        // SubmissionsCount turns to 0. We don't want to know how many times
        // allocation was retained by each submission.
        if (Pair.second)
          Elem.second.RefCount.increment();
      }
    }
    Kernel->SubmissionsCount++;
  }
  KernelsToBeSubmitted.clear();
}

ur_result_t ur_queue_handle_t_::signalEventFromCmdListIfLastEventDiscarded(
    ur_command_list_ptr_t CommandList) {
  // We signal new event at the end of command list only if we have queue with
  // discard_events property and the last command event is discarded.
  if (!(doReuseDiscardedEvents() && LastCommandEvent &&
        LastCommandEvent->IsDiscarded))
    return UR_RESULT_SUCCESS;

  // NOTE: We create this "glue" event not as internal so it is not
  // participating in the discarded events reset/reuse logic, but
  // with no host-visibility since it is not going to be waited
  // from the host.
  ur_event_handle_t Event;
  UR_CALL(createEventAndAssociateQueue(
      reinterpret_cast<ur_queue_handle_t>(this), &Event,
      UR_EXT_COMMAND_TYPE_USER, CommandList,
      /* IsInternal */ false, /* IsMultiDevice */ true,
      /* HostVisible */ false));
  UR_CALL(urEventReleaseInternal(Event));
  LastCommandEvent = Event;

  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (CommandList->first, Event->ZeEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_handle_t_::executeOpenCommandList(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  // If there are any commands still in the open command list for this
  // queue, then close and execute that command list now.
  if (hasOpenCommandList(IsCopy)) {
    adjustBatchSizeForPartialBatch(IsCopy);
    auto Res =
        executeCommandList(CommandBatch.OpenCommandList, false /*IsBlocking*/,
                           false /*OKToBatchCommand*/);
    CommandBatch.OpenCommandList = CommandListMap.end();
    return Res;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_handle_t_::resetCommandList(
    ur_command_list_ptr_t CommandList, bool MakeAvailable,
    std::vector<ur_event_handle_t> &EventListToCleanup, bool CheckStatus) {
  bool UseCopyEngine = CommandList->second.isCopy(this);

  // Immediate commandlists do not have an associated fence.
  if (CommandList->second.ZeFence != nullptr) {
    // Fence had been signalled meaning the associated command-list completed.
    // Reset the fence and put the command list into a cache for reuse in PI
    // calls.
    ZE2UR_CALL(zeFenceReset, (CommandList->second.ZeFence));
    ZE2UR_CALL(zeCommandListReset, (CommandList->first));
    CommandList->second.ZeFenceInUse = false;
    CommandList->second.IsClosed = false;
  }

  auto &EventList = CommandList->second.EventList;
  // Check if standard commandlist or fully synced in-order queue.
  // If one of those conditions is met then we are sure that all events are
  // completed so we don't need to check event status.
  if (!CheckStatus || CommandList->second.ZeFence != nullptr ||
      (isInOrderQueue() && !LastCommandEvent)) {
    // Remember all the events in this command list which needs to be
    // released/cleaned up and clear event list associated with command list.
    std::move(std::begin(EventList), std::end(EventList),
              std::back_inserter(EventListToCleanup));
    EventList.clear();
  } else if (!isDiscardEvents()) {
    // If events in the queue are discarded then we can't check their status.
    // Helper for checking of event completion
    auto EventCompleted = [](ur_event_handle_t Event) -> bool {
      std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
      ze_result_t ZeResult =
          Event->Completed
              ? ZE_RESULT_SUCCESS
              : ZE_CALL_NOCHECK(zeEventQueryStatus, (Event->ZeEvent));
      return ZeResult == ZE_RESULT_SUCCESS;
    };
    // Handle in-order specially as we can just in few checks (with binary
    // search) a completed event and then all events before it are also
    // done.
    if (isInOrderQueue()) {
      size_t Bisect = EventList.size();
      size_t Iter = 0;
      for (auto it = EventList.rbegin(); it != EventList.rend(); ++Iter) {
        if (!EventCompleted(*it)) {
          if (Bisect > 1 && Iter < 3) { // Heuristically limit by 3 checks
            Bisect >>= 1;
            it += Bisect;
            continue;
          }
          break;
        }
        // Bulk move of event up to "it" to the list ready for cleanup
        std::move(it, EventList.rend(), std::back_inserter(EventListToCleanup));
        EventList.erase(EventList.begin(), it.base());
        break;
      }
      return UR_RESULT_SUCCESS;
    }

    if (auto &completions = CommandList->second.completions; completions) {
      if (completions->tryCleanup(this, CommandList->first, EventList,
                                  EventListToCleanup) == UR_RESULT_SUCCESS) {
        return UR_RESULT_SUCCESS;
      }
    }

    // For immediate commandlist reset only those events that have signalled.
    for (auto it = EventList.begin(); it != EventList.end();) {
      // Break early as soon as we found first incomplete event because next
      // events are submitted even later. We are not trying to find all
      // completed events here because it may be costly. I.e. we are checking
      // only elements which are most likely completed because they were
      // submitted earlier. It is guaranteed that all events will be eventually
      // cleaned up at queue sync/release.
      if (!EventCompleted(*it))
        break;

      EventListToCleanup.push_back(std::move((*it)));
      it = EventList.erase(it);
    }
  }

  // Standard commandlists move in and out of the cache as they are recycled.
  // Immediate commandlists are always available.
  if (CommandList->second.ZeFence != nullptr && MakeAvailable) {
    std::scoped_lock<ur_mutex> Lock(this->Context->ZeCommandListCacheMutex);
    auto &ZeCommandListCache =
        UseCopyEngine
            ? this->Context->ZeCopyCommandListCache[this->Device->ZeDevice]
            : this->Context->ZeComputeCommandListCache[this->Device->ZeDevice];
    struct l0_command_list_cache_info ListInfo;
    ListInfo.ZeQueueDesc = CommandList->second.ZeQueueDesc;
    ListInfo.InOrderList = CommandList->second.IsInOrderList;
    ListInfo.IsImmediate = CommandList->second.IsImmediate;
    ZeCommandListCache.push_back({CommandList->first, ListInfo});
  }

  return UR_RESULT_SUCCESS;
}

bool ur_command_list_info_t::isCopy(ur_queue_handle_t Queue) const {
  return ZeQueueDesc.ordinal !=
         (uint32_t)Queue->Device
             ->QueueGroup
                 [ur_device_handle_t_::queue_group_info_t::type::Compute]
             .ZeOrdinal;
}

void ur_command_list_info_t::append(ur_event_handle_t Event) {
  if (completions) {
    completions->append(Event);
  }
  EventList.push_back(Event);
}

ur_command_list_ptr_t
ur_queue_handle_t_::eventOpenCommandList(ur_event_handle_t Event) {
  using IsCopy = bool;

  if (UsingImmCmdLists) {
    // When using immediate commandlists there are no open command lists.
    return CommandListMap.end();
  }

  if (hasOpenCommandList(IsCopy{false})) {
    const auto &ComputeEventList =
        ComputeCommandBatch.OpenCommandList->second.EventList;
    if (std::find(ComputeEventList.begin(), ComputeEventList.end(), Event) !=
        ComputeEventList.end())
      return ComputeCommandBatch.OpenCommandList;
  }
  if (hasOpenCommandList(IsCopy{true})) {
    const auto &CopyEventList =
        CopyCommandBatch.OpenCommandList->second.EventList;
    if (std::find(CopyEventList.begin(), CopyEventList.end(), Event) !=
        CopyEventList.end())
      return CopyCommandBatch.OpenCommandList;
  }
  return CommandListMap.end();
}

void ur_queue_handle_t_::ur_queue_group_t::setImmCmdList(
    ur_queue_handle_t queue, ze_command_list_handle_t ZeCommandList) {
  // An immediate command list was given to us but we don't have the queue
  // descriptor information. Create a dummy and note that it is not recycleable.
  ZeStruct<ze_command_queue_desc_t> ZeQueueDesc;

  ImmCmdLists = std::vector<ur_command_list_ptr_t>(
      1,
      Queue->CommandListMap
          .insert(std::pair<ze_command_list_handle_t, ur_command_list_info_t>{
              ZeCommandList,
              ur_command_list_info_t(nullptr, true, false, nullptr, ZeQueueDesc,
                                     queue->useCompletionBatching(), false,
                                     false, true)})
          .first);
}

ur_queue_handle_t_::ur_queue_group_t &
ur_queue_handle_t_::getQueueGroup(bool UseCopyEngine) {
  auto &Map = (UseCopyEngine ? CopyQueueGroupsByTID : ComputeQueueGroupsByTID);
  return Map.get();
}

// Return the index of the next queue to use based on a
// round robin strategy and the queue group ordinal.
uint32_t ur_queue_handle_t_::ur_queue_group_t::getQueueIndex(
    uint32_t *QueueGroupOrdinal, uint32_t *QueueIndex, bool QueryOnly) {
  auto CurrentIndex = NextIndex;

  if (!QueryOnly) {
    ++NextIndex;
    if (NextIndex > UpperIndex)
      NextIndex = LowerIndex;
  }

  // Find out the right queue group ordinal (first queue might be "main" or
  // "link")
  auto QueueType = Type;
  if (QueueType != queue_type::Compute)
    QueueType = (CurrentIndex == 0 && Queue->Device->hasMainCopyEngine())
                    ? queue_type::MainCopy
                    : queue_type::LinkCopy;

  *QueueGroupOrdinal = Queue->Device->QueueGroup[QueueType].ZeOrdinal;
  // Adjust the index to the L0 queue group since we represent "main" and
  // "link"
  // L0 groups with a single copy group ("main" would take "0" index).
  auto ZeCommandQueueIndex = CurrentIndex;
  if (QueueType == queue_type::LinkCopy && Queue->Device->hasMainCopyEngine()) {
    ZeCommandQueueIndex -= 1;
  }
  *QueueIndex = ZeCommandQueueIndex;

  return CurrentIndex;
}

// This function will return one of possibly multiple available native
// queues and the value of the queue group ordinal.
ze_command_queue_handle_t &
ur_queue_handle_t_::ur_queue_group_t::getZeQueue(uint32_t *QueueGroupOrdinal) {

  // QueueIndex is the proper L0 index.
  // Index is the plugins concept of index, with main and link copy engines in
  // one range.
  uint32_t QueueIndex;
  auto Index = getQueueIndex(QueueGroupOrdinal, &QueueIndex);

  ze_command_queue_handle_t &ZeQueue = ZeQueues[Index];
  if (ZeQueue)
    return ZeQueue;

  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  ZeCommandQueueDesc.ordinal = *QueueGroupOrdinal;
  ZeCommandQueueDesc.index = QueueIndex;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  const char *Priority = "Normal";
  if (Queue->isPriorityLow()) {
    ZeCommandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    Priority = "Low";
  } else if (Queue->isPriorityHigh()) {
    ZeCommandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    Priority = "High";
  }

  // Evaluate performance of explicit usage for "0" index.
  if (QueueIndex != 0) {
    ZeCommandQueueDesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
  }

  logger::debug("[getZeQueue]: create queue ordinal = {}, index = {} "
                "(round robin in [{}, {}]) priority = {}",
                ZeCommandQueueDesc.ordinal, ZeCommandQueueDesc.index,
                LowerIndex, UpperIndex, Priority);

  auto ZeResult = ZE_CALL_NOCHECK(
      zeCommandQueueCreate, (Queue->Context->ZeContext, Queue->Device->ZeDevice,
                             &ZeCommandQueueDesc, &ZeQueue));
  if (ZeResult) {
    die("[L0] getZeQueue: failed to create queue");
  }

  return ZeQueue;
}

int32_t ur_queue_handle_t_::ur_queue_group_t::getCmdQueueOrdinal(
    ze_command_queue_handle_t CmdQueue) {
  // Find out the right queue group ordinal (first queue might be "main" or
  // "link")
  auto QueueType = Type;
  if (QueueType != queue_type::Compute)
    QueueType = (ZeQueues[0] == CmdQueue && Queue->Device->hasMainCopyEngine())
                    ? queue_type::MainCopy
                    : queue_type::LinkCopy;
  return Queue->Device->QueueGroup[QueueType].ZeOrdinal;
}

bool ur_queue_handle_t_::useCompletionBatching() {
  static bool enabled = getenv_tobool(
      "UR_L0_IMMEDIATE_COMMANDLISTS_BATCH_EVENT_COMPLETIONS", false);
  return enabled && !isInOrderQueue() && UsingImmCmdLists;
}

// Helper function to create a new command-list to this queue and associated
// fence tracking its completion. This command list & fence are added to the
// map of command lists in this queue with ZeFenceInUse = false.
// The caller must hold a lock of the queue already.
ur_result_t ur_queue_handle_t_::createCommandList(
    bool UseCopyEngine, ur_command_list_ptr_t &CommandList,
    ze_command_queue_handle_t *ForcedCmdQueue) {

  ze_fence_handle_t ZeFence;
  ZeStruct<ze_fence_desc_t> ZeFenceDesc;
  ze_command_list_handle_t ZeCommandList;

  uint32_t QueueGroupOrdinal;
  auto &QGroup = getQueueGroup(UseCopyEngine);
  auto &ZeCommandQueue =
      ForcedCmdQueue ? *ForcedCmdQueue : QGroup.getZeQueue(&QueueGroupOrdinal);
  if (ForcedCmdQueue)
    QueueGroupOrdinal = QGroup.getCmdQueueOrdinal(ZeCommandQueue);

  ZeStruct<ze_command_list_desc_t> ZeCommandListDesc;
  ZeCommandListDesc.commandQueueGroupOrdinal = QueueGroupOrdinal;

  bool IsInOrderList = false;
  if (Device->useDriverInOrderLists() && isInOrderQueue()) {
    ZeCommandListDesc.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER;
    IsInOrderList = true;
  }

  ZE2UR_CALL(zeCommandListCreate, (Context->ZeContext, Device->ZeDevice,
                                   &ZeCommandListDesc, &ZeCommandList));

  ZE2UR_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
  ZeStruct<ze_command_queue_desc_t> ZeQueueDesc;
  ZeQueueDesc.ordinal = QueueGroupOrdinal;

  std::tie(CommandList, std::ignore) = CommandListMap.insert(
      std::pair<ze_command_list_handle_t, ur_command_list_info_t>(
          ZeCommandList,
          ur_command_list_info_t(
              ZeFence, false /*ZeFenceInUse*/, false /*IsClosed*/,
              ZeCommandQueue, ZeQueueDesc, useCompletionBatching(),
              true /*CanReuse*/, IsInOrderList, false /*IsImmediate*/)));

  UR_CALL(insertStartBarrierIfDiscardEventsMode(CommandList));
  UR_CALL(insertActiveBarriers(CommandList, UseCopyEngine));
  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_handle_t_::insertActiveBarriers(ur_command_list_ptr_t &CmdList,
                                         bool UseCopyEngine) {
  // Early exit if there are no active barriers.
  if (ActiveBarriers.empty())
    return UR_RESULT_SUCCESS;

  // Create a wait-list and retain events.
  _ur_ze_event_list_t ActiveBarriersWaitList;
  UR_CALL(ActiveBarriersWaitList.createAndRetainUrZeEventList(
      ActiveBarriers.vector().size(), ActiveBarriers.vector().data(),
      reinterpret_cast<ur_queue_handle_t>(this), UseCopyEngine));

  // We can now replace active barriers with the ones in the wait list.
  UR_CALL(ActiveBarriers.clear());

  if (ActiveBarriersWaitList.Length == 0) {
    return UR_RESULT_SUCCESS;
  }

  for (uint32_t I = 0; I < ActiveBarriersWaitList.Length; ++I) {
    auto &Event = ActiveBarriersWaitList.UrEventList[I];
    ActiveBarriers.add(Event);
  }

  ur_event_handle_t Event = nullptr;
  if (auto Res = createEventAndAssociateQueue(
          reinterpret_cast<ur_queue_handle_t>(this), &Event,
          UR_EXT_COMMAND_TYPE_USER, CmdList,
          /* IsInternal */ true, /* IsMultiDevice */ true))
    return Res;

  Event->WaitList = ActiveBarriersWaitList;
  Event->OwnNativeHandle = true;

  // If there are more active barriers, insert a barrier on the command-list. We
  // do not need an event for finishing so we pass nullptr.
  ZE2UR_CALL(zeCommandListAppendBarrier,
             (CmdList->first, nullptr, ActiveBarriersWaitList.Length,
              ActiveBarriersWaitList.ZeEventList));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_handle_t_::insertStartBarrierIfDiscardEventsMode(
    ur_command_list_ptr_t &CmdList) {
  // If current command list is different from the last command list then insert
  // a barrier waiting for the last command event.
  if (doReuseDiscardedEvents() && CmdList != LastUsedCommandList &&
      LastCommandEvent) {
    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CmdList->first, nullptr, 1, &(LastCommandEvent->ZeEvent)));
    LastCommandEvent = nullptr;
  }
  return UR_RESULT_SUCCESS;
}

// This is an experimental option that allows the use of copy engine, if
// available in the device, in Level Zero plugin for copy operations submitted
// to an in-order queue. The default is 1.
static const bool UseCopyEngineForInOrderQueue = [] {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE");
  const char *CopyEngineForInOrderQueue =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  return (!CopyEngineForInOrderQueue ||
          (std::stoi(CopyEngineForInOrderQueue) != 0));
}();

bool ur_queue_handle_t_::useCopyEngine(bool PreferCopyEngine) const {
  auto InitialCopyGroup = CopyQueueGroupsByTID.begin()->second;
  return PreferCopyEngine && InitialCopyGroup.ZeQueues.size() > 0 &&
         (!isInOrderQueue() || UseCopyEngineForInOrderQueue);
}

// This function will return one of po6ssibly multiple available
// immediate commandlists associated with this Queue.
ur_command_list_ptr_t &ur_queue_handle_t_::ur_queue_group_t::getImmCmdList() {

  uint32_t QueueIndex, QueueOrdinal;
  auto Index = getQueueIndex(&QueueOrdinal, &QueueIndex);

  if ((ImmCmdLists[Index] != Queue->CommandListMap.end()) &&
      (!Queue->CounterBasedEventsEnabled ||
       (Queue->CounterBasedEventsEnabled &&
        (ImmCmdLists[Index]->second.ZeQueueDesc.flags &
         ZE_COMMAND_QUEUE_FLAG_IN_ORDER))))
    return ImmCmdLists[Index];

  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  ZeCommandQueueDesc.ordinal = QueueOrdinal;
  ZeCommandQueueDesc.index = QueueIndex;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  bool isInOrderList = false;
  const char *Priority = "Normal";
  if (Queue->isPriorityLow()) {
    ZeCommandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    Priority = "Low";
  } else if (Queue->isPriorityHigh()) {
    ZeCommandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    Priority = "High";
  }

  // Evaluate performance of explicit usage for "0" index.
  if (QueueIndex != 0) {
    ZeCommandQueueDesc.flags |= ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
  }

  if (Queue->Device->useDriverInOrderLists() && Queue->isInOrderQueue()) {
    isInOrderList = true;
    ZeCommandQueueDesc.flags |= ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  }

  // Check if context's command list cache has an immediate command list with
  // matching index.
  ze_command_list_handle_t ZeCommandList = nullptr;
  {
    // Acquire lock to avoid race conditions.
    std::scoped_lock<ur_mutex> Lock(Queue->Context->ZeCommandListCacheMutex);
    // Under mutex since operator[] does insertion on the first usage for every
    // unique ZeDevice.
    auto &ZeCommandListCache =
        isCopy()
            ? Queue->Context->ZeCopyCommandListCache[Queue->Device->ZeDevice]
            : Queue->Context
                  ->ZeComputeCommandListCache[Queue->Device->ZeDevice];
    for (auto ZeCommandListIt = ZeCommandListCache.begin();
         ZeCommandListIt != ZeCommandListCache.end(); ++ZeCommandListIt) {
      const auto &Desc = (*ZeCommandListIt).second.ZeQueueDesc;
      if (Desc.index == ZeCommandQueueDesc.index &&
          Desc.flags == ZeCommandQueueDesc.flags &&
          Desc.mode == ZeCommandQueueDesc.mode &&
          Desc.priority == ZeCommandQueueDesc.priority) {
        ZeCommandList = (*ZeCommandListIt).first;
        ZeCommandListCache.erase(ZeCommandListIt);
        break;
      }
    }
  }

  // If cache didn't contain a command list, create one.
  if (!ZeCommandList) {
    logger::debug("[getZeQueue]: create queue ordinal = {}, index = {} "
                  "(round robin in [{}, {}]) priority = {}",
                  ZeCommandQueueDesc.ordinal, ZeCommandQueueDesc.index,
                  LowerIndex, UpperIndex, Priority);

    ZE_CALL_NOCHECK(zeCommandListCreateImmediate,
                    (Queue->Context->ZeContext, Queue->Device->ZeDevice,
                     &ZeCommandQueueDesc, &ZeCommandList));
  }

  ImmCmdLists[Index] =
      Queue->CommandListMap
          .insert(std::pair<ze_command_list_handle_t, ur_command_list_info_t>{
              ZeCommandList,
              ur_command_list_info_t(
                  nullptr, true, false, nullptr, ZeCommandQueueDesc,
                  Queue->useCompletionBatching(), true, isInOrderList, true)})
          .first;

  return ImmCmdLists[Index];
}

// Get value of the threshold for number of events in immediate command lists.
// If number of events in the immediate command list exceeds this threshold then
// cleanup process for those events is executed.
static const size_t ImmCmdListsEventCleanupThreshold = [] {
  const char *UrRet =
      std::getenv("UR_L0_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD");
  const char *PiRet = std::getenv(
      "SYCL_PI_LEVEL_ZERO_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD");
  const char *ImmCmdListsEventCleanupThresholdStr =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  static constexpr int Default = 1000;
  if (!ImmCmdListsEventCleanupThresholdStr)
    return Default;

  int Threshold = std::atoi(ImmCmdListsEventCleanupThresholdStr);

  // Basically disable threshold if negative value is provided.
  if (Threshold < 0)
    return INT_MAX;

  return Threshold;
}();

size_t ur_queue_handle_t_::getImmdCmmdListsEventCleanupThreshold() {
  return useCompletionBatching() ? CompletionEventsPerBatch
                                 : ImmCmdListsEventCleanupThreshold;
}
