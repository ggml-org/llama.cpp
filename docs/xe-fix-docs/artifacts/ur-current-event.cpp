//===--------- event.cpp - Level Zero Adapter -----------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <mutex>
#include <optional>
#include <string.h>

#include "command_buffer.hpp"
#include "common.hpp"
#include "common/device.hpp"
#include "event.hpp"
#include "logger/ur_logger.hpp"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"

namespace ur::level_zero::v1 {

void printZeEventList(const ur_ze_event_list_t &UrZeEventList) {
  if (UrL0Debug & UR_L0_DEBUG_BASIC) {
    std::stringstream ss;
    ss << "  NumEventsInWaitList " << UrZeEventList.Length << ":";

    for (uint32_t I = 0; I < UrZeEventList.Length; I++) {
      ss << " " << ur_cast<std::uintptr_t>(UrZeEventList.ZeEventList[I]);
    }
    UR_LOG(DEBUG, ss.str().c_str());
  }
}

// This is an experimental option that allows the use of multiple command lists
// when submitting barriers. The default is 0.
static const bool UseMultipleCmdlistBarriers = [] {
  const char *UrRet = std::getenv("UR_L0_USE_MULTIPLE_COMMANDLIST_BARRIERS");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS");
  const char *UseMultipleCmdlistBarriersFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  if (!UseMultipleCmdlistBarriersFlag)
    return true;
  return std::atoi(UseMultipleCmdlistBarriersFlag) > 0;
}();

bool WaitListEmptyOrAllEventsFromSameQueue(
    ur_queue_handle_t Queue, uint32_t NumEventsInWaitList,
    const ur_event_handle_t *EventWaitList) {
  if (!NumEventsInWaitList)
    return true;

  for (uint32_t i = 0; i < NumEventsInWaitList; ++i) {
    if (Queue != EventWaitList[i]->UrQueue)
      return false;
  }

  return true;
}

ur_result_t urEnqueueEventsWait(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands must be complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  if (EventWaitListInternal) {
    bool UseCopyEngine = false;

    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    ur_ze_event_list_t TmpWaitList = {};
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitListInternal, Queue, UseCopyEngine));

    // Get a new command list to be used on this call
    ur_command_list_ptr_t CommandList{};
    UR_CALL(Queue->Context->getAvailableCommandList(
        Queue, CommandList, UseCopyEngine, NumEventsInWaitList,
        EventWaitListInternal, false /*AllowBatching*/,
        nullptr /*ForceCmdQueue*/));

    ze_event_handle_t ZeEvent = nullptr;
    ur_event_handle_t InternalEvent;
    bool IsInternal = OutEventInternal == nullptr;
    ur_event_handle_t *Event =
        OutEventInternal ? OutEventInternal : &InternalEvent;
    UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_EVENTS_WAIT,
                                         CommandList, IsInternal, false));

    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;

    const auto &WaitList = (*Event)->WaitList;
    auto ZeCommandList = CommandList->first;
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (ZeCommandList, WaitList.Length, WaitList.ZeEventList));

    ZE2UR_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

    // Execute command list asynchronously as the event will be used
    // to track down its completion.
    return Queue->executeCommandList(CommandList, false /*IsBlocking*/,
                                     false /*OKToBatchCommand*/);
  }

  // If wait-list is empty, then this particular command should wait until
  // all previous enqueued commands to the command-queue have completed.
  //
  // TODO: find a way to do that without blocking the host.

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  if (OutEventInternal) {
    UR_CALL(createEventAndAssociateQueue(Queue, OutEventInternal,
                                         UR_COMMAND_EVENTS_WAIT,
                                         Queue->CommandListMap.end(), false,
                                         /* IsInternal */ false));
  }

  UR_CALL(Queue->executeAllOpenCommandLists());
  UR_CALL(Queue->synchronize());

  if (OutEventInternal) {
    auto OutEventLocal = *OutEventInternal;
    Queue->LastCommandEvent = OutEventLocal;

    if (!OutEventLocal->CounterBasedEventsEnabled)
      ZE2UR_CALL(zeEventHostSignal, (OutEventLocal->ZeEvent));
    OutEventLocal->Completed = true;
  }

  if (!Queue->UsingImmCmdLists) {
    resetCommandLists(Queue);
  }

  return UR_RESULT_SUCCESS;
}

// Control if wait with barrier is implemented by signal of an event
// as opposed by true barrier command for in-order queue.
static const bool InOrderBarrierBySignal = [] {
  const char *UrRet = std::getenv("UR_L0_IN_ORDER_BARRIER_BY_SIGNAL");
  return (UrRet ? std::atoi(UrRet) : true);
}();

ur_result_t urEnqueueEventsWaitWithBarrier(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands must be complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  return ur::level_zero::v1::urEnqueueEventsWaitWithBarrierExt(
      QueueOpque, nullptr, NumEventsInWaitList, EventWaitList, OutEvent);
}

ur_result_t urEnqueueEventsWaitWithBarrierExt(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in][optional] pointer to the extended enqueue
    const ur_exp_enqueue_ext_properties_t *EnqueueExtProp,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands must be complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  bool InterruptBasedEventsEnabled =
      EnqueueExtProp ? (EnqueueExtProp->flags &
                        UR_EXP_ENQUEUE_EXT_FLAG_LOW_POWER_EVENTS_SUPPORT) ||
                           Queue->InterruptBasedEventsEnabled
                     : Queue->InterruptBasedEventsEnabled;
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Helper function for appending a barrier to a command list.
  auto insertBarrierIntoCmdList =
      [&Queue](ur_command_list_ptr_t CmdList, ur_ze_event_list_t &EventWaitList,
               ur_event_handle_t &Event, bool IsInternal,
               bool InterruptBasedEventsEnabled) {
        (void)InterruptBasedEventsEnabled;
        UR_CALL(createEventAndAssociateQueue(
            Queue, &Event, UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, CmdList,
            IsInternal, /* IsMultiDevice */ false));

        Event->WaitList = EventWaitList;

        // For in-order queue we don't need a real barrier, just wait for
        // requested events in potentially different queues and add a "barrier"
        // event signal because it is already guaranteed that previous commands
        // in this queue are completed when the signal is started.
        //
        // Only consideration here is that when profiling is used, signalEvent
        // cannot be used if EventWaitList.Lenght == 0. In those cases, we need
        // to fallback directly to barrier to have correct timestamps. See here:
        // https://spec.oneapi.io/level-zero/latest/core/api.html?highlight=appendsignalevent#_CPPv430zeCommandListAppendSignalEvent24ze_command_list_handle_t17ze_event_handle_t
        //
        // TODO: this and other special handling of in-order queues to be
        // updated when/if Level Zero adds native support for in-order queues.
        //
        if (Queue->isInOrderQueue() && InOrderBarrierBySignal &&
            !Queue->isProfilingEnabled()) {
          if (EventWaitList.Length) {
            ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
                       (CmdList->first, EventWaitList.Length,
                        EventWaitList.ZeEventList));
          }
          ZE2UR_CALL(zeCommandListAppendSignalEvent,
                     (CmdList->first, Event->ZeEvent));
        } else {
          ZE2UR_CALL(zeCommandListAppendBarrier,
                     (CmdList->first, Event->ZeEvent, EventWaitList.Length,
                      EventWaitList.ZeEventList));
        }

        return UR_RESULT_SUCCESS;
      };

  // If the queue is in-order then each command in it effectively acts as a
  // barrier, so we don't need to do anything except if we were requested
  // a "barrier" event to be created. Or if we need to wait for events in
  // potentially different queues.
  //
  if (Queue->isInOrderQueue() && NumEventsInWaitList == 0 &&
      !OutEventInternal) {
    return UR_RESULT_SUCCESS;
  }

  ur_event_handle_t ResultEvent = nullptr;
  bool IsInternal = OutEventInternal == nullptr;

  // For in-order queue and wait-list which is empty or has events from
  // the same queue just use the last command event as the barrier event.
  // This optimization is disabled when profiling is enabled to ensure
  // accurate profiling values & the overhead that profiling incurs.
  if (Queue->isInOrderQueue() && !Queue->isProfilingEnabled() &&
      WaitListEmptyOrAllEventsFromSameQueue(Queue, NumEventsInWaitList,
                                            EventWaitListInternal) &&
      Queue->LastCommandEvent && !Queue->LastCommandEvent->IsDiscarded) {
    UR_CALL(
        ur::level_zero::v1::urEventRetain(v1_cast(Queue->LastCommandEvent)));
    ResultEvent = Queue->LastCommandEvent;
    if (OutEventInternal) {
      *OutEventInternal = ResultEvent;
    }
    return UR_RESULT_SUCCESS;
  }

  // Indicator for whether batching is allowed. This may be changed later in
  // this function, but allow it by default.
  bool OkToBatch = true;

  // If we have a list of events to make the barrier from, then we can create a
  // barrier on these and use the resulting event as our future barrier.
  // We use the same approach if
  // UR_L0_USE_MULTIPLE_COMMANDLIST_BARRIERS is not set to a
  // positive value.
  // We use the same approach if we have in-order queue because every command
  // depends on previous one, so we don't need to insert barrier to multiple
  // command lists.
  if (NumEventsInWaitList || !UseMultipleCmdlistBarriers ||
      Queue->isInOrderQueue()) {
    // Retain the events as they will be owned by the result event.
    ur_ze_event_list_t TmpWaitList;
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitListInternal, Queue,
        false /*UseCopyEngine=*/));

    // Get an arbitrary command-list in the queue.
    ur_command_list_ptr_t CmdList;
    UR_CALL(Queue->Context->getAvailableCommandList(
        Queue, CmdList, false /*UseCopyEngine=*/, NumEventsInWaitList,
        EventWaitListInternal, OkToBatch, nullptr /*ForcedCmdQueue*/));

    // Insert the barrier into the command-list and execute.
    UR_CALL(insertBarrierIntoCmdList(CmdList, TmpWaitList, ResultEvent,
                                     IsInternal, InterruptBasedEventsEnabled));

    UR_CALL(
        Queue->executeCommandList(CmdList, false /*IsBlocking*/, OkToBatch));

    // Because of the dependency between commands in the in-order queue we don't
    // need to keep track of any active barriers if we have in-order queue.
    if (UseMultipleCmdlistBarriers && !Queue->isInOrderQueue()) {
      auto UREvent = reinterpret_cast<ur_event_handle_t>(ResultEvent);
      // We must release the Active Barrier before we start the next one
      // otherwise we will leak an event that won't be released.
      UR_CALL(Queue->ActiveBarriers.clear());
      Queue->ActiveBarriers.add(UREvent);
    }

    if (OutEventInternal) {
      *OutEventInternal = ResultEvent;
    }
    return UR_RESULT_SUCCESS;
  }

  // Since there are no events to explicitly create a barrier for, we are
  // inserting a queue-wide barrier.

  // Command list(s) for putting barriers.
  std::vector<ur_command_list_ptr_t> CmdLists;

  // There must be at least one L0 queue.
  auto &ComputeGroup = Queue->ComputeQueueGroupsByTID.get();
  auto &CopyGroup = Queue->CopyQueueGroupsByTID.get();
  UR_ASSERT(!ComputeGroup.ZeQueues.empty() || !CopyGroup.ZeQueues.empty(),
            UR_RESULT_ERROR_INVALID_QUEUE);

  size_t NumQueues = 0;
  for (auto &QueueMap :
       {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
    for (auto &QueueGroup : QueueMap)
      NumQueues += QueueGroup.second.ZeQueues.size();

  OkToBatch = true;
  // Get an available command list tied to each command queue. We need
  // these so a queue-wide barrier can be inserted into each command
  // queue.
  CmdLists.reserve(NumQueues);
  for (auto &QueueMap :
       {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
    for (auto &QueueGroup : QueueMap) {
      bool UseCopyEngine =
          QueueGroup.second.Type !=
          ur::level_zero::v1::ur_queue_handle_t_::queue_type::Compute;
      if (Queue->UsingImmCmdLists) {
        // If immediate command lists are being used, each will act as their own
        // queue, so we must insert a barrier into each.
        for (auto &ImmCmdList : QueueGroup.second.ImmCmdLists)
          if (ImmCmdList != Queue->CommandListMap.end())
            CmdLists.push_back(ImmCmdList);
      } else {
        for (auto ZeQueue : QueueGroup.second.ZeQueues) {
          if (ZeQueue) {
            ur_command_list_ptr_t CmdList;
            UR_CALL(Queue->Context->getAvailableCommandList(
                Queue, CmdList, UseCopyEngine, NumEventsInWaitList,
                EventWaitListInternal, OkToBatch, &ZeQueue));
            CmdLists.push_back(CmdList);
          }
        }
      }
    }

  // If no activity has occurred on the queue then there will be no cmdlists.
  // We need one for generating an Event, so create one.
  if (CmdLists.size() == 0) {
    // Get any available command list.
    ur_command_list_ptr_t CmdList;
    UR_CALL(Queue->Context->getAvailableCommandList(
        Queue, CmdList, false /*UseCopyEngine=*/, NumEventsInWaitList,
        EventWaitListInternal, OkToBatch, nullptr /*ForcedCmdQueue*/));
    CmdLists.push_back(CmdList);
  }

  if (CmdLists.size() > 1) {
    // Insert a barrier into each unique command queue using the available
    // command-lists.
    std::vector<ur_event_handle_t> EventWaitVector(CmdLists.size());
    for (size_t I = 0; I < CmdLists.size(); ++I) {
      ur_ze_event_list_t waitlist;
      UR_CALL(insertBarrierIntoCmdList(CmdLists[I], waitlist,
                                       EventWaitVector[I], true /*IsInternal*/,
                                       InterruptBasedEventsEnabled));
    }
    // If there were multiple queues we need to create a "convergence" event to
    // be our active barrier. This convergence event is signalled by a barrier
    // on all the events from the barriers we have inserted into each queue.
    // Use the first command list as our convergence command list.
    ur_command_list_ptr_t &ConvergenceCmdList = CmdLists[0];

    // Create an event list. It will take ownership over all relevant events so
    // we relinquish ownership and let it keep all events it needs.
    ur_ze_event_list_t BaseWaitList;
    UR_CALL(BaseWaitList.createAndRetainUrZeEventList(
        EventWaitVector.size(),
        reinterpret_cast<const ur_event_handle_t *>(EventWaitVector.data()),
        Queue, ConvergenceCmdList->second.isCopy(Queue)));

    // Insert a barrier with the events from each command-queue into the
    // convergence command list. The resulting event signals the convergence of
    // all barriers.
    UR_CALL(insertBarrierIntoCmdList(ConvergenceCmdList, BaseWaitList,
                                     ResultEvent, IsInternal,
                                     InterruptBasedEventsEnabled));
  } else {
    // If there is only a single queue then insert a barrier and the single
    // result event can be used as our active barrier and used as the return
    // event. Take into account whether output event is discarded or not.
    ur_ze_event_list_t waitlist;
    UR_CALL(insertBarrierIntoCmdList(CmdLists[0], waitlist, ResultEvent,
                                     IsInternal, InterruptBasedEventsEnabled));
  }

  // Execute each command list so the barriers can be encountered.
  for (ur_command_list_ptr_t &CmdList : CmdLists) {
    bool IsCopy = CmdList->second.isCopy(Queue);
    const auto &CommandBatch =
        (IsCopy) ? Queue->CopyCommandBatch : Queue->ComputeCommandBatch;
    // Only batch if the matching CmdList is already open.
    OkToBatch = CommandBatch.OpenCommandList == CmdList;

    UR_CALL(
        Queue->executeCommandList(CmdList, false /*IsBlocking*/, OkToBatch));
  }

  UR_CALL(Queue->ActiveBarriers.clear());
  Queue->ActiveBarriers.add(ResultEvent);
  if (OutEventInternal) {
    *OutEventInternal = ResultEvent;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urEventGetInfo(
    /// [in] handle of the event object
    ::ur_event_handle_t EventOpque,
    /// [in] the name of the event property to query
    ur_event_info_t PropName,
    /// [in] size in bytes of the event property value
    size_t PropValueSize,
    /// [out][optional] value of the event property
    void *PropValue,
    size_t
        /// [out][optional] bytes returned in event property
        *PropValueSizeRet) {
  auto Event = v1_cast(EventOpque);
  UrReturnHelper ReturnValue(PropValueSize, PropValue, PropValueSizeRet);

  switch (PropName) {
  case UR_EVENT_INFO_COMMAND_QUEUE: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(ur_queue_handle_t{Event->UrQueue});
  }
  case UR_EVENT_INFO_CONTEXT: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(ur_context_handle_t{Event->Context});
  }
  case UR_EVENT_INFO_COMMAND_TYPE: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(ur_cast<ur_command_t>(Event->CommandType));
  }
  case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    // Check to see if the event's Queue has an open command list due to
    // batching. If so, go ahead and close and submit it, because it is
    // possible that this is trying to query some event's status that
    // is part of the batch.  This isn't strictly required, but it seems
    // like a reasonable thing to do.
    auto UrQueue = Event->UrQueue;
    if (UrQueue) {
      // Lock automatically releases when this goes out of scope.
      std::unique_lock<ur_shared_mutex> Lock(UrQueue->Mutex, std::try_to_lock);
      // If we fail to acquire the lock, it's possible that the queue might
      // already be waiting for this event in synchronize().
      if (Lock.owns_lock()) {
        const auto &OpenCommandList = UrQueue->eventOpenCommandList(Event);
        if (OpenCommandList != UrQueue->CommandListMap.end()) {
          UR_CALL(UrQueue->executeOpenCommandList(
              OpenCommandList->second.isCopy(UrQueue)));
        }
      }
    }

    // Level Zero has a much more explicit notion of command submission than
    // OpenCL. It doesn't happen unless the user submits a command list. We've
    // done it just above so the status is at least PI_EVENT_SUBMITTED.
    //
    // NOTE: We currently cannot tell if command is currently running, so
    // it will always show up "submitted" before it is finally "completed".
    //
    uint32_t Result = ur_cast<uint32_t>(UR_EVENT_STATUS_SUBMITTED);

    // Make sure that we query a host-visible event only.
    // If one wasn't yet created then don't create it here as well, and
    // just conservatively return that event is not yet completed.
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    auto HostVisibleEvent = Event->HostVisibleEvent;
    if (Event->Completed) {
      Result = UR_EVENT_STATUS_COMPLETE;
    } else if (HostVisibleEvent && checkL0LoaderTeardown()) {
      ze_result_t ZeResult;
      ZeResult =
          ZE_CALL_NOCHECK(zeEventQueryStatus, (HostVisibleEvent->ZeEvent));
      if (ZeResult == ZE_RESULT_SUCCESS) {
        Result = UR_EVENT_STATUS_COMPLETE;
      }
    }
    return ReturnValue(Result);
  }
  case UR_EVENT_INFO_REFERENCE_COUNT: {
    return ReturnValue(Event->RefCount.getCount());
  }
  default:
    UR_LOG(ERR, "Unsupported ParamName in urEventGetInfo: ParamName={}(0x{})",
           PropName, logger::toHex(PropName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urEventGetProfilingInfo(
    /// [in] handle of the event object
    ::ur_event_handle_t EventOpque,
    /// [in] the name of the profiling property to query
    ur_profiling_info_t PropName,
    /// [in] size in bytes of the profiling property value
    size_t PropValueSize,
    /// [out][optional] value of the profiling property
    void *PropValue,
    /// [out][optional] pointer to the actual size in bytes returned in
    /// propValue
    size_t *PropValueSizeRet) {
  auto Event = v1_cast(EventOpque);
  std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);

  // The event must either have profiling enabled or be recording timestamps.
  bool isTimestampedEvent = Event->IsTimestamped;
  if (!Event->isProfilingEnabled() && !isTimestampedEvent) {
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

  ur_device_handle_t Device =
      Event->UrQueue ? Event->UrQueue->Device : Event->Context->getDevices()[0];

  const double ZeTimerResolution = Device->getTimerResolution();
  const uint64_t TimestampMaxValue = Device->getTimestampMask();

  UrReturnHelper ReturnValue(PropValueSize, PropValue, PropValueSizeRet);

  // For timestamped events we have the timestamps ready directly on the event
  // handle, so we short-circuit the return. The tag is an empty command, so
  // all timestamps (queued, submit, start, end, complete) are the single
  // GPU-written completion time.
  // We don't support user events with timestamps due to requiring the UrQueue.
  if (isTimestampedEvent && Event->UrQueue) {
    // If RecordEventEndTimestamp on the event is non-zero it means it has
    // collected the result of the queue already. In that case it has been
    // adjusted and is ready for immediate return.
    if (Event->RecordEventEndTimestamp)
      return ReturnValue(Event->RecordEventEndTimestamp);

    // Otherwise we need to collect it from the queue.
    auto Entry = Event->UrQueue->EndTimeRecordings.find(Event);

    // Unexpected state if there is no end-time record.
    if (Entry == Event->UrQueue->EndTimeRecordings.end())
      return UR_RESULT_ERROR_UNKNOWN;
    auto &EndTimeRecording = Entry->second;

    // End time needs to be adjusted for resolution and valid bits. A single
    // timestamp has no separate start value to detect wrap-around against.
    uint64_t ContextEndTime =
        (EndTimeRecording & TimestampMaxValue) * ZeTimerResolution;

    // If the result is 0, we have not yet gotten results back and so we just
    // return it.
    if (ContextEndTime == 0)
      return ReturnValue(ContextEndTime);

    // Now that we have the result, there is no need to keep it in the queue
    // anymore, so we cache it on the event and evict the record from the
    // queue.
    Event->RecordEventEndTimestamp = ContextEndTime;
    Event->UrQueue->EndTimeRecordings.erase(Entry);

    return ReturnValue(ContextEndTime);
  }

  ze_kernel_timestamp_result_t tsResult;

  // A Command-buffer consists of three command-lists for which only a single
  // event is returned to users. The actual profiling information related to the
  // command-buffer should therefore be extrated from graph events themsleves.
  // The timestamps of these events are saved in a memory region attached to
  // event usning CommandData field. The timings must therefore be recovered
  // from this memory.
  if (Event->CommandType == UR_COMMAND_ENQUEUE_COMMAND_BUFFER_EXP) {
    if (Event->CommandData) {
      command_buffer_profiling_t *ProfilingsPtr;
      switch (PropName) {
      case UR_PROFILING_INFO_COMMAND_START: {
        ProfilingsPtr =
            static_cast<command_buffer_profiling_t *>(Event->CommandData);
        // Sync-point order does not necessarily match to the order of
        // execution. We therefore look for the first command executed.
        uint64_t MinStart = ProfilingsPtr->Timestamps[0].global.kernelStart;
        for (uint64_t i = 1; i < ProfilingsPtr->NumEvents; i++) {
          uint64_t Timestamp = ProfilingsPtr->Timestamps[i].global.kernelStart;
          if (Timestamp < MinStart) {
            MinStart = Timestamp;
          }
        }
        uint64_t ContextStartTime =
            (MinStart & TimestampMaxValue) * ZeTimerResolution;
        return ReturnValue(ContextStartTime);
      }
      case UR_PROFILING_INFO_COMMAND_END: {
        ProfilingsPtr =
            static_cast<command_buffer_profiling_t *>(Event->CommandData);
        // Sync-point order does not necessarily match to the order of
        // execution. We therefore look for the last command executed.
        uint64_t MaxEnd = ProfilingsPtr->Timestamps[0].global.kernelEnd;
        uint64_t LastStart = ProfilingsPtr->Timestamps[0].global.kernelStart;
        for (uint64_t i = 1; i < ProfilingsPtr->NumEvents; i++) {
          uint64_t Timestamp = ProfilingsPtr->Timestamps[i].global.kernelEnd;
          if (Timestamp > MaxEnd) {
            MaxEnd = Timestamp;
            LastStart = ProfilingsPtr->Timestamps[i].global.kernelStart;
          }
        }
        uint64_t ContextStartTime = (LastStart & TimestampMaxValue);
        uint64_t ContextEndTime = (MaxEnd & TimestampMaxValue);

        //
        // Handle a possible wrap-around (the underlying HW counter is <
        // 64-bit). Note, it will not report correct time if there were multiple
        // wrap arounds, and the longer term plan is to enlarge the capacity of
        // the HW timestamps.
        //
        if (ContextEndTime <= ContextStartTime) {
          ContextEndTime += TimestampMaxValue;
        }
        ContextEndTime *= ZeTimerResolution;
        return ReturnValue(ContextEndTime);
      }
      case UR_PROFILING_INFO_COMMAND_COMPLETE:
        UR_LOG(ERR, "urEventGetProfilingInfo: "
                    "UR_PROFILING_INFO_COMMAND_COMPLETE not supported");
        return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
      default:
        UR_LOG(ERR, "urEventGetProfilingInfo: not supported ParamName");
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    } else {
      return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
    }
  }

  switch (PropName) {
  case UR_PROFILING_INFO_COMMAND_START: {
    ZE2UR_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));
    uint64_t ContextStartTime =
        (tsResult.global.kernelStart & TimestampMaxValue) * ZeTimerResolution;
    return ReturnValue(ContextStartTime);
  }
  case UR_PROFILING_INFO_COMMAND_END: {
    ZE2UR_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));

    uint64_t ContextStartTime =
        (tsResult.global.kernelStart & TimestampMaxValue);
    uint64_t ContextEndTime = (tsResult.global.kernelEnd & TimestampMaxValue);

    //
    // Handle a possible wrap-around (the underlying HW counter is < 64-bit).
    // Note, it will not report correct time if there were multiple wrap
    // arounds, and the longer term plan is to enlarge the capacity of the
    // HW timestamps.
    //
    if (ContextEndTime <= ContextStartTime) {
      ContextEndTime += TimestampMaxValue;
    }
    ContextEndTime *= ZeTimerResolution;
    return ReturnValue(ContextEndTime);
  }
  case UR_PROFILING_INFO_COMMAND_QUEUED:
  case UR_PROFILING_INFO_COMMAND_SUBMIT:
    // Note: No users for this case
    // The "command_submit" time is implemented by recording submission
    // timestamp with a call to urDeviceGetGlobalTimestamps before command
    // enqueue.
    //
    return ReturnValue(uint64_t{0});
  case UR_PROFILING_INFO_COMMAND_COMPLETE:
    UR_LOG(ERR, "urEventGetProfilingInfo: UR_PROFILING_INFO_COMMAND_COMPLETE "
                "not supported");
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default:
    UR_LOG(ERR, "urEventGetProfilingInfo: not supported ParamName");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueTimestampRecordingExp(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] blocking or non-blocking enqueue
    bool Blocking,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out] return an event object that identifies this particular
    /// command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternalPtr = v1_cast(OutEvent);
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  bool UseCopyEngine = false;
  ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitListInternal, Queue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList,
      EventWaitListInternal,
      /* AllowBatching */ false, nullptr /*ForcedCmdQueue*/));

  UR_CALL(createEventAndAssociateQueue(
      Queue, OutEventInternalPtr, UR_COMMAND_TIMESTAMP_RECORDING_EXP,
      CommandList,
      /* IsInternal */ false, /* HostVisible */ true));
  auto OutEventInternal = *OutEventInternalPtr;
  ze_event_handle_t ZeEvent = OutEventInternal->ZeEvent;
  OutEventInternal->WaitList = TmpWaitList;

  // Reset the end timestamp, in case it has been previously used.
  OutEventInternal->RecordEventEndTimestamp = 0;

  // Mark this event as timestamped
  OutEventInternal->IsTimestamped = true;

  // Create a new entry in the queue's recordings.
  Queue->EndTimeRecordings[*OutEventInternalPtr] = 0;

  ZE2UR_CALL(zeCommandListAppendWriteGlobalTimestamp,
             (CommandList->first,
              &Queue->EndTimeRecordings[*OutEventInternalPtr], ZeEvent,
              OutEventInternal->WaitList.Length,
              OutEventInternal->WaitList.ZeEventList));

  UR_CALL(
      Queue->executeCommandList(CommandList, Blocking, false /* OkToBatch */));

  return UR_RESULT_SUCCESS;
}

ur_result_t
/// [in] number of events in the event list
urEventWait(uint32_t NumEvents,
            /// [in][range(0, numEvents)] pointer to a
            /// list of events to wait for completion
            const ::ur_event_handle_t *EventWaitListOpque) {
  auto EventWaitList = v1_cast(EventWaitListOpque);
  for (uint32_t I = 0; I < NumEvents; I++) {
    auto e = EventWaitList[I];
    auto UrQueue = e->UrQueue;
    if (UrQueue && UrQueue->ZeEventsScope == OnDemandHostVisibleProxy) {
      // Make sure to add all host-visible "proxy" event signals if needed.
      // This ensures that all signalling commands are submitted below and
      // thus proxy events can be waited without a deadlock.
      //
      ur::level_zero::v1::ur_event_handle_t_ *Event = e;
      if (!Event->hasExternalRefs())
        die("urEventWait must not be called for an internal event");

      ze_event_handle_t ZeHostVisibleEvent;
      if (auto Res = Event->getOrCreateHostVisibleEvent(ZeHostVisibleEvent))
        return Res;
    }
  }
  // Submit dependent open command lists for execution, if any
  for (uint32_t I = 0; I < NumEvents; I++) {
    ur::level_zero::v1::ur_event_handle_t_ *Event = EventWaitList[I];
    auto UrQueue = Event->UrQueue;
    if (UrQueue) {
      // Lock automatically releases when this goes out of scope.
      std::scoped_lock<ur_shared_mutex> lock(UrQueue->Mutex);

      UR_CALL(UrQueue->executeAllOpenCommandLists());
    }
  }
  std::unordered_set<ur_queue_handle_t> Queues;
  for (uint32_t I = 0; I < NumEvents; I++) {
    {
      ur::level_zero::v1::ur_event_handle_t_ *Event = EventWaitList[I];
      {
        std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
        if (!Event->hasExternalRefs())
          die("urEventWait must not be called for an internal event");

        if (!Event->Completed) {
          auto HostVisibleEvent = Event->HostVisibleEvent;
          if (!HostVisibleEvent)
            die("The host-visible proxy event missing");

          ze_event_handle_t ZeEvent = HostVisibleEvent->ZeEvent;
          UR_LOG(DEBUG, "ZeEvent = {}", ur_cast<std::uintptr_t>(ZeEvent));
          // If this event was an inner batched event, then sync with
          // the Queue instead of waiting on the event.
          if (HostVisibleEvent->IsInnerBatchedEvent && Event->ZeBatchedQueue) {
            ZE2UR_CALL(zeHostSynchronize, (Event->ZeBatchedQueue));
          } else {
            ZE2UR_CALL(zeHostSynchronize, (ZeEvent));
          }
          Event->Completed = true;
        }
      }
      if (auto Q = Event->UrQueue) {
        if (Q->UsingImmCmdLists && Q->isInOrderQueue())
          // Use information about waited event to cleanup completed events in
          // the in-order queue.
          CleanupEventsInImmCmdLists(
              Event->UrQueue, false /* QueueLocked */, false /* QueueSynced */,
              reinterpret_cast<ur_event_handle_t>(Event));
        else {
          // NOTE: we are cleaning up after the event here to free resources
          // sooner in case run-time is not calling urEventRelease soon enough.
          CleanupCompletedEvent(reinterpret_cast<ur_event_handle_t>(Event),
                                false /*QueueLocked*/,
                                false /*SetEventCompleted*/);
          // For the case when we have out-of-order queue or regular command
          // lists its more efficient to check fences so put the queue in the
          // set to cleanup later.
          Queues.insert(Q);
        }
      }
    }
  }

  // We waited some events above, check queue for signaled command lists and
  // reset them.
  for (auto &Q : Queues) {
    std::unique_lock<ur_shared_mutex> Lock(Q->Mutex);
    resetCommandLists(Q);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
/// [in] handle of the event object
urEventRetain(
    /** [in] handle of the event object */ ::ur_event_handle_t EventOpque) {
  auto Event = v1_cast(EventOpque);
  Event->RefCountExternal++;
  Event->RefCount.retain();

  return UR_RESULT_SUCCESS;
}

ur_result_t

urEventRelease(
    /** [in] handle of the event object */ ::ur_event_handle_t EventOpque) {
  auto Event = v1_cast(EventOpque);
  Event->RefCountExternal--;
  bool isEventsWaitCompleted =
      (Event->CommandType == UR_COMMAND_EVENTS_WAIT ||
       Event->CommandType == UR_COMMAND_EVENTS_WAIT_WITH_BARRIER) &&
      Event->Completed;
  bool isEventDeleted = false;
  UR_CALL(urEventReleaseInternal(Event, &isEventDeleted));
  // If this is a Completed Event Wait Out Event, then we need to cleanup the
  // event at user release and not at the time of completion.
  // Use CleanupCompletedEvent which is a no-op if the event was already
  // cleaned up (e.g. by CleanupEventListFromResetCmdList), preventing a
  // double-release of the internal reference count.
  if (isEventsWaitCompleted & !isEventDeleted) {
    UR_CALL(CleanupCompletedEvent(Event, false, false));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urEventGetNativeHandle(
    /// [in] handle of the event.
    ::ur_event_handle_t EventOpque,
    /// [out] a pointer to the native handle of the event.
    ::ur_native_handle_t *NativeEvent) {
  auto Event = v1_cast(EventOpque);
  {
    std::shared_lock<ur_shared_mutex> Lock(Event->Mutex);
    auto *ZeEvent = ur_cast<ze_event_handle_t *>(NativeEvent);
    *ZeEvent = Event->ZeEvent;
  }
  // Event can potentially be in an open command-list, make sure that
  // it is submitted for execution to avoid potential deadlock if
  // interop app is going to wait for it.
  auto Queue = Event->UrQueue;
  if (Queue) {
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);
    const auto &OpenCommandList = Queue->eventOpenCommandList(Event);
    if (OpenCommandList != Queue->CommandListMap.end()) {
      UR_CALL(
          Queue->executeOpenCommandList(OpenCommandList->second.isCopy(Queue)));
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urExtEventCreate(
    /// [in] handle of the context object
    ::ur_context_handle_t ContextOpque,
    ::ur_event_handle_t
        /// [out] pointer to the handle of the event object created.
        *Event) {
  auto Context = v1_cast(ContextOpque);
  auto EventInternal = v1_cast(Event);
  UR_CALL(EventCreate(Context, nullptr /*Queue*/, false /*IsMultiDevice*/,
                      true /*HostVisible*/, EventInternal,
                      false /*CounterBasedEventEnabled*/,
                      false /*ForceDisableProfiling*/, false));

  (*EventInternal)->RefCountExternal++;
  if (!(*EventInternal)->CounterBasedEventsEnabled)
    ZE2UR_CALL(zeEventHostSignal, ((*EventInternal)->ZeEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t urEventCreateWithNativeHandle(
    /// [in] the native handle of the event.
    ::ur_native_handle_t NativeEvent,
    /// [in] handle of the context object
    ::ur_context_handle_t ContextOpque,
    const ur_event_native_properties_t *Properties,
    /// [out] pointer to the handle of the event object created.
    ::ur_event_handle_t *Event) {
  auto Context = v1_cast(ContextOpque);
  auto EventInternal = v1_cast(Event);

  // we dont have urEventCreate, so use this check for now to know that
  // the call comes from urEventCreate()
  if (reinterpret_cast<ze_event_handle_t>(NativeEvent) == nullptr) {
    UR_CALL(EventCreate(Context, nullptr /*Queue*/, false /*IsMultiDevice*/,
                        true /*HostVisible*/, EventInternal,
                        false /*CounterBasedEventEnabled*/,
                        false /*ForceDisableProfiling*/, false));

    (*EventInternal)->RefCountExternal++;
    if (!(*EventInternal)->CounterBasedEventsEnabled)
      ZE2UR_CALL(zeEventHostSignal, ((*EventInternal)->ZeEvent));
    return UR_RESULT_SUCCESS;
  }

  auto ZeEvent = ur_cast<ze_event_handle_t>(NativeEvent);
  ur::level_zero::v1::ur_event_handle_t_ *UREvent{};
  try {
    UREvent = new ur::level_zero::v1::ur_event_handle_t_(
        ZeEvent, nullptr /* ZeEventPool */, Context, UR_EXT_COMMAND_TYPE_USER,
        Properties->isNativeHandleOwned);
    UREvent->RefCountExternal++;

  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  // Assume native event is host-visible, or otherwise we'd
  // need to create a host-visible proxy for it.
  UREvent->HostVisibleEvent = reinterpret_cast<ur_event_handle_t>(UREvent);

  // Unlike regular events managed by SYCL RT we don't have to wait for interop
  // events completion, and not need to do the their `cleanup()`. This in
  // particular guarantees that the extra `urEventRelease` is not called on
  // them. That release is needed to match the `urEventRetain` of regular events
  // made for waiting for event completion, but not this interop event.
  UREvent->CleanedUp = true;

  *EventInternal = UREvent;

  return UR_RESULT_SUCCESS;
}

ur_result_t urEventSetCallback(
    /// [in] handle of the event object
    ::ur_event_handle_t /*Event*/,
    /// [in] execution status of the event
    ur_execution_info_t /*ExecStatus*/,
    /// [in] execution status of the event
    ur_event_callback_t /*Notify*/,
    /// [in][out][optional] pointer to data to be passed to callback.
    void * /*UserData*/) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urEventCreateExp(::ur_context_handle_t /*hContext*/,
                             ::ur_device_handle_t /*hDevice*/,
                             const ur_exp_event_desc_t * /*pEventDesc*/,
                             ::ur_event_handle_t * /*phEvent*/) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCGetEventHandleExp(::ur_event_handle_t /*hEvent*/,
                                   void ** /*ppIPCEventHandleData*/,
                                   size_t * /*pIPCEventHandleDataSizeRet*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCPutEventHandleExp(::ur_context_handle_t /*hContext*/,
                                   void * /*pIPCEventHandleData*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCOpenEventHandleExp(::ur_context_handle_t /*hContext*/,
                                    const void * /*pIPCEventHandleData*/,
                                    size_t /*ipcEventHandleDataSize*/,
                                    ::ur_event_handle_t * /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_event_handle_t_::getOrCreateHostVisibleEvent(
    ze_event_handle_t &ZeHostVisibleEvent) {
  auto UrQueue = this->UrQueue;
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(UrQueue->Mutex,
                                                          this->Mutex);

  if (!HostVisibleEvent) {
    this->IsCreatingHostProxyEvent = true;
    if (UrQueue->ZeEventsScope != OnDemandHostVisibleProxy)
      die("getOrCreateHostVisibleEvent: missing host-visible event");

    // Submit the command(s) signalling the proxy event to the queue.
    // We have to first submit a wait for the device-only event for which this
    // proxy is created.
    //
    // Get a new command list to be used on this call

    // We want to batch these commands to avoid extra submissions (costly)
    bool OkToBatch = true;

    ur_command_list_ptr_t CommandList{};
    UR_CALL(UrQueue->Context->getAvailableCommandList(
        UrQueue, CommandList, false /* UseCopyEngine */, 0, nullptr, OkToBatch,
        nullptr /*ForcedCmdQueue*/))

    // Create a "proxy" host-visible event.
    UR_CALL(createEventAndAssociateQueue(
        UrQueue, &HostVisibleEvent, UR_EXT_COMMAND_TYPE_USER, CommandList,
        /* IsInternal */ false, /* IsMultiDevice */ false,
        /* HostVisible */ true));

    if (this->IsInnerBatchedEvent) {
      ZE2UR_CALL(zeCommandListAppendBarrier,
                 (CommandList->first, ZeEvent, 0, nullptr));
    } else {
      ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
                 (CommandList->first, 1, &ZeEvent));
    }
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (CommandList->first, HostVisibleEvent->ZeEvent));

    UR_CALL(UrQueue->executeCommandList(CommandList, false, OkToBatch))
    this->IsCreatingHostProxyEvent = false;
  }

  ZeHostVisibleEvent = HostVisibleEvent->ZeEvent;
  return UR_RESULT_SUCCESS;
}

/**
 * @brief Destructor for the ur_event_handle_t_ class.
 *
 * This destructor is responsible for cleaning up the event handle when the
 * object is destroyed. It checks if the event (`ZeEvent`) is valid and if the
 * event has been completed (`Completed`). If both conditions are met, it
 * further checks if the associated queue (`UrQueue`) is valid and if it is not
 * set to discard events. If all conditions are satisfied, it calls
 * `zeEventDestroy` to destroy the event.
 *
 * This ensures that resources are properly released and avoids potential memory
 * leaks or resource mismanagement.
 */
ur::level_zero::v1::ur_event_handle_t_::~ur_event_handle_t_() {
  if (this->ZeEvent && this->Completed && checkL0LoaderTeardown()) {
    if (this->UrQueue && !this->UrQueue->isDiscardEvents())
      ZE_CALL_NOCHECK(zeEventDestroy, (this->ZeEvent));
  }
}

ur_result_t urEventReleaseInternal(ur_event_handle_t Event,
                                   bool *isEventDeleted) {
  if (!Event->RefCount.release())
    return UR_RESULT_SUCCESS;

  if (Event->OriginAllocEvent) {
    urEventReleaseInternal(Event->OriginAllocEvent, isEventDeleted);
  }

  if (Event->CommandType == UR_COMMAND_MEM_UNMAP && Event->CommandData) {
    // Free the memory allocated in the urEnqueueMemBufferMap.
    if (auto Res = ZeMemFreeHelper(Event->Context, Event->CommandData))
      return Res;
    Event->CommandData = nullptr;
  }
  if (Event->CommandType == UR_COMMAND_ENQUEUE_COMMAND_BUFFER_EXP &&
      Event->CommandData) {
    // Free the memory extra event allocated for profiling purposed.
    command_buffer_profiling_t *ProfilingPtr =
        static_cast<command_buffer_profiling_t *>(Event->CommandData);
    delete[] ProfilingPtr->Timestamps;
    delete ProfilingPtr;
    Event->CommandData = nullptr;
  }
  if (Event->OwnNativeHandle) {
    if (DisableEventsCaching) {
      if (checkL0LoaderTeardown()) {
        auto ZeResult = ZE_CALL_NOCHECK(zeEventDestroy, (Event->ZeEvent));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && (ZeResult != ZE_RESULT_ERROR_UNINITIALIZED &&
                         ZeResult != ZE_RESULT_ERROR_UNKNOWN))
          return ze2urResult(ZeResult);
        if (ZeResult == ZE_RESULT_ERROR_UNKNOWN) {
          ZeResult = ZE_RESULT_ERROR_UNINITIALIZED;
        }
      }
      Event->ZeEvent = nullptr;
      auto Context = Event->Context;
      if (auto Res = Context->decrementUnreleasedEventsInPool(Event))
        return Res;
    }
  }
  // It is possible that host-visible event was never created.
  // In case it was check if that's different from this same event
  // and release a reference to it.
  if (Event->HostVisibleEvent && Event->HostVisibleEvent != Event) {
    // Decrement ref-count of the host-visible proxy event.
    UR_CALL(urEventReleaseInternal(Event->HostVisibleEvent, isEventDeleted));
  }

  // Save pointer to the queue before deleting/resetting event.
  auto Queue = Event->UrQueue;

  // If the event was a timestamp recording, we try to evict its entry in the
  // queue.
  if (Event->IsTimestamped) {
    auto Entry = Queue->EndTimeRecordings.find(Event);
    if (Entry != Queue->EndTimeRecordings.end()) {
      auto &EndTimeRecording = Entry->second;
      if (EndTimeRecording == 0) {
        // If the end time recording has not finished, we tell the queue that
        // the event is no longer alive to avoid invalid write-backs.
        Queue->EvictedEndTimeRecordings.insert(
            Queue->EndTimeRecordings.extract(Entry));
      } else {
        // Otherwise we evict the entry.
        Queue->EndTimeRecordings.erase(Entry);
      }
    }
  }

  // When we add an event to the cache we need to check whether profiling is
  // enabled or not, so we access properties of the queue and that's why queue
  // must released later.
  if (DisableEventsCaching || !Event->OwnNativeHandle) {
    delete Event;
    if (isEventDeleted) {
      *isEventDeleted = true;
    }
  } else {
    Event->Context->addEventToContextCache(Event);
  }

  // We intentionally incremented the reference counter when an event is
  // created so that we can avoid ur_queue_handle_t is released before the
  // associated ur_event_handle_t is released. Here we have to decrement it so
  // ur_queue_handle_t can be released successfully.
  if (Queue) {
    UR_CALL(urQueueReleaseInternal(Queue));
  }

  return UR_RESULT_SUCCESS;
}

// Helper function to implement zeHostSynchronize.
// The behavior is to avoid infinite wait during host sync under ZE_DEBUG.
// This allows for a much more responsive debugging of hangs.
//
template <typename T, typename Func>
ze_result_t zeHostSynchronizeImpl(Func Api, T Handle) {
  if (!UrL0Debug) {
    return Api(Handle, UINT64_MAX);
  }

  ze_result_t R;
  while ((R = Api(Handle, 1000)) == ZE_RESULT_NOT_READY)
    ;
  return R;
}

// Template function to do various types of host synchronizations.
// This is intended to be used instead of direct calls to specific
// Level-Zero synchronization APIs.
//
template <typename T> ze_result_t zeHostSynchronize(T Handle);
template <> ze_result_t zeHostSynchronize(ze_event_handle_t Handle) {
  return zeHostSynchronizeImpl(zeEventHostSynchronize, Handle);
}
template <> ze_result_t zeHostSynchronize(ze_command_queue_handle_t Handle) {
  return zeHostSynchronizeImpl(zeCommandQueueSynchronize, Handle);
}

// Perform any necessary cleanup after an event has been signalled.
// This currently makes sure to release any kernel that may have been used by
// the event, updates the last command event in the queue and cleans up all dep
// events of the event.
// If the caller locks queue mutex then it must pass 'true' to QueueLocked.
ur_result_t CleanupCompletedEvent(ur_event_handle_t Event, bool QueueLocked,
                                  bool SetEventCompleted) {
  ur_kernel_handle_t AssociatedKernel = nullptr;
  // List of dependent events.
  std::list<ur_event_handle_t> EventsToBeReleased;
  ur_queue_handle_t AssociatedQueue = nullptr;
  {
    // If the Event is already locked, then continue with the cleanup, otherwise
    // block on locking the event.
    std::unique_lock<ur_shared_mutex> EventLock(Event->Mutex, std::try_to_lock);
    if (!EventLock.owns_lock() && !Event->IsCreatingHostProxyEvent) {
      EventLock.lock();
    }
    if (SetEventCompleted)
      Event->Completed = true;
    // Exit early of event was already cleanedup.
    if (Event->CleanedUp)
      return UR_RESULT_SUCCESS;

    AssociatedQueue = Event->UrQueue;

    // Remember the kernel associated with this event if there is one. We are
    // going to release it later.
    if (Event->CommandType == UR_COMMAND_KERNEL_LAUNCH && Event->CommandData) {
      AssociatedKernel =
          reinterpret_cast<ur_kernel_handle_t>(Event->CommandData);
      Event->CommandData = nullptr;
    }

    // Make a list of all the dependent events that must have signalled
    // because this event was dependent on them.
    Event->WaitList.collectEventsForReleaseAndDestroyUrZeEventList(
        EventsToBeReleased);

    Event->CleanedUp = true;
  }

  auto ReleaseIndirectMem = [](ur_kernel_handle_t Kernel) {
    if (IndirectAccessTrackingEnabled) {
      // urKernelRelease is called by CleanupCompletedEvent(Event) as soon as
      // kernel execution has finished. This is the place where we need to
      // release memory allocations. If kernel is not in use (not submitted by
      // some other thread) then release referenced memory allocations. As a
      // result, memory can be deallocated and context can be removed from
      // container in the platform. That's why we need to lock a mutex here.
      ur_platform_handle_t Plt =
          v1_cast(Kernel->Program->Context)->getPlatform();
      std::scoped_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex);

      if (--Kernel->SubmissionsCount == 0) {
        // Kernel is not submitted for execution, release referenced memory
        // allocations.
        for (auto &MemAlloc : Kernel->MemAllocs) {
          // std::pair<void *const, MemAllocRecord> *, Hash
          USMFreeHelper(MemAlloc->second.Context, MemAlloc->first,
                        MemAlloc->second.OwnNativeHandle);
        }
        Kernel->MemAllocs.clear();
      }
    }
  };

  // We've reset event data members above, now cleanup resources.
  if (AssociatedKernel) {
    ReleaseIndirectMem(AssociatedKernel);
    UR_CALL(ur::level_zero::v1::urKernelRelease(v1_cast(AssociatedKernel)));
  }

  if (AssociatedQueue) {
    {
      // Lock automatically releases when this goes out of scope.
      std::unique_lock<ur_shared_mutex> QueueLock(AssociatedQueue->Mutex,
                                                  std::defer_lock);
      if (!QueueLocked)
        QueueLock.lock();

      // If this event was the LastCommandEvent in the queue, being used
      // to make sure that commands were executed in-order, remove this.
      // If we don't do this, the event can get released and freed leaving
      // a dangling pointer to this event.  It could also cause unneeded
      // already finished events to show up in the wait list.
      if (AssociatedQueue->LastCommandEvent == Event) {
        AssociatedQueue->LastCommandEvent = nullptr;
      }
    }

    // Release this event since we explicitly retained it on creation and
    // association with queue. Events which don't have associated queue doesn't
    // require this release because it means that they are not created using
    // createEventAndAssociateQueue, i.e. additional retain was not made.
    UR_CALL(urEventReleaseInternal(Event));
  }

  // The list of dependent events will be appended to as we walk it so that this
  // algorithm doesn't go recursive due to dependent events themselves being
  // dependent on other events forming a potentially very deep tree, and deep
  // recursion.  That turned out to be a significant problem with the recursive
  // code that preceded this implementation.
  while (!EventsToBeReleased.empty()) {
    ur_event_handle_t DepEvent = EventsToBeReleased.front();
    DepEvent->Completed = true;
    EventsToBeReleased.pop_front();

    ur_kernel_handle_t DepEventKernel = nullptr;
    {
      std::scoped_lock<ur_shared_mutex> DepEventLock(DepEvent->Mutex);
      DepEvent->WaitList.collectEventsForReleaseAndDestroyUrZeEventList(
          EventsToBeReleased);
      if (IndirectAccessTrackingEnabled) {
        // DepEvent has finished, we can release the associated kernel if there
        // is one. This is the earliest place we can do this and it can't be
        // done twice, so it is safe. Lock automatically releases when this goes
        // out of scope.
        // TODO: this code needs to be moved out of the guard.
        if (DepEvent->CommandType == UR_COMMAND_KERNEL_LAUNCH &&
            DepEvent->CommandData) {
          DepEventKernel =
              reinterpret_cast<ur_kernel_handle_t>(DepEvent->CommandData);
          DepEvent->CommandData = nullptr;
        }
      }
    }
    if (DepEventKernel) {
      ReleaseIndirectMem(DepEventKernel);
      UR_CALL(ur::level_zero::v1::urKernelRelease(v1_cast(DepEventKernel)));
    }
    UR_CALL(urEventReleaseInternal(DepEvent));
  }

  return UR_RESULT_SUCCESS;
}

// Helper function for creating a PI event.
// The "Queue" argument specifies the PI queue where a command is submitted.
// The "HostVisible" argument specifies if event needs to be allocated from
// a host-visible pool.
//
ur_result_t
EventCreate(ur_context_handle_t Context, ur_queue_handle_t Queue,
            bool IsMultiDevice, bool HostVisible, ur_event_handle_t *RetEvent,
            bool CounterBasedEventEnabled, bool ForceDisableProfiling,
            bool InterruptBasedEventEnabled, std::optional<bool> IsInternal) {
  bool ProfilingEnabled =
      ForceDisableProfiling ? false : (!Queue || Queue->isProfilingEnabled());
  bool UsingImmediateCommandlists = !Queue || Queue->UsingImmCmdLists;

  // Handle optional IsInternal parameter - default to false if not provided
  bool isInternalValue = IsInternal.value_or(false);

  ur_device_handle_t Device = nullptr;

  if (!IsMultiDevice && Queue) {
    Device = Queue->Device;
  }

  if (auto CachedEvent = Context->getEventFromContextCache(
          HostVisible, ProfilingEnabled, Device, CounterBasedEventEnabled,
          InterruptBasedEventEnabled)) {
    *RetEvent = CachedEvent;
    return UR_RESULT_SUCCESS;
  }

  ze_event_handle_t ZeEvent;
  ze_event_pool_handle_t ZeEventPool = {};

  size_t Index = 0;

  if (auto Res = Context->getFreeSlotInExistingOrNewPool(
          ZeEventPool, Index, HostVisible, ProfilingEnabled, Device,
          CounterBasedEventEnabled, UsingImmediateCommandlists,
          InterruptBasedEventEnabled, Queue, isInternalValue))
    return Res;

  ZeStruct<ze_event_desc_t> ZeEventDesc;
  ZeEventDesc.index = Index;
  ZeEventDesc.wait = 0;

  if (HostVisible || CounterBasedEventEnabled || InterruptBasedEventEnabled) {
    ZeEventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  } else {
    //
    // Set the scope to "device" for every event. This is sufficient for
    // global device access and peer device access. If needed to be seen on
    // the host we are doing special handling, see EventsScope options.
    //
    // TODO: see if "sub-device" (ZE_EVENT_SCOPE_FLAG_SUBDEVICE) can better be
    //       used in some circumstances.
    //
    ZeEventDesc.signal = 0;
  }

  ZE2UR_CALL(zeEventCreate, (ZeEventPool, &ZeEventDesc, &ZeEvent));

  try {
    *RetEvent = new ur::level_zero::v1::ur_event_handle_t_(
        ZeEvent, ZeEventPool, Context, UR_EXT_COMMAND_TYPE_USER, true);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  (*RetEvent)->CounterBasedEventsEnabled = CounterBasedEventEnabled;
  (*RetEvent)->InterruptBasedEventsEnabled = InterruptBasedEventEnabled;
  if (HostVisible)
    (*RetEvent)->HostVisibleEvent =
        reinterpret_cast<ur_event_handle_t>(*RetEvent);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur::level_zero::v1::ur_event_handle_t_::reset() {
  // Clean up timestamp recording entry from queue if this was a timestamped
  // event
  if (IsTimestamped && UrQueue) {
    auto *Queue = UrQueue;
    auto Entry = Queue->EndTimeRecordings.find(this);
    if (Entry != Queue->EndTimeRecordings.end()) {
      Queue->EndTimeRecordings.erase(Entry);
    }
  }

  UrQueue = nullptr;
  CleanedUp = false;
  Completed = false;
  CommandData = nullptr;
  CommandType = UR_EXT_COMMAND_TYPE_USER;
  WaitList = {};
  RefCountExternal = 0;
  RefCount.reset();
  CommandList = std::nullopt;
  completionBatch = std::nullopt;
  OriginAllocEvent = nullptr;
  IsTimestamped = false;
  RecordEventEndTimestamp = 0;

  if (!isHostVisible())
    HostVisibleEvent = nullptr;
  if (!CounterBasedEventsEnabled)
    ZE2UR_CALL(zeEventHostReset, (ZeEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_ze_event_list_t::createAndRetainUrZeEventList(
    uint32_t EventListLength, const ur_event_handle_t *EventList,
    ur_queue_handle_t CurQueue, bool UseCopyEngine) {
  this->Length = 0;
  this->ZeEventList = nullptr;
  this->UrEventList = nullptr;

  if (CurQueue->isInOrderQueue() && CurQueue->LastCommandEvent != nullptr) {
    if (CurQueue->UsingImmCmdLists) {
      if (ReuseDiscardedEvents && CurQueue->isDiscardEvents()) {
        // If queue is in-order with discarded events and if
        // new command list is different from the last used command list then
        // signal new event from the last immediate command list. We are going
        // to insert a barrier in the new command list waiting for that event.
        auto QueueGroup = CurQueue->getQueueGroup(UseCopyEngine);
        uint32_t QueueGroupOrdinal, QueueIndex;
        auto NextIndex =
            QueueGroup.getQueueIndex(&QueueGroupOrdinal, &QueueIndex,
                                     /*QueryOnly */ true);
        auto NextImmCmdList = QueueGroup.ImmCmdLists[NextIndex];
        if (CurQueue->LastUsedCommandList != CurQueue->CommandListMap.end() &&
            CurQueue->LastUsedCommandList != NextImmCmdList) {
          CurQueue->signalEventFromCmdListIfLastEventDiscarded(
              CurQueue->LastUsedCommandList);
        }
      }
    } else {
      // Ensure LastCommandEvent's batch is submitted if it is differrent
      // from the one this command is going to. If we reuse discarded events
      // then signalEventFromCmdListIfLastEventDiscarded will be called at batch
      // close if needed.
      const auto &OpenCommandList =
          CurQueue->eventOpenCommandList(CurQueue->LastCommandEvent);
      if (OpenCommandList != CurQueue->CommandListMap.end() &&
          OpenCommandList->second.isCopy(CurQueue) != UseCopyEngine) {

        if (auto Res = CurQueue->executeOpenCommandList(
                OpenCommandList->second.isCopy(CurQueue)))
          return Res;
      }
    }
  }

  // For in-order queues, every command should be executed only after the
  // previous command has finished. The event associated with the last
  // enqueued command is added into the waitlist to ensure in-order semantics.
  bool IncludeLastCommandEvent =
      CurQueue->isInOrderQueue() && CurQueue->LastCommandEvent != nullptr;

  // If the last event is discarded then we already have a barrier waiting for
  // that event, so must not include the last command event into the wait
  // list because it will cause waiting for event which was reset.
  if (ReuseDiscardedEvents && CurQueue->isDiscardEvents() &&
      CurQueue->LastCommandEvent && CurQueue->LastCommandEvent->IsDiscarded)
    IncludeLastCommandEvent = false;

  // If we are using L0 native implementation for handling in-order queues,
  // then we don't need to add the last enqueued event into the waitlist, as
  // the native driver implementation will already ensure in-order semantics.
  // The only exception is when a different immediate command was last used on
  // the same UR Queue.
  if (CurQueue->Device->Platform->allowDriverInOrderLists(
          true /*Only Allow Driver In Order List if requested*/) &&
      CurQueue->isInOrderQueue() && CurQueue->UsingImmCmdLists) {
    auto QueueGroup = CurQueue->getQueueGroup(UseCopyEngine);
    uint32_t QueueGroupOrdinal, QueueIndex;
    auto NextIndex = QueueGroup.getQueueIndex(&QueueGroupOrdinal, &QueueIndex,
                                              /*QueryOnly */ true);
    auto NextImmCmdList = QueueGroup.ImmCmdLists[NextIndex];
    IncludeLastCommandEvent &=
        CurQueue->LastUsedCommandList != CurQueue->CommandListMap.end() &&
        NextImmCmdList != CurQueue->LastUsedCommandList;
  }

  uint32_t TmpListLength = 0;
  bool CurQueueUnlocked = false;
  ur_event_handle_t PendingMultiDeviceEvent = nullptr;
  ur_event_handle_t PendingRetainedEvent = nullptr;

  // On failure leave *this empty: release every event retained so far, free
  // the arrays and make sure the caller's lock of CurQueue is held again.
  bool Committed = false;
  OnScopeExit Rollback([&]() {
    if (Committed)
      return;
    if (CurQueueUnlocked)
      CurQueue->Mutex.lock();
    std::list<ur_event_handle_t> EventsToBeReleased;
    this->Length = TmpListLength;
    collectEventsForReleaseAndDestroyUrZeEventList(EventsToBeReleased);
    for (ur_event_handle_t Event :
         {PendingMultiDeviceEvent, PendingRetainedEvent}) {
      if (Event)
        EventsToBeReleased.push_back(Event);
    }
    for (ur_event_handle_t Event : EventsToBeReleased)
      urEventReleaseInternal(Event);
  });

  try {
    if (IncludeLastCommandEvent) {
      this->ZeEventList = new ze_event_handle_t[EventListLength + 1];
      this->UrEventList = new ur_event_handle_t[EventListLength + 1];
      std::shared_lock<ur_shared_mutex> Lock(CurQueue->LastCommandEvent->Mutex);
      this->ZeEventList[0] = CurQueue->LastCommandEvent->ZeEvent;
      this->UrEventList[0] = CurQueue->LastCommandEvent;
      this->UrEventList[0]->RefCount.retain();
      TmpListLength = 1;
    } else if (EventListLength > 0) {
      this->ZeEventList = new ze_event_handle_t[EventListLength];
      this->UrEventList = new ur_event_handle_t[EventListLength];
    }

    // For in-order queue and wait-list which is empty or has events only from
    // the same queue then we don't need to wait on any other additional events
    if (CurQueue->Device->Platform->allowDriverInOrderLists(
            true /*Only Allow Driver In Order List if requested*/) &&
        CurQueue->isInOrderQueue() &&
        WaitListEmptyOrAllEventsFromSameQueue(CurQueue, EventListLength,
                                              EventList)) {
      this->Length = TmpListLength;
      Committed = true;
      return UR_RESULT_SUCCESS;
    }

    if (EventListLength > 0) {
      for (uint32_t I = 0; I < EventListLength; I++) {
        {
          std::shared_lock<ur_shared_mutex> Lock(EventList[I]->Mutex);
          if (EventList[I]->Completed)
            continue;

          // Poll of the host-visible events.
          auto HostVisibleEvent = EventList[I]->HostVisibleEvent;
          if (FilterEventWaitList && HostVisibleEvent) {
            auto Res = ZE_CALL_NOCHECK(zeEventQueryStatus,
                                       (HostVisibleEvent->ZeEvent));
            if (Res == ZE_RESULT_SUCCESS) {
              // Event has already completed, don't put it into the list
              continue;
            }
          }
        }

        auto Queue = EventList[I]->UrQueue;

        auto CurQueueDevice = CurQueue->Device;
        std::optional<std::unique_lock<ur_shared_mutex>> QueueLock =
            std::nullopt;
        // The caller of createAndRetainUrZeEventList must already hold
        // a lock of the CurQueue. However, if the CurQueue is different
        // then the Event's Queue, we need to drop that lock and
        // acquire the Event's Queue lock. This is done to avoid a lock
        // ordering issue.
        // For the rest of this scope, CurQueue cannot be accessed.
        // TODO: This solution is very error-prone. This requires a refactor
        // to either have fine-granularity locks inside of the queues or
        // to move any operations on queues other than CurQueue out
        // of this scope.
        if (Queue && Queue != CurQueue) {
          CurQueue->Mutex.unlock();
          CurQueueUnlocked = true;
          QueueLock = std::unique_lock<ur_shared_mutex>(Queue->Mutex);
        }

        if (Queue) {
          // If the event that is going to be waited is in an open batch
          // different from where this next command is going to be added,
          // then we have to force execute of that open command-list
          // to avoid deadlocks.
          //
          const auto &OpenCommandList =
              Queue->eventOpenCommandList(EventList[I]);
          if (OpenCommandList != Queue->CommandListMap.end()) {

            if (Queue == CurQueue &&
                OpenCommandList->second.isCopy(Queue) == UseCopyEngine) {
              // Don't force execute the batch yet since the new command
              // is going to the same open batch as the dependent event.
            } else {
              if (auto Res = Queue->executeOpenCommandList(
                      OpenCommandList->second.isCopy(Queue)))
                return Res;
            }
          }
        } else {
          // There is a dependency on an interop-event.
          // Similarily to the above to avoid dead locks ensure that
          // execution of all prior commands in the current command-
          // batch is visible to the host. This may not be the case
          // when we intended to have only last command in the batch
          // produce host-visible event, e.g.
          //
          //  event0 = interop event
          //  event1 = command1 (already in batch, no deps)
          //  event2 = command2 (is being added, dep on event0)
          //  event3 = signal host-visible event for the batch
          //  event1.wait()
          //  event0.signal()
          //
          // Make sure that event1.wait() will wait for a host-visible
          // event that is signalled before the command2 is enqueued.
          if (CurQueue->ZeEventsScope != AllHostVisible) {
            CurQueue->executeAllOpenCommandLists();
          }
        }

        ur_command_list_ptr_t CommandList;
        if (Queue && Queue->Device != CurQueueDevice) {
          // Get a command list prior to acquiring an event lock.
          // This prevents a potential deadlock with recursive
          // event locks.
          UR_CALL(Queue->Context->getAvailableCommandList(
              Queue, CommandList, false /*UseCopyEngine*/, 0, nullptr,
              true /*AllowBatching*/, nullptr /*ForcedCmdQueue*/));
        }

        std::shared_lock<ur_shared_mutex> Lock(EventList[I]->Mutex);

        ur_device_handle_t QueueRootDevice = nullptr;
        ur_device_handle_t CurrentQueueRootDevice = nullptr;
        if (Queue) {
          QueueRootDevice = Queue->Device;
          CurrentQueueRootDevice = CurQueueDevice;
          if (Queue->Device->isSubDevice()) {
            QueueRootDevice = Queue->Device->RootDevice;
          }
          if (CurQueueDevice->isSubDevice()) {
            CurrentQueueRootDevice = CurQueueDevice->RootDevice;
          }
        }

        if (Queue && QueueRootDevice != CurrentQueueRootDevice &&
            !EventList[I]->IsMultiDevice) {
          ze_event_handle_t MultiDeviceZeEvent = nullptr;
          ur_event_handle_t MultiDeviceEvent;
          bool IsInternal = true;
          bool IsMultiDevice = true;

          UR_CALL(createEventAndAssociateQueue(
              Queue, &MultiDeviceEvent, EventList[I]->CommandType, CommandList,
              IsInternal, IsMultiDevice));
          PendingMultiDeviceEvent = MultiDeviceEvent;
          MultiDeviceZeEvent = MultiDeviceEvent->ZeEvent;
          const auto &ZeCommandList = CommandList->first;
          EventList[I]->RefCount.retain();
          PendingRetainedEvent = EventList[I];

          // Append a Barrier to wait on the original event while signalling the
          // new multi device event.
          ZE2UR_CALL(
              zeCommandListAppendBarrier,
              (ZeCommandList, MultiDeviceZeEvent, 1u, &EventList[I]->ZeEvent));
          UR_CALL(Queue->executeCommandList(CommandList, /* IsBlocking */ false,
                                            /* OkToBatchCommand */ true));

          // Acquire lock of newly created MultiDeviceEvent to increase it's
          // RefCount
          std::shared_lock<ur_shared_mutex> Lock(MultiDeviceEvent->Mutex);

          this->ZeEventList[TmpListLength] = MultiDeviceZeEvent;
          this->UrEventList[TmpListLength] = MultiDeviceEvent;
          this->UrEventList[TmpListLength]->RefCount.retain();
          PendingMultiDeviceEvent = nullptr;
          PendingRetainedEvent = nullptr;
        } else {
          this->ZeEventList[TmpListLength] = EventList[I]->ZeEvent;
          this->UrEventList[TmpListLength] = EventList[I];
          this->UrEventList[TmpListLength]->RefCount.retain();
        }
        TmpListLength += 1;

        if (QueueLock.has_value()) {
          QueueLock.reset();
          CurQueue->Mutex.lock();
          CurQueueUnlocked = false;
        }
      }
    }

    this->Length = TmpListLength;
    Committed = true;

  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_ze_event_list_t::insert(ur_ze_event_list_t &Other) {
  if (this != &Other) {
    // save of the previous object values
    uint32_t PreLength = this->Length;
    ze_event_handle_t *PreZeEventList = this->ZeEventList;
    ur_event_handle_t *PreUrEventList = this->UrEventList;

    // allocate new memory
    uint32_t Length = PreLength + Other.Length;
    this->ZeEventList = new ze_event_handle_t[Length];
    this->UrEventList = new ur_event_handle_t[Length];

    // copy elements
    uint32_t TmpListLength = 0;
    for (uint32_t I = 0; I < PreLength; I++) {
      this->ZeEventList[TmpListLength] = std::move(PreZeEventList[I]);
      this->UrEventList[TmpListLength] = std::move(PreUrEventList[I]);
      TmpListLength += 1;
    }
    for (uint32_t I = 0; I < Other.Length; I++) {
      this->ZeEventList[TmpListLength] = std::move(Other.ZeEventList[I]);
      this->UrEventList[TmpListLength] = std::move(Other.UrEventList[I]);
      TmpListLength += 1;
    }
    this->Length = TmpListLength;

    // Free previous allocated memory
    delete[] PreZeEventList;
    delete[] PreUrEventList;
    delete[] Other.ZeEventList;
    delete[] Other.UrEventList;
    Other.ZeEventList = nullptr;
    Other.UrEventList = nullptr;
    Other.Length = 0;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_ze_event_list_t::collectEventsForReleaseAndDestroyUrZeEventList(
    std::list<ur_event_handle_t> &EventsToBeReleased) {
  // event wait lists are owned by events, this function is called with owning
  // event lock taken, hence it is thread safe
  for (uint32_t I = 0; I < Length; I++) {
    // Add the event to be released to the list
    EventsToBeReleased.push_back(UrEventList[I]);
  }
  Length = 0;

  if (ZeEventList != nullptr) {
    delete[] ZeEventList;
    ZeEventList = nullptr;
  }
  if (UrEventList != nullptr) {
    delete[] UrEventList;
    UrEventList = nullptr;
  }

  return UR_RESULT_SUCCESS;
}

// Tells if this event is with profiling capabilities.
bool ur::level_zero::v1::ur_event_handle_t_::isProfilingEnabled() const {
  return !UrQueue || // tentatively assume user events are profiling enabled
         (UrQueue->Properties & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0;
}

} // namespace ur::level_zero::v1
