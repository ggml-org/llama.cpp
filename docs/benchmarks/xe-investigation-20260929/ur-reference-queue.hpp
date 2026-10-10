//===--------- queue.hpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <list>
#include <map>
#include <optional>
#include <queue>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <ur/ur.hpp>
#include <ur_ddi.h>
#include <ze_api.h>
#include <zes_api.h>

#include "common.hpp"
#include "device.hpp"

extern "C" {
ur_result_t urQueueReleaseInternal(ur_queue_handle_t Queue);
} // extern "C"

struct ur_completion_batch;
using ur_completion_batch_list = std::list<ur_completion_batch>;
using ur_completion_batch_it = ur_completion_batch_list::iterator;

// Event completion batch for aggregating status checks of many events into
// a single one through a barrier. Batches can be continuously reused.
// Batches start empty and accumulate events, get sealed (which issues
// an asynchronous barrier on the command list), and then must be waited on
// for all the events to complete.
struct ur_completion_batch {
  ur_completion_batch();
  ~ur_completion_batch();

  enum state {
    EMPTY,        // use() -> ACCUMULATING
    ACCUMULATING, // append() -> full -> seal() -> SEALED
    SEALED,       // checkComplete() -> COMPLETED
    COMPLETED,    // reset() -> EMPTY
  };

  // Returns the state of the batch. Might be stale.
  state getState();

  // Return the most up-to-date state of the batch. Might query the state of the
  // underlying barrier event.
  state queryState();

  // Must be called on any completion batch prior to being used for events.
  void use();

  // Checks whether the batch is at capacity. This is a soft limit and can be
  // exceeded if necessary.
  bool isFull();

  // Appends an event to the batch.
  void append();

  // Seals the event batch and appends a barrier to the command list.
  // Adding any further events after this, but before reset, is undefined.
  ur_result_t seal(ur_queue_handle_t queue, ze_command_list_handle_t cmdlist);

  // Resets a complete batch back to an empty state. Cleanups internal state
  // but keeps allocated resources for reuse.
  ur_result_t reset();

private:
  // Checks whether all the events in the batch have completed. Might query
  // the underlying event status. Can only be called on a sealed batch.
  bool checkComplete();

  // Internal barrier event that is signaled on completion of the batched
  // events.
  ur_event_handle_t barrierEvent;

  // Current batch state. Don't use directly.
  state st;

  // Number of accumulated events.
  size_t numEvents;
};

// A collection of event completion batches. Manages querying event status
// in batches of events instead of individually, reducing the number of total
// queries necessary to determine whether a set of events have signaled.
struct ur_completion_batches {
  // This structure should never be copied because it contains a stable iterator
  // into a list. Copying it would likely result in unexpected behavior.
  ur_completion_batches(const ur_completion_batches &) = delete;
  ur_completion_batches &operator=(const ur_completion_batches &) = delete;
  ur_completion_batches(ur_completion_batches &&) = default;
  ur_completion_batches &operator=(ur_completion_batches &&) = default;

  ur_completion_batches();

  // Cleans up completed batches, and, if the currently active batch
  // is full, attempts to find an empty batch to be used as active.
  // If one is found, the current one is sealed, and the new one is
  // set as active. Otherwise, UR_RESULT_ERROR_OUT_OF_RESOURCES is
  // returned to indicate that there are no batches available.
  // This is safe, but will increase how many events are associated
  // with the active batch.
  ur_result_t tryCleanup(ur_queue_handle_t queue,
                         ze_command_list_handle_t cmdlist,
                         std::vector<ur_event_handle_t> &EventList,
                         std::vector<ur_event_handle_t> &EventListToCleanup);

  // Adds an event to the the active batch.
  // Ideally, all events that are appended here are then provided in the
  // vector for cleanup. Otherwise the event batch will simply ignore
  // missing events when it comes time for cleanup.
  void append(ur_event_handle_t event);

  // Resets all the batches without waiting for event completion.
  // Only safe when the command list was fully synchronized through
  // other means.
  void forceReset();

private:
  // Checks the state of all previously sealed batches. If any are complete,
  // moves the associated events from the EventList to EventListToCleanup,
  // and then resets the batch for reuse.
  ur_result_t cleanup(std::vector<ur_event_handle_t> &EventList,
                      std::vector<ur_event_handle_t> &EventListToCleanup);

  // Moves the completed events from EventList to EventListToCleanup.
  void moveCompletedEvents(ur_completion_batch_it it,
                           std::vector<ur_event_handle_t> &EventList,
                           std::vector<ur_event_handle_t> &EventListToCleanup);

  // Find or creates an empty batch. This might fail if there are now empty
  // batches and a batch limit has been reached.
  std::optional<ur_completion_batch_it> findFirstEmptyBatchOrCreate();

  ur_completion_batch_list batches;
  std::queue<ur_completion_batch_it> sealed;
  ur_completion_batch_it active;
};

ur_result_t resetCommandLists(ur_queue_handle_t Queue);
ur_result_t CleanupEventsInImmCmdLists(ur_queue_handle_t UrQueue,
                                       bool QueueLocked, bool QueueSynced,
                                       ur_event_handle_t CompletedEvent);

// Structure describing the specific use of a command-list in a queue.
// This is because command-lists are re-used across multiple queues
// in the same context.
struct ur_command_list_info_t {
  ur_command_list_info_t(ze_fence_handle_t ZeFence, bool ZeFenceInUse,
                         bool IsClosed, ze_command_queue_handle_t ZeQueue,
                         ZeStruct<ze_command_queue_desc_t> ZeQueueDesc,
                         bool UseCompletionBatching, bool CanReuse,
                         bool IsInOrderList, bool IsImmediate)
      : ZeFence(ZeFence), ZeFenceInUse(ZeFenceInUse), IsClosed(IsClosed),
        ZeQueue(ZeQueue), ZeQueueDesc(ZeQueueDesc),
        IsInOrderList(IsInOrderList), CanReuse(CanReuse),
        IsImmediate(IsImmediate) {
    if (UseCompletionBatching) {
      completions = ur_completion_batches();
    }
  }
  // The Level-Zero fence that will be signalled at completion.
  // Immediate commandlists do not have an associated fence.
  // A nullptr for the fence indicates that this is an immediate commandlist.
  ze_fence_handle_t ZeFence;
  // Record if the fence is in use.
  // This is needed to avoid leak of the tracked command-list if the fence
  // was not yet signaled at the time all events in that list were already
  // completed (we are polling the fence at events completion). The fence
  // may be still "in-use" due to sporadic delay in HW.
  bool ZeFenceInUse;

  // Indicates if command list is in closed state. This is needed to avoid
  // appending commands to the closed command list.
  bool IsClosed;

  // Record the queue to which the command list will be submitted.
  ze_command_queue_handle_t ZeQueue;

  // Record the queue descriptor fields used when creating the command list
  // because we cannot recover these fields from the command list. Immediate
  // command lists are recycled across queues and then all fields are used. For
  // standard command lists only the ordinal is used. For queues created through
  // the make_queue API the descriptor is unavailable so a dummy descriptor is
  // used and then this entry is marked as not eligible for recycling.
  ZeStruct<ze_command_queue_desc_t> ZeQueueDesc;
  // Indicates if this is an inorder list
  bool IsInOrderList;
  bool CanReuse;
  bool IsImmediate;

  // Helper functions to tell if this is a copy command-list.
  bool isCopy(ur_queue_handle_t Queue) const;

  // An optional event completion batching mechanism for out-of-order immediate
  // command lists.
  std::optional<ur_completion_batches> completions;

  // Keeps events created by commands submitted into this command-list.
  // TODO: use this for explicit wait/cleanup of events at command-list
  // completion.
  // TODO: use this for optimizing events in the same command-list, e.g.
  // only have last one visible to the host.
  std::vector<ur_event_handle_t> EventList;
  size_t size() const { return EventList.size(); }
  void append(ur_event_handle_t Event);
};

// The map type that would track all command-lists in a queue.
using ur_command_list_map_t =
    std::unordered_map<ze_command_list_handle_t, ur_command_list_info_t>;
// The iterator pointing to a specific command-list in use.
using ur_command_list_ptr_t = ur_command_list_map_t::iterator;

struct ur_queue_handle_t_ : _ur_object {
  ur_queue_handle_t_(std::vector<ze_command_queue_handle_t> &ComputeQueues,
                     std::vector<ze_command_queue_handle_t> &CopyQueues,
                     ur_context_handle_t Context, ur_device_handle_t Device,
                     bool OwnZeCommandQueue, ur_queue_flags_t Properties = 0,
                     int ForceComputeIndex = -1);

  using queue_type = ur_device_handle_t_::queue_group_info_t::type;
  // PI queue is in general a one to many mapping to L0 native queues.
  struct ur_queue_group_t {
    ur_queue_handle_t Queue;
    ur_queue_group_t() = delete;

    // The Queue argument captures the enclosing PI queue.
    // The Type argument specifies the type of this queue group.
    // The actual ZeQueues are populated at PI queue construction.
    ur_queue_group_t(ur_queue_handle_t Queue, queue_type Type)
        : Queue(Queue), Type(Type) {}

    // The type of the queue group.
    queue_type Type;
    bool isCopy() const { return Type != queue_type::Compute; }

    // Level Zero command queue handles.
    std::vector<ze_command_queue_handle_t> ZeQueues;

    // Immediate commandlist handles, one per Level Zero command queue handle.
    // These are created only once, along with the L0 queues (see above)
    // and reused thereafter.
    std::vector<ur_command_list_ptr_t> ImmCmdLists;

    // Return the index of the next queue to use based on a
    // round robin strategy and the queue group ordinal.
    // If QueryOnly is true then return index values but don't update internal
    // index data members of the queue.
    uint32_t getQueueIndex(uint32_t *QueueGroupOrdinal, uint32_t *QueueIndex,
                           bool QueryOnly = false);

    // Get the ordinal for a command queue handle.
    int32_t getCmdQueueOrdinal(ze_command_queue_handle_t CmdQueue);

    // This function will return one of possibly multiple available native
    // queues and the value of the queue group ordinal.
    ze_command_queue_handle_t &getZeQueue(uint32_t *QueueGroupOrdinal);

    // This function sets an immediate commandlist from the interop interface.
    void setImmCmdList(ur_queue_handle_t queue, ze_command_list_handle_t);

    // This function returns the next immediate commandlist to use.
    ur_command_list_ptr_t &getImmCmdList();

    // These indices are to filter specific range of the queues to use,
    // and to organize round-robin across them.
    uint32_t UpperIndex{0};
    uint32_t LowerIndex{0};
    uint32_t NextIndex{0};
  };

  // Helper class to facilitate per-thread queue groups
  // We maintain a hashtable of queue groups if requested to do them per-thread.
  // Otherwise it is just single entry used for all threads.
  struct pi_queue_group_by_tid_t
      : public std::unordered_map<std::thread::id, ur_queue_group_t> {
    bool PerThread = false;

    // Returns thread id if doing per-thread, or a generic id that represents
    // all the threads.
    std::thread::id tid() const {
      return PerThread ? std::this_thread::get_id() : std::thread::id();
    }

    // Make the specified queue group be the master
    void set(const ur_queue_group_t &QueueGroup) {
      const auto &Device = QueueGroup.Queue->Device;
      PerThread =
          Device->ImmCommandListUsed == ur_device_handle_t_::PerThreadPerQueue;
      assert(empty());
      insert({tid(), QueueGroup});
    }

    // Get a queue group to use for this thread
    ur_queue_group_t &get() {
      assert(!empty());
      auto It = find(tid());
      if (It != end()) {
        return It->second;
      }
      // Add new queue group for this thread initialized from a master entry.
      auto QueueGroup = begin()->second;
      // Create space for queues and immediate commandlists, which are created
      // on demand.
      QueueGroup.ZeQueues = std::vector<ze_command_queue_handle_t>(
          QueueGroup.ZeQueues.size(), nullptr);
      QueueGroup.ImmCmdLists = std::vector<ur_command_list_ptr_t>(
          QueueGroup.ZeQueues.size(), QueueGroup.Queue->CommandListMap.end());

      std::tie(It, std::ignore) = insert({tid(), QueueGroup});
      return It->second;
    }
  };

  // A map of compute groups containing compute queue handles, one per thread.
  // When a queue is accessed from multiple host threads, a separate queue group
  // is created for each thread. The key used for mapping is the thread ID.
  pi_queue_group_by_tid_t ComputeQueueGroupsByTID;

  // A group containing copy queue handles. The main copy engine, if available,
  // comes first followed by link copy engines, if available.
  // When a queue is accessed from multiple host threads, a separate queue group
  // is created for each thread. The key used for mapping is the thread ID.
  pi_queue_group_by_tid_t CopyQueueGroupsByTID;

  // Keeps the PI context to which this queue belongs.
  // This field is only set at ur_queue_handle_t_ creation time, and
  // cannot change. Therefore it can be accessed without holding a lock on this
  // ur_queue_handle_t_.
  const ur_context_handle_t Context;

  // Keeps the PI device to which this queue belongs.
  // This field is only set at ur_queue_handle_t_ creation time, and
  // cannot change. Therefore it can be accessed without holding a lock on this
  // ur_queue_handle_t_.
  const ur_device_handle_t Device;

  // A queue may use either standard or immediate commandlists. At queue
  // construction time this is set based on the device and any env var settings
  // that change the default for the device type. When an interop queue is
  // constructed, the caller chooses the type of commandlists to use.
  bool UsingImmCmdLists = false;

  // Scope of events used for events on the queue
  // Can be adjusted with UR_L0_DEVICE_SCOPE_EVENTS
  // for non-immediate command lists
  EventsScope ZeEventsScope = AllHostVisible;

  // Keeps track of the event associated with the last enqueued command into
  // this queue. this is used to add dependency with the last command to add
  // in-order semantics and updated with the latest event each time a new
  // command is enqueued.
  ur_event_handle_t LastCommandEvent = nullptr;

  // Indicates if we own the ZeCommandQueue or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnZeCommandQueue;

  // Keeps the properties of this queue.
  ur_queue_flags_t Properties;

  // Keeps track of whether we are using Counter-based Events
  bool CounterBasedEventsEnabled = false;

  bool InterruptBasedEventsEnabled = false;

  // Map of all command lists used in this queue.
  ur_command_list_map_t CommandListMap;

  // Helper data structure to hold all variables related to batching
  struct command_batch {
    // These two members are used to keep track of how often the
    // batching closes and executes a command list before reaching the
    // QueueComputeBatchSize limit, versus how often we reach the limit.
    // This info might be used to vary the QueueComputeBatchSize value.
    uint32_t NumTimesClosedEarly = {0};
    uint32_t NumTimesClosedFull = {0};

    // Open command list fields for batching commands into this queue.
    ur_command_list_ptr_t OpenCommandList{};

    // Approximate number of commands that are allowed to be batched for
    // this queue.
    // Added this member to the queue rather than using a global variable
    // so that future implementation could use heuristics to change this on
    // a queue specific basis. And by putting it in the queue itself, this
    // is thread safe because of the locking of the queue that occurs.
    uint32_t QueueBatchSize = {0};
  };

  // ComputeCommandBatch holds data related to batching of non-copy commands.
  // CopyCommandBatch holds data related to batching of copy commands.
  command_batch ComputeCommandBatch, CopyCommandBatch;

  // A helper structure to keep active barriers of the queue.
  // It additionally manages ref-count of events in this list.
  struct active_barriers {
    std::vector<ur_event_handle_t> Events;
    void add(ur_event_handle_t &Event);
    ur_result_t clear();
    bool empty() { return Events.empty(); }
    std::vector<ur_event_handle_t> &vector() { return Events; }
  };
  // A collection of currently active barriers.
  // These should be inserted into a command list whenever an available command
  // list is needed for a command.
  active_barriers ActiveBarriers;

  // Besides each PI object keeping a total reference count in
  // _ur_object::RefCount we keep special track of the queue *external*
  // references. This way we are able to tell when the queue is being finished
  // externally, and can wait for internal references to complete, and do proper
  // cleanup of the queue.
  // This counter doesn't track the lifetime of a queue object, it only tracks
  // the number of external references. I.e. even if it reaches zero a queue
  // object may not be destroyed and can be used internally in the plugin.
  // That's why we intentionally don't use atomic type for this counter to
  // enforce guarding with a mutex all the work involving this counter.
  uint32_t RefCountExternal{1};

  // Indicates that the queue is healthy and all operations on it are OK.
  bool Healthy{true};

  // The following data structures and methods are used only for handling
  // in-order queue with discard_events property. Some commands in such queue
  // may have discarded event. Which means that event is not visible outside of
  // the plugin. It is possible to reset and reuse discarded events in the same
  // in-order queue because of the dependency between commands. We don't have to
  // wait event completion to do this. We use the following 2-event model to
  // reuse events inside each command list:
  //
  // Operation1 = zeCommantListAppendMemoryCopy (signal ze_event1)
  // zeCommandListAppendBarrier(wait for ze_event1)
  // zeCommandListAppendEventReset(ze_event1)
  // # Create new ur_event_handle_t using ze_event1 and append to the cache.
  //
  // Operation2 = zeCommandListAppendMemoryCopy (signal ze_event2)
  // zeCommandListAppendBarrier(wait for ze_event2)
  // zeCommandListAppendEventReset(ze_event2)
  // # Create new ur_event_handle_t using ze_event2 and append to the cache.
  //
  // # Get ur_event_handle_t from the beginning of the cache because there are
  // two events there. So it is guaranteed that we do round-robin between two
  // # events - event from the last command is appended to the cache.
  // Operation3 = zeCommandListAppendMemoryCopy (signal ze_event1)
  // # The same ze_event1 is used for Operation1 and Operation3.
  //
  // When we switch to a different command list we need to signal new event and
  // wait for it in the new command list using barrier.
  // [CmdList1]
  // Operation1 = zeCommantListAppendMemoryCopy (signal event1)
  // zeCommandListAppendBarrier(wait for event1)
  // zeCommandListAppendEventReset(event1)
  // zeCommandListAppendSignalEvent(NewEvent)
  //
  // [CmdList2]
  // zeCommandListAppendBarrier(wait for NewEvent)
  //
  // This barrier guarantees that command list execution starts only after
  // completion of previous command list which signals aforementioned event. It
  // allows to reset and reuse same event handles inside all command lists in
  // scope of the queue. It means that we need 2 reusable events of each type
  // (host-visible and device-scope) per queue at maximum.

  // This data member keeps track of the last used command list and allows to
  // handle switch of immediate command lists because immediate command lists
  // are never closed unlike regular command lists.
  ur_command_list_ptr_t LastUsedCommandList = CommandListMap.end();

  // Vector of 2 lists of reusable events: host-visible and device-scope.
  // They are separated to allow faster access to stored events depending on
  // requested type of event. Each list contains events which can be reused
  // inside all command lists in the queue as described in the 2-event model.
  // Leftover events in the cache are relased at the queue destruction.
  std::vector<std::list<ur_event_handle_t>> EventCaches{2};
  std::vector<std::unordered_map<ur_device_handle_t, size_t>>
      EventCachesDeviceMap{2};

  // End-times enqueued are stored on the queue rather than on the event to
  // avoid the event objects having been destroyed prior to the write to the
  // end-time member.
  // RecordEventEndTimestamp is not adjusted for valid bits nor resolution, as
  // it is written asynchronously.
  std::map<ur_event_handle_t, uint64_t> EndTimeRecordings;
  // The event may die before the recording has been written back. In this case
  // we move it to a separate map to avoid conflicts.
  std::multimap<ur_event_handle_t, uint64_t> EvictedEndTimeRecordings;

  // Clear the end time recording timestamps entries.
  void clearEndTimeRecordings();

  // adjust the queue's batch size, knowing that the current command list
  // is being closed with a full batch.
  // For copy commands, IsCopy is set to 'true'.
  // For non-copy commands, IsCopy is set to 'false'.
  void adjustBatchSizeForFullBatch(bool IsCopy);

  // adjust the queue's batch size, knowing that the current command list
  // is being closed with only a partial batch of commands.
  // For copy commands, IsCopy is set to 'true'.
  // For non-copy commands, IsCopy is set to 'false'.
  void adjustBatchSizeForPartialBatch(bool IsCopy);

  // Attach a command list to this queue.
  // For non-immediate commandlist also close and execute it.
  // Note that this command list cannot be appended to after this.
  // The "IsBlocking" tells if the wait for completion is required.
  // If OKToBatchCommand is true, then this command list may be executed
  // immediately, or it may be left open for other future command to be
  // batched into.
  // If IsBlocking is true, then batching will not be allowed regardless
  // of the value of OKToBatchCommand
  //
  // For immediate commandlists, no close and execute is necessary.
  ur_result_t executeCommandList(ur_command_list_ptr_t CommandList,
                                 bool IsBlocking, bool OKToBatchCommand);

  // Helper method telling whether we need to reuse discarded event in this
  // queue.
  bool doReuseDiscardedEvents();

  // Append command to provided command list to wait and reset the last event if
  // it is discarded and create new ur_event_handle_t wrapper using the same
  // native event and put it to the cache. We call this method after each
  // command submission to make native event available to use by next commands.
  ur_result_t resetDiscardedEvent(ur_command_list_ptr_t);

  // Put ur_event_handle_t to the cache. Provided ur_event_handle_t object is
  // not used by any command but its ZeEvent is used by many ur_event_handle_t
  // objects. Commands to wait and reset ZeEvent must be submitted to the queue
  // before calling this method.
  ur_result_t addEventToQueueCache(ur_event_handle_t Event);

  // Returns true if any commands for this queue are allowed to
  // be batched together.
  // For copy commands, IsCopy is set to 'true'.
  // For non-copy commands, IsCopy is set to 'false'.
  bool isBatchingAllowed(bool IsCopy) const;

  // Returns true if the queue is a in-order queue.
  bool isInOrderQueue() const;

  // Returns true if the queue has discard events property.
  bool isDiscardEvents() const;

  // Returns true if the queue has low power events property.
  bool isLowPowerEvents() const;

  // Returns true if the queue has explicit priority set by user.
  bool isPriorityLow() const;
  bool isPriorityHigh() const;

  // Returns true if the queue has an explicitly selected submission mode.
  bool isBatchedSubmission() const;
  bool isImmediateSubmission() const;

  // Wait for all commandlists associated with this Queue to finish operations.
  [[nodiscard]] ur_result_t synchronize();

  // Get event from the queue's cache.
  // Returns nullptr if the cache doesn't contain any reusable events or if the
  // cache contains only one event which corresponds to the previous command and
  // can't be used for the current command because we can't use the same event
  // two times in a row and have to do round-robin between two events. Otherwise
  // it picks an event from the beginning of the cache and returns it. Event
  // from the last command is always appended to the end of the list.
  ur_event_handle_t getEventFromQueueCache(bool IsMultiDevice,
                                           bool HostVisible);

  // Returns true if an OpenCommandList has commands that need to be submitted.
  // If IsCopy is 'true', then the OpenCommandList containing copy commands is
  // checked. Otherwise, the OpenCommandList containing compute commands is
  // checked.
  bool hasOpenCommandList(bool IsCopy) const {
    const auto &CommandBatch =
        (IsCopy) ? CopyCommandBatch : ComputeCommandBatch;
    return CommandBatch.OpenCommandList != CommandListMap.end();
  }

  // Update map of memory references made by the kernels about to be submitted
  void CaptureIndirectAccesses();

  // Kernel is not necessarily submitted for execution during
  // urEnqueueKernelLaunch, it may be batched. That's why we need to save the
  // list of kernels which is going to be submitted but have not been submitted
  // yet. This is needed to capture memory allocations for each kernel with
  // indirect access in the list at the moment when kernel is really submitted
  // for execution.
  std::vector<ur_kernel_handle_t> KernelsToBeSubmitted;

  // Append command to the command list to signal new event if the last event in
  // the command list is discarded. While we submit commands in scope of the
  // same command list we can reset and reuse events but when we switch to a
  // different command list we currently need to signal new event and wait for
  // it in the new command list using barrier.
  ur_result_t signalEventFromCmdListIfLastEventDiscarded(ur_command_list_ptr_t);

  // If there is an open command list associated with this queue,
  // close it, execute it, and reset the corresponding OpenCommandList.
  // If IsCopy is 'true', then the OpenCommandList containing copy commands is
  // executed. Otherwise OpenCommandList containing compute commands is
  // executed.
  ur_result_t executeOpenCommandList(bool IsCopy);

  // Wrapper function to execute both OpenCommandLists (Copy and Compute).
  // This wrapper is helpful when all 'open' commands need to be executed.
  // Call-sites instances: urQuueueFinish, urQueueRelease, etc.
  ur_result_t executeAllOpenCommandLists() {
    using IsCopy = bool;
    if (auto Res = executeOpenCommandList(IsCopy{false}))
      return Res;
    if (auto Res = executeOpenCommandList(IsCopy{true}))
      return Res;
    return UR_RESULT_SUCCESS;
  }

  /// @brief Resets the command list and associated fence in the map and removes
  /// events from the command list.
  /// @param CommandList The caller must verify that this command list and fence
  /// have been signalled.
  /// @param MakeAvailable If the reset command list should be made available,
  /// then MakeAvailable needs to be set to true.
  /// @param EventListToCleanup  The EventListToCleanup contains a list of
  /// events from the command list which need to be cleaned up.
  /// @param CheckStatus Hint informing whether we need to check status of the
  /// events before removing them from the immediate command list. This is
  /// needed because immediate command lists are not associated with fences and
  /// in general status of the event needs to be checked.
  /// @return PI_SUCCESS if successful, PI error code otherwise.
  ur_result_t
  resetCommandList(ur_command_list_ptr_t CommandList, bool MakeAvailable,
                   std::vector<ur_event_handle_t> &EventListToCleanup,
                   bool CheckStatus = true);

  // Gets the open command containing the event, or CommandListMap.end()
  ur_command_list_ptr_t eventOpenCommandList(ur_event_handle_t Event);

  // Return the queue group to use based on standard/immediate commandlist mode,
  // and if immediate mode, the thread-specific group.
  ur_queue_group_t &getQueueGroup(bool UseCopyEngine);

  // Helper function to create a new command-list to this queue and associated
  // fence tracking its completion. This command list & fence are added to the
  // map of command lists in this queue with ZeFenceInUse = false.
  // The caller must hold a lock of the queue already.
  ur_result_t
  createCommandList(bool UseCopyEngine, ur_command_list_ptr_t &CommandList,
                    ze_command_queue_handle_t *ForcedCmdQueue = nullptr);

  // Inserts a barrier waiting for all unfinished events in ActiveBarriers into
  // CmdList. Any finished events will be removed from ActiveBarriers.
  ur_result_t insertActiveBarriers(ur_command_list_ptr_t &CmdList,
                                   bool UseCopyEngine);

  // This function considers multiple factors including copy engine
  // availability and user preference and returns a boolean that is used to
  // specify if copy engine will eventually be used for a particular command.
  bool useCopyEngine(bool PreferCopyEngine = true) const;

  // Insert a barrier waiting for the last command event into the beginning of
  // command list. This barrier guarantees that command list execution starts
  // only after completion of previous command list which signals aforementioned
  // event. It allows to reset and reuse same event handles inside all command
  // lists in the queue.
  ur_result_t
  insertStartBarrierIfDiscardEventsMode(ur_command_list_ptr_t &CmdList);

  // returns true if queue has profiling enabled
  bool isProfilingEnabled() {
    return ((this->Properties & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0);
  }

  // Checks whether this queue supports and uses event completion batching.
  // Can be true only when using out-of-order immediate command lists.
  bool useCompletionBatching();

  // Threshold for cleaning up the EventList for immediate command lists.
  size_t getImmdCmmdListsEventCleanupThreshold();

  // Pointer to the unified handle.
  ur_queue_handle_t_ *UnifiedHandle;
};

// This helper function creates a ur_event_handle_t and associate a
// ur_queue_handle_t. Note that the caller of this function must have acquired
// lock on the Queue that is passed in.
// \param Queue ur_queue_handle_t to associate with a new event.
// \param Event a pointer to hold the newly created ur_event_handle_t
// \param CommandType various command type determined by the caller
// \param CommandList is the command list where the event is added
// \param IsInternal tells if the event is internal, i.e. visible in the L0
//        plugin only.
// \param IsMultiDevice Indicates that this event must be visible by
//        multiple devices.
// \param ForceHostVisible tells if the event must be created in
//        the host-visible pool
ur_result_t createEventAndAssociateQueue(
    ur_queue_handle_t Queue, ur_event_handle_t *Event, ur_command_t CommandType,
    ur_command_list_ptr_t CommandList, bool IsInternal, bool IsMultiDevice,
    std::optional<bool> HostVisible = std::nullopt);

// This helper function checks to see if an event for a command can be included
// at the end of a command list batch. This will only be true if the event does
// not have dependencies or the dependencies are not for events which exist in
// this batch.
bool eventCanBeBatched(ur_queue_handle_t Queue, bool UseCopyEngine,
                       uint32_t NumEventsInWaitList,
                       const ur_event_handle_t *EventWaitList);

// This helper function checks to see if a signal event at the end of a command
// should be set. If the Queue is out of order and the command has no
// dependencies, then this command can be enqueued without a signal event set in
// a command list batch. The signal event will be appended at the end of the
// batch to be signalled at the end of the command list.
ur_result_t setSignalEvent(ur_queue_handle_t Queue, bool UseCopyEngine,
                           ze_event_handle_t *ZeEvent, ur_event_handle_t *Event,
                           uint32_t NumEventsInWaitList,
                           const ur_event_handle_t *EventWaitList,
                           ze_command_queue_handle_t ZeQueue);

// Helper function to perform the necessary cleanup of the events from reset cmd
// list.
ur_result_t CleanupEventListFromResetCmdList(
    std::vector<ur_event_handle_t> &EventListToCleanup,
    bool QueueLocked = false);
