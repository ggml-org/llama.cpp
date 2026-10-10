//===--------- memory.cpp - Level Zero Adapter ----------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <string.h>
#include <ur/ur.hpp>

#include "common/helpers/memory_helpers.hpp"
#include "common/image_common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "logger/ur_logger.hpp"
#include "queue.hpp"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"

namespace ur::level_zero::v1 {

// Default to using compute engine for fill operation, but allow to
// override this with an environment variable.
static bool PreferCopyEngine = [] {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_FILL");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL");
  return (UrRet ? std::stoi(UrRet) : (PiRet ? std::stoi(PiRet) : 0));
}();

// Helper function to check if a pointer is a device pointer.
bool IsDevicePointer(ur_context_handle_t Context, const void *Ptr) {
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  // Query memory type of the pointer
  ZE2UR_CALL(zeMemGetAllocProperties,
             (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
              &ZeDeviceHandle));

  return (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_DEVICE);
}

// Helper function to check if a pointer is a shared pointer.
bool IsSharedPointer(ur_context_handle_t Context, const void *Ptr) {
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  // Query memory type of the pointer
  ZE2UR_CALL(zeMemGetAllocProperties,
             (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
              &ZeDeviceHandle));

  return (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_SHARED);
}

// Shared by all memory read/write/copy PI interfaces.
// PI interfaces must have queue's and destination buffer's mutexes locked for
// exclusive use and source buffer's mutex locked for shared use on entry.
ur_result_t enqueueMemCopyHelper(ur_command_t CommandType,
                                 ur_queue_handle_t Queue, void *Dst,
                                 ur_bool_t BlockingWrite, size_t Size,
                                 const void *Src, uint32_t NumEventsInWaitList,
                                 const ur_event_handle_t *EventWaitList,
                                 ur_event_handle_t *OutEvent,
                                 bool PreferCopyEngine) {
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, EventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal, false));
  UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                         NumEventsInWaitList, EventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  UR_LOG(DEBUG, "calling zeCommandListAppendMemoryCopy() with ZeEvent {}",
         ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList(WaitList);

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (ZeCommandList, Dst, Src, Size, ZeEvent, WaitList.Length,
              WaitList.ZeEventList));

  UR_CALL(Queue->executeCommandList(CommandList, BlockingWrite, OkToBatch));

  return UR_RESULT_SUCCESS;
}

// Shared by all memory read/write/copy rect PI interfaces.
// PI interfaces must have queue's and destination buffer's mutexes locked for
// exclusive use and source buffer's mutex locked for shared use on entry.
ur_result_t enqueueMemCopyRectHelper(
    ur_command_t CommandType, ur_queue_handle_t Queue, const void *SrcBuffer,
    void *DstBuffer, ur_rect_offset_t SrcOrigin, ur_rect_offset_t DstOrigin,
    ur_rect_region_t Region, size_t SrcRowPitch, size_t DstRowPitch,
    size_t SrcSlicePitch, size_t DstSlicePitch, ur_bool_t Blocking,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *OutEvent, bool PreferCopyEngine) {
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, EventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal, false));
  UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                         NumEventsInWaitList, EventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  UR_LOG(DEBUG, "calling zeCommandListAppendMemoryCopy() with ZeEvent {}",
         ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList(WaitList);

  auto ZeParams = ur2zeRegionParams(SrcOrigin, DstOrigin, Region, SrcRowPitch,
                                    DstRowPitch, SrcSlicePitch, DstSlicePitch);

  ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
             (ZeCommandList, DstBuffer, &ZeParams.dstRegion, ZeParams.dstPitch,
              ZeParams.dstSlicePitch, SrcBuffer, &ZeParams.srcRegion,
              ZeParams.srcPitch, ZeParams.srcSlicePitch, ZeEvent,
              WaitList.Length, WaitList.ZeEventList));

  UR_LOG(DEBUG, "calling zeCommandListAppendMemoryCopyRegion()");

  UR_CALL(Queue->executeCommandList(CommandList, Blocking, OkToBatch));

  return UR_RESULT_SUCCESS;
}

// PI interfaces must have queue's and buffer's mutexes locked on entry.
static ur_result_t enqueueMemFillHelper(ur_command_t CommandType,
                                        ur_queue_handle_t Queue, void *Ptr,
                                        const void *Pattern, size_t PatternSize,
                                        size_t Size,
                                        uint32_t NumEventsInWaitList,
                                        const ur_event_handle_t *EventWaitList,
                                        ur_event_handle_t *OutEvent) {
  auto &Device = Queue->Device;

  // Make sure that pattern size matches the capability of the copy queues.
  // Check both main and link groups as we don't known which one will be used.
  //
  if (PreferCopyEngine && Device->hasCopyEngine()) {
    if (Device->hasMainCopyEngine() &&
        Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::MainCopy]
                .ZeProperties.maxMemoryFillPatternSize < PatternSize) {
      PreferCopyEngine = false;
    }
    if (Device->hasLinkCopyEngine() &&
        Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::LinkCopy]
                .ZeProperties.maxMemoryFillPatternSize < PatternSize) {
      PreferCopyEngine = false;
    }
  }

  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);
  if (!UseCopyEngine) {
    // Pattern size must fit the compute queue capabilities.
    UR_ASSERT(
        PatternSize <=
            Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
                .ZeProperties.maxMemoryFillPatternSize,
        UR_RESULT_ERROR_INVALID_VALUE);
  }

  ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  ur_command_list_ptr_t CommandList{};
  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, EventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal, false));
  UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                         NumEventsInWaitList, EventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  // PatternSize must be a power of two for zeCommandListAppendMemoryFill.
  // When it's not, the fill is emulated with zeCommandListAppendMemoryCopy.
  if (isPowerOf2(PatternSize)) {
    ZE2UR_CALL(zeCommandListAppendMemoryFill,
               (ZeCommandList, Ptr, Pattern, PatternSize, Size, ZeEvent,
                WaitList.Length, WaitList.ZeEventList));

    UR_LOG(DEBUG, "calling zeCommandListAppendMemoryFill() with ZeEvent {}",
           ur_cast<uint64_t>(ZeEvent));
    printZeEventList(WaitList);

    // Execute command list asynchronously, as the event will be used
    // to track down its completion.
    UR_CALL(Queue->executeCommandList(CommandList, false /*IsBlocking*/,
                                      OkToBatch));
  } else {
    // Copy pattern into every entry in memory array pointed by Ptr.
    uint32_t NumOfCopySteps = Size / PatternSize;
    const void *Src = Pattern;

    for (uint32_t step = 0; step < NumOfCopySteps; ++step) {
      void *Dst = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(Ptr) +
                                           step * PatternSize);
      ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                 (ZeCommandList, Dst, Src, PatternSize, ZeEvent,
                  WaitList.Length, WaitList.ZeEventList));
    }

    UR_LOG(DEBUG, "calling zeCommandListAppendMemoryCopy() with ZeEvent {}",
           ur_cast<uint64_t>(ZeEvent));
    printZeEventList(WaitList);

    // Execute command list synchronously.
    UR_CALL(
        Queue->executeCommandList(CommandList, true /*IsBlocking*/, OkToBatch));
  }

  return UR_RESULT_SUCCESS;
}

// If indirect access tracking is enabled then performs reference counting,
// otherwise just calls zeMemAllocHost.
static ur_result_t ZeHostMemAllocHelper(void **ResultPtr,
                                        ur_context_handle_t UrContext,
                                        size_t Size) {
  ur_platform_handle_t Plt = UrContext->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while
    // we are in the process of allocating a memory, this is needed to
    // properly capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with
    // indirect access, that is why explicitly retain context to be sure
    // that it is released after all memory allocations in this context are
    // released.
    UR_CALL(ur::level_zero::v1::urContextRetain(v1_cast(UrContext)));
  }

  ZeStruct<ze_host_mem_alloc_desc_t> ZeDesc;
  ZeDesc.flags = 0;
  ZE2UR_CALL(zeMemAllocHost,
             (UrContext->ZeContext, &ZeDesc, Size, 1, ResultPtr));

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    UrContext->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(UrContext));
  }
  return UR_RESULT_SUCCESS;
}

// Helper function to implement image read/write/copy.
// PI interfaces must have queue's and destination image's mutexes locked for
// exclusive use and source image's mutex locked for shared use on entry.
static ur_result_t enqueueMemImageCommandHelper(
    ur_command_t CommandType, ur_queue_handle_t Queue,
    const void *Src, // image or ptr
    void *Dst,       // image or ptr
    ur_bool_t IsBlocking, ur_rect_offset_t *SrcOrigin,
    ur_rect_offset_t *DstOrigin, ur_rect_region_t *Region, size_t RowPitch,
    size_t SlicePitch, uint32_t NumEventsInWaitList,
    const ur_event_handle_t *EventWaitList, ur_event_handle_t *OutEvent,
    bool PreferCopyEngine = false) {
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, EventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal, false));
  UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                         NumEventsInWaitList, EventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (CommandType == UR_COMMAND_MEM_IMAGE_READ) {
    ur_image *SrcMem = ur_cast<ur_image *>(const_cast<void *>(Src));

    ze_image_region_t ZeSrcRegion;
    UR_CALL(getImageRegionHelper(SrcMem->ZeImageDesc, SrcOrigin, Region,
                                 ZeSrcRegion));

    // TODO: Level Zero does not support row_pitch/slice_pitch for images yet.
    // Check that SYCL RT did not want pitch larger than default.
    UR_ASSERT(SrcMem->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

#ifndef NDEBUG
    auto SrcImage = SrcMem;
    const ze_image_desc_t &ZeImageDesc = SrcImage->ZeImageDesc;
    UR_ASSERT(
        RowPitch == 0 ||
            // special case RGBA image pitch equal to region's width
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
             RowPitch == 4 * 4 * ZeSrcRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
             RowPitch == 4 * 2 * ZeSrcRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
             RowPitch == 4 * ZeSrcRegion.width),
        UR_RESULT_ERROR_INVALID_IMAGE_SIZE);
#endif
    UR_ASSERT(SlicePitch == 0 || SlicePitch == RowPitch * ZeSrcRegion.height,
              UR_RESULT_ERROR_INVALID_IMAGE_SIZE);

    char *ZeHandleSrc = nullptr;
    UR_CALL(SrcMem->getZeHandle(
        ZeHandleSrc, ur::level_zero::v1::ur_mem_handle_t_::read_only,
        Queue->Device, EventWaitList, NumEventsInWaitList));
    ZE2UR_CALL(zeCommandListAppendImageCopyToMemory,
               (ZeCommandList, Dst, ur_cast<ze_image_handle_t>(ZeHandleSrc),
                &ZeSrcRegion, ZeEvent, WaitList.Length, WaitList.ZeEventList));
  } else if (CommandType == UR_COMMAND_MEM_IMAGE_WRITE) {
    ur_image *DstMem = ur_cast<ur_image *>(Dst);
    ze_image_region_t ZeDstRegion;
    UR_CALL(getImageRegionHelper(DstMem->ZeImageDesc, DstOrigin, Region,
                                 ZeDstRegion));

    // TODO: Level Zero does not support row_pitch/slice_pitch for images yet.
    // Check that SYCL RT did not want pitch larger than default.
    UR_ASSERT(DstMem->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

#ifndef NDEBUG
    auto DstImage = static_cast<ur_image *>(DstMem);
    const ze_image_desc_t &ZeImageDesc = DstImage->ZeImageDesc;
    UR_ASSERT(
        RowPitch == 0 ||
            // special case RGBA image pitch equal to region's width
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
             RowPitch == 4 * 4 * ZeDstRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
             RowPitch == 4 * 2 * ZeDstRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
             RowPitch == 4 * ZeDstRegion.width),
        UR_RESULT_ERROR_INVALID_IMAGE_SIZE);
#endif
    UR_ASSERT(SlicePitch == 0 || SlicePitch == RowPitch * ZeDstRegion.height,
              UR_RESULT_ERROR_INVALID_IMAGE_SIZE);

    char *ZeHandleDst = nullptr;
    UR_CALL(DstMem->getZeHandle(
        ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
        Queue->Device, EventWaitList, NumEventsInWaitList));
    ZE2UR_CALL(zeCommandListAppendImageCopyFromMemory,
               (ZeCommandList, ur_cast<ze_image_handle_t>(ZeHandleDst), Src,
                &ZeDstRegion, ZeEvent, WaitList.Length, WaitList.ZeEventList));
  } else if (CommandType == UR_COMMAND_MEM_IMAGE_COPY) {
    ur_image *SrcImage = ur_cast<ur_image *>(const_cast<void *>(Src));
    ur_image *DstImage = ur_cast<ur_image *>(Dst);

    ze_image_region_t ZeSrcRegion;
    UR_CALL(getImageRegionHelper(SrcImage->ZeImageDesc, SrcOrigin, Region,
                                 ZeSrcRegion));
    ze_image_region_t ZeDstRegion;
    UR_CALL(getImageRegionHelper(DstImage->ZeImageDesc, DstOrigin, Region,
                                 ZeDstRegion));

    char *ZeHandleSrc = nullptr;
    char *ZeHandleDst = nullptr;
    UR_CALL(SrcImage->getZeHandle(
        ZeHandleSrc, ur::level_zero::v1::ur_mem_handle_t_::read_only,
        Queue->Device, EventWaitList, NumEventsInWaitList));
    UR_CALL(DstImage->getZeHandle(
        ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
        Queue->Device, EventWaitList, NumEventsInWaitList));
    ZE2UR_CALL(zeCommandListAppendImageCopyRegion,
               (ZeCommandList, ur_cast<ze_image_handle_t>(ZeHandleDst),
                ur_cast<ze_image_handle_t>(ZeHandleSrc), &ZeDstRegion,
                &ZeSrcRegion, ZeEvent, 0, nullptr));
  } else {
    UR_LOG(ERR, "enqueueMemImageUpdate: unsupported image command type");
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  UR_CALL(Queue->executeCommandList(CommandList, IsBlocking, OkToBatch));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueMemBufferRead(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the buffer object
    ::ur_mem_handle_t hBufferOpque,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingRead,
    /// [in] offset in bytes in the buffer object
    size_t offset,
    /// [in] size in bytes of data being read
    size_t size,
    /// [in] pointer to host memory where data is to be read into
    void *pDst,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *phEventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *phEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto Src = v1_cast(hBufferOpque);
  auto phEventWaitListInternal = v1_cast(phEventWaitList);
  auto phEventInternal = v1_cast(phEvent);

  std::shared_lock<ur_shared_mutex> SrcLock(Src->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, Queue->Mutex);

  char *ZeHandleSrc = nullptr;
  UR_CALL(Src->getZeHandle(
      ZeHandleSrc, ur::level_zero::v1::ur_mem_handle_t_::read_only,
      Queue->Device, phEventWaitListInternal, numEventsInWaitList));
  return enqueueMemCopyHelper(UR_COMMAND_MEM_BUFFER_READ, Queue, pDst,
                              blockingRead, size, ZeHandleSrc + offset,
                              numEventsInWaitList, phEventWaitListInternal,
                              phEventInternal, true /* PreferCopyEngine */);
}

ur_result_t urEnqueueMemBufferWrite(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the buffer object
    ::ur_mem_handle_t hBufferOpque,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingWrite,
    /// [in] offset in bytes in the buffer object
    size_t offset,
    /// [in] size in bytes of data being written
    size_t size,
    /// [in] pointer to host memory where data is to be written from
    const void *pSrc,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *phEventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *phEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto Buffer = v1_cast(hBufferOpque);
  auto phEventWaitListInternal = v1_cast(phEventWaitList);
  auto phEventInternal = v1_cast(phEvent);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  UR_CALL(Buffer->getZeHandle(
      ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
      Queue->Device, phEventWaitListInternal, numEventsInWaitList));
  return enqueueMemCopyHelper(UR_COMMAND_MEM_BUFFER_WRITE, Queue,
                              ZeHandleDst + offset, // dst
                              blockingWrite, size,
                              pSrc, // src
                              numEventsInWaitList, phEventWaitListInternal,
                              phEventInternal, true /* PreferCopyEngine */);
}

ur_result_t urEnqueueMemBufferReadRect(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the buffer object
    ::ur_mem_handle_t hBufferOpque,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingRead,
    /// [in] 3D offset in the buffer
    ur_rect_offset_t bufferOffset,
    /// [in] 3D offset in the host region
    ur_rect_offset_t hostOffset,
    /// [in] 3D rectangular region descriptor: width, height, depth
    ur_rect_region_t region,
    /// [in] length of each row in bytes in the buffer object
    size_t bufferRowPitch,
    /// [in] length of each 2D slice in bytes in the buffer object being read
    size_t bufferSlicePitch,
    /// [in] length of each row in bytes in the host memory region pointed by
    /// dst
    size_t hostRowPitch,
    /// [in] length of each 2D slice in bytes in the host memory region pointed
    /// by dst
    size_t hostSlicePitch,
    /// [in] pointer to host memory where data is to be read into
    void *pDst,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *phEventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *phEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto Buffer = v1_cast(hBufferOpque);
  auto phEventWaitListInternal = v1_cast(phEventWaitList);
  auto phEventInternal = v1_cast(phEvent);

  std::shared_lock<ur_shared_mutex> SrcLock(Buffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, Queue->Mutex);

  char *ZeHandleSrc;
  UR_CALL(Buffer->getZeHandle(
      ZeHandleSrc, ur::level_zero::v1::ur_mem_handle_t_::read_only,
      Queue->Device, phEventWaitListInternal, numEventsInWaitList));
  return enqueueMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_READ_RECT, Queue, ZeHandleSrc, pDst, bufferOffset,
      hostOffset, region, bufferRowPitch, hostRowPitch, bufferSlicePitch,
      hostSlicePitch, blockingRead, numEventsInWaitList,
      phEventWaitListInternal, phEventInternal, false /*PreferCopyEngine*/);
}

ur_result_t urEnqueueMemBufferWriteRect(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the buffer object
    ::ur_mem_handle_t hBufferOpque,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingWrite,
    /// [in] 3D offset in the buffer
    ur_rect_offset_t bufferOffset,
    /// [in] 3D offset in the host region
    ur_rect_offset_t hostOffset,
    /// [in] 3D rectangular region descriptor: width, height, depth
    ur_rect_region_t region,
    /// [in] length of each row in bytes in the buffer object
    size_t bufferRowPitch,
    /// [in] length of each 2D slice in bytes in the buffer object being written
    size_t bufferSlicePitch,
    /// [in] length of each row in bytes in the host memory region pointed by
    /// src
    size_t hostRowPitch,
    /// [in] length of each 2D slice in bytes in the host memory region pointed
    /// by src
    size_t hostSlicePitch,
    /// [in] pointer to host memory where data is to be written from
    void *pSrc,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] points to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *phEventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *phEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto Buffer = v1_cast(hBufferOpque);
  auto phEventWaitListInternal = v1_cast(phEventWaitList);
  auto phEventInternal = v1_cast(phEvent);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  UR_CALL(Buffer->getZeHandle(
      ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
      Queue->Device, phEventWaitListInternal, numEventsInWaitList));
  return enqueueMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_WRITE_RECT, Queue,
      const_cast<char *>(static_cast<const char *>(pSrc)), ZeHandleDst,
      hostOffset, bufferOffset, region, hostRowPitch, bufferRowPitch,
      hostSlicePitch, bufferSlicePitch, blockingWrite, numEventsInWaitList,
      phEventWaitListInternal, phEventInternal, false /*PreferCopyEngine*/);
}

ur_result_t urEnqueueMemBufferCopy(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the src buffer object
    ::ur_mem_handle_t BufferSrcOpque,
    /// [in] handle of the dest buffer object
    ::ur_mem_handle_t BufferDstOpque,
    /// [in] offset into hBufferSrc to begin copying from
    size_t SrcOffset,
    /// [in] offset info hBufferDst to begin copying into
    size_t DstOffset,
    /// [in] size in bytes of data being copied
    size_t Size,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  ur_buffer *SrcBuffer = ur_cast<ur_buffer *>(BufferSrcOpque);
  ur_buffer *DstBuffer = ur_cast<ur_buffer *>(BufferDstOpque);

  UR_ASSERT(!SrcBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(!DstBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, DstBuffer->Mutex, Queue->Mutex);

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  char *ZeHandleSrc = nullptr;
  UR_CALL(SrcBuffer->getZeHandle(
      ZeHandleSrc, ur::level_zero::v1::ur_mem_handle_t_::read_only,
      Queue->Device, EventWaitListInternal, NumEventsInWaitList));
  char *ZeHandleDst = nullptr;
  UR_CALL(DstBuffer->getZeHandle(
      ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
      Queue->Device, EventWaitListInternal, NumEventsInWaitList));

  return enqueueMemCopyHelper(
      UR_COMMAND_MEM_BUFFER_COPY, Queue, ZeHandleDst + DstOffset,
      false, // blocking
      Size, ZeHandleSrc + SrcOffset, NumEventsInWaitList, EventWaitListInternal,
      OutEventInternal, PreferCopyEngine);
}

ur_result_t urEnqueueMemBufferCopyRect(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the source buffer object
    ::ur_mem_handle_t BufferSrcOpque,
    /// [in] handle of the dest buffer object
    ::ur_mem_handle_t BufferDstOpque,
    /// [in] 3D offset in the source buffer
    ur_rect_offset_t SrcOrigin,
    /// [in] 3D offset in the destination buffer
    ur_rect_offset_t DstOrigin,
    /// [in] source 3D rectangular region descriptor: width, height, depth
    ur_rect_region_t SrcRegion,
    /// [in] length of each row in bytes in the source buffer object
    size_t SrcRowPitch,
    /// [in] length of each 2D slice in bytes in the source buffer object
    size_t SrcSlicePitch,
    /// [in] length of each row in bytes in the destination buffer object
    size_t DstRowPitch,
    /// [in] length of each 2D slice in bytes in the destination buffer object
    size_t DstSlicePitch,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  ur_buffer *SrcBuffer = ur_cast<ur_buffer *>(BufferSrcOpque);
  ur_buffer *DstBuffer = ur_cast<ur_buffer *>(BufferDstOpque);

  UR_ASSERT(!SrcBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(!DstBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, DstBuffer->Mutex, Queue->Mutex);

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  char *ZeHandleSrc = nullptr;
  UR_CALL(SrcBuffer->getZeHandle(
      ZeHandleSrc, ur::level_zero::v1::ur_mem_handle_t_::read_only,
      Queue->Device, EventWaitListInternal, NumEventsInWaitList));
  char *ZeHandleDst = nullptr;
  UR_CALL(DstBuffer->getZeHandle(
      ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
      Queue->Device, EventWaitListInternal, NumEventsInWaitList));

  return enqueueMemCopyRectHelper(UR_COMMAND_MEM_BUFFER_COPY_RECT, Queue,
                                  ZeHandleSrc, ZeHandleDst, SrcOrigin,
                                  DstOrigin, SrcRegion, SrcRowPitch,
                                  DstRowPitch, SrcSlicePitch, DstSlicePitch,
                                  false, // blocking
                                  NumEventsInWaitList, EventWaitListInternal,
                                  OutEventInternal, PreferCopyEngine);
}

ur_result_t urEnqueueMemBufferFill(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the buffer object
    ::ur_mem_handle_t BufferOpque,
    /// [in] pointer to the fill pattern
    const void *Pattern,
    /// [in] size in bytes of the pattern
    size_t PatternSize,
    /// [in] offset into the buffer
    size_t Offset,
    /// [in] fill size in bytes, must be a multiple of patternSize
    size_t Size,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto Buffer = v1_cast(BufferOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  ur_buffer *UrBuffer = reinterpret_cast<ur_buffer *>(BufferOpque);
  UR_CALL(UrBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                Queue->Device, EventWaitListInternal,
                                NumEventsInWaitList));
  return enqueueMemFillHelper(
      UR_COMMAND_MEM_BUFFER_FILL, Queue, ZeHandleDst + Offset,
      Pattern,     // It will be interpreted as an 8-bit value,
      PatternSize, // which is indicated with this pattern_size==1
      Size, NumEventsInWaitList, EventWaitListInternal, OutEventInternal);
}

ur_result_t urEnqueueMemImageRead(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the image object
    ::ur_mem_handle_t ImageOpque,
    /// [in] indicates blocking (true), non-blocking (false)
    bool BlockingRead,
    /// [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_offset_t Origin,
    /// [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
    /// image
    ur_rect_region_t Region,
    /// [in] length of each row in bytes
    size_t RowPitch,
    /// [in] length of each 2D slice of the 3D image
    size_t SlicePitch,
    /// [in] pointer to host memory where image is to be read into
    void *Dst,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto Image = v1_cast(ImageOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Image->Mutex);
  return enqueueMemImageCommandHelper(
      UR_COMMAND_MEM_IMAGE_READ, Queue, ImageOpque, Dst, BlockingRead, &Origin,
      nullptr, &Region, RowPitch, SlicePitch, NumEventsInWaitList,
      EventWaitListInternal, OutEventInternal);
}

ur_result_t urEnqueueMemImageWrite(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the image object
    ::ur_mem_handle_t ImageOpque,
    /// [in] indicates blocking (true), non-blocking (false)
    bool BlockingWrite,
    /// [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_offset_t Origin,
    /// [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
    /// image
    ur_rect_region_t Region,
    /// [in] length of each row in bytes
    size_t RowPitch,
    /// [in] length of each 2D slice of the 3D image
    size_t SlicePitch,
    /// [in] pointer to host memory where image is to be read into
    void *Src,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto Image = v1_cast(ImageOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Image->Mutex);
  return enqueueMemImageCommandHelper(
      UR_COMMAND_MEM_IMAGE_WRITE, Queue, Src, ImageOpque, BlockingWrite,
      nullptr, &Origin, &Region, RowPitch, SlicePitch, NumEventsInWaitList,
      EventWaitListInternal, OutEventInternal);
}

ur_result_t urEnqueueMemImageCopy(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the src image object
    ::ur_mem_handle_t ImageSrcOpque,
    /// [in] handle of the dest image object
    ::ur_mem_handle_t ImageDstOpque,
    /// [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
    /// image
    ur_rect_offset_t SrcOrigin,
    /// [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D, or
    /// 3D image
    ur_rect_offset_t DstOrigin,
    /// [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
    /// image
    ur_rect_region_t Region,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto ImageSrc = v1_cast(ImageSrcOpque);
  auto ImageDst = v1_cast(ImageDstOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  std::shared_lock<ur_shared_mutex> SrcLock(ImageSrc->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, ImageDst->Mutex, Queue->Mutex);
  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  // Images are always allocated on device.
  bool PreferCopyEngine = false;
  return enqueueMemImageCommandHelper(
      UR_COMMAND_MEM_IMAGE_COPY, Queue, ImageSrcOpque, ImageDstOpque,
      false, // is_blocking
      &SrcOrigin, &DstOrigin, &Region,
      0, // row pitch
      0, // slice pitch
      NumEventsInWaitList, EventWaitListInternal, OutEventInternal,
      PreferCopyEngine);
}

ur_result_t urEnqueueMemBufferMap(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the buffer object
    ::ur_mem_handle_t BufOpque,
    /// [in] indicates blocking (true), non-blocking (false)
    bool BlockingMap,
    /// [in] flags for read, write, readwrite mapping
    ur_map_flags_t MapFlags,
    /// [in] offset in bytes of the buffer region being mapped
    size_t Offset,
    /// [in] size in bytes of the buffer region being mapped
    size_t Size,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent,
    /// [in,out] return mapped pointer. TODO: move it before
    /// numEventsInWaitList?
    void **RetMap) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  auto Buffer = ur_cast<ur_buffer *>(BufOpque);

  UR_ASSERT(!Buffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEventInternal == nullptr;
  ur_event_handle_t *Event =
      OutEventInternal ? OutEventInternal : &InternalEvent;
  ze_event_handle_t ZeEvent = nullptr;

  bool UseCopyEngine = false;
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    ur_ze_event_list_t TmpWaitList;
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitListInternal, Queue, UseCopyEngine));

    UR_CALL(createEventAndAssociateQueue(
        Queue, Event, UR_COMMAND_MEM_BUFFER_MAP, Queue->CommandListMap.end(),
        IsInternal, false));

    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;
  }

  // Translate the host access mode info.
  ur::level_zero::v1::ur_mem_handle_t_::access_mode_t AccessMode =
      ur::level_zero::v1::ur_mem_handle_t_::unknown;
  if (MapFlags & UR_MAP_FLAG_WRITE_INVALIDATE_REGION)
    AccessMode = ur::level_zero::v1::ur_mem_handle_t_::write_only;
  else {
    if (MapFlags & UR_MAP_FLAG_READ) {
      AccessMode = ur::level_zero::v1::ur_mem_handle_t_::read_only;
      if (MapFlags & UR_MAP_FLAG_WRITE)
        AccessMode = ur::level_zero::v1::ur_mem_handle_t_::read_write;
    } else if (MapFlags & UR_MAP_FLAG_WRITE)
      AccessMode = ur::level_zero::v1::ur_mem_handle_t_::write_only;
  }

  UR_ASSERT(AccessMode != ur::level_zero::v1::ur_mem_handle_t_::unknown,
            UR_RESULT_ERROR_INVALID_VALUE);

  // TODO: Level Zero is missing the memory "mapping" capabilities, so we are
  // left to doing new memory allocation and a copy (read) on discrete devices.
  // For integrated devices, we have allocated the buffer in host memory so no
  // actions are needed here except for synchronizing on incoming events.
  // A host-to-host copy is done if a host pointer had been supplied during
  // buffer creation on integrated devices.
  //
  // TODO: for discrete, check if the input buffer is already allocated
  // in shared memory and thus is accessible from the host as is.
  // Can we get SYCL RT to predict/allocate in shared memory
  // from the beginning?

  // For integrated devices the buffer has been allocated in host memory.
  if (Buffer->OnHost) {
    // Wait on incoming events before doing the copy
    if (NumEventsInWaitList > 0)
      UR_CALL(
          ur::level_zero::v1::urEventWait(NumEventsInWaitList, EventWaitList));

    if (Queue->isInOrderQueue())
      UR_CALL(ur::level_zero::v1::urQueueFinish(QueueOpque));

    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);

    char *ZeHandleSrc;
    UR_CALL(Buffer->getZeHandle(ZeHandleSrc, AccessMode, Queue->Device,
                                EventWaitListInternal, NumEventsInWaitList));

    if (Buffer->MapHostPtr) {
      *RetMap = Buffer->MapHostPtr + Offset;
      if (ZeHandleSrc != Buffer->MapHostPtr &&
          AccessMode != ur::level_zero::v1::ur_mem_handle_t_::write_only) {
        memcpy(*RetMap, ZeHandleSrc + Offset, Size);
      }
    } else {
      *RetMap = ZeHandleSrc + Offset;
    }

    auto Res = Buffer->Mappings.insert({*RetMap, {Offset, Size}});
    // False as the second value in pair means that mapping was not inserted
    // because mapping already exists.
    if (!Res.second) {
      UR_LOG(ERR, "urEnqueueMemBufferMap: duplicate mapping detected");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    // Signal this event
    if (!(*Event)->CounterBasedEventsEnabled)
      ZE2UR_CALL(zeEventHostSignal, (ZeEvent));
    (*Event)->Completed = true;
    return UR_RESULT_SUCCESS;
  }

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  if (Buffer->MapHostPtr) {
    *RetMap = Buffer->MapHostPtr + Offset;
  } else {
    // TODO: use USM host allocator here
    // TODO: Do we even need every map to allocate new host memory?
    //       In the case when the buffer is "OnHost" we use single allocation.
    UR_CALL(ZeHostMemAllocHelper(RetMap, Queue->Context, Size));
  }

  // Take a shortcut if the host is not going to read buffer's data.
  if (AccessMode == ur::level_zero::v1::ur_mem_handle_t_::write_only) {
    (*Event)->Completed = true;
  } else {
    // For discrete devices we need a command list
    ur_command_list_ptr_t CommandList{};
    UR_CALL(Queue->Context->getAvailableCommandList(
        Queue, CommandList, UseCopyEngine, NumEventsInWaitList,
        EventWaitListInternal, false /*AllowBatching*/,
        nullptr /*ForcedCmdQueue*/));

    // Add the event to the command list.
    CommandList->second.append(reinterpret_cast<ur_event_handle_t>(*Event));
    (*Event)->RefCount.retain();

    const auto &ZeCommandList = CommandList->first;
    const auto &WaitList = (*Event)->WaitList;

    char *ZeHandleSrc;
    UR_CALL(Buffer->getZeHandle(ZeHandleSrc, AccessMode, Queue->Device,
                                EventWaitListInternal, NumEventsInWaitList));

    UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                           NumEventsInWaitList, EventWaitListInternal,
                           CommandList->second.ZeQueue));

    ZE2UR_CALL(zeCommandListAppendMemoryCopy,
               (ZeCommandList, *RetMap, ZeHandleSrc + Offset, Size, ZeEvent,
                WaitList.Length, WaitList.ZeEventList));

    UR_CALL(Queue->executeCommandList(CommandList, BlockingMap,
                                      false /*OKToBatchCommand*/));
  }

  auto Res = Buffer->Mappings.insert({*RetMap, {Offset, Size}});
  // False as the second value in pair means that mapping was not inserted
  // because mapping already exists.
  if (!Res.second) {
    UR_LOG(ERR, "urEnqueueMemBufferMap: duplicate mapping detected");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueMemUnmap(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] handle of the memory (buffer or image) object
    ::ur_mem_handle_t MemOpque,
    /// [in] mapped host address
    void *MappedPtr,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitListOpque,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEventOpque) {
  auto Queue = v1_cast(QueueOpque);
  auto Mem = v1_cast(MemOpque);
  auto EventWaitListInternal = v1_cast(EventWaitListOpque);
  auto OutEventInternal = v1_cast(OutEventOpque);
  UR_ASSERT(!Mem->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  auto Buffer = static_cast<ur_buffer *>(Mem);

  bool UseCopyEngine = false;

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEventInternal == nullptr;
  ur_event_handle_t *Event =
      OutEventInternal ? OutEventInternal : &InternalEvent;
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    ur_ze_event_list_t TmpWaitList;
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitListInternal, Queue, UseCopyEngine));

    UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_MEM_UNMAP,
                                         Queue->CommandListMap.end(),
                                         IsInternal, false));
    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;
  }

  ur_buffer::Mapping MapInfo = {};
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);
    auto It = Buffer->Mappings.find(MappedPtr);
    if (It == Buffer->Mappings.end()) {
      UR_LOG(ERR, "urEnqueueMemUnmap: unknown memory mapping");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    MapInfo = It->second;
    Buffer->Mappings.erase(It);

    // NOTE: we still have to free the host memory allocated/returned by
    // piEnqueueMemBufferMap, but can only do so after the above copy
    // is completed. Instead of waiting for It here (blocking), we shall
    // do so in piEventRelease called for the pi_event tracking the unmap.
    // In the case of an integrated device, the map operation does not allocate
    // any memory, so there is nothing to free. This is indicated by a nullptr.
    (*Event)->CommandData =
        (Buffer->OnHost ? nullptr : (Buffer->MapHostPtr ? nullptr : MappedPtr));
  }

  // For integrated devices the buffer is allocated in host memory.
  if (Buffer->OnHost) {
    // Wait on incoming events before doing the copy
    if (NumEventsInWaitList > 0)
      UR_CALL(ur::level_zero::v1::urEventWait(NumEventsInWaitList,
                                              EventWaitListOpque));

    if (Queue->isInOrderQueue())
      UR_CALL(ur::level_zero::v1::urQueueFinish(QueueOpque));

    char *ZeHandleDst;
    UR_CALL(Buffer->getZeHandle(
        ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
        Queue->Device, EventWaitListInternal, NumEventsInWaitList));

    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);
    if (Buffer->MapHostPtr)
      memcpy(ZeHandleDst + MapInfo.Offset, MappedPtr, MapInfo.Size);

    // Signal this event if it is not using counter based events
    if (!(*Event)->CounterBasedEventsEnabled)
      ZE2UR_CALL(zeEventHostSignal, (ZeEvent));
    (*Event)->Completed = true;
    return UR_RESULT_SUCCESS;
  }

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList,
      EventWaitListInternal, false /*AllowBatching*/,
      nullptr /*ForcedCmdQueue*/));

  CommandList->second.append(reinterpret_cast<ur_event_handle_t>(*Event));
  (*Event)->RefCount.retain();

  const auto &ZeCommandList = CommandList->first;

  // TODO: Level Zero is missing the memory "mapping" capabilities, so we are
  // left to doing copy (write back to the device).
  //
  // NOTE: Keep this in sync with the implementation of
  // piEnqueueMemBufferMap.

  char *ZeHandleDst;
  UR_CALL(Buffer->getZeHandle(
      ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
      Queue->Device, EventWaitListInternal, NumEventsInWaitList));

  UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                         NumEventsInWaitList, EventWaitListInternal,
                         CommandList->second.ZeQueue));

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (ZeCommandList, ZeHandleDst + MapInfo.Offset, MappedPtr,
              MapInfo.Size, ZeEvent, (*Event)->WaitList.Length,
              (*Event)->WaitList.ZeEventList));

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  UR_CALL(Queue->executeCommandList(CommandList, false /*IsBlocking*/,
                                    false /*OKToBatchCommand*/));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueUSMMemcpy(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] blocking or non-blocking copy
    bool Blocking,
    /// [in] pointer to the destination USM memory object
    void *Dst,
    /// [in] pointer to the source USM memory object
    const void *Src,
    /// [in] size in bytes to be copied
    size_t Size,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Device to Device copies are found to execute slower on copy engine
  // (versus compute engine).
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Src) ||
                          !IsDevicePointer(Queue->Context, Dst);
  // For better performance, Copy Engines are not preferred given Shared
  // pointers on DG2.
  if (Queue->Device->isDG2() && (IsSharedPointer(Queue->Context, Src) ||
                                 IsSharedPointer(Queue->Context, Dst))) {
    PreferCopyEngine = false;
  }

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper( // TODO: do we need a new command type for this?
      UR_COMMAND_MEM_BUFFER_COPY, Queue, Dst, Blocking, Size, Src,
      NumEventsInWaitList, EventWaitListInternal, OutEventInternal,
      PreferCopyEngine);
}

ur_result_t urEnqueueUSMPrefetch(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] pointer to the USM memory object
    const void *Mem,
    /// [in] size in bytes to be fetched
    size_t Size,
    /// [in] USM prefetch flags
    ur_usm_migration_flags_t Flags,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto OutEventInternal = v1_cast(OutEvent);
  switch (Flags) {
  case UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE:
    break;
  case UR_USM_MIGRATION_FLAG_DEVICE_TO_HOST:
    UR_LOG(WARN,
           "enqueueUSMPrefetch: L0 does not support prefetch to host yet");
    break;
  default:
    UR_LOG(ERR, "enqueueUSMPrefetch: invalid USM migration flag");
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  bool UseCopyEngine = false;

  // Please note that the following code should be run before the
  // subsequent getAvailableCommandList() call so that there is no
  // dead-lock from waiting unsubmitted events in an open batch.
  // The createAndRetainUrZeEventList() has the proper side-effect
  // of submitting batches with dependent events.
  //
  ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitListInternal, Queue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  // TODO: Change UseCopyEngine argument to 'true' once L0 backend
  // support is added
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList,
      EventWaitListInternal, false /*AllowBatching*/,
      nullptr /*ForcedCmdQueue*/));

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEventInternal == nullptr;
  ur_event_handle_t *Event =
      OutEventInternal ? OutEventInternal : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_USM_PREFETCH,
                                       CommandList, IsInternal, false));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &WaitList = (*Event)->WaitList;
  const auto &ZeCommandList = CommandList->first;
  if (WaitList.Length) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }
  // TODO: Support migration flags after L0 backend support is added
  if (Flags == UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE) {
    ZE2UR_CALL(zeCommandListAppendMemoryPrefetch, (ZeCommandList, Mem, Size));
  }

  // TODO: Level Zero does not have a completion "event" with the prefetch API,
  // so manually add command to signal our event.
  ZE2UR_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  UR_CALL(Queue->executeCommandList(CommandList, false /*IsBlocking*/,
                                    false /*OKToBatchCommand*/));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueUSMAdvise(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] pointer to the USM memory object
    const void *Mem,
    /// [in] size in bytes to be advised
    size_t Size,
    /// [in] USM memory advice
    ur_usm_advice_flags_t Advice,
    /// [in,out][optional] return an event object that identifies this
    /// particular command instance.
    ::ur_event_handle_t *OutEvent) {
  auto Queue = v1_cast(QueueOpque);
  auto OutEventInternal = v1_cast(OutEvent);
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  auto ZeAdvice = ur_cast<ze_memory_advice_t>(Advice);

  bool UseCopyEngine = false;

  ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(0, nullptr, Queue,
                                                   UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  // UseCopyEngine is set to 'false' here.
  // TODO: Additional analysis is required to check if this operation will
  // run faster on copy engines.
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, 0, nullptr, false /*AllowBatching*/,
      nullptr /*ForcedCmdQueue*/));

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent{};
  bool IsInternal = OutEventInternal == nullptr;
  ur_event_handle_t *Event =
      OutEventInternal ? OutEventInternal : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_USM_ADVISE,
                                       CommandList, IsInternal, false));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  ZE2UR_CALL(zeCommandListAppendMemAdvise,
             (ZeCommandList, Queue->Device->ZeDevice, Mem, Size, ZeAdvice));

  // TODO: Level Zero does not have a completion "event" with the advise API,
  // so manually add command to signal our event.
  ZE2UR_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  UR_CALL(Queue->executeCommandList(CommandList, false /*IsBlocking*/,
                                    false /*OKToBatchCommand*/));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueUSMFill2D(
    /// [in] handle of the queue to submit to.
    ::ur_queue_handle_t /*Queue*/,
    /// [in] pointer to memory to be filled.
    void * /*Mem*/,
    /// [in] the total width of the destination memory including padding.
    size_t /*Pitch*/,
    /// [in] the size in bytes of the pattern.
    size_t /*PatternSize*/,
    /// [in] pointer with the bytes of the pattern to set.
    const void * /*Pattern*/,
    /// [in] the width in bytes of each row to fill.
    size_t /*Width*/,
    /// [in] the height of the columns to fill.
    size_t /*Height*/,
    /// [in] size of the event wait list
    uint32_t /*NumEventsInWaitList*/,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ::ur_event_handle_t * /*EventWaitList*/,
    /// [in,out][optional] return an event object that identifies this
    /// particular kernel execution instance.
    ::ur_event_handle_t * /*OutEvent*/) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urEnqueueUSMMemcpy2D(
    /// [in] handle of the queue to submit to.
    ::ur_queue_handle_t QueueOpque,
    /// [in] indicates if this operation should block the host.
    bool Blocking,
    /// [in] pointer to memory where data will be copied.
    void *Dst,
    /// [in] the total width of the source memory including padding.
    size_t DstPitch,
    /// [in] pointer to memory to be copied.
    const void *Src,
    /// [in] the total width of the source memory including padding.
    size_t SrcPitch,
    /// [in] the width in bytes of each row to be copied.
    size_t Width,
    /// [in] the height of columns to be copied.
    size_t Height,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ::ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular kernel execution instance.
    ::ur_event_handle_t *Event) {
  ur_rect_offset_t ZeroOffset{0, 0, 0};
  ur_rect_region_t Region{Width, Height, 0};

  auto Queue = v1_cast(QueueOpque);
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Device to Device copies are found to execute slower on copy engine
  // (versus compute engine).
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Src) ||
                          !IsDevicePointer(Queue->Context, Dst);
  // For better performance, Copy Engines are not preferred given Shared
  // pointers on DG2.
  if (Queue->Device->isDG2() && (IsSharedPointer(Queue->Context, Src) ||
                                 IsSharedPointer(Queue->Context, Dst))) {
    PreferCopyEngine = false;
  }

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyRectHelper( // TODO: do we need a new command type for
                                   // this?
      UR_COMMAND_MEM_BUFFER_COPY_RECT, Queue, Src, Dst, ZeroOffset, ZeroOffset,
      Region, SrcPitch, DstPitch, 0, /*SrcSlicePitch=*/
      0,                             /*DstSlicePitch=*/
      Blocking, NumEventsInWaitList, v1_cast(EventWaitList), v1_cast(Event),
      PreferCopyEngine);
}

ur_result_t urMemImageCreate(
    /// [in] handle of the context object
    ::ur_context_handle_t ContextOpque,
    /// [in] allocation and usage information flags
    ur_mem_flags_t Flags,
    /// [in] pointer to image format specification
    const ur_image_format_t *ImageFormat,
    /// [in] pointer to image description
    const ur_image_desc_t *ImageDesc,
    /// [in] pointer to the buffer data
    void *Host,
    /// [out] pointer to handle of image object created
    ::ur_mem_handle_t *Mem) {
  auto Context = v1_cast(ContextOpque);
  // TODO: implement read-only, write-only
  if ((Flags & UR_MEM_FLAG_READ_WRITE) == 0) {
    die("urMemImageCreate: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDesc(ImageFormat, ImageDesc, ZeImageDesc));

  // Currently we have the "0" device in context with mutliple root devices to
  // own the image.
  // TODO: Implement explicit copying for acessing the image from other devices
  // in the context.
  ur_device_handle_t Device = Context->getDevices()[0];
  ze_image_handle_t ZeImage;
  ZE2UR_CALL(zeImageCreate,
             (Context->ZeContext, Device->ZeDevice, &ZeImageDesc, &ZeImage));

  UR_CALL(createUrMemFromZeImage(Context, ZeImage, /*OwnZeMemHandle*/ true,
                                 ZeImageDesc, Mem));

  if ((Flags & UR_MEM_FLAG_USE_HOST_POINTER) != 0 ||
      (Flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) != 0) {
    // Initialize image synchronously with immediate offload.
    // zeCommandListAppendImageCopyFromMemory must not be called from
    // simultaneous threads with the same command list handle, so we need
    // exclusive lock.
    std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
    ZE2UR_CALL(zeCommandListAppendImageCopyFromMemory,
               (Context->ZeCommandListInit, ZeImage, Host, nullptr, nullptr, 0,
                nullptr));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemImageCreateWithNativeHandle(
    /// [in] the native handle to the memory.
    ::ur_native_handle_t NativeMem,
    /// [in] handle of the context object.
    ::ur_context_handle_t ContextOpque,
    /// [in] pointer to image format specification.
    [[maybe_unused]] const ur_image_format_t *ImageFormat,
    /// [in] pointer to image description.
    [[maybe_unused]] const ur_image_desc_t *ImageDesc,
    /// [in][optional] pointer to native memory creation properties.
    const ur_mem_native_properties_t *Properties, ::ur_mem_handle_t *Mem) {
  auto Context = v1_cast(ContextOpque);
  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  ze_image_handle_t ZeHImage = ur_cast<ze_image_handle_t>(NativeMem);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  ur_result_t Res = ur2zeImageDesc(ImageFormat, ImageDesc, ZeImageDesc);
  if (Res != UR_RESULT_SUCCESS) {
    *Mem = nullptr;
    return Res;
  }

  auto OwnNativeHandle = Properties ? Properties->isNativeHandleOwned : false;
  UR_CALL(createUrMemFromZeImage(Context, ZeHImage, OwnNativeHandle,
                                 ZeImageDesc, Mem));

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemBufferCreate(
    /// [in] handle of the context object
    ::ur_context_handle_t ContextOpque,
    /// [in] allocation and usage information flags
    ur_mem_flags_t Flags,
    /// [in] size in bytes of the memory object to be allocated
    size_t Size, const ur_buffer_properties_t *Properties,
    /// [out] pointer to handle of the memory buffer created
    ::ur_mem_handle_t *RetBuffer) {
  auto Context = v1_cast(ContextOpque);
  if (Flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
    // Having PI_MEM_FLAGS_HOST_PTR_ALLOC for buffer requires allocation of
    // pinned host memory, see:
    // sycl/doc/extensions/supported/sycl_ext_oneapi_use_pinned_host_memory_property.asciidoc
    // We are however missing such functionality in Level Zero, so we just
    // ignore the flag for now.
    //
  }

  void *Host = nullptr;
  if (Properties) {
    Host = Properties->pHost;
  }

  bool HostPtrImported = false;
  if (Flags & UR_MEM_FLAG_USE_HOST_POINTER)
    HostPtrImported =
        maybeImportUSM(Context->getPlatform()->ZeDriverHandleExpTranslated,
                       Context->ZeContext, Host, Size);

  auto releaseImport = [Context](void *Ptr) {
    ZeUSMImport.doZeUSMRelease(
        Context->getPlatform()->ZeDriverHandleExpTranslated, Ptr);
  };
  std::unique_ptr<void, decltype(releaseImport)> importGuard(
      HostPtrImported ? Host : nullptr, releaseImport);

  ur_buffer *Buffer = nullptr;
  auto HostPtrOrNull = (Flags & UR_MEM_FLAG_USE_HOST_POINTER)
                           ? reinterpret_cast<char *>(Host)
                           : nullptr;

  std::unique_ptr<ur_buffer, void (*)(ur_buffer *)> bufferGuard = {
      nullptr, [](ur_buffer *handle) {
        if (handle)
          handle->free();
        delete handle;
      }};

  try {
    Buffer = new ur_buffer(Context, Size, HostPtrOrNull, HostPtrImported);
    bufferGuard.reset(Buffer);
    importGuard.release();
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  // Initialize the buffer with user data
  if (Host) {
    if ((Flags & UR_MEM_FLAG_USE_HOST_POINTER) != 0 ||
        (Flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) != 0) {

      // We don't yet know which device needs this buffer, so make the first
      // device in the context be the master, and hold the initial valid
      // allocation.
      char *ZeHandleDst;
      UR_CALL(Buffer->getZeHandle(
          ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only,
          Context->getDevices()[0], nullptr, 0u));
      if (Buffer->OnHost) {
        // Do a host to host copy.
        // For an imported HostPtr the copy is unneeded.
        if (!HostPtrImported)
          memcpy(ZeHandleDst, Host, Size);
      } else {
        // Initialize the buffer synchronously with immediate offload
        // zeCommandListAppendMemoryCopy must not be called from simultaneous
        // threads with the same command list handle, so we need exclusive lock.
        std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
        ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                   (Context->ZeCommandListInit, ZeHandleDst, Host, Size,
                    nullptr, 0, nullptr));
      }
    } else if (Flags == 0 || (Flags == UR_MEM_FLAG_READ_WRITE)) {
      // Nothing more to do.
    } else
      die("urMemBufferCreate: not implemented");
  }

  *RetBuffer = v1_cast(static_cast<ur_mem_handle_t_ *>(bufferGuard.release()));
  return UR_RESULT_SUCCESS;
}

ur_result_t urMemRetain(
    /// [in] handle of the memory object to get access
    ::ur_mem_handle_t MemOpque) {
  v1_cast(MemOpque)->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t urMemRelease(
    /// [in] handle of the memory object to release
    ::ur_mem_handle_t MemOpque) {
  auto UrMem = v1_cast(MemOpque);
  if (!UrMem->RefCount.release())
    return UR_RESULT_SUCCESS;

  if (UrMem->isImage()) {
    char *ZeHandleImage;
    auto Image = static_cast<ur_image *>(UrMem);
    if (Image->OwnNativeHandle) {
      UR_CALL(UrMem->getZeHandle(
          ZeHandleImage, ur::level_zero::v1::ur_mem_handle_t_::write_only,
          nullptr, nullptr, 0u));
      if (checkL0LoaderTeardown()) {
        auto ZeResult = ZE_CALL_NOCHECK(
            zeImageDestroy, (ur_cast<ze_image_handle_t>(ZeHandleImage)));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && (ZeResult != ZE_RESULT_ERROR_UNINITIALIZED &&
                         ZeResult != ZE_RESULT_ERROR_UNKNOWN))
          return ze2urResult(ZeResult);
        if (ZeResult == ZE_RESULT_ERROR_UNKNOWN) {
          ZeResult = ZE_RESULT_ERROR_UNINITIALIZED;
        }
      }
    }
    delete Image;
  } else {
    auto Buffer = static_cast<ur_buffer *>(UrMem);
    Buffer->free();
    delete Buffer;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemBufferPartition(

    /// [in] handle of the buffer object to allocate from
    ::ur_mem_handle_t BufferOpque,
    /// [in] allocation and usage information flags
    ur_mem_flags_t Flags,
    /// [in] buffer creation type
    ur_buffer_create_type_t /*BufferCreateType*/,
    /// [in] pointer to buffer create region information
    const ur_buffer_region_t *BufferCreateInfo,
    /// [out] pointer to the handle of sub buffer created
    ::ur_mem_handle_t *RetMem) {
  auto Buffer = v1_cast(BufferOpque);
  UR_ASSERT(BufferOpque && !Buffer->isImage() &&
                !(static_cast<ur_buffer *>(Buffer))->isSubBuffer(),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  std::shared_lock<ur_shared_mutex> Guard(Buffer->Mutex);

  if (Flags != UR_MEM_FLAG_READ_WRITE) {
    die("urMemBufferPartition: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }

  std::unique_ptr<ur_buffer> partitionedBuffer;
  try {
    partitionedBuffer = std::make_unique<ur_buffer>(
        static_cast<ur_buffer *>(Buffer), BufferCreateInfo->origin,
        BufferCreateInfo->size);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  *RetMem =
      v1_cast(static_cast<ur_mem_handle_t_ *>(partitionedBuffer.release()));
  return UR_RESULT_SUCCESS;
}

ur_result_t urMemGetNativeHandle(
    /// [in] handle of the mem.
    ::ur_mem_handle_t MemOpque,
    /// [in] handle of the device.
    ::ur_device_handle_t,
    /// [out] a pointer to the native handle of the mem.
    ::ur_native_handle_t *NativeMem) {
  auto Mem = v1_cast(MemOpque);
  std::shared_lock<ur_shared_mutex> Guard(Mem->Mutex);
  char *ZeHandle = nullptr;
  UR_CALL(Mem->getZeHandle(ZeHandle,
                           ur::level_zero::v1::ur_mem_handle_t_::read_write,
                           nullptr, nullptr, 0u));
  *NativeMem = ur_cast<ur_native_handle_t>(ZeHandle);

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemBufferCreateWithNativeHandle(
    /// [in] the native handle to the memory.
    ::ur_native_handle_t NativeMem,
    /// [in] handle of the context object.
    ::ur_context_handle_t ContextOpque,
    /// [in][optional] pointer to native memory creation properties.
    const ur_mem_native_properties_t *Properties,
    /// [out] pointer to handle of buffer memory object created.
    ::ur_mem_handle_t *Mem) {
  auto Context = v1_cast(ContextOpque);
  bool OwnNativeHandle = Properties ? Properties->isNativeHandleOwned : false;

  std::unique_ptr<ur_buffer, void (*)(ur_buffer *)> Buffer = {
      nullptr, [](ur_buffer *handle) {
        if (handle) {
          handle->DeviceMappedHostNativePtr = nullptr;
          handle->free();
          delete handle;
        }
      }};

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  // Get base of the allocation
  void *Ptr = ur_cast<void *>(NativeMem);

  void *Base;
  size_t Size;
  ZE2UR_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, &Base, &Size));

  UR_ASSERT(Ptr == Base, UR_RESULT_ERROR_INVALID_VALUE);

  ze_device_handle_t ZeDevice;
  ZeStruct<ze_memory_allocation_properties_t> MemoryAttrs;
  UR_CALL(getMemoryAttrs(Context->ZeContext, Ptr, &ZeDevice, &MemoryAttrs));
  if (MemoryAttrs.type == ZE_MEMORY_TYPE_UNKNOWN) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  ur_device_handle_t Device{};
  if (ZeDevice) {
    Device = Context->getPlatform()->getDeviceFromNativeHandle(ZeDevice);
    UR_ASSERT(Context->isValidDevice(Device), UR_RESULT_ERROR_INVALID_CONTEXT);
  }

  try {
    Buffer.reset(new ur_buffer(Context, Size, Device,
                               ur_cast<char *>(NativeMem),
                               /*OwnZeMemHandle=*/false));
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  if (Device) {
    // If this allocation is on a device, then we re-use it for the buffer.
    // Nothing to do.
  } else if (Buffer->OnHost) {
    // If this is host allocation and buffer always stays on host there
    // nothing more to do.
  } else {
    // In all other cases (shared allocation, or host allocation that cannot
    // represent the buffer in this context) copy the data to a newly
    // created device allocation.
    char *ZeHandleDst;
    UR_CALL(Buffer->getZeHandle(
        ZeHandleDst, ur::level_zero::v1::ur_mem_handle_t_::write_only, Device,
        nullptr, 0u));

    // Indicate that this buffer has the device buffer mapped to a native buffer
    // and track the native pointer such that the memory is synced later at
    // memory free.
    Buffer->DeviceMappedHostNativePtr = Ptr;
    // zeCommandListAppendMemoryCopy must not be called from simultaneous
    // threads with the same command list handle, so we need exclusive lock.
    std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
    ZE2UR_CALL(zeCommandListAppendMemoryCopy,
               (Context->ZeCommandListInit, ZeHandleDst, Ptr, Size, nullptr, 0,
                nullptr));
  }

  ur_platform_handle_t Plt = Context->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  // If we don't own the native handle then we can't control deallocation of
  // that memory so there is no point of keeping track of the memory
  // allocation for deferred memory release in the mode when indirect access
  // tracking is enabled.
  if (IndirectAccessTrackingEnabled && OwnNativeHandle) {
    // We need to keep track of all memory allocations in the context
    ContextsLock.lock();
    try {
      Context->MemAllocs.emplace(
          std::piecewise_construct, std::forward_as_tuple(Ptr),
          std::forward_as_tuple(Context, OwnNativeHandle));
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    // Retain context to be sure that it is released after all memory
    // allocations in this context are released.
    UR_CALL(ur::level_zero::v1::urContextRetain(ContextOpque));
  }

  if (OwnNativeHandle)
    Buffer->Allocations[Device].ReleaseAction =
        ur_buffer::allocation_t::free_native;

  *Mem = v1_cast(static_cast<ur_mem_handle_t_ *>(Buffer.release()));
  return UR_RESULT_SUCCESS;
}

ur_result_t urMemGetInfo(
    /// [in] handle to the memory object being queried.
    ::ur_mem_handle_t MemoryOpque,
    /// [in] type of the info to retrieve.
    ur_mem_info_t MemInfoType,
    /// [in] the number of bytes of memory pointed to by pMemInfo.
    size_t PropSize,
    /// [out][optional] array of bytes holding the info. If propSize is less
    /// than the real number of bytes needed to return the info then the
    /// ::UR_RESULT_ERROR_INVALID_SIZE error is returned and pMemInfo is not
    /// used.
    void *MemInfo,
    /// [out][optional] pointer to the actual size in bytes of data queried by
    /// pMemInfo.
    size_t *PropSizeRet) {
  auto Buffer = reinterpret_cast<ur_buffer *>(MemoryOpque);
  std::shared_lock<ur_shared_mutex> Lock(Buffer->Mutex);
  UrReturnHelper ReturnValue(PropSize, MemInfo, PropSizeRet);

  switch (MemInfoType) {
  case UR_MEM_INFO_CONTEXT: {
    return ReturnValue(Buffer->UrContext);
  }
  case UR_MEM_INFO_SIZE: {
    // Get size of the allocation
    return ReturnValue(size_t{Buffer->Size});
  }
  case UR_MEM_INFO_REFERENCE_COUNT: {
    return ReturnValue(Buffer->RefCount.getCount());
  }
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemImageGetInfo(
    /// [in] handle to the image object being queried.
    ::ur_mem_handle_t /*Memory*/,
    /// [in] type of image info to retrieve.
    ur_image_info_t /*ImgInfoType*/,
    /// [in] the number of bytes of memory pointer to by pImgInfo.
    size_t /*PropSize*/,
    /// [out][optional] array of bytes holding the info. If propSize is less
    /// than the real number of bytes needed to return the info then the
    /// ::UR_RESULT_ERROR_INVALID_SIZE error is returned and pImgInfo is not
    /// used.
    void * /*ImgInfo*/,
    /// [out][optional] pointer to the actual size in bytes of data queried by
    /// pImgInfo.
    size_t * /*PropSizeRet*/) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urEnqueueUSMFill(
    /// [in] handle of the queue object
    ::ur_queue_handle_t QueueOpque,
    /// [in] pointer to USM memory object
    void *Ptr,
    /// [in] the size in bytes of the pattern. Must be a power of 2 and less
    /// than or equal to width.
    size_t PatternSize,
    /// [in] pointer with the bytes of the pattern to set.
    const void *Pattern,
    /// [in] size in bytes to be set. Must be a multiple of patternSize.
    size_t Size,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ::ur_event_handle_t *EventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ::ur_event_handle_t *Event) {
  auto Queue = v1_cast(QueueOpque);
  auto EventWaitListInternal = v1_cast(EventWaitList);
  auto EventInternal = v1_cast(Event);
  std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);

  return enqueueMemFillHelper(
      // TODO: do we need a new command type for USM memset?
      UR_COMMAND_MEM_BUFFER_FILL, Queue, Ptr,
      Pattern,     // It will be interpreted as an 8-bit value,
      PatternSize, // which is indicated with this pattern_size==1
      Size, NumEventsInWaitList, EventWaitListInternal, EventInternal);
}

/// Host Pipes
ur_result_t urEnqueueReadHostPipe(
    ::ur_queue_handle_t /*hQueue*/, ::ur_program_handle_t /*hProgram*/,
    const char * /*pipe_symbol*/, bool /*blocking*/, void * /*pDst*/,
    size_t /*size*/, uint32_t /*numEventsInWaitList*/,
    const ::ur_event_handle_t * /*phEventWaitList*/,
    ::ur_event_handle_t * /*phEvent*/) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urEnqueueWriteHostPipe(
    ::ur_queue_handle_t /*hQueue*/, ::ur_program_handle_t /*hProgram*/,
    const char * /*pipe_symbol*/, bool /*blocking*/, void * /*pSrc*/,
    size_t /*size*/, uint32_t /*numEventsInWaitList*/,
    const ::ur_event_handle_t * /*phEventWaitList*/,
    ::ur_event_handle_t * /*phEvent*/) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCGetMemHandleExp(::ur_context_handle_t, void *pMem,
                                 void **ppIPCMemHandleData,
                                 size_t *pIPCMemHandleDataSizeRet) {
  umf_memory_pool_handle_t umfPool;
  auto urRet = umf::umf2urResult(umfPoolByPtr(pMem, &umfPool));
  if (urRet)
    return urRet;

  // Fast path for returning the size of the handle only.
  if (!ppIPCMemHandleData)
    return umf::umf2urResult(
        umfPoolGetIPCHandleSize(umfPool, pIPCMemHandleDataSizeRet));

  size_t fallbackUMFHandleSize = 0;
  size_t *umfHandleSize = pIPCMemHandleDataSizeRet != nullptr
                              ? pIPCMemHandleDataSizeRet
                              : &fallbackUMFHandleSize;
  return umf::umf2urResult(umfGetIPCHandle(
      pMem, reinterpret_cast<umf_ipc_handle_t *>(ppIPCMemHandleData),
      umfHandleSize));
}

ur_result_t urIPCPutMemHandleExp(::ur_context_handle_t,
                                 void *pIPCMemHandleData) {
  return umf::umf2urResult(
      umfPutIPCHandle(reinterpret_cast<umf_ipc_handle_t>(pIPCMemHandleData)));
}

ur_result_t urIPCOpenMemHandleExp(::ur_context_handle_t hContextOpque,
                                  ::ur_device_handle_t hDevice,
                                  void *pIPCMemHandleData,
                                  size_t ipcMemHandleDataSize, void **ppMem) {
  auto hContext = v1_cast(hContextOpque);
  auto *pool = hContext->DefaultPool.getPool(
      usm::pool_descriptor{v1_cast(&hContext->DefaultPool), hContextOpque,
                           hDevice, UR_USM_TYPE_DEVICE, false});
  if (!pool)
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  umf_memory_pool_handle_t umfPool = pool->UmfPool.get();

  size_t umfHandleSize = 0;
  auto urRet =
      umf::umf2urResult(umfPoolGetIPCHandleSize(umfPool, &umfHandleSize));
  if (urRet)
    return urRet;

  if (umfHandleSize != ipcMemHandleDataSize)
    return UR_RESULT_ERROR_INVALID_VALUE;

  umf_ipc_handler_handle_t umfIPCHandler;
  urRet = umf::umf2urResult(umfPoolGetIPCHandler(umfPool, &umfIPCHandler));
  if (urRet)
    return urRet;

  return umf::umf2urResult(umfOpenIPCHandle(
      umfIPCHandler, reinterpret_cast<umf_ipc_handle_t>(pIPCMemHandleData),
      ppMem));
}

ur_result_t urIPCCloseMemHandleExp(::ur_context_handle_t, void *pMem) {
  return umf::umf2urResult(umfCloseIPCHandle(pMem));
}

// If indirect access tracking is enabled then performs reference counting,
// otherwise just calls zeMemAllocDevice.
static ur_result_t ZeDeviceMemAllocHelper(void **ResultPtr,
                                          ur_context_handle_t Context,
                                          ur_device_handle_t Device,
                                          size_t Size) {
  ur_platform_handle_t Plt = Device->Platform;
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while
    // we are in the process of allocating a memory, this is needed to
    // properly capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with
    // indirect access, that is why explicitly retain context to be sure
    // that it is released after all memory allocations in this context are
    // released.
    UR_CALL(ur::level_zero::v1::urContextRetain(v1_cast(Context)));
  }

  ze_device_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;
  ZE2UR_CALL(zeMemAllocDevice, (Context->ZeContext, &ZeDesc, Size, 1,
                                Device->ZeDevice, ResultPtr));

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*ResultPtr),
                               std::forward_as_tuple(Context));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_buffer::getBufferZeHandle(char *&ZeHandle,
                                         access_mode_t AccessMode,
                                         ur_device_handle_t Device,
                                         const ur_event_handle_t *phWaitEvents,
                                         uint32_t numWaitEvents) {

  // NOTE: There might be no valid allocation at all yet and we get
  // here from piEnqueueKernelLaunch that would be doing the buffer
  // initialization. In this case the Device is not null as kernel
  // launch is always on a specific device.
  if (!Device)
    Device = LastDeviceWithValidAllocation;
  // If the device is still not selected then use the first one in
  // the context of the buffer.
  if (!Device)
    Device = UrContext->getDevices()[0];

  auto &Allocation = Allocations[Device];

  if (this->isFreed) {
    die("getZeHandle() buffer already released, no valid handles.");
  }

  // Sub-buffers don't maintain own allocations but rely on parent buffer.
  if (SubBuffer) {
    // Verify that the Parent Buffer is still valid or if it has been freed.
    if (SubBuffer->Parent && !SubBuffer->Parent->isFreed) {
      UR_CALL(SubBuffer->Parent->getZeHandle(ZeHandle, AccessMode, Device,
                                             phWaitEvents, numWaitEvents));
      ZeHandle += SubBuffer->Origin;
      // Still store the allocation info in the PI sub-buffer for
      // getZeHandlePtr to work. At least zeKernelSetArgumentValue needs to
      // be given a pointer to the allocation handle rather than its value.
      //
      Allocation.ZeHandle = ZeHandle;
      Allocation.ReleaseAction = allocation_t::keep;
      LastDeviceWithValidAllocation = Device;
      return UR_RESULT_SUCCESS;
    } else {
      // Return an error if the parent buffer is already gone.
      die("getZeHandle() SubBuffer's parent already released, no valid "
          "handles.");
    }
  }

  // First handle case where the buffer is represented by only
  // a single host allocation.
  if (OnHost) {
    auto &HostAllocation = Allocations[nullptr];
    // The host allocation may already exists, e.g. with imported
    // host ptr, or in case of interop buffer.
    if (!HostAllocation.ZeHandle) {
      if (DisjointPoolConfigInstance.EnableBuffers) {
        HostAllocation.ReleaseAction = allocation_t::free;
        ur_usm_desc_t USMDesc{};
        USMDesc.align = getAlignment();
        ur_usm_pool_handle_t Pool{};
        UR_CALL(ur::level_zero::v1::urUSMHostAlloc(
            v1_cast(UrContext), &USMDesc, v1_cast(Pool), Size,
            reinterpret_cast<void **>(&ZeHandle)));
      } else {
        HostAllocation.ReleaseAction = allocation_t::free_native;
        UR_CALL(ZeHostMemAllocHelper(reinterpret_cast<void **>(&ZeHandle),
                                     UrContext, Size));
      }
      HostAllocation.ZeHandle = ZeHandle;
      HostAllocation.Valid = true;
    }
    Allocation = HostAllocation;
    Allocation.ReleaseAction = allocation_t::keep;
    ZeHandle = Allocation.ZeHandle;
    LastDeviceWithValidAllocation = Device;
    return UR_RESULT_SUCCESS;
  }

  // Peform actual device allocation as needed.
  if (!Allocation.ZeHandle) {
    // Create device allocation
    if (DisjointPoolConfigInstance.EnableBuffers) {
      Allocation.ReleaseAction = allocation_t::free;
      ur_usm_desc_t USMDesc{};
      USMDesc.align = getAlignment();
      ur_usm_pool_handle_t Pool{};
      UR_CALL(ur::level_zero::v1::urUSMDeviceAlloc(
          v1_cast(UrContext), common_cast(Device), &USMDesc, v1_cast(Pool),
          Size, reinterpret_cast<void **>(&ZeHandle)));
    } else {
      Allocation.ReleaseAction = allocation_t::free_native;
      UR_CALL(ZeDeviceMemAllocHelper(reinterpret_cast<void **>(&ZeHandle),
                                     UrContext, Device, Size));
    }
    Allocation.ZeHandle = ZeHandle;
  } else {
    ZeHandle = Allocation.ZeHandle;
  }

  // If some prior access invalidated this allocation then make it valid again.
  if (!Allocation.Valid) {
    // LastDeviceWithValidAllocation should always have valid allocation.
    if (Device == LastDeviceWithValidAllocation)
      die("getZeHandle: last used allocation is not valid");

    // For write-only access the allocation contents is not going to be used.
    // So don't do anything to make it "valid".
    bool NeedCopy =
        AccessMode != ur::level_zero::v1::ur_mem_handle_t_::write_only;
    // It's also possible that the buffer doesn't have a valid allocation
    // yet presumably when it is passed to a kernel that will perform
    // it's intialization.
    if (NeedCopy && !LastDeviceWithValidAllocation) {
      NeedCopy = false;
    }
    char *ZeHandleSrc = nullptr;
    if (NeedCopy) {
      UR_CALL(getZeHandle(
          ZeHandleSrc, ur::level_zero::v1::ur_mem_handle_t_::read_only,
          LastDeviceWithValidAllocation, phWaitEvents, numWaitEvents));
      // It's possible with the single root-device contexts that
      // the buffer is represented by the single root-device
      // allocation and then skip the copy to itself.
      if (ZeHandleSrc == ZeHandle)
        NeedCopy = false;
    }

    std::unique_ptr<ze_event_handle_t, void (*)(ze_event_handle_t *)>
        eventListGuard = {nullptr,
                          [](ze_event_handle_t *handle) { delete[] handle; }};

    if (NeedCopy) {
      // Wait on all dependency events passed in to ensure that the memory which
      // is being init is updated correctly.
      ur_ze_event_list_t waitlist;
      waitlist.ZeEventList = nullptr;
      waitlist.Length = 0;
      uint32_t EventListIndex = 0;
      for (unsigned i = 0; i < numWaitEvents; ++i) {
        if (phWaitEvents[i]->HostVisibleEvent) {
          ZE2UR_CALL(zeEventHostSynchronize,
                     (phWaitEvents[i]->ZeEvent, UINT64_MAX));
        } else {
          // Generate the waitlist for the Copy calls based on the passed in
          // dependencies, if they exist for device only.
          if (waitlist.ZeEventList == nullptr) {
            waitlist.ZeEventList = new ze_event_handle_t[numWaitEvents];
            eventListGuard.reset(waitlist.ZeEventList);
          }
          waitlist.ZeEventList[EventListIndex] = phWaitEvents[i]->ZeEvent;
          waitlist.Length++;
          EventListIndex++;
        }
      }
      if (waitlist.Length > 0) {
        ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
                   (UrContext->ZeCommandListInit, waitlist.Length,
                    waitlist.ZeEventList));
      }

      // Copy valid buffer data to this allocation.
      // TODO: see if we should better use peer's device allocation used
      // directly, if that capability is reported with zeDeviceCanAccessPeer,
      // instead of maintaining a separate allocation and performing
      // explciit copies.
      //
      // zeCommandListAppendMemoryCopy must not be called from simultaneous
      // threads with the same command list handle, so we need exclusive lock.
      ze_bool_t P2P = false;
      ZE2UR_CALL(
          zeDeviceCanAccessPeer,
          (Device->ZeDevice, LastDeviceWithValidAllocation->ZeDevice, &P2P));
      if (!P2P) {
        // P2P copy is not possible, so copy through the host.
        auto &HostAllocation = Allocations[nullptr];
        // The host allocation may already exists, e.g. with imported
        // host ptr, or in case of interop buffer.
        if (!HostAllocation.ZeHandle) {
          void *ZeHandleHost;
          if (DisjointPoolConfigInstance.EnableBuffers) {
            HostAllocation.ReleaseAction = allocation_t::free;
            ur_usm_desc_t USMDesc{};
            USMDesc.align = getAlignment();
            ur_usm_pool_handle_t Pool{};
            UR_CALL(ur::level_zero::v1::urUSMHostAlloc(v1_cast(UrContext),
                                                       &USMDesc, v1_cast(Pool),
                                                       Size, &ZeHandleHost));
          } else {
            HostAllocation.ReleaseAction = allocation_t::free_native;
            UR_CALL(ZeHostMemAllocHelper(&ZeHandleHost, UrContext, Size));
          }
          HostAllocation.ZeHandle = reinterpret_cast<char *>(ZeHandleHost);
          HostAllocation.Valid = false;
        }
        std::scoped_lock<ur_mutex> Lock(UrContext->ImmediateCommandListMutex);
        if (!HostAllocation.Valid) {
          ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                     (UrContext->ZeCommandListInit, HostAllocation.ZeHandle,
                      ZeHandleSrc, Size, nullptr, 0u, nullptr));
          // Mark the host allocation data  as valid so it can be reused.
          // It will be invalidated below if the current access is not
          // read-only.
          HostAllocation.Valid = true;
        }
        ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                   (UrContext->ZeCommandListInit, ZeHandle,
                    HostAllocation.ZeHandle, Size, nullptr, 0u, nullptr));
      } else {
        // Perform P2P copy.
        std::scoped_lock<ur_mutex> Lock(UrContext->ImmediateCommandListMutex);
        ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                   (UrContext->ZeCommandListInit, ZeHandle, ZeHandleSrc, Size,
                    nullptr, 0u, nullptr));
      }
    }
    Allocation.Valid = true;
    LastDeviceWithValidAllocation = Device;
  }

  // Invalidate other allocations that would become not valid if
  // this access is not read-only.
  if (AccessMode != ur::level_zero::v1::ur_mem_handle_t_::read_only) {
    for (auto &Alloc : Allocations) {
      if (Alloc.first != LastDeviceWithValidAllocation)
        Alloc.second.Valid = false;
    }
  }

  UR_LOG(DEBUG, "getZeHandle(pi_device{{{}}}) = {}", (void *)Device,
         (void *)Allocation.ZeHandle);
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_buffer::free() {
  for (auto &Alloc : Allocations) {
    auto &ZeHandle = Alloc.second.ZeHandle;
    // It is possible that the real allocation wasn't made if the buffer
    // wasn't really used in this location.
    if (!ZeHandle)
      continue;

    switch (Alloc.second.ReleaseAction) {
    case allocation_t::keep:
      break;
    case allocation_t::free: {
      ur_platform_handle_t Plt = UrContext->getPlatform();
      std::scoped_lock<ur_shared_mutex> Lock(IndirectAccessTrackingEnabled
                                                 ? Plt->ContextsMutex
                                                 : UrContext->Mutex);

      // If this memory was allocated as a proxy device buffer, then we must
      // copy the final memory contents back to the original native pointer
      // before releasing the buffer memory.
      if (DeviceMappedHostNativePtr != nullptr) {
        // zeCommandListAppendMemoryCopy must not be called from simultaneous
        // threads with the same command list handle, so we need exclusive lock.
        std::scoped_lock<ur_mutex> Lock(UrContext->ImmediateCommandListMutex);
        ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                   (UrContext->ZeCommandListInit, DeviceMappedHostNativePtr,
                    ZeHandle, Size, nullptr, 0, nullptr));
      }
      UR_CALL(USMFreeHelper(UrContext, ZeHandle));
      break;
    }
    case allocation_t::free_native:
      UR_CALL(ZeMemFreeHelper(UrContext, ZeHandle));
      break;
    case allocation_t::unimport:
      ZeUSMImport.doZeUSMRelease(
          UrContext->getPlatform()->ZeDriverHandleExpTranslated, ZeHandle);
      break;
    default:
      die("ur_buffer::free(): Unhandled release action");
    }
    ZeHandle = nullptr; // don't leave hanging pointers
    this->isFreed = true;
  }
  return UR_RESULT_SUCCESS;
}

// Buffer constructor
ur_buffer::ur_buffer(ur_context_handle_t Context, size_t Size, char *HostPtr,
                     bool ImportedHostPtr = false)
    : ur_mem_handle_t_(mem_type_t::buffer, Context), Size(Size) {

  // We treat integrated devices (physical memory shared with the CPU)
  // differently from discrete devices (those with distinct memories).
  // For integrated devices, allocating the buffer in the host memory
  // enables automatic access from the device, and makes copying
  // unnecessary in the map/unmap operations. This improves performance.
  OnHost = Context->getDevices().size() == 1 &&
           Context->getDevices()[0]->ZeDeviceProperties->flags &
               ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

  // Fill the host allocation data.
  if (HostPtr) {
    MapHostPtr = HostPtr;
    // If this host ptr is imported to USM then use this as a host
    // allocation for this buffer.
    if (ImportedHostPtr) {
      Allocations[nullptr].ZeHandle = HostPtr;
      Allocations[nullptr].Valid = true;
      Allocations[nullptr].ReleaseAction = ur_buffer::allocation_t::unimport;
    }
  }

  // This initialization does not end up with any valid allocation yet.
  LastDeviceWithValidAllocation = nullptr;
}

ur_buffer::ur_buffer(ur_context_handle_t Context, ur_device_handle_t Device,
                     size_t Size)
    : ur_mem_handle_t_(mem_type_t::buffer, Context, Device), Size(Size) {}

// Interop-buffer constructor
ur_buffer::ur_buffer(ur_context_handle_t Context, size_t Size,
                     ur_device_handle_t Device, char *ZeMemHandle,
                     bool OwnZeMemHandle)
    : ur_mem_handle_t_(mem_type_t::buffer, Context, Device), Size(Size) {

  // Device == nullptr means host allocation
  Allocations[Device].ZeHandle = ZeMemHandle;
  Allocations[Device].Valid = true;
  Allocations[Device].ReleaseAction =
      OwnZeMemHandle ? allocation_t::free_native : allocation_t::keep;

  // Check if this buffer can always stay on host
  OnHost = false;
  if (!Device) { // Host allocation
    if (Context->getDevices().size() == 1 &&
        Context->getDevices()[0]->ZeDeviceProperties->flags &
            ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
      OnHost = true;
      MapHostPtr = ZeMemHandle; // map to this allocation
    }
  }
  LastDeviceWithValidAllocation = Device;
}

ur_buffer::~ur_buffer() {
  if (isSubBuffer())
    try {
      ur::level_zero::v1::urMemRelease(
          reinterpret_cast<::ur_mem_handle_t>(SubBuffer->Parent));
    } catch (...) {
      UR_LOG_SAFE(ERR, "UR buffer destructor: Error releasing parent buffer");
    }
}

ur_result_t ur::level_zero::v1::ur_mem_handle_t_::getZeHandle(
    char *&ZeHandle, access_mode_t mode, ur_device_handle_t Device,
    const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents) {
  switch (mem_type) {
  case ur::level_zero::v1::ur_mem_handle_t_::image:
    return reinterpret_cast<ur_image *>(this)->getImageZeHandle(
        ZeHandle, mode, Device, phWaitEvents, numWaitEvents);
  case ur::level_zero::v1::ur_mem_handle_t_::buffer:
    return reinterpret_cast<ur_buffer *>(this)->getBufferZeHandle(
        ZeHandle, mode, Device, phWaitEvents, numWaitEvents);
  default:
    die("ur::level_zero::v1::ur_mem_handle_t_::getZeHandle: Unhandled memory "
        "type");
  }
  ur::unreachable();
}

ur_result_t ur::level_zero::v1::ur_mem_handle_t_::getZeHandlePtr(
    char **&ZeHandlePtr, access_mode_t mode, ur_device_handle_t Device,
    const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents) {
  switch (mem_type) {
  case ur::level_zero::v1::ur_mem_handle_t_::image:
    return reinterpret_cast<ur_image *>(this)->getImageZeHandlePtr(
        ZeHandlePtr, mode, Device, phWaitEvents, numWaitEvents);
  case ur::level_zero::v1::ur_mem_handle_t_::buffer:
    return reinterpret_cast<ur_buffer *>(this)->getBufferZeHandlePtr(
        ZeHandlePtr, mode, Device, phWaitEvents, numWaitEvents);
  default:
    die("ur::level_zero::v1::ur_mem_handle_t_::getZeHandle: Unhandled memory "
        "type");
  }
  ur::unreachable();
}

ur_result_t ur_buffer::getBufferZeHandlePtr(
    char **&ZeHandlePtr, access_mode_t AccessMode, ur_device_handle_t Device,
    const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents) {
  char *ZeHandle;
  UR_CALL(
      getZeHandle(ZeHandle, AccessMode, Device, phWaitEvents, numWaitEvents));
  ZeHandlePtr = &Allocations[Device].ZeHandle;
  return UR_RESULT_SUCCESS;
}

size_t ur_buffer::getAlignment() const {
  // Choose an alignment that is at most 64 and is the next power of 2
  // for sizes less than 64.
  auto Alignment = Size;
  if (Alignment > 32UL)
    Alignment = 64UL;
  else if (Alignment > 16UL)
    Alignment = 32UL;
  else if (Alignment > 8UL)
    Alignment = 16UL;
  else if (Alignment > 4UL)
    Alignment = 8UL;
  else if (Alignment > 2UL)
    Alignment = 4UL;
  else if (Alignment > 1UL)
    Alignment = 2UL;
  else
    Alignment = 1UL;
  return Alignment;
}

} // namespace ur::level_zero::v1
