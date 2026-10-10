/*
 * Copyright (C) 2018-2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "shared/source/command_stream/command_stream_receiver.h"

#include "shared/source/aub/aub_helper.h"
#include "shared/source/command_container/implicit_scaling.h"
#include "shared/source/command_stream/aub_subcapture_status.h"
#include "shared/source/command_stream/front_end_controller.h"
#include "shared/source/command_stream/scratch_space_controller.h"
#include "shared/source/command_stream/submission_status.h"
#include "shared/source/command_stream/submissions_aggregator.h"
#include "shared/source/command_stream/tag_allocation_layout.h"
#include "shared/source/debug_settings/debug_settings_manager.h"
#include "shared/source/device/device.h"
#include "shared/source/direct_submission/direct_submission_controller.h"
#include "shared/source/execution_environment/execution_environment.h"
#include "shared/source/execution_environment/root_device_environment.h"
#include "shared/source/gmm_helper/cache_settings_helper.h"
#include "shared/source/gmm_helper/page_table_mngr.h"
#include "shared/source/helpers/api_specific_config.h"
#include "shared/source/helpers/array_count.h"
#include "shared/source/helpers/compiler_product_helper.h"
#include "shared/source/helpers/flat_batch_buffer_helper.h"
#include "shared/source/helpers/flush_stamp.h"
#include "shared/source/helpers/gfx_core_helper.h"
#include "shared/source/helpers/pause_on_gpu_properties.h"
#include "shared/source/helpers/ray_tracing_helper.h"
#include "shared/source/host_function/host_function_allocator.h"
#include "shared/source/host_function/host_function_worker_interface.h"
#include "shared/source/memory_manager/allocation_properties.h"
#include "shared/source/memory_manager/internal_allocation_storage.h"
#include "shared/source/memory_manager/memory_manager.h"
#include "shared/source/memory_manager/surface.h"
#include "shared/source/os_interface/os_context.h"
#include "shared/source/os_interface/os_interface.h"
#include "shared/source/os_interface/os_thread.h"
#include "shared/source/os_interface/product_helper.h"
#include "shared/source/os_interface/sys_calls_common.h"
#include "shared/source/release_helpers/compiler_release_helper/compiler_release_helper.h"
#include "shared/source/release_helpers/release_helper/release_helper.h"
#include "shared/source/utilities/buffer_pool_allocator.inl"
#include "shared/source/utilities/hw_timestamps.h"
#include "shared/source/utilities/perf_counter.h"
#include "shared/source/utilities/pool_allocators.h"
#include "shared/source/utilities/tag_allocator.h"
#include "shared/source/utilities/wait_util.h"

#include "aubstream/hardware_context.h"

#include <array>
#include <iostream>

namespace NEO {

// Global table of CommandStreamReceiver factories for HW and tests
CommandStreamReceiverCreateFunc commandStreamReceiverFactory[2 * NEO::maxCoreEnumValue] = {};

namespace {
bool gpuAddressInAllocationRange(const GraphicsAllocation &allocation, uint64_t gpuAddress) {
    if (gpuAddress < allocation.getGpuAddress()) {
        return false;
    }

    const auto offsetInAllocation = ptrDiff(gpuAddress, allocation.getGpuAddress());
    return offsetInAllocation < allocation.getUnderlyingBufferSize();
}

GraphicsAllocation *getGraphicsAllocationForTagNode(MultiGraphicsAllocation &multiAllocation, uint64_t tagGpuAddress, uint32_t rootDeviceIndex) {
    auto *rootDeviceAllocation = multiAllocation.getGraphicsAllocation(rootDeviceIndex);
    if ((rootDeviceAllocation != nullptr) && gpuAddressInAllocationRange(*rootDeviceAllocation, tagGpuAddress)) {
        return rootDeviceAllocation;
    }

    for (auto *allocation : multiAllocation.getGraphicsAllocations()) {
        if ((allocation != nullptr) && gpuAddressInAllocationRange(*allocation, tagGpuAddress)) {
            return allocation;
        }
    }

    if (rootDeviceAllocation != nullptr) {
        return rootDeviceAllocation;
    }

    return multiAllocation.getDefaultGraphicsAllocation();
}

bool tryGetTagNodeChunkOffsetInAllocation(const TagNodeBase &tagNode, const GraphicsAllocation &gfxAllocation, uint64_t tagOffset, size_t chunkSize, uint64_t &chunkOffset) {
    const auto allocationCpuBase = gfxAllocation.getUnderlyingBuffer();
    if ((allocationCpuBase != nullptr) &&
        byteRangeContains(allocationCpuBase, gfxAllocation.getUnderlyingBufferSize(), tagNode.getCpuBase())) {
        chunkOffset = ptrDiff(tagNode.getCpuBase(), allocationCpuBase) + tagOffset;
    } else {
        if (tagNode.getGpuAddress() < gfxAllocation.getGpuAddress()) {
            return false;
        }
        chunkOffset = ptrDiff(tagNode.getGpuAddress(), gfxAllocation.getGpuAddress()) + tagOffset;
    }

    if (chunkOffset > gfxAllocation.getUnderlyingBufferSize()) {
        return false;
    }
    const auto availableChunkSize = gfxAllocation.getUnderlyingBufferSize() - static_cast<size_t>(chunkOffset);
    return chunkSize <= availableChunkSize;
}
} // namespace

CommandStreamReceiver::CommandStreamReceiver(ExecutionEnvironment &executionEnvironment,
                                             uint32_t rootDeviceIndex,
                                             const DeviceBitfield deviceBitfield)
    : executionEnvironment(executionEnvironment),
      debugConfirmationFunction([]() { std::cin.get(); }),
      rootDeviceIndex(rootDeviceIndex),
      deviceBitfield(deviceBitfield) {
    residencyAllocations.reserve(startingResidencyContainerSize);

    latestSentStatelessMocsConfig = CacheSettings::unknownMocs;
    lastSentSliceCount = QueueSliceCount::defaultSliceCount;
    lastAdditionalKernelExecInfo = AdditionalKernelExecInfo::notSet;
    submissionAggregator.reset(new SubmissionAggregator());
    if (ApiSpecificConfig::getApiType() == ApiSpecificConfig::L0) {
        this->dispatchMode = DispatchMode::immediateDispatch;
    }
    if (debugManager.flags.CsrDispatchMode.get()) {
        this->dispatchMode = (DispatchMode)debugManager.flags.CsrDispatchMode.get();
    }
    flushStamp.reset(new FlushStampTracker(true));
    for (int i = 0; i < IndirectHeap::Type::numTypes; ++i) {
        indirectHeap[i] = nullptr;
    }
    internalAllocationStorage = std::make_unique<InternalAllocationStorage>(*this);
    const auto &hwInfo = peekHwInfo();
    uint32_t subDeviceCount = static_cast<uint32_t>(deviceBitfield.count());
    auto &gfxCoreHelper = getGfxCoreHelper();
    auto &rootDeviceEnvironment = peekRootDeviceEnvironment();
    bool platformImplicitScaling = gfxCoreHelper.platformSupportsImplicitScaling(rootDeviceEnvironment);
    if (NEO::ImplicitScalingHelper::isImplicitScalingEnabled(deviceBitfield, platformImplicitScaling) &&
        subDeviceCount > 1 &&
        debugManager.flags.EnableStaticPartitioning.get() != 0) {
        this->activePartitions = subDeviceCount;
        this->staticWorkPartitioningEnabled = true;
    }
    this->streamProperties.initSupport(rootDeviceEnvironment);
    auto &productHelper = getProductHelper();
    productHelper.fillFrontEndPropertiesSupportStructure(feSupportFlags, hwInfo);
    productHelper.fillPipelineSelectPropertiesSupportStructure(pipelineSupportFlags, hwInfo);
    productHelper.fillStateBaseAddressPropertiesSupportStructure(sbaSupportFlags);
    this->doubleSbaWa = productHelper.isAdditionalStateBaseAddressWARequired(hwInfo);
    this->l1CachePolicyData.init(productHelper);

    registeredClients.reserve(16);

    auto &compilerProductHelper = rootDeviceEnvironment.getHelper<CompilerProductHelper>();
    this->heaplessModeEnabled = compilerProductHelper.isHeaplessModeEnabled(hwInfo);
    this->hostFunctionWorkerMode = static_cast<HostFunctionWorkerMode>(debugManager.flags.HostFunctionWorkMode.get());
    this->skipPreemptionAllocation = getOSInterface() && getOSInterface()->getDriverModel() && getOSInterface()->getDriverModel()->isLatePreemptionStartSupported(hwInfo);
}

CommandStreamReceiver::~CommandStreamReceiver() {
    if (userPauseConfirmation) {
        {
            std::unique_lock<SpinLock> lock{debugPauseStateLock};
            *debugPauseStateAddress = DebugPauseState::terminate;
        }

        userPauseConfirmation->join();
    }

    for (int i = 0; i < IndirectHeap::Type::numTypes; ++i) {
        if (indirectHeap[i] != nullptr) {
            releaseIndirectHeap(static_cast<IndirectHeap::Type>(i));
            delete indirectHeap[i];
            indirectHeap[i] = nullptr;
        }
    }
    cleanupResources();

    internalAllocationStorage->cleanAllocationList(-1, REUSABLE_ALLOCATION);
    internalAllocationStorage->cleanAllocationList(-1, TEMPORARY_ALLOCATION);
    internalAllocationStorage->cleanAllocationList(-1, DEFERRED_DEALLOCATION);
    getMemoryManager()->unregisterEngineForCsr(this);
}

SubmissionStatus CommandStreamReceiver::submitBatchBuffer(BatchBuffer &batchBuffer, ResidencyContainer &allocationsForResidency) {
    this->latestSentTaskCount = taskCount + 1;

    SubmissionStatus retVal = this->flush(batchBuffer, allocationsForResidency);

    if (retVal != NEO::SubmissionStatus::success) {
        return retVal;
    }
    if (!isUpdateTagFromWaitEnabled()) {
        this->latestFlushedTaskCount = taskCount + 1;
    }
    taskCount++;

    return retVal;
}

void CommandStreamReceiver::makeResident(MultiGraphicsAllocation &gfxAllocation) {
    makeResident(*gfxAllocation.getGraphicsAllocation(rootDeviceIndex));
}

void CommandStreamReceiver::makeResident(GraphicsAllocation &gfxAllocation) {
    auto submissionTaskCount = this->taskCount + 1;

    gfxAllocation.updateTaskCount(submissionTaskCount, osContext->getContextId());

    if (gfxAllocation.isResidencyTaskCountBelow(submissionTaskCount, osContext->getContextId())) {
        auto pushAllocations = this->pushAllocationsForMakeResident;

        if (debugManager.flags.MakeEachAllocationResident.get() != -1) {
            pushAllocations = !debugManager.flags.MakeEachAllocationResident.get();
        }

        if (pushAllocations) {
            this->getResidencyAllocations().push_back(&gfxAllocation);
        }

        if (this->dispatchMode == DispatchMode::batchedDispatch) {
            checkForNewResources(submissionTaskCount, gfxAllocation.getTaskCount(osContext->getContextId()), gfxAllocation);
            if (!gfxAllocation.isResident(osContext->getContextId())) {
                this->totalMemoryUsed += gfxAllocation.getUnderlyingBufferSize();
            }
        }
    }

    gfxAllocation.updateResidencyTaskCount(submissionTaskCount, osContext->getContextId());
}

void CommandStreamReceiver::processEviction() {}

void CommandStreamReceiver::makeNonResident(GraphicsAllocation &gfxAllocation) {
    if (gfxAllocation.isResident(osContext->getContextId())) {
        if (gfxAllocation.peekEvictable() && !gfxAllocation.isAlwaysResident(osContext->getContextId())) {
            this->addToEvictionContainer(gfxAllocation);
        } else {
            gfxAllocation.setEvictable(true);
        }
    }

    if (!gfxAllocation.isAlwaysResident(this->osContext->getContextId())) {
        gfxAllocation.releaseResidencyInOsContext(this->osContext->getContextId());
    }
}

void CommandStreamReceiver::makeSurfacePackNonResident(ResidencyContainer &allocationsForResidency, bool clearAllocations) {
    for (auto &surface : allocationsForResidency) {
        this->makeNonResident(*surface);
    }
    if (clearAllocations) {
        allocationsForResidency.clear();
    }
    this->processEviction();
}

SubmissionStatus CommandStreamReceiver::processResidency(ResidencyContainer &allocationsForResidency, uint32_t handleId) {
    return SubmissionStatus::success;
}

void CommandStreamReceiver::makeResidentHostPtrAllocation(GraphicsAllocation *gfxAllocation) {
    makeResident(*gfxAllocation);
}

void CommandStreamReceiver::makeResidentPreemptionAllocation() {
    PRINT_STRING(debugManager.flags.PrintLateMidThreadPreemptionStartInfo.get(), stdout, "Late Mid Thread Preemption Start: Try making preemption allocation resident\n");

    auto preemptionAllocation = getPreemptionAllocation();
    if (preemptionAllocation && !getSkipPreemptionAllocation()) {
        PRINT_STRING(debugManager.flags.PrintLateMidThreadPreemptionStartInfo.get(), stdout, "Late Mid Thread Preemption Start: Preemption allocation is resident\n");
        makeResident(*preemptionAllocation);
    }
}

WaitStatus CommandStreamReceiver::waitForTaskCount(TaskCountType requiredTaskCount) {
    auto address = getTagAddress();
    if (!skipResourceCleanup() && address) {
        this->downloadTagAllocation(requiredTaskCount);
        return baseWaitFunction(address, WaitParams{false, false, false, 0}, requiredTaskCount);
    }

    return WaitStatus::ready;
}

WaitStatus CommandStreamReceiver::waitForTaskCountAndCleanAllocationList(TaskCountType requiredTaskCount, uint32_t allocationUsage) {
    WaitStatus waitStatus{WaitStatus::ready};
    auto &list = allocationUsage == TEMPORARY_ALLOCATION ? internalAllocationStorage->getTemporaryAllocations() : internalAllocationStorage->getAllocationsForReuse();
    if (!list.peekIsEmpty()) {
        waitStatus = this->CommandStreamReceiver::waitForTaskCount(requiredTaskCount);
    }
    internalAllocationStorage->cleanAllocationList(requiredTaskCount, allocationUsage);
    if (allocationUsage == TEMPORARY_ALLOCATION) {
        internalAllocationStorage->cleanAllocationList(requiredTaskCount, DEFERRED_DEALLOCATION);
    }

    return waitStatus;
}

WaitStatus CommandStreamReceiver::waitForTaskCountAndCleanTemporaryAllocationList(TaskCountType requiredTaskCount) {
    return waitForTaskCountAndCleanAllocationList(requiredTaskCount, TEMPORARY_ALLOCATION);
}

void CommandStreamReceiver::createHostFunctionWorker() {

    if (this->hostFunctionWorker != nullptr) {
        return;
    }

    bool skipHostFunctionExecution = getType() == NEO::CommandStreamReceiverType::aub ||
                                     getType() == NEO::CommandStreamReceiverType::nullAub;

    auto *rootDeviceEnvironment = this->executionEnvironment.rootDeviceEnvironments[rootDeviceIndex].get();

    HostFunctionFactory::createAndSetHostFunctionWorker(this->hostFunctionWorkerMode,
                                                        skipHostFunctionExecution,
                                                        this,
                                                        rootDeviceEnvironment);

    auto *streamer = &this->getHostFunctionStreamer();
    this->hostFunctionWorker->start(streamer);
}

HostFunctionWorker *CommandStreamReceiver::getHostFunctionWorker() {
    return this->hostFunctionWorker;
}

void CommandStreamReceiver::ensureCommandBufferAllocation(LinearStream &commandStream, size_t minimumRequiredSize, size_t additionalAllocationSize) {
    if (commandStream.getAvailableSpace() >= minimumRequiredSize) {
        return;
    }

    auto alignment = MemoryConstants::pageSize64k;

    if (debugManager.flags.ForceCommandBufferAlignment.get() != -1) {
        alignment = debugManager.flags.ForceCommandBufferAlignment.get() * MemoryConstants::kiloByte;
    }

    const auto allocationSize = alignUp(minimumRequiredSize + additionalAllocationSize, alignment);
    constexpr static auto allocationType = AllocationType::commandBuffer;

    GraphicsAllocation *allocation = nullptr;

    if (this->device && CommandBufferPoolAllocator::isEnabled(getProductHelper())) {
        auto ctx = createDeferredFreeContext();
        auto &poolAllocator = this->device->getCommandBufferPoolAllocator();
        allocation = poolAllocator.allocate(allocationSize, &ctx);
    }

    if (allocation == nullptr) {
        allocation = this->getInternalAllocationStorage()->obtainReusableAllocation(allocationSize, allocationType).release();
    }
    if (allocation == nullptr) {
        const AllocationProperties commandStreamAllocationProperties{rootDeviceIndex, true, allocationSize, allocationType,
                                                                     isMultiOsContextCapable(), false, osContext->getDeviceBitfield()};
        allocation = this->getMemoryManager()->allocateGraphicsMemoryWithProperties(commandStreamAllocationProperties);
    }
    DEBUG_BREAK_IF(allocation == nullptr);

    if (auto oldAllocation = commandStream.getGraphicsAllocation(); oldAllocation != nullptr) {
        releaseCommandBufferAllocation(oldAllocation);
        this->ucResourceRequiresTagUpdate = true;
    }

    commandStream.replaceBuffer(allocation->getUnderlyingBuffer(), allocationSize - additionalAllocationSize);
    commandStream.replaceGraphicsAllocation(allocation);
}

void CommandStreamReceiver::preallocateAllocation(AllocationType type, size_t size) {
    const AllocationProperties commandStreamAllocationProperties{rootDeviceIndex, true, size, type,
                                                                 isMultiOsContextCapable(), false, deviceBitfield};
    auto allocation = this->getMemoryManager()->allocateGraphicsMemoryWithProperties(commandStreamAllocationProperties);
    if (allocation) {
        this->makeResident(*allocation);
        getInternalAllocationStorage()->storeAllocation(std::unique_ptr<GraphicsAllocation>(allocation), REUSABLE_ALLOCATION);
    }
}

void CommandStreamReceiver::preallocateCommandBuffer() {
    preallocateAllocation(AllocationType::commandBuffer, MemoryConstants::pageSize64k);
}

void CommandStreamReceiver::preallocateInternalHeap() {
    preallocateAllocation(AllocationType::internalHeap, HeapSize::getDefaultHeapSize(IndirectHeapType::indirectObject));
}

void CommandStreamReceiver::fillReusableAllocationsList() {
    std::call_once(preallocateResourcesFlag, [&] {
        auto lock = obtainUniqueOwnership();
        auto &productHelper = getProductHelper();

        if (!CommandBufferPoolAllocator::isEnabled(productHelper)) {
            auto &gfxCoreHelper = getGfxCoreHelper();
            auto amountToFill = gfxCoreHelper.getAmountOfAllocationsToFill();
            for (auto i = 0u; i < amountToFill; i++) {
                preallocateCommandBuffer();
            }
        }

        const bool isBcs = EngineHelpers::isBcs(this->osContext->getEngineType());
        auto internalHeapsToFill = isBcs ? 0u : productHelper.getInternalHeapsPreallocated();
        const auto internalHeapPreallocationSize = HeapSize::getDefaultHeapSize(IndirectHeapType::indirectObject);
        const bool internalHeapPoolHandlesPreallocationSize = InternalHeapPoolAllocator::isEnabled(productHelper) &&
                                                              (internalHeapPreallocationSize <= InternalHeapPoolTraits::maxAllocationSize);
        if (!internalHeapPoolHandlesPreallocationSize) {
            for (auto i = 0u; i < internalHeapsToFill; i++) {
                preallocateInternalHeap();
            }
        }
    });
}

void CommandStreamReceiver::requestPreallocation() {
    auto &productHelper = getProductHelper();

    if (CommandBufferPoolAllocator::isEnabled(productHelper)) {
        return;
    }

    auto preallocationsPerQueue = productHelper.getCommandBuffersPreallocatedPerCommandQueue();
    if (debugManager.flags.SetAmountOfReusableAllocationsPerCmdQueue.get() != -1) {
        preallocationsPerQueue = debugManager.flags.SetAmountOfReusableAllocationsPerCmdQueue.get();
    }
    auto lock = obtainUniqueOwnership();
    requestedPreallocationsAmount += preallocationsPerQueue;
    const int64_t amountToPreallocate = static_cast<int64_t>(requestedPreallocationsAmount.load()) - preallocatedAmount;
    DEBUG_BREAK_IF(amountToPreallocate > preallocationsPerQueue);
    if (amountToPreallocate > 0) {
        for (auto i = 0u; i < amountToPreallocate; i++) {
            preallocateCommandBuffer();
        }
        preallocatedAmount += static_cast<uint32_t>(amountToPreallocate);
    }
}

void CommandStreamReceiver::releasePreallocationRequest() {
    auto &productHelper = getProductHelper();

    if (CommandBufferPoolAllocator::isEnabled(productHelper)) {
        return;
    }

    auto preallocationsPerQueue = productHelper.getCommandBuffersPreallocatedPerCommandQueue();
    if (debugManager.flags.SetAmountOfReusableAllocationsPerCmdQueue.get() != -1) {
        preallocationsPerQueue = debugManager.flags.SetAmountOfReusableAllocationsPerCmdQueue.get();
    }
    DEBUG_BREAK_IF(preallocationsPerQueue > requestedPreallocationsAmount);
    requestedPreallocationsAmount -= preallocationsPerQueue;
}

uint8_t CommandStreamReceiver::getUmdPowerHintValue() const {
    return this->osContext ? this->osContext->getUmdPowerHintValue() : 0u;
}

bool CommandStreamReceiver::initializeResources(const PreemptionMode preemptionMode) {
    if (!resourcesInitialized) {
        auto lock = obtainUniqueOwnership();
        if (!resourcesInitialized) {
            if (!osContext->ensureContextInitialized()) {
                return false;
            }

            if (preemptionMode == NEO::PreemptionMode::MidThread &&
                !this->getPreemptionAllocation()) {
                this->createPreemptionAllocation();
            }
            if (getProductHelper().isFrontEndControllerEnabled() && device->getFrontEndController() && !allocateFrontEndAllocation(device->getFrontEndController()->getFrontEndAllocationSize())) {
                return false;
            }
            this->resourcesInitialized = true;

            initializeEngine();
        }
    }

    return true;
}

MemoryManager *CommandStreamReceiver::getMemoryManager() const {
    DEBUG_BREAK_IF(!executionEnvironment.memoryManager);
    return executionEnvironment.memoryManager.get();
}

LinearStream &CommandStreamReceiver::getCS(size_t minRequiredSize) {
    constexpr static auto additionalAllocationSize = MemoryConstants::cacheLineSize + CSRequirements::csOverfetchSize;
    ensureCommandBufferAllocation(this->commandStream, minRequiredSize, additionalAllocationSize);
    return commandStream;
}

OSInterface *CommandStreamReceiver::getOSInterface() const {
    return executionEnvironment.rootDeviceEnvironments[rootDeviceIndex]->osInterface.get();
}

GmmHelper *CommandStreamReceiver::peekGmmHelper() const {
    return executionEnvironment.rootDeviceEnvironments[rootDeviceIndex]->gmmHelper.get();
}

uint64_t CommandStreamReceiver::getWorkPartitionAllocationGpuAddress() const {
    if (isStaticWorkPartitioningEnabled()) {
        return getWorkPartitionAllocation()->getGpuAddress();
    }
    return 0;
}

bool CommandStreamReceiver::isRcs() const {
    return this->osContext->getEngineType() == aub_stream::ENGINE_RCS;
}

bool CommandStreamReceiver::skipResourceCleanup() const {
    return ((this->getOSInterface() && this->getOSInterface()->getDriverModel() && this->getOSInterface()->getDriverModel()->skipResourceCleanup()) || forceSkipResourceCleanupRequired);
}

bool CommandStreamReceiver::isGpuHangDetected() const {
    if (debugManager.flags.DisableGpuHangDetection.get()) {
        return false;
    }
    auto isGpuHang = this->osContext && this->getOSInterface() && this->getOSInterface()->getDriverModel() && this->getOSInterface()->getDriverModel()->isGpuHangDetected(*osContext);
    if (isGpuHang) {
        this->getOSInterface()->getDriverModel()->getDeviceState();
    }
    return isGpuHang;
}

void CommandStreamReceiver::cleanupResources() {
    waitForTaskCountAndCleanAllocationList(this->latestFlushedTaskCount, TEMPORARY_ALLOCATION);
    waitForTaskCountAndCleanAllocationList(this->latestFlushedTaskCount, REUSABLE_ALLOCATION);

    if (debugSurface) {
        getMemoryManager()->freeGraphicsMemory(debugSurface);
        debugSurface = nullptr;
    }

    if (commandStream.getCpuBase()) {
        auto *csAllocation = commandStream.getGraphicsAllocation();
        bool freedToPool = false;
        if (this->device && csAllocation && csAllocation->getParentAllocation()) {
            auto &poolAllocator = this->device->getCommandBufferPoolAllocator();
            if (poolAllocator.isPoolBuffer(csAllocation->getParentAllocation())) {
                DEBUG_BREAK_IF(!csAllocation->isView());
                poolAllocator.free(csAllocation);
                freedToPool = true;
            }
        }
        if (!freedToPool) {
            getMemoryManager()->freeGraphicsMemory(csAllocation);
        }
        commandStream.replaceGraphicsAllocation(nullptr);
        commandStream.replaceBuffer(nullptr, 0);
    }

    if (hostFunctionWorker) {
        cleanupHostFunctionWorker();
    }

    if (tagsMultiAllocation) {
        // Null tag address to prevent waiting for tag update when freeing it
        tagAllocation = nullptr;
        tagAddress = nullptr;
        DEBUG_BREAK_IF(tagAllocation != nullptr);
        DEBUG_BREAK_IF(tagAddress != nullptr);

        for (auto graphicsAllocation : tagsMultiAllocation->getGraphicsAllocations()) {
            getMemoryManager()->freeGraphicsMemory(graphicsAllocation);
        }
        delete tagsMultiAllocation;
        tagsMultiAllocation = nullptr;
    }

    if (globalFenceAllocation) {
        getMemoryManager()->freeGraphicsMemory(globalFenceAllocation);
        globalFenceAllocation = nullptr;
    }

    if (preemptionAllocation) {
        getMemoryManager()->freeGraphicsMemory(preemptionAllocation);
        preemptionAllocation = nullptr;
    }

    if (clearColorAllocation) {
        getMemoryManager()->freeGraphicsMemory(clearColorAllocation);
        clearColorAllocation = nullptr;
    }

    if (workPartitionAllocation) {
        getMemoryManager()->freeGraphicsMemory(workPartitionAllocation);
        workPartitionAllocation = nullptr;
    }

    if (frontEndAllocation) {
        getMemoryManager()->freeGraphicsMemory(frontEndAllocation);
        frontEndAllocation = nullptr;
    }

    releaseGlobalStatelessHeap();
    for (auto &alloc : ownedPrivateAllocations) {
        getMemoryManager()->freeGraphicsMemory(alloc.second);
    }
    ownedPrivateAllocations.clear();
}

void CommandStreamReceiver ::cleanupHostFunctionWorker() {

    if (hostFunctionWorker) {
        hostFunctionWorker->finish();

        if (hostFunctionWorkerMode != HostFunctionWorkerMode::schedulerWithThreadPool) {
            delete hostFunctionWorker;
        }

        hostFunctionWorker = nullptr;
    }
}

WaitStatus CommandStreamReceiver::waitForCompletionWithTimeout(const WaitParams &params, TaskCountType taskCountToWait) {
    bool printWaitForCompletion = debugManager.flags.LogWaitingForCompletion.get();
    if (printWaitForCompletion) {
        printTagAddressContent(taskCountToWait, params.waitTimeout, true);
    }

    TaskCountType latestSentTaskCount = this->latestFlushedTaskCount;
    if (latestSentTaskCount < taskCountToWait) {
        if (!this->flushBatchedSubmissions()) {
            const auto isGpuHang{isGpuHangDetected()};
            return isGpuHang ? WaitStatus::gpuHang : WaitStatus::notReady;
        }
    }

    auto retCode = baseWaitFunction(getTagAddress(), params, taskCountToWait);
    if (printWaitForCompletion) {
        printTagAddressContent(taskCountToWait, params.waitTimeout, false);
    }
    return retCode;
}

bool CommandStreamReceiver::checkGpuHangDetected(TimeType currentTime, TimeType &lastHangCheckTime) const {
    std::chrono::microseconds elapsedTimeSinceGpuHangCheck = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - lastHangCheckTime);

    if (elapsedTimeSinceGpuHangCheck.count() >= gpuHangCheckPeriod.count()) {
        lastHangCheckTime = currentTime;
        if (isGpuHangDetected()) {
            return true;
        }
    }
    return false;
}

WaitStatus CommandStreamReceiver::baseWaitFunction(volatile TagAddressType *pollAddress, const WaitParams &params, TaskCountType taskCountToWait) {
    std::chrono::high_resolution_clock::time_point waitStartTime, lastHangCheckTime, currentTime;
    int64_t timeDiff = 0;

    TaskCountType latestSentTaskCount = this->latestFlushedTaskCount;
    if (latestSentTaskCount < taskCountToWait) {
        if (this->flushTagUpdate() != NEO::SubmissionStatus::success) {
            return WaitStatus::notReady;
        }
    }
    volatile TagAddressType *partitionAddress = pollAddress;

    waitStartTime = std::chrono::high_resolution_clock::now();
    lastHangCheckTime = waitStartTime;
    for (uint32_t i = 0; i < activePartitions; i++) {
        while (*partitionAddress < taskCountToWait && (!params.enableTimeout || timeDiff <= params.waitTimeout)) {
            this->downloadTagAllocation(taskCountToWait);

            if (!params.indefinitelyPoll && WaitUtils::waitFunction(partitionAddress, taskCountToWait, timeDiff)) {
                break;
            }

            currentTime = std::chrono::high_resolution_clock::now();
            if (checkGpuHangDetected(currentTime, lastHangCheckTime)) {
                return WaitStatus::gpuHang;
            }

            timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - waitStartTime).count();
        }

        partitionAddress = ptrOffset(partitionAddress, this->immWritePostSyncWriteOffset);
    }

    partitionAddress = pollAddress;
    for (uint32_t i = 0; i < activePartitions; i++) {
        if (*partitionAddress < taskCountToWait) {
            return WaitStatus::notReady;
        }
        partitionAddress = ptrOffset(partitionAddress, this->immWritePostSyncWriteOffset);
    }

    return WaitStatus::ready;
}

void CommandStreamReceiver::setTagAllocation(GraphicsAllocation *allocation) {
    this->tagAllocation = allocation;
    UNRECOVERABLE_IF(allocation == nullptr);
    this->tagAddress = reinterpret_cast<TagAddressType *>(allocation->getUnderlyingBuffer());
    this->debugPauseStateAddress = reinterpret_cast<DebugPauseState *>(
        reinterpret_cast<uint8_t *>(allocation->getUnderlyingBuffer()) + TagAllocationLayout::debugPauseStateAddressOffset);
    this->ucTagAddress = static_cast<TagAddressType *>(ptrOffset(allocation->getUnderlyingBuffer(), TagAllocationLayout::ucTagAddressOffset));
}

MultiGraphicsAllocation &CommandStreamReceiver::createMultiAllocationInSystemMemoryPool(AllocationType allocationType) {
    RootDeviceIndicesContainer rootDeviceIndices;

    rootDeviceIndices.pushUnique(rootDeviceIndex);

    auto maxRootDeviceIndex = static_cast<uint32_t>(this->executionEnvironment.rootDeviceEnvironments.size() - 1);
    auto allocations = new MultiGraphicsAllocation(maxRootDeviceIndex);

    AllocationProperties unifiedMemoryProperties{rootDeviceIndex, MemoryConstants::pageSize, allocationType, systemMemoryBitfield};

    this->getMemoryManager()->createMultiGraphicsAllocationInSystemMemoryPool(rootDeviceIndices, unifiedMemoryProperties, *allocations);
    return *allocations;
}

bool CommandStreamReceiver::ensureTagAllocationForRootDeviceIndex(uint32_t rootDeviceIndex) {
    UNRECOVERABLE_IF(!tagsMultiAllocation);
    if (rootDeviceIndex >= tagsMultiAllocation->getGraphicsAllocations().size()) {
        return false;
    }
    if (tagsMultiAllocation->getGraphicsAllocation(rootDeviceIndex)) {
        return true;
    }
    AllocationProperties allocationProperties{rootDeviceIndex, MemoryConstants::pageSize, AllocationType::tagBuffer, systemMemoryBitfield};
    allocationProperties.flags.allocateMemory = false;
    auto graphicsAllocation = this->getMemoryManager()->createGraphicsAllocationFromExistingStorage(allocationProperties, tagAllocation->getUnderlyingBuffer(), *tagsMultiAllocation);
    if (!graphicsAllocation) {
        return false;
    }
    tagsMultiAllocation->addAllocation(graphicsAllocation);
    return true;
}

FlushStamp CommandStreamReceiver::obtainCurrentFlushStamp() const {
    return flushStamp->peekStamp();
}

void CommandStreamReceiver::setRequiredScratchSizes(uint32_t newRequiredScratchSlot0Size, uint32_t newRequiredScratchSlot1Size) {
    if (newRequiredScratchSlot0Size > requiredScratchSlot0Size) {
        requiredScratchSlot0Size = newRequiredScratchSlot0Size;
    }
    if (newRequiredScratchSlot1Size > requiredScratchSlot1Size) {
        requiredScratchSlot1Size = newRequiredScratchSlot1Size;
    }
}

GraphicsAllocation *CommandStreamReceiver::getScratchAllocation() {
    return scratchSpaceController->getScratchSpaceSlot0Allocation();
}

void CommandStreamReceiver::overwriteFlatBatchBufferHelper(FlatBatchBufferHelper *newHelper) {
    flatBatchBufferHelper.reset(newHelper);
}

void CommandStreamReceiver::initProgrammingFlags() {
    isPreambleSent = false;
    gsbaFor32BitProgrammed = false;
    bindingTableBaseAddressRequired = true;
    mediaVfeStateDirty = true;
    lastVmeSubslicesConfig = false;
    stateComputeModeDirty = true;

    lastPreemptionMode = PreemptionMode::Initial;

    latestSentStatelessMocsConfig = CacheSettings::unknownMocs;
    this->streamProperties.stateBaseAddress.statelessMocs = {};
}

void CommandStreamReceiver::programForAubSubCapture(bool wasActiveInPreviousEnqueue, bool isActive) {
    if (!wasActiveInPreviousEnqueue && isActive) {
        // force CSR reprogramming upon subcapture activation
        this->initProgrammingFlags();
    }
    if (wasActiveInPreviousEnqueue && !isActive) {
        // flush BB upon subcapture deactivation
        this->flushBatchedSubmissions();
    }
}

ResidencyContainer &CommandStreamReceiver::getResidencyAllocations() {
    return this->residencyAllocations;
}

PrivateAllocsToReuseContainer &CommandStreamReceiver::getOwnedPrivateAllocations() {
    return this->ownedPrivateAllocations;
}

AubSubCaptureStatus CommandStreamReceiver::checkAndActivateAubSubCapture(const std::string &kernelName) { return {false, false}; }

void CommandStreamReceiver::addAubComment(const char *comment) {}

bool CommandStreamReceiver::isTlbFlushRequiredForStateCacheFlush() {
    return false;
}

void CommandStreamReceiver::downloadAllocation(GraphicsAllocation &gfxAllocation) {
    if (this->downloadAllocationImpl) {
        this->downloadAllocationImpl(gfxAllocation, 0, gfxAllocation.getUnderlyingBufferSize());
    }
}

bool CommandStreamReceiver::enqueueWaitForPagingFence(uint64_t pagingFenceValue) {
    auto controller = this->executionEnvironment.directSubmissionController.get();
    if (this->isAnyDirectSubmissionEnabled() && controller) {
        controller->enqueueWaitForPagingFence(this, pagingFenceValue);
        return true;
    }
    return false;
}

void CommandStreamReceiver::drainPagingFenceQueue() {
    auto controller = this->executionEnvironment.directSubmissionController.get();
    if (this->isAnyDirectSubmissionEnabled() && controller) {
        controller->drainPagingFenceQueue();
    }
}

// Returns a unique identifier for the context group this CSR belongs to
uint32_t CommandStreamReceiver::getContextGroupId() const {
    const OsContext &osContext = this->getOsContext();
    // If the context is part of a group, use the primary context's id as the group id
    if (osContext.isPartOfContextGroup()) {
        const OsContext *primary = osContext.getPrimaryContext();
        if (primary != nullptr) {
            return primary->getContextId();
        }
    }
    // Otherwise, use this context's id
    return osContext.getContextId();
}

GraphicsAllocation *CommandStreamReceiver::allocateDebugSurface(size_t size) {
    UNRECOVERABLE_IF(debugSurface != nullptr);
    if (primaryCsr) {
        return nullptr;
    }
    debugSurface = getMemoryManager()->allocateGraphicsMemoryWithProperties({rootDeviceIndex, size, AllocationType::debugContextSaveArea, getOsContext().getDeviceBitfield()});
    return debugSurface;
}

GraphicsAllocation *CommandStreamReceiver::allocateFrontEndAllocation(size_t size) {
    UNRECOVERABLE_IF(frontEndAllocation != nullptr);
    AllocationProperties properties{rootDeviceIndex, size, AllocationType::preemption, getOsContext().getDeviceBitfield()};
    properties.alignment = MemoryConstants::pageSize64k;

    frontEndAllocation = getMemoryManager()->allocateGraphicsMemoryWithProperties(properties);
    if (frontEndAllocation) {
        memset(frontEndAllocation->getUnderlyingBuffer(), 0, size);
    }
    return frontEndAllocation;
}

void *CommandStreamReceiver::getIndirectHeapCurrentPtr(IndirectHeapType heapType) const {
    auto heap = indirectHeap[heapType];
    if (heap) {
        return heap->getSpace(0);
    }
    return nullptr;
}

void CommandStreamReceiver::signalHostFunctionWorker(uint32_t nHostFunctions) {
    hostFunctionWorker->submit(nHostFunctions);
}

void CommandStreamReceiver::ensureHostFunctionWorkerStarted(HostFunctionAllocator *allocator) {
    if (!this->hostFunctionWorkerStarted.load(std::memory_order_acquire)) {
        startHostFunctionWorker(allocator);
    }
}

void CommandStreamReceiver::startHostFunctionWorker(HostFunctionAllocator *allocator) {
    auto lock = obtainHostFunctionWorkerStartLock();
    if (this->hostFunctionWorkerStarted.load(std::memory_order_relaxed)) {
        return;
    }

    createHostFunctionStreamer(allocator);
    createHostFunctionWorker();

    this->hostFunctionWorkerStarted.store(true, std::memory_order_release);
}

void CommandStreamReceiver::createHostFunctionStreamer(HostFunctionAllocator *allocator) {

    auto nPartitions = this->activePartitions;
    auto partitionOffset = this->immWritePostSyncWriteOffset;
    HostFunctionChunk chunk = allocator->obtainChunk();
    UNRECOVERABLE_IF(chunk.cpuPtr == nullptr);
    auto hostFunctionIdAddress = chunk.cpuPtr;

    auto useSemaphore64bCmd = peekRootDeviceEnvironment().getCompilerReleaseHelper().isAvailableSemaphore64(peekHwInfo());

    auto dcFlushRequired = this->getDcFlushSupport();
    this->hostFunctionStreamer = std::make_unique<HostFunctionStreamer>(this,
                                                                        chunk.allocation,
                                                                        hostFunctionIdAddress,
                                                                        this->downloadAllocationImpl,
                                                                        this->uploadAllocationChunkImpl,
                                                                        nPartitions,
                                                                        partitionOffset,
                                                                        isTbxMode(),
                                                                        dcFlushRequired,
                                                                        useSemaphore64bCmd,
                                                                        allocator->getTbxWriteMutex());
}

HostFunctionStreamer &CommandStreamReceiver::getHostFunctionStreamer() {
    return *hostFunctionStreamer.get();
}

std::unique_lock<CommandStreamReceiver::MutexType> CommandStreamReceiver::obtainHostAllocationLock() {
    return std::unique_lock<CommandStreamReceiver::MutexType>(this->hostAllocationMutex);
}

void CommandStreamReceiver::makeResidentHostFunctionAllocation() {
    auto &streamer = this->getHostFunctionStreamer();
    auto *hostFunctionAllocation = streamer.getHostFunctionIdAllocation();
    if (hostFunctionAllocation) {
        this->makeResident(*hostFunctionAllocation);
    }
}

IndirectHeap &CommandStreamReceiver::getIndirectHeap(IndirectHeap::Type heapType,
                                                     size_t minRequiredSize) {
    DEBUG_BREAK_IF(static_cast<uint32_t>(heapType) >= arrayCount(indirectHeap));
    auto &heap = indirectHeap[heapType];
    GraphicsAllocation *heapMemory = nullptr;

    if (heap) {
        heapMemory = heap->getGraphicsAllocation();
    }

    if (heap && heap->getAvailableSpace() < minRequiredSize && heapMemory) {
        releaseHeapAllocation(heapMemory);
        heap->replaceBuffer(nullptr, 0);
        heap->replaceGraphicsAllocation(nullptr);
        heapMemory = nullptr;
        this->heapStorageRequiresRecyclingTag = true;
    }

    if (!heapMemory) {
        allocateHeapMemory(heapType, minRequiredSize, heap);
    }

    return *heap;
}

void CommandStreamReceiver::allocateHeapMemory(IndirectHeap::Type heapType,
                                               size_t minRequiredSize, IndirectHeap *&indirectHeap) {
    size_t reservedSize = 0;
    auto defaultHeapSize = HeapSize::getHeapSize(HeapSize::getDefaultHeapSize(heapType));
    bool requireInternalHeap = IndirectHeap::Type::indirectObject == heapType ? canUse4GbHeaps() : false;

    if (debugManager.flags.AddPatchInfoCommentsForAUBDump.get()) {
        requireInternalHeap = false;
    }

    minRequiredSize += reservedSize;

    size_t allocationSize;
    if (IndirectHeap::Type::surfaceState == heapType) {
        DEBUG_BREAK_IF(minRequiredSize > defaultSshSize - MemoryConstants::pageSize);
        allocationSize = defaultSshSize;
    } else {
        allocationSize = alignUp(std::max(defaultHeapSize, minRequiredSize), MemoryConstants::pageSize);
    }

    auto allocationType = AllocationType::linearStream;
    if (requireInternalHeap) {
        allocationType = AllocationType::internalHeap;
    }

    GraphicsAllocation *heapMemory = nullptr;
    if (device) {
        auto &productHelper = device->getProductHelper();
        if (allocationType == AllocationType::linearStream && LinearStreamPoolAllocator::isEnabled(productHelper)) {
            auto ctx = createDeferredFreeContext();
            heapMemory = device->getLinearStreamPoolAllocator().allocate(allocationSize, &ctx);
        } else if (allocationType == AllocationType::internalHeap && InternalHeapPoolAllocator::isEnabled(productHelper)) {
            auto ctx = createDeferredFreeContext();
            heapMemory = device->getInternalHeapPoolAllocator().allocate(allocationSize, &ctx);
        }
    }
    if (!heapMemory) {
        heapMemory = internalAllocationStorage->obtainReusableAllocation(allocationSize, allocationType).release();
    }
    if (!heapMemory) {
        heapMemory = getMemoryManager()->allocateGraphicsMemoryWithProperties({rootDeviceIndex, true, allocationSize, allocationType,
                                                                               isMultiOsContextCapable(), false, osContext->getDeviceBitfield()});
    }

    size_t finalHeapSize;
    if (IndirectHeap::Type::surfaceState == heapType) {
        finalHeapSize = defaultSshSize - MemoryConstants::pageSize;
    } else {
        finalHeapSize = heapMemory->getUnderlyingBufferSize();
    }

    if (indirectHeap) {
        indirectHeap->replaceBuffer(heapMemory->getUnderlyingBuffer(), finalHeapSize);
        indirectHeap->replaceGraphicsAllocation(heapMemory);
    } else {
        indirectHeap = new IndirectHeap(heapMemory, requireInternalHeap);
        indirectHeap->overrideMaxSize(finalHeapSize);
    }
    scratchSpaceController->reserveHeap(heapType, indirectHeap);
}

void CommandStreamReceiver::releaseIndirectHeap(IndirectHeap::Type heapType) {
    DEBUG_BREAK_IF(static_cast<uint32_t>(heapType) >= arrayCount(indirectHeap));
    auto &heap = indirectHeap[heapType];

    if (heap) {
        auto heapMemory = heap->getGraphicsAllocation();
        if (heapMemory != nullptr) {
            releaseHeapAllocation(heapMemory);
            heap->replaceBuffer(nullptr, 0);
            heap->replaceGraphicsAllocation(nullptr);
        }
    }
}

void CommandStreamReceiver::releaseHeapAllocation(GraphicsAllocation *heapMemory) {
    if (!heapMemory->isView()) {
        internalAllocationStorage->storeAllocation(std::unique_ptr<GraphicsAllocation>(heapMemory), REUSABLE_ALLOCATION);
        return;
    }

    if (device) {
        auto *parent = heapMemory->getParentAllocation();
        if (parent && device->getLinearStreamPoolAllocator().isPoolBuffer(parent)) {
            device->getLinearStreamPoolAllocator().free(heapMemory, peekTaskCount(), osContext->getContextId());
            return;
        }
        if (parent && device->getInternalHeapPoolAllocator().isPoolBuffer(parent)) {
            device->getInternalHeapPoolAllocator().free(heapMemory, peekTaskCount(), osContext->getContextId());
            return;
        }
    }

    delete heapMemory;
}

void CommandStreamReceiver::releaseCommandBufferAllocation(GraphicsAllocation *commandBufferMemory) {
    if (commandBufferMemory->isView()) {
        if (device) {
            auto &poolAllocator = device->getCommandBufferPoolAllocator();
            auto parent = commandBufferMemory->getParentAllocation();
            if (parent && poolAllocator.isPoolBuffer(parent)) {
                poolAllocator.free(commandBufferMemory, peekTaskCount(), osContext->getContextId());
                return;
            }
        }
    }

    internalAllocationStorage->storeAllocation(std::unique_ptr<GraphicsAllocation>(commandBufferMemory), REUSABLE_ALLOCATION);
}

void *CommandStreamReceiver::asyncDebugBreakConfirmation(void *arg) {
    auto self = reinterpret_cast<CommandStreamReceiver *>(arg);

    do {
        auto debugPauseStateValue = DebugPauseState::waitingForUserStartConfirmation;
        if (debugManager.flags.PauseOnGpuMode.get() != PauseOnGpuProperties::PauseMode::AfterWorkload) {
            do {
                {
                    std::unique_lock<SpinLock> lock{self->debugPauseStateLock};
                    debugPauseStateValue = *self->debugPauseStateAddress;
                }

                if (debugPauseStateValue == DebugPauseState::terminate) {
                    return nullptr;
                }
                std::this_thread::yield();
            } while (debugPauseStateValue != DebugPauseState::waitingForUserStartConfirmation);
            std::cout << "Debug break: Press enter to start workload" << std::endl;
            self->debugConfirmationFunction();
            debugPauseStateValue = DebugPauseState::hasUserStartConfirmation;
            {
                std::unique_lock<SpinLock> lock{self->debugPauseStateLock};
                *self->debugPauseStateAddress = debugPauseStateValue;
            }
        }

        if (debugManager.flags.PauseOnGpuMode.get() != PauseOnGpuProperties::PauseMode::BeforeWorkload) {
            do {
                {
                    std::unique_lock<SpinLock> lock{self->debugPauseStateLock};
                    debugPauseStateValue = *self->debugPauseStateAddress;
                }
                if (debugPauseStateValue == DebugPauseState::terminate) {
                    return nullptr;
                }
                std::this_thread::yield();
            } while (debugPauseStateValue != DebugPauseState::waitingForUserEndConfirmation);

            std::cout << "Debug break: Workload ended, press enter to continue" << std::endl;
            self->debugConfirmationFunction();

            {
                std::unique_lock<SpinLock> lock{self->debugPauseStateLock};
                *self->debugPauseStateAddress = DebugPauseState::hasUserEndConfirmation;
            }
        }
    } while (debugManager.flags.PauseOnEnqueue.get() == PauseOnGpuProperties::DebugFlagValues::OnEachEnqueue || debugManager.flags.PauseOnBlitCopy.get() == PauseOnGpuProperties::DebugFlagValues::OnEachEnqueue);
    return nullptr;
}

bool CommandStreamReceiver::initializeTagAllocation() {
    this->tagsMultiAllocation = &this->createMultiAllocationInSystemMemoryPool(AllocationType::tagBuffer);

    auto tagAllocation = tagsMultiAllocation->getGraphicsAllocation(rootDeviceIndex);
    if (!tagAllocation) {
        return false;
    }

    this->setTagAllocation(tagAllocation);

    auto initValue = debugManager.flags.EnableNullHardware.get() ? static_cast<uint32_t>(-1) : initialHardwareTag;
    auto tagAddress = this->tagAddress;
    auto ucTagAddress = this->ucTagAddress;
    auto completionFence = reinterpret_cast<TaskCountType *>(getCompletionAddress());
    UNRECOVERABLE_IF(!completionFence);
    uint32_t subDevices = static_cast<uint32_t>(this->deviceBitfield.count());
    for (uint32_t i = 0; i < subDevices; i++) {
        *tagAddress = initValue;
        tagAddress = ptrOffset(tagAddress, this->immWritePostSyncWriteOffset);
        *ucTagAddress = initValue;
        ucTagAddress = ptrOffset(ucTagAddress, this->immWritePostSyncWriteOffset);
        *completionFence = 0;
        completionFence = ptrOffset(completionFence, this->immWritePostSyncWriteOffset);
    }
    *this->debugPauseStateAddress = debugManager.flags.EnableNullHardware.get() ? DebugPauseState::disabled : DebugPauseState::waitingForFirstSemaphore;

    PRINT_STRING(debugManager.flags.PrintTagAllocationAddress.get(), stdout,
                 "\nCreated tag allocation %p for engine %u\n",
                 this->tagAddress, static_cast<uint32_t>(osContext->getEngineType()));

    if (debugManager.flags.PauseOnEnqueue.get() != -1 || debugManager.flags.PauseOnBlitCopy.get() != -1) {
        userPauseConfirmation = Thread::createFunc(CommandStreamReceiver::asyncDebugBreakConfirmation, reinterpret_cast<void *>(this));
    }

    this->barrierCountTagAddress = ptrOffset(this->tagAddress, TagAllocationLayout::barrierCountOffset);

    return true;
}

bool CommandStreamReceiver::createWorkPartitionAllocation(const Device &device) {
    if (!staticWorkPartitioningEnabled) {
        return false;
    }
    UNRECOVERABLE_IF(device.getNumGenericSubDevices() < 2);

    AllocationProperties properties{this->rootDeviceIndex, true, 4096u, AllocationType::workPartitionSurface, true, false, deviceBitfield};
    this->workPartitionAllocation = getMemoryManager()->allocateGraphicsMemoryWithProperties(properties);
    if (this->workPartitionAllocation == nullptr) {
        return false;
    }

    uint32_t logicalId = 0;
    auto copySrc = std::make_unique<std::array<uint32_t, 2>>();
    for (uint32_t deviceIndex = 0; deviceIndex < deviceBitfield.size(); deviceIndex++) {
        if (!deviceBitfield.test(deviceIndex)) {
            continue;
        }

        *copySrc = {logicalId++, deviceIndex};
        DeviceBitfield copyBitfield{};
        copyBitfield.set(deviceIndex);
        auto copySuccess = MemoryTransferHelper::transferMemoryToAllocationBanks(true, device, workPartitionAllocation, 0, copySrc.get(), 2 * sizeof(uint32_t), copyBitfield);

        if (!copySuccess) {
            return false;
        }
    }

    return true;
}

bool CommandStreamReceiver::createGlobalFenceAllocation() {
    const auto &gfxCoreHelper = this->getGfxCoreHelper();
    const auto &hwInfo = this->peekHwInfo();
    const auto &productHelper = this->getProductHelper();
    if (!gfxCoreHelper.isFenceAllocationRequired(hwInfo, productHelper)) {
        return true;
    }

    DEBUG_BREAK_IF(this->globalFenceAllocation != nullptr);
    this->globalFenceAllocation = getMemoryManager()->allocateGraphicsMemoryWithProperties({rootDeviceIndex, MemoryConstants::pageSize, AllocationType::globalFence, osContext->getDeviceBitfield()});
    return this->globalFenceAllocation != nullptr;
}

bool CommandStreamReceiver::createPreemptionAllocation() {
    auto &rootDeviceEnvironment = executionEnvironment.rootDeviceEnvironments[rootDeviceIndex];
    if (EngineHelpers::isBcs(osContext->getEngineType()) || rootDeviceEnvironment->debugger.get()) {
        return true;
    }
    auto hwInfo = rootDeviceEnvironment->getHardwareInfo();
    auto &gfxCoreHelper = getGfxCoreHelper();
    size_t preemptionSurfaceSize = hwInfo->capabilityTable.requiredPreemptionSurfaceSize;
    if (debugManager.flags.OverrideCsrAllocationSize.get() > 0) {
        preemptionSurfaceSize = debugManager.flags.OverrideCsrAllocationSize.get();
    }
    UNRECOVERABLE_IF(preemptionSurfaceSize == 0);
    AllocationProperties properties{rootDeviceIndex, true, preemptionSurfaceSize, AllocationType::preemption, isMultiOsContextCapable(), false, deviceBitfield};
    properties.flags.uncacheable = hwInfo->workaroundTable.flags.waCSRUncachable;
    properties.alignment = gfxCoreHelper.getPreemptionAllocationAlignment();
    this->preemptionAllocation = getMemoryManager()->allocateGraphicsMemoryWithProperties(properties);
    return this->preemptionAllocation != nullptr;
}

std::unique_lock<CommandStreamReceiver::MutexType> CommandStreamReceiver::obtainUniqueOwnership() {
    return std::unique_lock<CommandStreamReceiver::MutexType>(this->ownershipMutex);
}

std::unique_lock<CommandStreamReceiver::MutexType> CommandStreamReceiver::tryObtainUniqueOwnership() {
    return std::unique_lock<CommandStreamReceiver::MutexType>(this->ownershipMutex, std::try_to_lock);
}
std::unique_lock<CommandStreamReceiver::MutexType> CommandStreamReceiver::obtainHostPtrSurfaceCreationLock() {
    return std::unique_lock<CommandStreamReceiver::MutexType>(this->hostPtrSurfaceCreationMutex);
}

std::unique_lock<CommandStreamReceiver::MutexType> CommandStreamReceiver::obtainHostFunctionWorkerStartLock() {
    return std::unique_lock<CommandStreamReceiver::MutexType>(this->hostFunctionWorkerStartMutex);
}

AllocationsList &CommandStreamReceiver::getTemporaryAllocations() { return internalAllocationStorage->getTemporaryAllocations(); }
AllocationsList &CommandStreamReceiver::getAllocationsForReuse() { return internalAllocationStorage->getAllocationsForReuse(); }
AllocationsList &CommandStreamReceiver::getDeferredAllocations() { return internalAllocationStorage->getDeferredAllocations(); }

bool CommandStreamReceiver::createAllocationForHostSurface(HostPtrSurface &surface, bool requiresL3Flush) {
    auto csrLock = obtainUniqueOwnership();
    std::unique_lock<decltype(hostPtrSurfaceCreationMutex)> lock = this->obtainHostPtrSurfaceCreationLock();
    auto allocation = internalAllocationStorage->obtainTemporaryAllocationWithPtr(surface.getSurfaceSize(), surface.getMemoryPointer(), AllocationType::externalHostPtr);

    if (allocation == nullptr) {
        auto memoryManager = getMemoryManager();
        AllocationProperties properties{rootDeviceIndex,
                                        false, // allocateMemory
                                        surface.getSurfaceSize(), AllocationType::externalHostPtr,
                                        false, // isMultiStorageAllocation
                                        osContext->getDeviceBitfield()};
        properties.flags.flushL3RequiredForRead = properties.flags.flushL3RequiredForWrite = requiresL3Flush;
        allocation.reset(memoryManager->allocateGraphicsMemoryWithProperties(properties, surface.getMemoryPointer()));
        if (allocation == nullptr && surface.peekIsPtrCopyAllowed()) {
            // Try with no host pointer allocation and copy
            allocation.reset(memoryManager->allocateInternalGraphicsMemoryWithHostCopy(rootDeviceIndex,
                                                                                       internalAllocationStorage->getDeviceBitfield(),
                                                                                       surface.getMemoryPointer(),
                                                                                       surface.getSurfaceSize()));
        }
    }

    if (allocation == nullptr) {
        return false;
    }
    allocation->incrementHostPtrTaskCountAssignment();
    allocation->updateTaskCount(0u, osContext->getContextId());
    surface.setAllocation(allocation.get());
    internalAllocationStorage->storeAllocation(std::move(allocation), TEMPORARY_ALLOCATION);
    return true;
}

TagAllocatorBase *CommandStreamReceiver::getEventTsAllocator() {
    if (profilingTimeStampAllocator.get() == nullptr) {
        RootDeviceIndicesContainer rootDeviceIndices = {rootDeviceIndex};
        profilingTimeStampAllocator = std::make_unique<TagAllocator<HwTimeStamps>>(rootDeviceIndices, getMemoryManager(), getPreferredTagPoolSize(), MemoryConstants::cacheLineSize,
                                                                                   sizeof(HwTimeStamps), 0, false, true, osContext->getDeviceBitfield());
        setupTagAllocatorForSimulation(*profilingTimeStampAllocator);
    }
    return profilingTimeStampAllocator.get();
}

TagAllocatorBase *CommandStreamReceiver::getEventPerfCountAllocator(const uint32_t tagSize) {
    if (perfCounterAllocator.get() == nullptr) {
        RootDeviceIndicesContainer rootDeviceIndices = {rootDeviceIndex};
        perfCounterAllocator = std::make_unique<TagAllocator<HwPerfCounter>>(
            rootDeviceIndices, getMemoryManager(), getPreferredTagPoolSize(), MemoryConstants::cacheLineSize, tagSize, 0, false, true, osContext->getDeviceBitfield());
        setupTagAllocatorForSimulation(*perfCounterAllocator);
    }
    return perfCounterAllocator.get();
}

void CommandStreamReceiver::setupTagAllocatorForSimulation(TagAllocatorBase &tagAllocator) {
    if (!isTbxMode() && !isAubMode()) {
        tagAllocator.setTagNodeUpdateCallback({});
        tagAllocator.setTagPoolCreatedCallback({});
        return;
    }

    tagAllocator.setTagNodeUpdateCallback([this](TagNodeBase &tagNode, uint64_t tagOffset, size_t chunkSize) {
        this->writeTagAllocationChunkToSimulation(tagNode, tagOffset, chunkSize);
    });

    auto uploadTagPoolToSimulation = [this](MultiGraphicsAllocation &multiAllocation) {
        for (auto *gfxAllocation : multiAllocation.getGraphicsAllocations()) {
            if (gfxAllocation == nullptr || gfxAllocation->isSimulationInitialUploadDone()) {
                continue;
            }
            const size_t fullAllocationSize = gfxAllocation->getUnderlyingBufferSize();
            if (fullAllocationSize == 0) {
                gfxAllocation->setSimulationInitialUploadDone(true);
                continue;
            }
            writeAllocationChunkToSimulation(*gfxAllocation, 0u, fullAllocationSize);
            gfxAllocation->setSimulationInitialUploadDone(true);
        }
    };

    tagAllocator.setTagPoolCreatedCallback(uploadTagPoolToSimulation);

    for (const auto &multiAllocation : tagAllocator.getGfxAllocations()) {
        if (multiAllocation != nullptr) {
            uploadTagPoolToSimulation(*multiAllocation);
        }
    }
}

void CommandStreamReceiver::writeAllocationChunkToSimulation(GraphicsAllocation &gfxAllocation, uint64_t chunkOffset, size_t chunkSize) {
    if ((!isTbxMode() && !isAubMode()) || chunkSize == 0) {
        return;
    }

    [[maybe_unused]] auto ownership = obtainUniqueOwnership();
    setWritableForSimulation(true, gfxAllocation);

    const bool useChunkCopy = isChunkCopySupportedForSimulation();
    const uint64_t selectedChunkOffset = useChunkCopy ? chunkOffset : 0;
    const size_t selectedChunkSize = useChunkCopy ? chunkSize : 0;
    [[maybe_unused]] const bool writeStatus = writeMemory(gfxAllocation,
                                                          useChunkCopy,
                                                          selectedChunkOffset,
                                                          selectedChunkSize);

    if (AubHelper::isOneTimeAubWritableAllocationType(gfxAllocation.getAllocationType())) {
        setWritableForSimulation(false, gfxAllocation);
    }
    DEBUG_BREAK_IF(!writeStatus);
}

void CommandStreamReceiver::writeTagAllocationChunkToSimulation(TagNodeBase &tagNode, uint64_t tagOffset, size_t chunkSize) {
    if (chunkSize == 0) {
        return;
    }

    auto *multiAllocation = tagNode.getBaseGraphicsAllocation();
    if (multiAllocation == nullptr) {
        return;
    }

    auto *gfxAllocation = getGraphicsAllocationForTagNode(*multiAllocation, tagNode.getGpuAddress(), rootDeviceIndex);
    if (gfxAllocation == nullptr) {
        return;
    }

    uint64_t chunkOffset = 0;
    if (!tryGetTagNodeChunkOffsetInAllocation(tagNode, *gfxAllocation, tagOffset, chunkSize, chunkOffset)) {
        DEBUG_BREAK_IF(true);
        return;
    }

    writeAllocationChunkToSimulation(*gfxAllocation, chunkOffset, chunkSize);
}

uint32_t CommandStreamReceiver::getPreferredTagPoolSize() const {
    if (debugManager.flags.DisableTimestampPacketOptimizations.get()) {
        return 1;
    }

    return 2048;
}

bool CommandStreamReceiver::expectMemory(const void *gfxAddress, const void *srcAddress,
                                         size_t length, uint32_t compareOperation) {
    auto isMemoryEqual = (memcmp(gfxAddress, srcAddress, length) == 0);
    auto isEqualMemoryExpected = (compareOperation == aub_stream::CompareOperationValues::CompareEqual);

    return (isMemoryEqual == isEqualMemoryExpected);
}

bool CommandStreamReceiver::needsPageTableManager() const {
    auto hwInfo = executionEnvironment.rootDeviceEnvironments[rootDeviceIndex]->getHardwareInfo();
    auto &productHelper = getProductHelper();

    if (pageTableManager.get() != nullptr) {
        return false;
    }
    return productHelper.isPageTableManagerSupported(*hwInfo);
}

void CommandStreamReceiver::printDeviceIndex() {
    PRINT_STRING(debugManager.flags.PrintDeviceAndEngineIdOnSubmission.get(), stdout,
                 "%u: Submission to RootDevice Index: %u, Sub-Devices Mask: %lu, EngineId: %u (%s, %s), Priority: %s\n",
                 SysCalls::getProcessId(),
                 this->getRootDeviceIndex(),
                 this->osContext->getDeviceBitfield().to_ulong(),
                 this->osContext->getEngineType(),
                 EngineHelpers::engineTypeToString(this->osContext->getEngineType()).c_str(),
                 EngineHelpers::engineUsageToString(this->osContext->getEngineUsage()).c_str(),
                 this->osContext->hasPriorityLevel() ? std::to_string(this->osContext->getPriorityLevel()).c_str() : "std::nullopt");
}

void CommandStreamReceiver::checkForNewResources(TaskCountType submittedTaskCount, TaskCountType allocationTaskCount, GraphicsAllocation &gfxAllocation) {
    if (useNewResourceImplicitFlush) {
        if (allocationTaskCount == GraphicsAllocation::objectNotUsed && !GraphicsAllocation::isIsaAllocationType(gfxAllocation.getAllocationType())) {
            newResources = true;
            PRINT_STRING(debugManager.flags.ProvideVerboseImplicitFlush.get(), stdout,
                         "New resource detected of type %llu\n", static_cast<unsigned long long>(gfxAllocation.getAllocationType()));
        }
    }
}

bool CommandStreamReceiver::checkImplicitFlushForGpuIdle() {
    if (useGpuIdleImplicitFlush) {
        if (this->taskCount == *getTagAddress()) {
            return true;
        }
    }
    return false;
}

bool CommandStreamReceiver::getAndClearIsWalkerWithProfilingEnqueued() {
    bool retVal = this->isWalkerWithProfilingEnqueued;
    this->isWalkerWithProfilingEnqueued = false;
    return retVal;
}

void CommandStreamReceiver::downloadTagAllocation(TaskCountType taskCountToWait) {
    if (this->getTagAllocation()) {
        if (taskCountToWait && taskCountToWait <= this->peekLatestFlushedTaskCount()) {
            this->downloadAllocation(*this->getTagAllocation());
        }
    }
}

bool CommandStreamReceiver::testTaskCountReady(volatile TagAddressType *pollAddress, TaskCountType taskCountToWait) {
    this->downloadTagAllocation(taskCountToWait);
    for (uint32_t i = 0; i < activePartitions; i++) {
        if (!WaitUtils::waitFunction(pollAddress, taskCountToWait, 0)) {
            return false;
        }

        pollAddress = ptrOffset(pollAddress, this->immWritePostSyncWriteOffset);
    }

    downloadAllocations(true);

    return true;
}

const HardwareInfo &CommandStreamReceiver::peekHwInfo() const {
    return *peekRootDeviceEnvironment().getHardwareInfo();
}

const RootDeviceEnvironment &CommandStreamReceiver::peekRootDeviceEnvironment() const {
    return *executionEnvironment.rootDeviceEnvironments[rootDeviceIndex];
}

const GfxCoreHelper &CommandStreamReceiver::getGfxCoreHelper() const {
    return peekRootDeviceEnvironment().getHelper<GfxCoreHelper>();
}

const ProductHelper &CommandStreamReceiver::getProductHelper() const {
    return peekRootDeviceEnvironment().getHelper<ProductHelper>();
}

const ReleaseHelper &CommandStreamReceiver::getReleaseHelper() const {
    return peekRootDeviceEnvironment().getReleaseHelper();
}

TaskCountType CommandStreamReceiver::getCompletionValue(const GraphicsAllocation &gfxAllocation) {
    if (completionFenceValuePointer) {
        return *completionFenceValuePointer;
    }
    auto osContextId = osContext->getContextId();
    return gfxAllocation.getTaskCount(osContextId);
}

void CommandStreamReceiver::printTagAddressContent(TaskCountType taskCountToWait, int64_t waitTimeout, bool start) {
    if (getType() == NEO::CommandStreamReceiverType::aub) {
        if (start) {
            PRINT_STRING(true, stdout, "\nAub dump wait for task count %llu", taskCountToWait);
        } else {
            PRINT_STRING(true, stdout, "\nAub dump wait completed.");
        }
        return;
    }
    auto postSyncAddress = getTagAddress();
    if (start) {
        PRINT_STRING(true, stdout,
                     "\nWaiting for task count %llu at location %p with timeout %llx. Current value:",
                     taskCountToWait, postSyncAddress, waitTimeout);
    } else {
        PRINT_STRING(true, stdout,
                     "%s", "\nWaiting completed. Current value:");
    }

    for (uint32_t i = 0; i < activePartitions; i++) {
        PRINT_STRING(true, stdout, " %u", *postSyncAddress);
        postSyncAddress = ptrOffset(postSyncAddress, this->immWritePostSyncWriteOffset);
    }
    PRINT_STRING(true, stdout, "%s", "\n");
}

bool CommandStreamReceiver::isTbxMode() const {
    return (getType() == NEO::CommandStreamReceiverType::tbx || getType() == NEO::CommandStreamReceiverType::tbxWithAub);
}

bool CommandStreamReceiver::isAubMode() const {
    return (getType() == NEO::CommandStreamReceiverType::aub || getType() == NEO::CommandStreamReceiverType::tbxWithAub || getType() == NEO::CommandStreamReceiverType::hardwareWithAub || getType() == NEO::CommandStreamReceiverType::nullAub);
}

bool CommandStreamReceiver::isHardwareMode() const {
    return (getType() == NEO::CommandStreamReceiverType::hardware);
}

TaskCountType CompletionStamp::getTaskCountFromSubmissionStatusError(SubmissionStatus status) {
    switch (status) {
    case SubmissionStatus::outOfHostMemory:
        return CompletionStamp::outOfHostMemory;
    case SubmissionStatus::outOfMemory:
        return CompletionStamp::outOfDeviceMemory;
    case SubmissionStatus::failed:
        return CompletionStamp::failed;
    case SubmissionStatus::unsupported:
        return CompletionStamp::unsupported;
    default:
        return 0;
    }
}
uint64_t CommandStreamReceiver::getBarrierCountGpuAddress() const { return ptrOffset(this->tagAllocation->getGpuAddress(), TagAllocationLayout::barrierCountOffset); }
uint64_t CommandStreamReceiver::getDebugPauseStateGPUAddress() const { return tagAllocation->getGpuAddress() + TagAllocationLayout::debugPauseStateAddressOffset; }
uint64_t CommandStreamReceiver::getUcTagGPUAddress() const { return tagAllocation->getGpuAddress() + TagAllocationLayout::ucTagAddressOffset; }
uint64_t CommandStreamReceiver::getCompletionAddress() const {
    uint64_t completionFenceAddress = castToUint64(const_cast<TagAddressType *>(tagAddress));
    if (completionFenceAddress == 0) {
        return 0;
    }
    completionFenceAddress += TagAllocationLayout::completionFenceOffset;
    return completionFenceAddress;
}

void CommandStreamReceiver::createGlobalStatelessHeap() {
    if (this->globalStatelessHeapAllocation == nullptr) {
        auto lock = obtainUniqueOwnership();
        if (this->globalStatelessHeapAllocation == nullptr) {
            constexpr size_t heapSize = 16 * MemoryConstants::kiloByte;
            constexpr AllocationType allocationType = AllocationType::linearStream;

            if (device) {
                auto &productHelper = device->getProductHelper();
                if (LinearStreamPoolAllocator::isEnabled(productHelper)) {
                    this->globalStatelessHeapAllocation = device->getLinearStreamPoolAllocator().allocate(heapSize);
                }
            }

            if (this->globalStatelessHeapAllocation == nullptr) {
                AllocationProperties properties{rootDeviceIndex, true, heapSize, allocationType,
                                                isMultiOsContextCapable(), false, osContext->getDeviceBitfield()};

                this->globalStatelessHeapAllocation = getMemoryManager()->allocateGraphicsMemoryWithProperties(properties);
            }

            if (this->globalStatelessHeapAllocation != nullptr) {
                this->globalStatelessHeap = std::make_unique<IndirectHeap>(this->globalStatelessHeapAllocation);
            }
        }
    }
}

void CommandStreamReceiver::releaseGlobalStatelessHeap() {
    auto *allocation = globalStatelessHeapAllocation;
    globalStatelessHeapAllocation = nullptr;
    globalStatelessHeap.reset();

    if (allocation == nullptr) {
        return;
    }

    if (allocation->isView()) {
        if (device) {
            auto &poolAllocator = device->getLinearStreamPoolAllocator();
            auto *parent = allocation->getParentAllocation();
            if (parent && poolAllocator.isPoolBuffer(parent)) {
                poolAllocator.free(allocation);
                return;
            }
        }

        delete allocation;
        return;
    }

    getMemoryManager()->freeGraphicsMemory(allocation);
}

bool CommandStreamReceiver::isRayTracingStateProgramingNeeded(Device &device) const {
    return device.rayTracingIsInitialized() && getBtdCommandDirty();
}

void CommandStreamReceiver::registerClient(void *client) {
    std::unique_lock<MutexType> lock(registeredClientsMutex);
    auto element = std::find(registeredClients.begin(), registeredClients.end(), client);
    if (element == registeredClients.end()) {
        registeredClients.push_back(client);
        numClients++;
    }
}

void CommandStreamReceiver::unregisterClient(void *client) {
    std::unique_lock<MutexType> lock(registeredClientsMutex);

    auto element = std::find(registeredClients.begin(), registeredClients.end(), client);
    if (element != registeredClients.end()) {
        registeredClients.erase(element);
        numClients--;
    }
}

void CommandStreamReceiver::ensurePrimaryCsrInitialized(Device &device) {
    auto csrToInitialize = primaryCsr ? primaryCsr : this;
    csrToInitialize->initializeDeviceWithFirstSubmission(device);
}

void CommandStreamReceiver::addToEvictionContainer(GraphicsAllocation &gfxAllocation) {}

DeferredFreeContext CommandStreamReceiver::createDeferredFreeContext() const {
    DeferredFreeContext ctx{};
    ctx.tagAddress = getTagAddress();
    ctx.ucTagAddress = getUcTagAddress();
    ctx.contextId = getOsContext().getContextId();
    ctx.partitionCount = getActivePartitions();
    ctx.tagOffset = getImmWritePostSyncWriteOffset();
    return ctx;
}

} // namespace NEO
