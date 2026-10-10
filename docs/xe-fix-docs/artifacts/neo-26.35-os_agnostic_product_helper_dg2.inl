/*
 * Copyright (C) 2021-2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "shared/source/command_stream/command_stream_receiver.h"
#include "shared/source/helpers/api_specific_config.h"
#include "shared/source/helpers/gfx_core_helper.h"
#include "shared/source/helpers/local_memory_access_modes.h"
#include "shared/source/memory_manager/memory_manager.h"
#include "shared/source/unified_memory/usm_memory_support.h"

#include "aubstream/product_family.h"

namespace NEO {
template <>
void ProductHelperHw<gfxProduct>::adjustSamplerState(void *sampler, const HardwareInfo &hwInfo) const {
    using SAMPLER_STATE = typename XeHpgCoreFamily::SAMPLER_STATE;
    auto samplerState = reinterpret_cast<SAMPLER_STATE *>(sampler);
    if (debugManager.flags.ForceSamplerLowFilteringPrecision.get()) {
        samplerState->setLowQualityFilter(SAMPLER_STATE::LOW_QUALITY_FILTER_ENABLE);
    }

    auto isMirrorAddressMode = SAMPLER_STATE::TEXTURE_COORDINATE_MODE_MIRROR == samplerState->getTcxAddressControlMode();
    auto isNearestFilter = SAMPLER_STATE::MIN_MODE_FILTER_NEAREST == samplerState->getMinModeFilter();
    auto isWorkaroundRequired = (DG2::isG10(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_C, hwInfo, *this)) ||
                                (DG2::isG11(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_B, hwInfo, *this));
    if (isNearestFilter && isMirrorAddressMode && isWorkaroundRequired) {
        samplerState->setRAddressMinFilterRoundingEnable(true);
        samplerState->setRAddressMagFilterRoundingEnable(true);
    }
}

template <>
uint32_t ProductHelperHw<gfxProduct>::getHwRevIdFromStepping(uint32_t stepping, const HardwareInfo &hwInfo) const {
    switch (stepping) {
    case REVISION_A0:
        return 0x0;
    case REVISION_A1:
        return 0x1;
    case REVISION_B:
        return 0x4;
    case REVISION_C:
        return 0x8;
    default:
        return CommonConstants::invalidStepping;
    }
}

template <>
uint32_t ProductHelperHw<gfxProduct>::getSteppingFromHwRevId(const HardwareInfo &hwInfo) const {
    switch (hwInfo.platform.usRevId) {
    case 0x0:
        return REVISION_A0;
    case 0x1:
        return REVISION_A1;
    case 0x4:
        return REVISION_B;
    case 0x8:
        return REVISION_C;
    default:
        return CommonConstants::invalidStepping;
    }
}

template <>
bool ProductHelperHw<gfxProduct>::isAcquireGlobalFenceInDirectSubmissionRequired(const HardwareInfo &hwInfo) const {
    return true;
}

template <>
bool ProductHelperHw<gfxProduct>::isAdditionalStateBaseAddressWARequired(const HardwareInfo &hwInfo) const {
    if (DG2::isG10(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_B, REVISION_C, hwInfo, *this)) {
        return true;
    }
    if (DG2::isG11(hwInfo)) {
        return true;
    }
    return false;
}

template <>
bool ProductHelperHw<gfxProduct>::isMaxThreadsForWorkgroupWARequired(const HardwareInfo &hwInfo) const {
    return DG2::isG10(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_A1, hwInfo, *this);
}

template <>
void ProductHelperHw<gfxProduct>::setForceNonCoherent(void *const statePtr, const StateComputeModeProperties &properties) const {
    using STATE_COMPUTE_MODE = typename XeHpgCoreFamily::STATE_COMPUTE_MODE;
    using FORCE_NON_COHERENT = typename STATE_COMPUTE_MODE::FORCE_NON_COHERENT;

    STATE_COMPUTE_MODE &stateComputeMode = *static_cast<STATE_COMPUTE_MODE *>(statePtr);
    FORCE_NON_COHERENT coherencyValue = (properties.isCoherencyRequired.value == 1) ? FORCE_NON_COHERENT::FORCE_NON_COHERENT_FORCE_DISABLED
                                                                                    : FORCE_NON_COHERENT::FORCE_NON_COHERENT_FORCE_GPU_NON_COHERENT;
    stateComputeMode.setForceNonCoherent(coherencyValue);

    auto mask = stateComputeMode.getMaskBits();
    mask |= XeHpgCoreFamily::stateComputeModeForceNonCoherentMask;
    stateComputeMode.setMaskBits(mask);
}

template <>
bool ProductHelperHw<gfxProduct>::isDefaultEngineTypeAdjustmentRequired(const HardwareInfo &hwInfo) const {
    return DG2::isG10(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_B, hwInfo, *this);
}

template <>
bool ProductHelperHw<gfxProduct>::isDisableOverdispatchAvailable(const HardwareInfo &hwInfo) const {
    if (DG2::isG10(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_B, hwInfo, *this)) {
        return false;
    }
    return true;
}

template <>
LocalMemoryAccessMode ProductHelperHw<gfxProduct>::getDefaultLocalMemoryAccessMode(const HardwareInfo &hwInfo) const {
    if (DG2::isG10(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_B, hwInfo, *this)) {
        return LocalMemoryAccessMode::cpuAccessDisallowed;
    }
    return LocalMemoryAccessMode::defaultMode;
}

template <>
bool ProductHelperHw<gfxProduct>::isBlitterForImagesSupported() const {
    return true;
}

template <>
bool ProductHelperHw<gfxProduct>::isTile64With3DSurfaceOnBCSSupported(const HardwareInfo &hwInfo) const {
    if (DG2::isG10(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_C, hwInfo, *this)) {
        return false;
    }
    if (DG2::isG11(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_B, hwInfo, *this)) {
        return false;
    }
    return true;
}

template <>
uint32_t ProductHelperHw<gfxProduct>::computeMaxNeededSubSliceSpace(const HardwareInfo &hwInfo) const {
    const uint32_t highestEnabledSlice = NEO::GfxCoreHelper::getHighestEnabledSlice(hwInfo);

    UNRECOVERABLE_IF(highestEnabledSlice == 0);
    UNRECOVERABLE_IF(hwInfo.gtSystemInfo.MaxSlicesSupported == 0);
    auto subSlicesPerSlice = hwInfo.gtSystemInfo.MaxSubSlicesSupported / hwInfo.gtSystemInfo.MaxSlicesSupported;
    auto maxSubSlice = std::max(highestEnabledSlice * subSlicesPerSlice, hwInfo.gtSystemInfo.MaxSubSlicesSupported);

    return maxSubSlice;
}

template <>
bool ProductHelperHw<gfxProduct>::isCpuCopyNecessary(const void *ptr, MemoryManager *memoryManager) const {
    if (memoryManager) {
        if constexpr (is32bit) {
            return memoryManager->isWCMemory(ptr);
        } else {
            return false;
        }
    } else {
        return false;
    }
}
template <>
bool ProductHelperHw<gfxProduct>::isStorageInfoAdjustmentRequired() const {
    if constexpr (is32bit) {
        return true;
    } else {
        return false;
    }
}

template <>
bool ProductHelperHw<gfxProduct>::isResolveDependenciesByPipeControlsSupported() const {
    if (debugManager.flags.ResolveDependenciesViaPipeControls.get() != -1) {
        return debugManager.flags.ResolveDependenciesViaPipeControls.get() == 1;
    }
    return true;
}

template <>
bool ProductHelperHw<gfxProduct>::useLocalPreferredForCacheableBuffers() const {
    return true;
}

template <>
std::optional<aub_stream::ProductFamily> ProductHelperHw<gfxProduct>::getAubStreamProductFamily() const {
    return aub_stream::ProductFamily::Dg2;
};
template <>
bool ProductHelperHw<gfxProduct>::isFusedEuDisabledForDpas(bool kernelHasDpasInstructions, const uint32_t *lws, const uint32_t *groupCount, const HardwareInfo &hwInfo) const {
    auto isAcm = DG2::isG10(hwInfo) || DG2::isG11(hwInfo) || DG2::isG12(hwInfo);
    if (!kernelHasDpasInstructions || !isAcm) {
        return false;
    } else if (lws == nullptr) {
        return true;
    } else if (size_t lwsCount = lws[0] * lws[1] * lws[2]; lwsCount > 1 && (lwsCount & 1) != 0) {
        return true;
    } else if (lwsCount > 1) {
        return false;
    } else if (groupCount == nullptr || (groupCount[0] & 1) != 0) {
        return true;
    } else {
        return false;
    }
}

template <>
bool ProductHelperHw<gfxProduct>::isCalculationForDisablingEuFusionWithDpasNeeded(const HardwareInfo &hwInfo) const {
    return DG2::isG10(hwInfo) || DG2::isG11(hwInfo) || DG2::isG12(hwInfo);
}

template <>
bool ProductHelperHw<gfxProduct>::disableL3CacheForDebug(const HardwareInfo &hwInfo) const {
    return DG2::isG10(hwInfo) && GfxCoreHelper::isWorkaroundRequired(REVISION_A0, REVISION_B, hwInfo, *this);
}

template <>
void ProductHelperHw<gfxProduct>::adjustNumberOfCcs(HardwareInfo &hwInfo) const {
    hwInfo.gtSystemInfo.CCSInfo.NumberOfCCSEnabled = 1;
}

template <>
bool ProductHelperHw<gfxProduct>::isDeviceUsmAllocationReuseSupported() const {
    return false;
}

template <>
uint64_t ProductHelperHw<gfxProduct>::getHostMemCapabilitiesValue() const {
    return (UnifiedSharedMemoryFlags::access);
}

template <>
bool ProductHelperHw<gfxProduct>::isCompressionForbidden(const HardwareInfo &hwInfo) const {
    return isCompressionForbiddenCommon(false);
}

template <>
bool ProductHelperHw<gfxProduct>::isDeviceUsmPoolAllocatorSupported() const {
    return ApiSpecificConfig::OCL == ApiSpecificConfig::getApiType();
}

template <>
bool ProductHelperHw<gfxProduct>::isHostUsmPoolAllocatorSupported() const {
    return ApiSpecificConfig::OCL == ApiSpecificConfig::getApiType();
}

template <>
bool ProductHelperHw<gfxProduct>::scanFullTopologyBitmap() const {
    return true;
}

} // namespace NEO
