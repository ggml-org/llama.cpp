/*
 * Copyright (C) 2025-2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "shared/source/utilities/pool_allocator_traits.h"

#include "shared/source/debug_settings/debug_settings_manager.h"
#include "shared/source/device/device.h"
#include "shared/source/os_interface/device_factory.h"
#include "shared/source/os_interface/product_helper.h"

namespace NEO {

AllocationProperties TimestampPoolTraits::createAllocationProperties(Device *device, size_t poolSize) {
    return AllocationProperties{device->getRootDeviceIndex(),
                                poolSize,
                                allocationType,
                                device->getDeviceBitfield()};
}

AllocationProperties GlobalSurfacePoolTraits::createAllocationProperties(Device *device, size_t poolSize) {
    return AllocationProperties{device->getRootDeviceIndex(),
                                true, // allocateMemory
                                poolSize,
                                allocationType,
                                false, // isMultiStorageAllocation
                                device->getDeviceBitfield()};
}

AllocationProperties ConstantSurfacePoolTraits::createAllocationProperties(Device *device, size_t poolSize) {
    return AllocationProperties{device->getRootDeviceIndex(),
                                true, // allocateMemory
                                poolSize,
                                allocationType,
                                false, // isMultiStorageAllocation
                                device->getDeviceBitfield()};
}

AllocationProperties CommandBufferPoolTraits::createAllocationProperties(Device *device, size_t poolSize) {
    return AllocationProperties{device->getRootDeviceIndex(),
                                true, // allocateMemory
                                poolSize,
                                allocationType,
                                (device->getNumGenericSubDevices() > 1u), // multiOsContextCapable
                                false,
                                device->getDeviceBitfield()};
}

AllocationProperties LinearStreamPoolTraits::createAllocationProperties(Device *device, size_t poolSize) {
    return AllocationProperties{device->getRootDeviceIndex(),
                                true, // allocateMemory
                                poolSize,
                                allocationType,
                                (device->getNumGenericSubDevices() > 1u), // multiOsContextCapable
                                false,
                                device->getDeviceBitfield()};
}

AllocationProperties InternalHeapPoolTraits::createAllocationProperties(Device *device, size_t poolSize) {
    return AllocationProperties{device->getRootDeviceIndex(),
                                true, // allocateMemory
                                poolSize,
                                allocationType,
                                (device->getNumGenericSubDevices() > 1u), // multiOsContextCapable
                                false,
                                device->getDeviceBitfield()};
}

bool CommandBufferPoolTraits::isEnabled(const ProductHelper &productHelper) {
    auto forceEnable = debugManager.flags.EnableCommandBufferPoolAllocator.get();
    if (forceEnable != -1) {
        return forceEnable == 1;
    }
    return productHelper.is2MBLocalMemAlignmentEnabled();
}

bool LinearStreamPoolTraits::isEnabled(const ProductHelper &productHelper) {
    auto forceEnable = debugManager.flags.EnableLinearStreamPoolAllocator.get();
    if (forceEnable != -1) {
        return forceEnable == 1;
    }
    return productHelper.is2MBLocalMemAlignmentEnabled();
}

bool InternalHeapPoolTraits::isEnabled(const ProductHelper &productHelper) {
    auto forceEnable = debugManager.flags.EnableInternalHeapPoolAllocator.get();
    if (forceEnable != -1) {
        return forceEnable == 1;
    }
    return productHelper.is2MBLocalMemAlignmentEnabled();
}

} // namespace NEO
