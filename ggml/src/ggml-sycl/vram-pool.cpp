//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "vram-pool.hpp"

#include "common.hpp"
#include "ggml-impl.h"

namespace ggml_sycl {

vram_pool::vram_pool(sycl::queue & queue, size_t budget) :
    queue_(queue),
    device_id_(ggml_sycl_get_device_id_from_queue(queue)),
    budget_(budget) {
    GGML_LOG_INFO("[SYCL] VRAM pool created with %.2f GB budget\n", budget / (1024.0 * 1024.0 * 1024.0));
}

vram_pool::~vram_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count  = allocations_.size();
    size_t                      failed = 0;

    for (auto & [id, alloc] : allocations_) {
        if (alloc.ptr) {
            try {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, alloc.size);
                sycl::free(alloc.ptr, queue_);
            } catch (const sycl::exception & e) {
                GGML_LOG_ERROR("[SYCL] Failed to free tensor_id %llu: %s\n", (unsigned long long) id, e.what());
                failed++;
            }
        }
    }
    allocations_.clear();
    used_ = 0;

    if (count > 0) {
        GGML_LOG_INFO("[SYCL] VRAM pool destroyed, released %zu allocations%s\n", count,
                      failed > 0 ? " (some failed)" : "");
    }
}

void * vram_pool::allocate(size_t size, uint64_t tensor_id, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Validate alignment is power of 2 and non-zero
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        GGML_LOG_ERROR("[SYCL] Invalid alignment: %zu (must be power of 2)\n", alignment);
        return nullptr;
    }

    // Check for potential overflow before alignment calculation
    if (size > SIZE_MAX - alignment) {
        GGML_LOG_ERROR("[SYCL] Requested size too large: %zu\n", size);
        return nullptr;
    }

    // Round up size to alignment
    size = (size + alignment - 1) & ~(alignment - 1);

    // Check if already allocated
    auto it = allocations_.find(tensor_id);
    if (it != allocations_.end()) {
        // Check if requested size matches (after alignment)
        if (it->second.size != size) {
            GGML_LOG_ERROR("[SYCL] tensor_id %llu already allocated with size %zu, requested %zu\n",
                           (unsigned long long) tensor_id, it->second.size, size);
            return nullptr;
        }
        return it->second.ptr;  // Return existing allocation
    }

    // Check budget
    if (used_ + size > budget_) {
        GGML_LOG_WARN("[SYCL] Allocation of %zu bytes exceeds budget (used: %zu, budget: %zu)\n", size, used_, budget_);
        return nullptr;
    }

    // Allocate device memory
    void * ptr = nullptr;
    try {
        ggml_sycl::unified_cache_add_runtime_bytes(device_id_, size);
        ptr = ggml_sycl_malloc_device(size, queue_, "vram_pool");
    } catch (const sycl::exception & e) {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, size);
        GGML_LOG_ERROR("[SYCL] VRAM allocation failed: %s\n", e.what());
        return nullptr;
    }

    if (!ptr) {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, size);
        GGML_LOG_ERROR("[SYCL] malloc_device returned nullptr for size %zu\n", size);
        return nullptr;
    }

    allocations_[tensor_id] = { ptr, size };
    used_ += size;

    return ptr;
}

void vram_pool::deallocate(uint64_t tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocations_.find(tensor_id);
    if (it == allocations_.end()) {
        GGML_LOG_WARN("[SYCL] Attempted to deallocate non-existent tensor_id %llu\n", (unsigned long long) tensor_id);
        return;
    }

    try {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, it->second.size);
        sycl::free(it->second.ptr, queue_);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[SYCL] Failed to deallocate tensor_id %llu: %s\n", (unsigned long long) tensor_id, e.what());
    }
    used_ -= it->second.size;
    allocations_.erase(it);
}

bool vram_pool::is_allocated(uint64_t tensor_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.find(tensor_id) != allocations_.end();
}

void * vram_pool::get(uint64_t tensor_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = allocations_.find(tensor_id);
    return it != allocations_.end() ? it->second.ptr : nullptr;
}

size_t vram_pool::allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.size();
}

}  // namespace ggml_sycl
