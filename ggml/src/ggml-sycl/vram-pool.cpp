//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "vram-pool.hpp"

#include "ggml-impl.h"

namespace ggml_sycl {

vram_pool::vram_pool(sycl::queue & queue, size_t budget) : queue_(queue), budget_(budget) {
    GGML_LOG_INFO("[SYCL] VRAM pool created with %.2f GB budget\n", budget / (1024.0 * 1024.0 * 1024.0));
}

vram_pool::~vram_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = allocations_.size();

    for (auto & [id, alloc] : allocations_) {
        if (alloc.ptr) {
            sycl::free(alloc.ptr, queue_);
        }
    }
    allocations_.clear();
    used_ = 0;

    GGML_LOG_INFO("[SYCL] VRAM pool destroyed, released %zu allocations\n", count);
}

void * vram_pool::allocate(size_t size, uint64_t tensor_id, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Round up size to alignment
    size = (size + alignment - 1) & ~(alignment - 1);

    // Check if already allocated
    auto it = allocations_.find(tensor_id);
    if (it != allocations_.end()) {
        return it->second.ptr;  // Return existing allocation
    }

    // Check budget
    if (used_ + size > budget_) {
        return nullptr;
    }

    // Allocate device memory
    void * ptr = nullptr;
    try {
        ptr = sycl::malloc_device(size, queue_);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[SYCL] VRAM allocation failed: %s\n", e.what());
        return nullptr;
    }

    if (!ptr) {
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
        return;
    }

    sycl::free(it->second.ptr, queue_);
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
