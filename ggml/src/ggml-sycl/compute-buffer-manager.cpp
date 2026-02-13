//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Compute Buffer Manager implementation
// Part of unified memory system (epic llama.cpp-v3n, task llama.cpp-6s5)
//

#include "compute-buffer-manager.hpp"

#include "common.hpp"
#include "unified-cache.hpp"

#include <algorithm>
#include <cstdio>

namespace ggml_sycl {

ComputeBufferManager::ComputeBufferManager(sycl::queue & queue) :
    queue_(queue),
    device_id_(ggml_sycl_get_device_id_from_queue(queue)) {}

ComputeBufferManager::~ComputeBufferManager() {
    // Free all pooled buffers
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (auto & buf : pool_) {
        if (buf.ptr != nullptr) {
            if (!ggml_sycl::unified_free(buf.handle)) {
                fprintf(stderr, "[ComputeBufferManager] Warning: failed to free buffer %p (%zu bytes)\n",
                        buf.ptr, buf.size);
            }
        }
    }
    pool_.clear();

    // Free scratch buffer
    {
        std::lock_guard<std::mutex> scratch_lock(scratch_mutex_);
        if (scratch_ptr_ != nullptr) {
            if (!ggml_sycl::unified_free(scratch_handle_)) {
                fprintf(stderr, "[ComputeBufferManager] Warning: failed to free scratch %p (%zu bytes)\n",
                        scratch_ptr_, scratch_capacity_);
            }
            scratch_ptr_      = nullptr;
            scratch_size_     = 0;
            scratch_capacity_ = 0;
            scratch_handle_   = {};
        }
    }
}

void * ComputeBufferManager::allocate(size_t size, const char * name) {
    if (size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);
    total_allocations_++;

    // Try to find an existing free buffer that can satisfy this request
    ComputeBuffer * buf = find_free_buffer(size);
    if (buf != nullptr) {
        buf->in_use     = true;
        buf->alloc_time = current_time();
        buf->name       = name;
        pool_hits_++;
        return buf->ptr;
    }

    // No suitable buffer in pool - allocate new one
    pool_misses_++;
    void * ptr = allocate_new_buffer(size);
    if (ptr == nullptr) {
        return nullptr;  // OOM
    }

    // Add to pool
    ComputeBuffer new_buf;
    new_buf.ptr        = ptr;
    new_buf.size       = size;
    new_buf.in_use     = true;
    new_buf.alloc_time = current_time();
    new_buf.name       = name;
    unified_lookup(ptr, &new_buf.handle);
    pool_.push_back(new_buf);

    return ptr;
}

void ComputeBufferManager::release(void * ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);

    for (auto & buf : pool_) {
        if (buf.ptr == ptr) {
            buf.in_use = false;
            buf.name   = nullptr;
            return;
        }
    }

    // Pointer not found in pool - this is a bug
    fprintf(stderr, "[ComputeBufferManager] Warning: release called on unknown pointer %p\n", ptr);
}

void * ComputeBufferManager::get_scratch(size_t min_size) {
    if (min_size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(scratch_mutex_);

    if (min_size <= scratch_capacity_) {
        scratch_size_ = min_size;
        return scratch_ptr_;
    }

    // Need to grow scratch
    grow_scratch(min_size);
    return scratch_ptr_;
}

EvictionPriority ComputeBufferManager::get_priority(void * ptr) const {
    // All compute buffers are P0 (CRITICAL) - never evicted
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (const auto & buf : pool_) {
        if (buf.ptr == ptr) {
            return EvictionPriority::P0_COMPUTE;
        }
    }

    // Check scratch
    std::lock_guard<std::mutex> scratch_lock(scratch_mutex_);
    if (ptr == scratch_ptr_) {
        return EvictionPriority::P0_COMPUTE;
    }

    // Unknown pointer - return P0 anyway (conservative)
    return EvictionPriority::P0_COMPUTE;
}

bool ComputeBufferManager::is_pinned(void * ptr) const {
    // All compute buffers are pinned (P0 = never evicted)
    return is_valid(ptr);
}

bool ComputeBufferManager::is_valid(void * ptr) const {
    if (ptr == nullptr) {
        return false;
    }

    // Check pool
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        for (const auto & buf : pool_) {
            if (buf.ptr == ptr) {
                return true;
            }
        }
    }

    // Check scratch
    {
        std::lock_guard<std::mutex> lock(scratch_mutex_);
        if (ptr == scratch_ptr_) {
            return true;
        }
    }

    return false;
}

bool ComputeBufferManager::try_evict_for_space(size_t bytes_needed) {
    (void) bytes_needed;
    // P0 buffers are NEVER evicted - always return false
    return false;
}

size_t ComputeBufferManager::pool_total_size() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    size_t                      total = 0;
    for (const auto & buf : pool_) {
        total += buf.size;
    }

    // Include scratch in total
    std::lock_guard<std::mutex> scratch_lock(scratch_mutex_);
    total += scratch_capacity_;

    return total;
}

size_t ComputeBufferManager::pool_used_size() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    size_t                      used = 0;
    for (const auto & buf : pool_) {
        if (buf.in_use) {
            used += buf.size;
        }
    }
    return used;
}

void ComputeBufferManager::print_stats() const {
    printf("=== Compute Buffer Manager Stats ===\n");
    printf("Total allocations: %zu\n", num_allocations());
    printf("Pool hits:         %zu\n", num_pool_hits());
    printf("Pool misses:       %zu\n", num_pool_misses());

    float hit_rate = 0.0f;
    if (num_allocations() > 0) {
        hit_rate = 100.0f * static_cast<float>(num_pool_hits()) / static_cast<float>(num_allocations());
    }
    printf("Pool hit rate:     %.1f%%\n", hit_rate);

    printf("Pool total size:   %zu bytes (%.2f MB)\n", pool_total_size(), pool_total_size() / (1024.0 * 1024.0));
    printf("Pool used size:    %zu bytes (%.2f MB)\n", pool_used_size(), pool_used_size() / (1024.0 * 1024.0));
    printf("Scratch capacity:  %zu bytes (%.2f MB)\n", get_scratch_capacity(),
           get_scratch_capacity() / (1024.0 * 1024.0));
    printf("=====================================\n");
}

ComputeBuffer * ComputeBufferManager::find_free_buffer(size_t size) {
    // Find smallest free buffer that can satisfy the request
    // This minimizes wasted memory from using oversized buffers
    ComputeBuffer * best      = nullptr;
    size_t          best_size = SIZE_MAX;

    for (auto & buf : pool_) {
        if (!buf.in_use && buf.size >= size) {
            if (buf.size < best_size) {
                best      = &buf;
                best_size = buf.size;
            }
        }
    }

    return best;
}

void * ComputeBufferManager::allocate_new_buffer(size_t size) {
    ggml_sycl::alloc_request req;
    req.queue                          = &queue_;
    req.device                         = device_id_;
    req.size                           = size;
    req.intent.role                    = ggml_sycl::alloc_role::COMPUTE;
    req.intent.category                = ggml_sycl::runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    ggml_sycl::alloc_handle handle;
    if (!ggml_sycl::unified_alloc(req, &handle)) {
        fprintf(stderr, "[ComputeBufferManager] allocation failed for %zu bytes\n", size);
        return nullptr;
    }
    return handle.ptr;
}

void ComputeBufferManager::grow_scratch(size_t new_size) {
    // Round up to 16 MB boundary for better allocation efficiency
    constexpr size_t ALIGNMENT = 16 * 1024 * 1024;  // 16 MB
    new_size                   = ((new_size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

    // Free old scratch if exists
    if (scratch_ptr_ != nullptr) {
        if (!ggml_sycl::unified_free(scratch_handle_)) {
            fprintf(stderr, "[ComputeBufferManager] Warning: failed to free old scratch %p (%zu bytes)\n",
                    scratch_ptr_, scratch_capacity_);
        }
        scratch_ptr_    = nullptr;
        scratch_handle_ = {};
    }

    // Allocate new scratch
    ggml_sycl::alloc_request req;
    req.queue                          = &queue_;
    req.device                         = device_id_;
    req.size                           = new_size;
    req.intent.role                    = ggml_sycl::alloc_role::COMPUTE;
    req.intent.category                = ggml_sycl::runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    if (!ggml_sycl::unified_alloc(req, &scratch_handle_) || scratch_handle_.ptr == nullptr) {
        fprintf(stderr, "[ComputeBufferManager] scratch allocation failed for %zu bytes\n", new_size);
        scratch_ptr_      = nullptr;
        scratch_capacity_ = 0;
        scratch_size_     = 0;
        scratch_handle_   = {};
        return;
    }
    scratch_ptr_      = scratch_handle_.ptr;
    scratch_capacity_ = new_size;
    scratch_size_     = new_size;
}

uint64_t ComputeBufferManager::current_time() {
    return time_counter_.fetch_add(1);
}

}  // namespace ggml_sycl
