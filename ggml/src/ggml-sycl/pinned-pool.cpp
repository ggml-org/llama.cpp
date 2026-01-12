//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "pinned-pool.hpp"

#include "ggml-impl.h"

namespace ggml_sycl {

pinned_chunk_pool::pinned_chunk_pool(sycl::queue & queue, size_t budget) : queue_(queue), budget_(budget) {
    GGML_LOG_INFO("[SYCL] Pinned chunk pool created with %.1f GB budget\n", budget / (1024.0 * 1024.0 * 1024.0));
}

pinned_chunk_pool::~pinned_chunk_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      chunk_count = chunks_.size();

    for (auto & c : chunks_) {
        if (c.base) {
            sycl::free(c.base, queue_);
        }
    }
    chunks_.clear();
    total_allocated_ = 0;

    GGML_LOG_INFO("[SYCL] Pinned chunk pool destroyed, released %zu chunks\n", chunk_count);
}

void * pinned_chunk_pool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Round up size to alignment
    size = (size + alignment - 1) & ~(alignment - 1);

    // Try existing chunks first
    for (auto & c : chunks_) {
        size_t aligned_offset = (c.used + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + size <= c.size) {
            void * ptr = static_cast<char *>(c.base) + aligned_offset;
            c.used     = aligned_offset + size;
            return ptr;
        }
    }

    // Need new chunk - check budget
    if (total_allocated_ + CHUNK_SIZE > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }

    if (!grow()) {
        return nullptr;
    }

    // Allocate from new chunk
    auto & c   = chunks_.back();
    void * ptr = c.base;
    c.used     = size;
    return ptr;
}

void pinned_chunk_pool::deallocate(void * ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find containing chunk
    for (auto & c : chunks_) {
        char * base = static_cast<char *>(c.base);
        char * p    = static_cast<char *>(ptr);
        if (p >= base && p < base + c.size) {
            c.freed += size;
            // Note: bump allocator doesn't reclaim individual allocations.
            // Chunk is only released when pool is destroyed.
            return;
        }
    }
}

size_t pinned_chunk_pool::allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

size_t pinned_chunk_pool::chunk_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return chunks_.size();
}

bool pinned_chunk_pool::grow() {
    void * ptr = nullptr;
    try {
        ptr = sycl::malloc_host(CHUNK_SIZE, queue_);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[SYCL] Failed to allocate 8GB pinned chunk: %s\n", e.what());
        return false;
    }

    if (!ptr) {
        GGML_LOG_ERROR("[SYCL] Failed to allocate 8GB pinned chunk (nullptr)\n");
        return false;
    }

    chunks_.push_back({ ptr, CHUNK_SIZE, 0, 0 });
    total_allocated_ += CHUNK_SIZE;

    GGML_LOG_INFO("[SYCL] Allocated pinned chunk %zu (total: %.1f GB)\n", chunks_.size(),
                  total_allocated_ / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace ggml_sycl
