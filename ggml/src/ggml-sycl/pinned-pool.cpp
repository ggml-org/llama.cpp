//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "pinned-pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>

#include "ggml-impl.h"

namespace ggml_sycl {
namespace {

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

size_t resolve_chunk_size() {
    const char * env = std::getenv("GGML_SYCL_PINNED_CHUNK_MB");
    if (!env || env[0] == '\0') {
        return pinned_chunk_pool::CHUNK_SIZE;
    }

    char * end = nullptr;
    long   mb  = std::strtol(env, &end, 10);
    if (end == env || mb <= 0) {
        GGML_LOG_WARN("[SYCL] Invalid GGML_SYCL_PINNED_CHUNK_MB='%s', using default chunk size\n", env);
        return pinned_chunk_pool::CHUNK_SIZE;
    }

    return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
}

size_t resolve_alloc_timeout_ms() {
    const char * env = std::getenv("GGML_SYCL_PINNED_ALLOC_TIMEOUT_MS");
    if (!env || env[0] == '\0') {
        return 0;
    }

    char * end = nullptr;
    long   ms  = std::strtol(env, &end, 10);
    if (end == env || ms <= 0) {
        GGML_LOG_WARN("[SYCL] Invalid GGML_SYCL_PINNED_ALLOC_TIMEOUT_MS='%s', disabling timeout\n", env);
        return 0;
    }

    return static_cast<size_t>(ms);
}

}  // namespace

pinned_chunk_pool::pinned_chunk_pool(sycl::queue & queue, size_t budget)
    : queue_(queue), budget_(budget), chunk_size_(resolve_chunk_size()), alloc_timeout_ms_(resolve_alloc_timeout_ms()) {
    GGML_LOG_INFO("[SYCL] Pinned chunk pool created with %.1f GB budget (chunk=%.1f MB)\n",
                  budget / (1024.0 * 1024.0 * 1024.0), chunk_size_ / (1024.0 * 1024.0));
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
    size = align_up(size, alignment);

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
    size_t new_chunk_size = std::max(chunk_size_, align_up(size, DEFAULT_ALIGNMENT));
    if (total_allocated_ + new_chunk_size > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }

    if (!grow(size)) {
        return nullptr;
    }

    // Allocate from new chunk
    auto & c   = chunks_.back();
    if (size > c.size) {
        GGML_LOG_ERROR("[SYCL] Pinned pool allocation %zu exceeds chunk size %zu\n", size, c.size);
        return nullptr;
    }
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

bool pinned_chunk_pool::grow(size_t min_size) {
    void * ptr = nullptr;
    size_t chunk_size = std::max(chunk_size_, align_up(min_size, DEFAULT_ALIGNMENT));

    if (alloc_timeout_ms_ > 0) {
        auto future = std::async(std::launch::async, [&]() {
            return sycl::malloc_host(chunk_size, queue_);
        });

        const auto status = future.wait_for(std::chrono::milliseconds(alloc_timeout_ms_));
        if (status != std::future_status::ready) {
            GGML_LOG_ERROR("[SYCL] Pinned chunk allocation timed out after %zu ms (size=%zu). Aborting.\n",
                           alloc_timeout_ms_, chunk_size);
            std::fflush(stderr);
            std::abort();
        }

        try {
            ptr = future.get();
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes): %s\n", chunk_size, e.what());
            return false;
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes): %s\n", chunk_size, e.what());
            return false;
        }
    } else {
        try {
            ptr = sycl::malloc_host(chunk_size, queue_);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes): %s\n", chunk_size, e.what());
            return false;
        }
    }

    if (!ptr) {
        GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes, nullptr)\n", chunk_size);
        return false;
    }

    chunks_.push_back({ ptr, chunk_size, 0, 0 });
    total_allocated_ += chunk_size;

    GGML_LOG_INFO("[SYCL] Allocated pinned chunk %zu (size=%.1f MB, total=%.1f GB)\n",
                  chunks_.size(), chunk_size / (1024.0 * 1024.0),
                  total_allocated_ / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace ggml_sycl
