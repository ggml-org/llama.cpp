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

#include "common.hpp"
#include "ggml-impl.h"

namespace ggml_sycl {
namespace {

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool pinned_trace_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_PINNED_TRACE");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    return enabled != 0;
}

size_t resolve_chunk_size() {
    const char * env = std::getenv("GGML_SYCL_PINNED_CHUNK_MB");
    if (!env || env[0] == '\0') {
        size_t chunk = pinned_chunk_pool::CHUNK_SIZE;
        // Keep default chunks moderate to avoid sudden multi-GB pinned allocations
        // when only small host-staging/fallback buffers are needed.
        chunk = std::min<size_t>(chunk, 256ull * 1024ull * 1024ull);
        return chunk;
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

constexpr uint8_t k_pinned_guard_pattern = 0xA5;
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
    size_t guard_size = 0;
    if (const char * env = std::getenv("GGML_SYCL_HOST_CACHE_GUARD")) {
        if (std::atoi(env) != 0) {
            guard_size = 64;
        }
    }
    const size_t alloc_size = size + guard_size;
    if (pinned_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] pinned alloc request: size=%zu align=%zu guard=%zu alloc=%zu chunks=%zu used=%.1f GB\n",
                      size, alignment, guard_size, alloc_size, chunks_.size(),
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0));
    }

    // Try existing chunks first
    for (auto & c : chunks_) {
        size_t aligned_offset = (c.used + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + alloc_size <= c.size) {
            void * ptr = static_cast<char *>(c.base) + aligned_offset;
            c.used     = aligned_offset + alloc_size;
            c.alloc_count++;
            if (guard_size > 0) {
                std::memset(static_cast<uint8_t *>(ptr) + size, k_pinned_guard_pattern, guard_size);
            }
            return ptr;
        }
    }

    // Need new chunk - check budget
    size_t new_chunk_size = std::max(chunk_size_, align_up(alloc_size, DEFAULT_ALIGNMENT));
    if (total_allocated_ + new_chunk_size > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }

    if (pinned_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] pinned pool grow: min=%zu chunk=%zu total=%.1f GB budget=%.1f GB\n",
                      alloc_size, new_chunk_size, total_allocated_ / (1024.0 * 1024.0 * 1024.0),
                      budget_ / (1024.0 * 1024.0 * 1024.0));
    }
    if (!grow(alloc_size)) {
        return nullptr;
    }

    // Allocate from new chunk
    auto & c   = chunks_.back();
    if (alloc_size > c.size) {
        GGML_LOG_ERROR("[SYCL] Pinned pool allocation %zu exceeds chunk size %zu\n", size, c.size);
        return nullptr;
    }
    void * ptr = c.base;
    c.used     = alloc_size;
    c.alloc_count++;
    if (guard_size > 0) {
        std::memset(static_cast<uint8_t *>(ptr) + size, k_pinned_guard_pattern, guard_size);
    }
    return ptr;
}

void pinned_chunk_pool::deallocate(void * ptr, size_t /* size */) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find containing chunk and track outstanding allocations
    for (auto & c : chunks_) {
        char * base = static_cast<char *>(c.base);
        char * p    = static_cast<char *>(ptr);
        if (p >= base && p < base + c.size) {
            c.freed++;  // Track number of frees, not bytes (avoids alignment mismatch)
            // Reclaim chunk when all allocations have been freed.
            // alloc_count tracks total allocations from this chunk.
            // When freed == alloc_count, the entire chunk is unused → reset.
            // This is critical for MoE warmup profiling which stages entire
            // layers (~538 MB each) through pinned staging buffers. Without
            // reclamation, 36 layers × 538 MB = 19.4 GB of pinned memory
            // is allocated and never reused (bump allocator pathology).
            if (c.freed >= c.alloc_count) {
                c.used        = 0;
                c.freed       = 0;
                c.alloc_count = 0;
            }
            return;
        }
    }
}

size_t pinned_chunk_pool::pre_allocate(size_t total_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Calculate how many chunks are needed to cover total_bytes,
    // minus capacity already available in existing chunks.
    size_t existing_capacity = 0;
    for (const auto & c : chunks_) {
        existing_capacity += (c.size - c.used);
    }
    if (existing_capacity >= total_bytes) {
        GGML_LOG_INFO("[SYCL] Pinned pool pre_allocate: already have %.1f MB free (need %.1f MB)\n",
                      existing_capacity / (1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0));
        return 0;
    }

    const size_t deficit       = total_bytes - existing_capacity;
    const size_t chunks_needed = (deficit + chunk_size_ - 1) / chunk_size_;
    size_t       chunks_grown  = 0;

    for (size_t i = 0; i < chunks_needed; i++) {
        if (total_allocated_ + chunk_size_ > budget_) {
            GGML_LOG_WARN("[SYCL] Pinned pool pre_allocate: budget exhausted after %zu/%zu chunks\n",
                          chunks_grown, chunks_needed);
            break;
        }
        if (!grow(chunk_size_)) {
            GGML_LOG_WARN("[SYCL] Pinned pool pre_allocate: grow failed at chunk %zu/%zu\n",
                          chunks_grown, chunks_needed);
            break;
        }
        chunks_grown++;
    }

    GGML_LOG_INFO("[SYCL] Pinned pool pre_allocate: grew %zu chunks for %.1f MB working set (total=%.1f GB)\n",
                  chunks_grown, total_bytes / (1024.0 * 1024.0), total_allocated_ / (1024.0 * 1024.0 * 1024.0));
    return chunks_grown;
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
        if (pinned_trace_enabled()) {
            GGML_LOG_INFO("[SYCL] pinned chunk malloc_host begin: size=%zu timeout=%zu ms\n", chunk_size,
                          alloc_timeout_ms_);
        }
        auto ctx    = queue_.get_context();
        auto future = std::async(std::launch::async, [&, ctx]() {
            return ggml_sycl_malloc_host(chunk_size, ctx, "pinned_chunk");
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
        if (pinned_trace_enabled()) {
            GGML_LOG_INFO("[SYCL] pinned chunk malloc_host begin: size=%zu\n", chunk_size);
        }
        try {
            ptr = ggml_sycl_malloc_host(chunk_size, queue_.get_context(), "pinned_chunk");
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes): %s\n", chunk_size, e.what());
            return false;
        }
    }

    if (!ptr) {
        GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes, nullptr)\n", chunk_size);
        return false;
    }
    if (pinned_trace_enabled()) {
        const sycl::usm::alloc alloc_type = ggml_sycl_get_alloc_type(ptr);
        const char *           alloc_name = alloc_type == sycl::usm::alloc::host ? "host" :
                                  alloc_type == sycl::usm::alloc::shared ? "shared" :
                                  alloc_type == sycl::usm::alloc::device ? "device" : "unknown";
        GGML_LOG_INFO("[SYCL] pinned chunk malloc_host ok: ptr=%p type=%s size=%.1f MB\n", ptr, alloc_name,
                      chunk_size / (1024.0 * 1024.0));
    }

    chunks_.push_back({ ptr, chunk_size, 0, 0, 0 });
    total_allocated_ += chunk_size;

    GGML_LOG_INFO("[SYCL] Allocated pinned chunk %zu (size=%.1f MB, total=%.1f GB)\n",
                  chunks_.size(), chunk_size / (1024.0 * 1024.0),
                  total_allocated_ / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace ggml_sycl
