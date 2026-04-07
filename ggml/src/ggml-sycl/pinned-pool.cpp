//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "pinned-pool.hpp"

#include "common.hpp"
#include "ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>

namespace ggml_sycl {
namespace {

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool host_zone_debug_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_HOST_ZONE_DEBUG");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    return enabled != 0;
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
        chunk        = std::min<size_t>(chunk, 256ull * 1024ull * 1024ull);
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

const char * host_zone_name(host_zone_id zone) {
    switch (zone) {
        case host_zone_id::WEIGHT:
            return "WEIGHT";
        case host_zone_id::KV:
            return "KV";
        case host_zone_id::STAGING:
            return "STAGING";
        case host_zone_id::SCRATCH:
            return "SCRATCH";
        default:
            return "UNKNOWN";
    }
}

pinned_chunk_pool::pinned_chunk_pool(sycl::queue & queue, size_t budget) :
    queue_(queue),
    budget_(budget),
    chunk_size_(resolve_chunk_size()),
    alloc_timeout_ms_(resolve_alloc_timeout_ms()) {
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

    if (zones_configured_) {
        GGML_LOG_WARN("[SYCL] legacy pinned pool allocate(%zu) after host zones were configured; rejecting\n", size);
        return nullptr;
    }

    // Round up size to alignment
    size              = align_up(size, alignment);
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
        GGML_LOG_INFO("[SYCL] pinned pool grow: min=%zu chunk=%zu total=%.1f GB budget=%.1f GB\n", alloc_size,
                      new_chunk_size, total_allocated_ / (1024.0 * 1024.0 * 1024.0),
                      budget_ / (1024.0 * 1024.0 * 1024.0));
    }
    if (!grow(alloc_size)) {
        return nullptr;
    }

    // Allocate from new chunk
    auto & c = chunks_.back();
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

    if (zones_configured_) {
        return;
    }

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
            GGML_LOG_WARN("[SYCL] Pinned pool pre_allocate: budget exhausted after %zu/%zu chunks\n", chunks_grown,
                          chunks_needed);
            break;
        }
        if (!grow(chunk_size_)) {
            GGML_LOG_WARN("[SYCL] Pinned pool pre_allocate: grow failed at chunk %zu/%zu\n", chunks_grown,
                          chunks_needed);
            break;
        }
        chunks_grown++;
    }

    GGML_LOG_INFO("[SYCL] Pinned pool pre_allocate: grew %zu chunks for %.1f MB working set (total=%.1f GB)\n",
                  chunks_grown, total_bytes / (1024.0 * 1024.0), total_allocated_ / (1024.0 * 1024.0 * 1024.0));
    return chunks_grown;
}

size_t pinned_chunk_pool::pre_allocate_all(size_t model_weight_bytes) {
    constexpr double k_headroom_factor = 1.2;
    const size_t     total_need = static_cast<size_t>(static_cast<double>(model_weight_bytes) * k_headroom_factor);

    GGML_LOG_INFO(
        "[SYCL] Pinned pool pre_allocate_all: "
        "model=%.1f MB, total=%.1f MB (%.0f%% headroom)\n",
        model_weight_bytes / (1024.0 * 1024.0), total_need / (1024.0 * 1024.0), (k_headroom_factor - 1.0) * 100.0);

    return pre_allocate(total_need);
}

void pinned_chunk_pool::configure_zones(size_t weight_bytes,
                                        size_t kv_bytes,
                                        size_t staging_bytes,
                                        size_t scratch_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    weight_bytes  = align_up(weight_bytes, DEFAULT_ALIGNMENT);
    kv_bytes      = align_up(kv_bytes, DEFAULT_ALIGNMENT);
    staging_bytes = align_up(staging_bytes, DEFAULT_ALIGNMENT);
    scratch_bytes = align_up(scratch_bytes, DEFAULT_ALIGNMENT);

    flat_spans_.clear();
    size_t logical_cursor = 0;
    for (size_t i = 0; i < chunks_.size(); ++i) {
        const size_t chunk_start = align_up(chunks_[i].used, DEFAULT_ALIGNMENT);
        if (chunk_start >= chunks_[i].size) {
            continue;
        }
        const size_t span_size = chunks_[i].size - chunk_start;
        flat_spans_.push_back({ logical_cursor, i, chunk_start, span_size });
        logical_cursor += span_size;
    }

    const size_t total_zone_bytes = weight_bytes + kv_bytes + staging_bytes + scratch_bytes;
    GGML_ASSERT(total_zone_bytes <= logical_cursor && "host zones exceed pre-allocated pinned pool capacity");

    size_t start    = 0;
    auto   set_zone = [&](host_zone_id zone, size_t size) {
        auto & z = zones_[static_cast<size_t>(zone)];
        z.start  = start;
        z.size   = size;
        z.used.store(0, std::memory_order_relaxed);
        start += size;
    };

    set_zone(host_zone_id::WEIGHT, weight_bytes);
    set_zone(host_zone_id::KV, kv_bytes);
    set_zone(host_zone_id::STAGING, staging_bytes);
    set_zone(host_zone_id::SCRATCH, scratch_bytes);
    zones_configured_ = true;

    GGML_LOG_INFO(
        "[HOST-ARENA] configured pinned pool zones: WEIGHT=%.1f MB KV=%.1f MB STAGING=%.1f MB SCRATCH=%.1f MB "
        "(free-cap=%.1f MB)\n",
        weight_bytes / (1024.0 * 1024.0), kv_bytes / (1024.0 * 1024.0), staging_bytes / (1024.0 * 1024.0),
        scratch_bytes / (1024.0 * 1024.0), logical_cursor / (1024.0 * 1024.0));
}

void * pinned_chunk_pool::zone_alloc(host_zone_id zone, size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }

    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT)) {
        return nullptr;
    }

    const size_t aligned_size = align_up(size, alignment);
    auto &       z            = zones_[zi];
    if (!zones_configured_ || z.size == 0) {
        return nullptr;
    }

    while (true) {
        size_t cur_used  = z.used.load(std::memory_order_relaxed);
        size_t candidate = align_up(cur_used, alignment);
        if (candidate + aligned_size > z.size) {
            return nullptr;
        }

        const size_t logical_off = z.start + candidate;

        const zone_chunk_span * span = nullptr;
        {
            // Binary search for the span containing offset
            auto it = std::upper_bound(flat_spans_.begin(), flat_spans_.end(), logical_off,
                [](size_t off, const zone_chunk_span& s) { return off < s.logical_start; });
            if (it != flat_spans_.begin()) {
                --it;
                span = &(*it);
            }
        }
        if (span == nullptr || logical_off < span->logical_start) {
            return nullptr;
        }

        const size_t span_off = logical_off - span->logical_start;
        if (span_off + aligned_size > span->span_size) {
            const size_t next_used = (span->logical_start + span->span_size) - z.start;
            if (z.used.compare_exchange_weak(cur_used, next_used, std::memory_order_relaxed)) {
                continue;
            }
            continue;
        }

        if (!z.used.compare_exchange_weak(cur_used, candidate + aligned_size, std::memory_order_relaxed)) {
            continue;
        }

        auto * ptr = static_cast<uint8_t *>(chunks_[span->chunk_idx].base) + span->chunk_start + span_off;
        if (host_zone_debug_enabled()) {
            GGML_LOG_INFO("[HOST-ZONE] alloc zone=%s size=%zu used=%zu/%zu ptr=%p\n", host_zone_name(zone), size,
                          candidate + aligned_size, z.size, ptr);
        }
        return ptr;
    }
}

void pinned_chunk_pool::zone_reset(host_zone_id zone) {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return;
    }
    zones_[zi].used.store(0, std::memory_order_relaxed);
    if (host_zone_debug_enabled()) {
        GGML_LOG_INFO("[HOST-ZONE] reset zone=%s\n", host_zone_name(zone));
    }
}

size_t pinned_chunk_pool::zone_used(host_zone_id zone) const {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return 0;
    }
    return zones_[zi].used.load(std::memory_order_relaxed);
}

size_t pinned_chunk_pool::zone_capacity(host_zone_id zone) const {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return 0;
    }
    return zones_[zi].size;
}

bool pinned_chunk_pool::contains(const void * ptr) const {
    if (!ptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const char *                p = static_cast<const char *>(ptr);
    for (const auto & c : chunks_) {
        const char * base = static_cast<const char *>(c.base);
        if (p >= base && p < base + c.size) {
            return true;
        }
    }
    return false;
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
    void * ptr        = nullptr;
    size_t chunk_size = std::max(chunk_size_, align_up(min_size, DEFAULT_ALIGNMENT));

    if (alloc_timeout_ms_ > 0) {
        if (pinned_trace_enabled()) {
            GGML_LOG_INFO("[SYCL] pinned chunk malloc_host begin: size=%zu timeout=%zu ms\n", chunk_size,
                          alloc_timeout_ms_);
        }
        auto ctx    = queue_.get_context();
        auto future = std::async(std::launch::async,
                                 [&, ctx]() { return ggml_sycl_malloc_host(chunk_size, ctx, "pinned_chunk"); });

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
        const char *           alloc_name = alloc_type == sycl::usm::alloc::host   ? "host" :
                                            alloc_type == sycl::usm::alloc::shared ? "shared" :
                                            alloc_type == sycl::usm::alloc::device ? "device" :
                                                                                     "unknown";
        GGML_LOG_INFO("[SYCL] pinned chunk malloc_host ok: ptr=%p type=%s size=%.1f MB\n", ptr, alloc_name,
                      chunk_size / (1024.0 * 1024.0));
    }

    chunks_.push_back({ ptr, chunk_size, 0, 0, 0 });
    total_allocated_ += chunk_size;

    GGML_LOG_INFO("[SYCL] Allocated pinned chunk %zu (size=%.1f MB, total=%.1f GB)\n", chunks_.size(),
                  chunk_size / (1024.0 * 1024.0), total_allocated_ / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace ggml_sycl
