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
    size_t                      chunk_count = chunks_.size() + runtime_chunks_.size();

    for (auto & c : chunks_) {
        if (c.base) {
            sycl::free(c.base, queue_);
        }
    }
    for (auto & c : runtime_chunks_) {
        if (c.base) {
            sycl::free(c.base, queue_);
        }
    }
    chunks_.clear();
    runtime_chunks_.clear();
    total_allocated_ = 0;

    GGML_LOG_INFO("[SYCL] Pinned chunk pool destroyed, released %zu chunks\n", chunk_count);
}

segmented_buffer pinned_chunk_pool::allocate_segmented(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    segmented_buffer            result;
    result.total_size = size;

    size = align_up(size, alignment);

    for (auto & c : chunks_) {
        size_t aligned_offset = (c.used + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + size <= c.size) {
            void * ptr = static_cast<char *>(c.base) + aligned_offset;
            c.used     = aligned_offset + size;
            c.alloc_count++;
            result.segments.push_back({ ptr, size });
            return result;
        }
    }

    // Need to grow - allocate a new chunk
    size_t new_chunk_size = std::min(chunk_size_, align_up(size, DEFAULT_ALIGNMENT));
    if (total_allocated_ + new_chunk_size > budget_) {
        return {};
    }

    if (!grow_into(chunks_, new_chunk_size, false)) {
        return {};
    }

    auto & c = chunks_.back();
    if (size > c.size) {
        return {};
    }
    void * ptr = c.base;
    c.used     = size;
    c.alloc_count++;
    result.segments.push_back({ ptr, size });
    return result;
}

void * pinned_chunk_pool::allocate(size_t size, size_t alignment) {
    segmented_buffer buf = allocate_segmented(size, alignment);
    return buf.segments.empty() ? nullptr : buf.segments[0].ptr;
}

void * pinned_chunk_pool::allocate_runtime(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocate_from_chunks(runtime_chunks_, size, alignment, true);
}

void pinned_chunk_pool::deallocate(void * ptr, size_t /* size */) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (deallocate_from_chunks(runtime_chunks_, ptr)) {
        return;
    }
    (void) deallocate_from_chunks(chunks_, ptr);
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

    const size_t total_zone_bytes   = weight_bytes + kv_bytes + staging_bytes + scratch_bytes;
    auto         rebuild_flat_spans = [&]() {
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
        return logical_cursor;
    };

    size_t logical_cursor = rebuild_flat_spans();
    while (logical_cursor < total_zone_bytes && total_allocated_ + chunk_size_ <= budget_) {
        if (!grow(chunk_size_)) {
            break;
        }
        logical_cursor = rebuild_flat_spans();
    }

    if (total_zone_bytes > logical_cursor) {
        GGML_LOG_WARN(
            "[HOST-ARENA] pinned pool capacity %.1f MB is below requested zone footprint %.1f MB; "
            "disabling host zones and falling back to runtime pinned allocations\n",
            logical_cursor / (1024.0 * 1024.0), total_zone_bytes / (1024.0 * 1024.0));
        for (auto & zone : zones_) {
            zone.start = 0;
            zone.size  = 0;
            zone.used.store(0, std::memory_order_relaxed);
        }
        flat_spans_.clear();
        zones_configured_ = false;
        return;
    }

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

segmented_buffer pinned_chunk_pool::zone_alloc_segmented(host_zone_id zone, size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    segmented_buffer            result;
    result.total_size = size;

    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT)) {
        return {};
    }

    auto & z = zones_[zi];
    if (!zones_configured_ || z.size == 0) {
        return {};
    }

    size_t remaining = size;
    while (remaining > 0) {
        size_t cur_used  = z.used.load(std::memory_order_relaxed);
        size_t candidate = align_up(cur_used, alignment);
        if (candidate >= z.size) {
            GGML_LOG_ERROR("[HOST-ZONE] segmented alloc failed: zone=%s exhausted (used=%zu/%zu, remaining=%zu)\n",
                           host_zone_name(zone), candidate, z.size, remaining);
            return {};
        }

        size_t logical_off = z.start + candidate;

        // Find the span that contains logical_off
        const zone_chunk_span * span = nullptr;
        {
            auto it = std::upper_bound(flat_spans_.begin(), flat_spans_.end(), logical_off,
                                       [](size_t off, const zone_chunk_span & s) { return off < s.logical_start; });
            if (it != flat_spans_.begin()) {
                --it;
                span = &(*it);
            }
        }

        if (span == nullptr || logical_off < span->logical_start) {
            GGML_LOG_ERROR("[HOST-ZONE] segmented alloc: no span for logical_off=%zu in zone=%s\n", logical_off,
                           host_zone_name(zone));
            return {};
        }

        // Calculate how much space is available in this span
        size_t span_remaining = span->span_size - (logical_off - span->logical_start);
        if (span_remaining == 0) {
            // Move to next span - advance used to the start of the next span
            auto next_it =
                std::upper_bound(flat_spans_.begin(), flat_spans_.end(), logical_off,
                                 [](size_t off, const zone_chunk_span & s) { return off < s.logical_start; });
            if (next_it == flat_spans_.end()) {
                GGML_LOG_ERROR("[HOST-ZONE] segmented alloc: no more spans for zone=%s (remaining=%zu)\n",
                               host_zone_name(zone), remaining);
                return {};
            }
            size_t next_span_start = next_it->logical_start;
            size_t new_used        = next_span_start - z.start;
            z.used.store(new_used, std::memory_order_relaxed);
            continue;
        }

        // Determine the sub-allocation size: min of remaining request, chunk_size, and span_remaining
        // Round span_remaining down to alignment to keep sub-allocations aligned
        size_t max_in_span = span_remaining >= alignment ? (span_remaining / alignment) * alignment : span_remaining;
        size_t alloc_size  = std::min({ remaining, chunk_size_, max_in_span });
        size_t aligned_alloc_size = align_up(alloc_size, alignment);

        // Verify it fits in the zone
        if (candidate + aligned_alloc_size > z.size) {
            GGML_LOG_ERROR("[HOST-ZONE] segmented alloc failed: zone=%s size=%zu used=%zu/%zu\n", host_zone_name(zone),
                           alloc_size, candidate + aligned_alloc_size, z.size);
            return {};
        }

        if (!z.used.compare_exchange_weak(cur_used, candidate + aligned_alloc_size, std::memory_order_relaxed)) {
            continue;
        }

        void * ptr = static_cast<uint8_t *>(chunks_[span->chunk_idx].base) + span->chunk_start +
                     (logical_off - span->logical_start);
        result.segments.push_back({ ptr, alloc_size });
        remaining -= alloc_size;
    }
    return result;
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

void pinned_chunk_pool::zone_rollback(host_zone_id zone, size_t saved_used) {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return;
    }
    auto & z   = zones_[zi];
    size_t cur = z.used.load(std::memory_order_relaxed);
    if (saved_used > cur) {
        GGML_LOG_WARN("[HOST-ZONE] rollback zone=%s: saved_used=%zu > current_used=%zu, skipping\n",
                      host_zone_name(zone), saved_used, cur);
        return;
    }
    z.used.store(saved_used, std::memory_order_relaxed);
    if (host_zone_debug_enabled()) {
        GGML_LOG_INFO("[HOST-ZONE] rollback zone=%s used %zu -> %zu\n", host_zone_name(zone), cur, saved_used);
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

bool pinned_chunk_pool::grow_zone(host_zone_id zone, size_t additional_bytes) {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Calculate how many chunks we need to add
    size_t chunks_needed = (additional_bytes + chunk_size_ - 1) / chunk_size_;
    if (chunks_needed == 0) {
        return true;
    }

    // Check budget before allocating
    if (total_allocated_ + chunks_needed * chunk_size_ > budget_) {
        GGML_LOG_WARN(
            "[SYCL] Pinned pool grow_zone: budget exhausted for zone %zu "
            "(need %zu chunks, budget=%.1f GB, used=%.1f GB)\n",
            zi, chunks_needed, budget_ / (1024.0 * 1024.0 * 1024.0), total_allocated_ / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    // Record the old end of the pool (where new chunks will start)
    size_t old_chunk_count = chunks_.size();

    // Allocate new chunks
    for (size_t i = 0; i < chunks_needed; i++) {
        if (!grow(chunk_size_)) {
            GGML_LOG_WARN("[SYCL] Pinned pool grow_zone: grow failed at chunk %zu/%zu\n", i, chunks_needed);
            break;
        }
    }

    size_t new_chunks_added = chunks_.size() - old_chunk_count;
    if (new_chunks_added == 0) {
        return false;
    }

    // Extend the zone's span to include the new chunks
    size_t additional_capacity = new_chunks_added * chunk_size_;
    zones_[zi].size += additional_capacity;

    // Update flat_spans_ to include the new chunks in this zone
    // Compute logical cursor: zone start + old zone size (before growth)
    size_t logical_cursor = zones_[zi].start + (zones_[zi].size - additional_capacity);
    for (size_t i = old_chunk_count; i < chunks_.size(); i++) {
        flat_spans_.push_back({ logical_cursor, i, 0ULL, chunks_[i].size });
        logical_cursor += chunks_[i].size;
    }

    GGML_LOG_INFO(
        "[SYCL] Pinned pool grow_zone: zone %zu grown by %.1f MB "
        "(%zu new chunks, zone now %.1f MB)\n",
        zi, additional_capacity / (1024.0 * 1024.0), new_chunks_added, zones_[zi].size / (1024.0 * 1024.0));
    return true;
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
    for (const auto & c : runtime_chunks_) {
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
    return chunks_.size() + runtime_chunks_.size();
}

bool pinned_chunk_pool::grow(size_t min_size) {
    return grow_into(chunks_, min_size, false);
}

void * pinned_chunk_pool::allocate_from_chunks(std::vector<chunk> & chunks,
                                               size_t               size,
                                               size_t               alignment,
                                               bool                 runtime_pool) {
    size              = align_up(size, alignment);
    size_t guard_size = 0;
    if (const char * env = std::getenv("GGML_SYCL_HOST_CACHE_GUARD")) {
        if (std::atoi(env) != 0) {
            guard_size = 64;
        }
    }
    const size_t alloc_size = size + guard_size;
    if (pinned_trace_enabled()) {
        GGML_LOG_INFO(
            "[SYCL] pinned alloc request: size=%zu align=%zu guard=%zu alloc=%zu chunks=%zu used=%.1f GB mode=%s\n",
            size, alignment, guard_size, alloc_size, chunks.size(), total_allocated_ / (1024.0 * 1024.0 * 1024.0),
            runtime_pool ? "runtime" : "zone-base");
    }

    for (auto & c : chunks) {
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

    size_t new_chunk_size = std::max(chunk_size_, align_up(alloc_size, DEFAULT_ALIGNMENT));
    if (total_allocated_ + new_chunk_size > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }

    if (pinned_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] pinned pool grow: min=%zu chunk=%zu total=%.1f GB budget=%.1f GB mode=%s\n", alloc_size,
                      new_chunk_size, total_allocated_ / (1024.0 * 1024.0 * 1024.0),
                      budget_ / (1024.0 * 1024.0 * 1024.0), runtime_pool ? "runtime" : "zone-base");
    }
    if (!grow_into(chunks, alloc_size, runtime_pool)) {
        return nullptr;
    }

    auto & c = chunks.back();
    if (alloc_size > c.size) {
        GGML_LOG_ERROR(
            "[SYCL] Pinned pool allocation %zu MB exceeds chunk size %zu MB. "
            "For large allocations, use allocate_segmented() or pre-allocate more chunks.\n",
            alloc_size / (1024 * 1024), c.size / (1024 * 1024));
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

bool pinned_chunk_pool::deallocate_from_chunks(std::vector<chunk> & chunks, void * ptr) {
    for (auto & c : chunks) {
        char * base = static_cast<char *>(c.base);
        char * p    = static_cast<char *>(ptr);
        if (p >= base && p < base + c.size) {
            c.freed++;
            if (c.freed >= c.alloc_count) {
                c.used        = 0;
                c.freed       = 0;
                c.alloc_count = 0;
            }
            return true;
        }
    }
    return false;
}

bool pinned_chunk_pool::grow_into(std::vector<chunk> & chunks, size_t min_size, bool runtime_pool) {
    void * ptr        = nullptr;
    // Use chunk_size_ as the default, but allow larger chunks when the
    // allocation exceeds chunk_size_ (e.g., 615 MB reorder buffers for
    // MoE models).  Level Zero's ~11 GB per-allocation limit is the
    // real cap, not chunk_size_.
    size_t chunk_size = align_up(min_size, DEFAULT_ALIGNMENT);
    if (chunk_size < chunk_size_) {
        chunk_size = chunk_size_;
    }

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

    chunks.push_back({ ptr, chunk_size, 0, 0, 0 });
    total_allocated_ += chunk_size;

    GGML_LOG_INFO("[SYCL] Allocated pinned %s chunk %zu (size=%.1f MB, total=%.1f GB)\n",
                  runtime_pool ? "runtime" : "base", chunks.size(), chunk_size / (1024.0 * 1024.0),
                  total_allocated_ / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace ggml_sycl
