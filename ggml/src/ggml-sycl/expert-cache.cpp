//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "expert-cache.hpp"

#include "common.hpp"
#include "ggml-impl.h"
#include "pinned-pool.hpp"

namespace ggml_sycl {

expert_cache::expert_cache(sycl::queue & compute_queue, pinned_chunk_pool & pool, size_t vram_budget) :
    compute_queue_(compute_queue),
    copy_queue_(compute_queue.get_context(), compute_queue.get_device()),
    pinned_pool_(pool),
    device_id_(ggml_sycl_get_device_id_from_queue(compute_queue)),
    vram_budget_(vram_budget) {
    GGML_LOG_INFO("[SYCL] Expert cache created with %.1f GB VRAM budget\n", vram_budget / (1024.0 * 1024.0 * 1024.0));
}

expert_cache::~expert_cache() {
    // Wait for any pending async prefetch operations to complete
    // before freeing VRAM buffers to avoid undefined behavior
    copy_queue_.wait();

    // Free all VRAM allocations
    for (auto & [key, entry] : vram_entries_) {
        if (entry.vram_ptr) {
            ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, entry.size);
            sycl::free(entry.vram_ptr, compute_queue_);
        }
    }

    // Report final statistics
    size_t total    = cache_hits_ + cache_misses_;
    float  hit_rate = total > 0 ? 100.0f * cache_hits_ / total : 0.0f;
    GGML_LOG_INFO("[SYCL] Expert cache destroyed (hits=%zu, misses=%zu, hit_rate=%.1f%%)\n", cache_hits_, cache_misses_,
                  hit_rate);
}

void expert_cache::register_expert(int layer, int expert_id, void * host_ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    expert_key                  key{ layer, expert_id };
    host_entries_[key] = { host_ptr, size };
}

void * expert_cache::get_expert(int layer, int expert_id, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    expert_key                  key{ layer, expert_id };

    // Check VRAM cache first
    auto it = vram_entries_.find(key);
    if (it != vram_entries_.end()) {
        // Cache hit - update LRU and frequency
        it->second.last_access = time_++;
        it->second.access_count++;
        cache_hits_++;
        return it->second.vram_ptr;
    }

    // Cache miss
    cache_misses_++;

    // Get host pointer
    auto host_it = host_entries_.find(key);
    if (host_it == host_entries_.end()) {
        GGML_LOG_ERROR("[SYCL] Expert not registered: layer=%d expert=%d\n", layer, expert_id);
        return nullptr;
    }

    void * host_ptr = host_it->second.host_ptr;

    // Allocate VRAM (may evict)
    void * vram_ptr = allocate_vram(size);
    if (!vram_ptr) {
        // VRAM exhausted even after eviction - return host pointer for direct access
        // This is a fallback; performance will degrade but correctness maintained
        return host_ptr;
    }

    // Copy from pinned host to VRAM (synchronous for get_expert)
    compute_queue_.memcpy(vram_ptr, host_ptr, size).wait();

    // Record new entry
    vram_entries_[key] = { vram_ptr, size, time_++, 1 };
    return vram_ptr;
}

bool expert_cache::is_cached_in_vram(int layer, int expert_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vram_entries_.find({ layer, expert_id }) != vram_entries_.end();
}

void expert_cache::prefetch(const std::vector<expert_key> & experts, size_t expert_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto & key : experts) {
        // Skip if already cached
        if (vram_entries_.find(key) != vram_entries_.end()) {
            continue;
        }

        // Skip if not registered
        auto host_it = host_entries_.find(key);
        if (host_it == host_entries_.end()) {
            continue;
        }

        // Try to allocate VRAM
        void * vram_ptr = allocate_vram(expert_size);
        if (!vram_ptr) {
            break;  // VRAM full, stop prefetching
        }

        // Async copy using separate queue
        copy_queue_.memcpy(vram_ptr, host_it->second.host_ptr, expert_size);

        // Record entry (prefetched entries start with access_count=0)
        vram_entries_[key] = { vram_ptr, expert_size, time_++, 0 };
    }
    // Don't wait - let transfers overlap with compute
}

void expert_cache::wait_prefetch() {
    copy_queue_.wait();
}

size_t expert_cache::entries_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vram_entries_.size();
}

float expert_cache::compute_score(const vram_entry & e) const {
    // Combined LRU + frequency scoring
    // Higher score = more likely to keep
    //
    // recency: inversely proportional to time since last access
    // frequency: number of accesses
    //
    // Weight: 30% recency, 70% frequency
    // This favors frequently-used experts over recently-used-once experts
    float recency   = 1.0f / static_cast<float>(time_ - e.last_access + 1);
    float frequency = static_cast<float>(e.access_count);
    return 0.3f * recency + 0.7f * frequency;
}

void * expert_cache::allocate_vram(size_t size) {
    // Evict until we have space
    while (vram_used_ + size > vram_budget_) {
        if (vram_entries_.empty()) {
            return nullptr;  // Nothing to evict, budget exhausted
        }
        evict_lowest_score();
    }

    // Allocate device memory
    void * ptr = nullptr;
    try {
        ggml_sycl::unified_cache_add_runtime_bytes(device_id_, size);
        ptr = ggml_sycl_malloc_device(size, compute_queue_, "expert_cache_vram");
    } catch (const sycl::exception & e) {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, size);
        GGML_LOG_ERROR("[SYCL] Expert cache VRAM allocation failed: %s\n", e.what());
        return nullptr;
    }

    if (!ptr) {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, size);
        return nullptr;
    }
    vram_used_ += size;
    return ptr;
}

void expert_cache::evict_lowest_score() {
    if (vram_entries_.empty()) {
        return;
    }

    // Find entry with lowest score
    auto  worst       = vram_entries_.begin();
    float worst_score = compute_score(worst->second);

    for (auto it = vram_entries_.begin(); it != vram_entries_.end(); ++it) {
        float score = compute_score(it->second);
        if (score < worst_score) {
            worst       = it;
            worst_score = score;
        }
    }

    // Free VRAM and remove entry
    ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, worst->second.size);
    sycl::free(worst->second.vram_ptr, compute_queue_);
    vram_used_ -= worst->second.size;
    vram_entries_.erase(worst);
}

}  // namespace ggml_sycl
