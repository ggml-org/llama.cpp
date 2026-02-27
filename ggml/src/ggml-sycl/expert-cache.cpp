//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "expert-cache.hpp"

#include "common.hpp"
#include "ggml-impl.h"
#include "unified-cache.hpp"

#include <algorithm>
#include <cstdlib>

namespace ggml_sycl {

// ---------------------------------------------------------------------------
// Env var: GGML_SYCL_EXPERT_CACHE_MB overrides default budget.
// ---------------------------------------------------------------------------
static size_t get_expert_cache_budget_override() {
    const char * env = std::getenv("GGML_SYCL_EXPERT_CACHE_MB");
    if (env) {
        int mb = std::atoi(env);
        if (mb > 0) {
            return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
        }
    }
    return 0;  // 0 = use default
}

// ---------------------------------------------------------------------------
// ExpertCache implementation
// ---------------------------------------------------------------------------

void ExpertCache::init(int device_id, size_t vram_budget_bytes, sycl::queue & q) {
    std::unique_lock lock(mutex_);

    if (pool_) {
        GGML_LOG_WARN("[EXPERT-CACHE] init() called on already-initialized cache, ignoring\n");
        return;
    }

    device_id_ = device_id;
    queue_     = &q;

    // Check env var override
    size_t override_bytes = get_expert_cache_budget_override();
    if (override_bytes > 0) {
        vram_budget_bytes = override_bytes;
        GGML_LOG_INFO("[EXPERT-CACHE] Budget overridden by GGML_SYCL_EXPERT_CACHE_MB: %zu MB\n",
                      static_cast<size_t>(vram_budget_bytes / (1024ULL * 1024ULL)));
    }

    pool_size_ = vram_budget_bytes;

    if (pool_size_ == 0) {
        GGML_LOG_WARN("[EXPERT-CACHE] Zero budget, expert cache disabled\n");
        return;
    }

    // Allocate contiguous VRAM pool
    try {
        pool_ = sycl::malloc_device(pool_size_, q);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[EXPERT-CACHE] Failed to allocate %.1f MB VRAM pool: %s\n",
                       pool_size_ / (1024.0 * 1024.0), e.what());
        pool_      = nullptr;
        pool_size_ = 0;
        return;
    }

    if (!pool_) {
        GGML_LOG_ERROR("[EXPERT-CACHE] sycl::malloc_device returned nullptr for %.1f MB\n",
                       pool_size_ / (1024.0 * 1024.0));
        pool_size_ = 0;
        return;
    }

    // Track budget via unified cache
    unified_cache_add_runtime_bytes(device_id_, pool_size_, runtime_category::EXPERT_CACHE);

    GGML_LOG_INFO("[EXPERT-CACHE] Initialized: pool=%.1f MB device=%d\n",
                  pool_size_ / (1024.0 * 1024.0), device_id_);
}

void ExpertCache::shutdown() {
    std::unique_lock lock(mutex_);

    if (!pool_) {
        return;
    }

    // Report final statistics
    uint64_t total    = hits_ + misses_;
    float    hr       = total > 0 ? 100.0f * hits_ / total : 0.0f;
    GGML_LOG_INFO("[EXPERT-CACHE] Shutting down: cached=%zu/%d hit_rate=%.1f%% pool=%.1f MB\n",
                  lookup_map_.size(), n_slots_, hr, pool_size_ / (1024.0 * 1024.0));

    // Release budget tracking
    unified_cache_sub_runtime_bytes(device_id_, pool_size_, runtime_category::EXPERT_CACHE);

    // Free contiguous pool
    if (queue_) {
        sycl::free(pool_, *queue_);
    }

    pool_      = nullptr;
    pool_size_ = 0;
    n_slots_   = 0;
    slot_size_ = 0;
    slots_.clear();
    lookup_map_.clear();
    hits_   = 0;
    misses_ = 0;
    global_token_ = 0;
}

ExpertCache::~ExpertCache() {
    shutdown();
}

void ExpertCache::register_expert(int layer_idx, int expert_idx,
                                  const void * host_ptr, size_t bytes) {
    std::unique_lock lock(mutex_);

    int64_t key = make_key(layer_idx, expert_idx);
    host_entries_[key] = { host_ptr, bytes };

    // On first registration, determine slot size.
    // All experts must be the same size (or smaller than slot_size_).
    size_t aligned = align_up(bytes);
    if (slot_size_ == 0) {
        slot_size_ = aligned;
        // Now we know how many slots fit in the pool
        if (pool_size_ > 0 && pool_) {
            n_slots_ = static_cast<int>(pool_size_ / slot_size_);
            slots_.resize(n_slots_);
            // Initialize device_ptr for each slot into the contiguous pool
            auto * base = static_cast<char *>(pool_);
            for (int i = 0; i < n_slots_; i++) {
                slots_[i].device_ptr = base + static_cast<size_t>(i) * slot_size_;
            }
            GGML_LOG_INFO("[EXPERT-CACHE] Slot size=%.1f KB, total_slots=%d\n",
                          slot_size_ / 1024.0, n_slots_);
        }
    }
}

ExpertLookup ExpertCache::lookup(int layer_idx, int expert_idx) const {
    std::shared_lock lock(mutex_);

    int64_t key = make_key(layer_idx, expert_idx);

    ExpertLookup result{};
    result.device_ptr = nullptr;
    result.host_ptr   = nullptr;
    result.is_cached  = false;

    // Always set host_ptr if registered
    auto host_it = host_entries_.find(key);
    if (host_it != host_entries_.end()) {
        result.host_ptr = const_cast<void *>(host_it->second.ptr);
    }

    // Check VRAM cache
    auto it = lookup_map_.find(key);
    if (it != lookup_map_.end()) {
        int slot_idx = it->second;
        result.device_ptr = slots_[slot_idx].device_ptr;
        result.is_cached  = true;
    }

    return result;
}

void * ExpertCache::ensure_cached(int layer_idx, int expert_idx, sycl::queue & q) {
    std::unique_lock lock(mutex_);

    int64_t key = make_key(layer_idx, expert_idx);

    // Check if already cached
    auto it = lookup_map_.find(key);
    if (it != lookup_map_.end()) {
        int slot_idx = it->second;
        // Update stats
        slots_[slot_idx].frequency++;
        slots_[slot_idx].last_access = global_token_;
        recompute_score(slots_[slot_idx], global_token_);
        hits_++;
        return slots_[slot_idx].device_ptr;
    }

    // Cache miss
    misses_++;

    // Get host source
    auto host_it = host_entries_.find(key);
    if (host_it == host_entries_.end()) {
        GGML_LOG_ERROR("[EXPERT-CACHE] Expert not registered: layer=%d expert=%d\n",
                       layer_idx, expert_idx);
        return nullptr;
    }

    const void * host_src  = host_it->second.ptr;
    size_t       src_bytes = host_it->second.size;

    if (n_slots_ == 0) {
        // Pool not initialized (maybe zero budget); return host ptr as fallback
        return const_cast<void *>(host_src);
    }

    // Find a free slot or evict
    int target_slot = -1;

    // First: look for an empty slot
    for (int i = 0; i < n_slots_; i++) {
        if (slots_[i].layer_idx == -1) {
            target_slot = i;
            break;
        }
    }

    // No free slot: evict lowest score
    if (target_slot == -1) {
        target_slot = find_eviction_candidate();
        if (target_slot == -1) {
            // Should not happen if n_slots_ > 0, but be safe
            return const_cast<void *>(host_src);
        }

        // Remove old entry from lookup map
        int64_t old_key = make_key(slots_[target_slot].layer_idx,
                                   slots_[target_slot].expert_idx);
        lookup_map_.erase(old_key);
    }

    // Synchronous H2D copy into slot
    q.memcpy(slots_[target_slot].device_ptr, host_src, src_bytes).wait();

    // Update slot metadata
    slots_[target_slot].layer_idx   = layer_idx;
    slots_[target_slot].expert_idx  = expert_idx;
    slots_[target_slot].size_bytes  = src_bytes;
    slots_[target_slot].frequency   = 1;
    slots_[target_slot].last_access = global_token_;
    recompute_score(slots_[target_slot], global_token_);

    // Register in lookup map
    lookup_map_[key] = target_slot;

    return slots_[target_slot].device_ptr;
}

sycl::event ExpertCache::prefetch_async(int layer_idx, int expert_idx, sycl::queue & q) {
    std::unique_lock lock(mutex_);

    int64_t key = make_key(layer_idx, expert_idx);

    // Already cached? Return completed event.
    auto it = lookup_map_.find(key);
    if (it != lookup_map_.end()) {
        return sycl::event();
    }

    // Get host source
    auto host_it = host_entries_.find(key);
    if (host_it == host_entries_.end()) {
        return sycl::event();
    }

    const void * host_src  = host_it->second.ptr;
    size_t       src_bytes = host_it->second.size;

    if (n_slots_ == 0) {
        return sycl::event();
    }

    // Find a free slot or evict
    int target_slot = -1;
    for (int i = 0; i < n_slots_; i++) {
        if (slots_[i].layer_idx == -1) {
            target_slot = i;
            break;
        }
    }

    if (target_slot == -1) {
        target_slot = find_eviction_candidate();
        if (target_slot == -1) {
            return sycl::event();
        }
        int64_t old_key = make_key(slots_[target_slot].layer_idx,
                                   slots_[target_slot].expert_idx);
        lookup_map_.erase(old_key);
    }

    // Async H2D copy (non-blocking)
    sycl::event evt = q.memcpy(slots_[target_slot].device_ptr, host_src, src_bytes);

    // Update slot metadata (prefetched entries start with frequency=0)
    slots_[target_slot].layer_idx   = layer_idx;
    slots_[target_slot].expert_idx  = expert_idx;
    slots_[target_slot].size_bytes  = src_bytes;
    slots_[target_slot].frequency   = 0;
    slots_[target_slot].last_access = global_token_;
    recompute_score(slots_[target_slot], global_token_);

    lookup_map_[key] = target_slot;

    return evt;
}

void ExpertCache::update_score(int layer_idx, int expert_idx, uint64_t token_counter) {
    std::unique_lock lock(mutex_);

    global_token_ = token_counter;

    int64_t key = make_key(layer_idx, expert_idx);
    auto it = lookup_map_.find(key);
    if (it == lookup_map_.end()) {
        return;
    }

    int slot_idx = it->second;
    slots_[slot_idx].frequency++;
    slots_[slot_idx].last_access = token_counter;
    recompute_score(slots_[slot_idx], token_counter);
}

size_t ExpertCache::cached_count() const {
    std::shared_lock lock(mutex_);
    return lookup_map_.size();
}

size_t ExpertCache::total_slots() const {
    std::shared_lock lock(mutex_);
    return static_cast<size_t>(n_slots_);
}

float ExpertCache::hit_rate() const {
    std::shared_lock lock(mutex_);
    uint64_t total = hits_ + misses_;
    if (total == 0) {
        return 0.0f;
    }
    return static_cast<float>(hits_) / static_cast<float>(total);
}

size_t ExpertCache::vram_budget() const {
    std::shared_lock lock(mutex_);
    return pool_size_;
}

size_t ExpertCache::vram_used() const {
    std::shared_lock lock(mutex_);
    return lookup_map_.size() * slot_size_;
}

size_t ExpertCache::cache_hits() const {
    std::shared_lock lock(mutex_);
    return static_cast<size_t>(hits_);
}

size_t ExpertCache::cache_misses() const {
    std::shared_lock lock(mutex_);
    return static_cast<size_t>(misses_);
}

size_t ExpertCache::entries_count() const {
    std::shared_lock lock(mutex_);
    return lookup_map_.size();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void ExpertCache::recompute_score(ExpertSlot & slot, uint64_t current_token) const {
    // score = alpha * frequency + beta * (token_counter - last_access)
    // alpha = 1.0, beta = -0.01
    // Higher score = more valuable (less likely to evict)
    constexpr float alpha = 1.0f;
    constexpr float beta  = -0.01f;

    float freq_term    = alpha * static_cast<float>(slot.frequency);
    float recency_term = beta * static_cast<float>(current_token - slot.last_access);
    slot.score = freq_term + recency_term;
}

int ExpertCache::find_eviction_candidate() const {
    if (n_slots_ == 0) {
        return -1;
    }

    int   best_idx   = -1;
    float best_score = std::numeric_limits<float>::max();

    for (int i = 0; i < n_slots_; i++) {
        if (slots_[i].layer_idx == -1) {
            continue;  // Empty slot, skip
        }
        if (slots_[i].score < best_score) {
            best_score = slots_[i].score;
            best_idx   = i;
        }
    }

    return best_idx;
}

int64_t ExpertCache::make_key(int layer, int expert) const {
    return static_cast<int64_t>(layer) * MAX_EXPERTS_PER_LAYER + expert;
}

} // namespace ggml_sycl
