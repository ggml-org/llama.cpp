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
#include <cmath>
#include <cstdlib>
#include <limits>

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
// Env var: GGML_SYCL_EXPERT_CACHE_WARMUP=N overrides warmup token count.
// ---------------------------------------------------------------------------
static int get_warmup_token_count() {
    const char * env = std::getenv("GGML_SYCL_EXPERT_CACHE_WARMUP");
    if (env) {
        int n = std::atoi(env);
        if (n >= 0) {
            return n;  // 0 = disable warmup
        }
    }
    return 32;  // Default: profile first 32 tokens
}

// ---------------------------------------------------------------------------
// ExpertCache implementation
// ---------------------------------------------------------------------------

size_t ExpertCache::default_budget(int device_id) {
    // 50% of remaining VRAM after dense layers (weights + runtime allocations).
    size_t available = unified_cache_available_for_compute(device_id);
    return available / 2;
}

void ExpertCache::init(int device_id, size_t vram_budget_bytes, sycl::queue & q) {
    std::unique_lock lock(mutex_);

    if (pool_) {
        GGML_LOG_WARN("[EXPERT-CACHE] init() called on already-initialized cache, ignoring\n");
        return;
    }

    device_id_ = device_id;
    queue_     = &q;

    // Check env var override first
    size_t override_bytes = get_expert_cache_budget_override();
    if (override_bytes > 0) {
        vram_budget_bytes = override_bytes;
        GGML_LOG_INFO("[EXPERT-CACHE] Budget overridden by GGML_SYCL_EXPERT_CACHE_MB: %zu MB\n",
                      static_cast<size_t>(vram_budget_bytes / (1024ULL * 1024ULL)));
    } else if (vram_budget_bytes == 0) {
        // Use default: 50% of remaining VRAM after dense layers
        vram_budget_bytes = default_budget(device_id);
        GGML_LOG_INFO("[EXPERT-CACHE] Using default budget (50%% of remaining VRAM): %.1f MB\n",
                      vram_budget_bytes / (1024.0 * 1024.0));
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
        GGML_LOG_ERROR("[EXPERT-CACHE] Failed to allocate %.1f MB VRAM pool: %s\n", pool_size_ / (1024.0 * 1024.0),
                       e.what());
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

    // Read warm-start configuration
    warmup_target_ = get_warmup_token_count();
    warmup_active_ = (warmup_target_ > 0);
    warmup_done_   = !warmup_active_;

    // Read eviction alpha (LFU vs recency blend)
    const char * alpha_env = std::getenv("GGML_SYCL_EXPERT_EVICT_ALPHA");
    if (alpha_env) {
        float a = static_cast<float>(std::atof(alpha_env));
        if (a >= 0.0f && a <= 1.0f) {
            evict_alpha_ = a;
        }
    }

    GGML_LOG_INFO(
        "[EXPERT-CACHE] Initialized: pool=%.1f MB device=%d warmup=%d tokens "
        "evict_alpha=%.2f (%.0f%% freq, %.0f%% recency)\n",
        pool_size_ / (1024.0 * 1024.0), device_id_, warmup_target_, evict_alpha_, evict_alpha_ * 100.0f,
        (1.0f - evict_alpha_) * 100.0f);
}

void ExpertCache::shutdown() {
    std::unique_lock lock(mutex_);

    if (!pool_) {
        return;
    }

    // Report final statistics
    uint64_t total = hits_ + misses_;
    float    hr    = total > 0 ? 100.0f * hits_ / total : 0.0f;
    GGML_LOG_INFO("[EXPERT-CACHE] Shutting down: cached=%zu/%d hit_rate=%.1f%% pool=%.1f MB\n", lookup_map_.size(),
                  n_slots_, hr, pool_size_ / (1024.0 * 1024.0));

    // Skip SYCL resource cleanup if runtime is shutting down
    // (static destruction order fiasco — SYCL context may be invalid)
    if (!ggml_sycl::ggml_sycl_is_shutting_down()) {
        unified_cache_sub_runtime_bytes(device_id_, pool_size_, runtime_category::EXPERT_CACHE);

        if (queue_) {
            try {
                sycl::free(pool_, *queue_);
            } catch (...) {
                // SYCL runtime may be partially torn down
            }
        }
    }

    pool_      = nullptr;
    pool_size_ = 0;
    n_slots_   = 0;
    slot_size_ = 0;
    slots_.clear();
    lookup_map_.clear();
    hits_          = 0;
    misses_        = 0;
    global_token_  = 0;
    evict_alpha_   = 0.7f;
    current_layer_ = -1;
    co_activation_.clear();
    warmup_freq_.clear();
    warmup_tokens_  = 0;
    warmup_active_  = false;
    warmup_done_    = false;
    rolling_idx_    = 0;
    rolling_count_  = 0;
    last_log_token_ = 0;
}

ExpertCache::~ExpertCache() {
    shutdown();
}

void ExpertCache::register_expert(int layer_idx, int expert_idx, const void * host_ptr, size_t bytes) {
    std::unique_lock lock(mutex_);

    int64_t key        = make_key(layer_idx, expert_idx);
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
            GGML_LOG_INFO("[EXPERT-CACHE] Slot size=%.1f KB, total_slots=%d\n", slot_size_ / 1024.0, n_slots_);
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
        int slot_idx      = it->second;
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
        auto & slot = slots_[it->second];
        // Update stats (Least-Stale: only last_access matters for eviction)
        slot.frequency++;
        slot.last_access = global_token_;
        hits_++;
        return slot.device_ptr;
    }

    // Cache miss
    misses_++;

    // Get host source
    auto host_it = host_entries_.find(key);
    if (host_it == host_entries_.end()) {
        GGML_LOG_ERROR("[EXPERT-CACHE] Expert not registered: layer=%d expert=%d\n", layer_idx, expert_idx);
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
        int64_t old_key = make_key(slots_[target_slot].layer_idx, slots_[target_slot].expert_idx);
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
        int64_t old_key = make_key(slots_[target_slot].layer_idx, slots_[target_slot].expert_idx);
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

    lookup_map_[key] = target_slot;

    return evt;
}

PrefetchResult ExpertCache::prefetch_batch_async(const std::vector<std::pair<int, int>> & experts,
                                                 sycl::queue &                            dma_queue) {
    PrefetchResult result;
    result.n_submitted = 0;

    if (n_slots_ == 0) {
        return result;
    }

    // Collect DMA work items under lock, then submit outside lock.
    struct DmaWork {
        expert_key   ek;
        void *       dst;  // device slot pointer
        const void * src;  // host source pointer
        size_t       bytes;
    };

    std::vector<DmaWork> work;
    work.reserve(experts.size());

    // ---------------------------------------------------------------
    // Phase 1: exclusive lock — plan evictions, reserve slots, update metadata.
    // ---------------------------------------------------------------
    {
        std::unique_lock lock(mutex_);

        for (const auto & [layer_idx, expert_idx] : experts) {
            int64_t key = make_key(layer_idx, expert_idx);

            // Already cached? Skip.
            if (lookup_map_.find(key) != lookup_map_.end()) {
                continue;
            }

            // Get host source.
            auto host_it = host_entries_.find(key);
            if (host_it == host_entries_.end()) {
                continue;
            }

            const void * host_src  = host_it->second.ptr;
            size_t       src_bytes = host_it->second.size;

            // Find a free slot or evict.
            int target_slot = find_empty_slot();
            if (target_slot == -1) {
                target_slot = find_eviction_candidate();
                if (target_slot == -1) {
                    continue;  // No slots available
                }
                // Remove old entry from lookup map.
                int64_t old_key = make_key(slots_[target_slot].layer_idx, slots_[target_slot].expert_idx);
                lookup_map_.erase(old_key);
            }

            // Reserve the slot: update metadata NOW so concurrent lookups
            // see the slot as occupied (prevents double-allocation).
            slots_[target_slot].layer_idx   = layer_idx;
            slots_[target_slot].expert_idx  = expert_idx;
            slots_[target_slot].size_bytes  = src_bytes;
            slots_[target_slot].frequency   = 0;
            slots_[target_slot].last_access = global_token_;

            lookup_map_[key] = target_slot;

            // Record DMA work for Phase 2.
            DmaWork w;
            w.ek    = { layer_idx, expert_idx };
            w.dst   = slots_[target_slot].device_ptr;
            w.src   = host_src;
            w.bytes = src_bytes;
            work.push_back(w);

            misses_++;
        }
    }
    // ---------------------------------------------------------------
    // Phase 2: lock released — submit all H2D DMAs as non-blocking ops.
    // This avoids blocking concurrent lookup() calls during DMA submission.
    // ---------------------------------------------------------------

    result.events.reserve(work.size());
    for (const auto & w : work) {
        sycl::event ev = dma_queue.memcpy(w.dst, w.src, w.bytes);
        result.events.push_back({ w.ek, ev });
        result.n_submitted++;
    }

    return result;
}

void ExpertCache::update_score(int layer_idx, int expert_idx, uint64_t token_counter) {
    std::unique_lock lock(mutex_);

    global_token_ = token_counter;

    int64_t key = make_key(layer_idx, expert_idx);
    auto    it  = lookup_map_.find(key);
    if (it == lookup_map_.end()) {
        return;
    }

    auto & slot      = slots_[it->second];
    slot.last_access = token_counter;
    slot.frequency++;
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
    uint64_t         total = hits_ + misses_;
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

float ExpertCache::rolling_hit_rate() const {
    std::shared_lock lock(mutex_);
    if (rolling_count_ == 0) {
        return 0.0f;
    }
    int total_hits   = 0;
    int total_misses = 0;
    int n            = std::min(rolling_count_, ROLLING_WINDOW);
    for (int i = 0; i < n; i++) {
        total_hits += rolling_buf_[i].hits;
        total_misses += rolling_buf_[i].misses;
    }
    int total = total_hits + total_misses;
    return total > 0 ? static_cast<float>(total_hits) / static_cast<float>(total) : 0.0f;
}

bool ExpertCache::is_warmup_phase() const {
    std::shared_lock lock(mutex_);
    return warmup_active_;
}

void ExpertCache::record_access_batch(int         current_layer,
                                      const int * expert_ids,
                                      int         n_experts,
                                      uint64_t    token_counter) {
    std::unique_lock lock(mutex_);

    current_layer_ = current_layer;
    global_token_  = token_counter;

    // Track per-token hits/misses for rolling window
    int token_hits   = 0;
    int token_misses = 0;

    for (int i = 0; i < n_experts; i++) {
        int64_t key = make_key(current_layer, expert_ids[i]);

        // Track warmup frequency
        if (warmup_active_) {
            warmup_freq_[key]++;
        }

        // Track hit/miss for rolling stats
        auto it = lookup_map_.find(key);
        if (it != lookup_map_.end()) {
            token_hits++;
        } else {
            token_misses++;
        }

        // Track co-activation: for each pair (i, j) where i < j,
        // record that these experts were activated together.
        for (int j = i + 1; j < n_experts; j++) {
            int64_t key_j    = make_key(current_layer, expert_ids[j]);
            // Encode pair as sorted (min, max) key
            int64_t pair_key = (std::min(key, key_j) << 32) | (std::max(key, key_j) & 0xFFFFFFFF);
            co_activation_[pair_key]++;
        }
    }

    // Update rolling window
    rolling_buf_[rolling_idx_] = { token_hits, token_misses };
    rolling_idx_               = (rolling_idx_ + 1) % ROLLING_WINDOW;
    if (rolling_count_ < ROLLING_WINDOW) {
        rolling_count_++;
    }

    // Warmup phase tracking
    if (warmup_active_) {
        warmup_tokens_++;
        if (warmup_tokens_ >= warmup_target_) {
            finish_warmup_locked();  // mutex_ already held by caller
            return;
        }
    }

    // Periodic logging
    maybe_log_stats();
}

void ExpertCache::finish_warmup() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    finish_warmup_locked();
}

void ExpertCache::finish_warmup_locked() {
    if (!warmup_active_ || warmup_done_) {
        return;
    }

    warmup_active_ = false;
    warmup_done_   = true;

    if (!queue_ || n_slots_ == 0) {
        GGML_LOG_INFO("[EXPERT-CACHE] Warm-start: no pool available, skipping bulk-load\n");
        warmup_freq_.clear();
        return;
    }

    // Sort experts by frequency (descending) and bulk-load top-N into VRAM
    std::vector<std::pair<int64_t, uint32_t>> sorted_freq(warmup_freq_.begin(), warmup_freq_.end());
    std::sort(sorted_freq.begin(), sorted_freq.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });

    int n_loaded = 0;
    for (const auto & [key, freq] : sorted_freq) {
        if (n_loaded >= n_slots_) {
            break;  // Pool full
        }

        // Already cached?
        if (lookup_map_.find(key) != lookup_map_.end()) {
            continue;
        }

        // Find host source
        auto host_it = host_entries_.find(key);
        if (host_it == host_entries_.end()) {
            continue;
        }

        // Find a free slot
        int target_slot = -1;
        for (int i = 0; i < n_slots_; i++) {
            if (slots_[i].layer_idx == -1) {
                target_slot = i;
                break;
            }
        }
        if (target_slot == -1) {
            break;  // No more free slots
        }

        // H2D copy
        const void * host_src  = host_it->second.ptr;
        size_t       src_bytes = host_it->second.size;
        queue_->memcpy(slots_[target_slot].device_ptr, host_src, src_bytes).wait();

        // Decode layer/expert from key
        int layer_idx  = static_cast<int>(key / MAX_EXPERTS_PER_LAYER);
        int expert_idx = static_cast<int>(key % MAX_EXPERTS_PER_LAYER);

        slots_[target_slot].layer_idx   = layer_idx;
        slots_[target_slot].expert_idx  = expert_idx;
        slots_[target_slot].size_bytes  = src_bytes;
        slots_[target_slot].frequency   = freq;
        slots_[target_slot].last_access = global_token_;

        lookup_map_[key] = target_slot;
        n_loaded++;
    }

    GGML_LOG_INFO("[EXPERT-CACHE] Warm-start complete: loaded %d/%d experts after %d tokens\n", n_loaded, n_slots_,
                  warmup_tokens_);

    warmup_freq_.clear();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

int ExpertCache::find_eviction_candidate() const {
    // Hybrid LFU+Staleness eviction policy.
    // Combined score = alpha * log2(1 + frequency) + (1 - alpha) * recency
    // where recency = 1.0 / (1 + staleness).
    // Evict the slot with the LOWEST combined score.
    // Tiebreaker: prefer evicting experts from layers furthest from current compute.
    int   best_slot  = -1;
    float best_score = std::numeric_limits<float>::max();

    for (int i = 0; i < n_slots_; i++) {
        if (slots_[i].layer_idx < 0) {
            continue;  // Empty slot
        }

        const uint64_t staleness       = global_token_ - slots_[i].last_access;
        const float    recency_score   = 1.0f / (1.0f + static_cast<float>(staleness));
        const float    frequency_score = std::log2(1.0f + static_cast<float>(slots_[i].frequency));
        const float    combined        = evict_alpha_ * frequency_score + (1.0f - evict_alpha_) * recency_score;

        if (combined < best_score) {
            best_score = combined;
            best_slot  = i;
        } else if (combined == best_score && best_slot >= 0) {
            // Tiebreaker: prefer slot from a layer further from current compute layer
            const int dist_new  = std::abs(slots_[i].layer_idx - current_layer_);
            const int dist_best = std::abs(slots_[best_slot].layer_idx - current_layer_);
            if (dist_new > dist_best) {
                best_slot  = i;
                best_score = combined;
            }
        }
    }

    return best_slot;
}

void ExpertCache::maybe_log_stats() {
    // Called with mutex_ held.
    if (global_token_ - last_log_token_ < log_interval_) {
        return;
    }
    last_log_token_ = global_token_;

    uint64_t total = hits_ + misses_;
    float    hr    = total > 0 ? 100.0f * static_cast<float>(hits_) / static_cast<float>(total) : 0.0f;

    // Compute rolling hit rate (inline, mutex already held)
    float rhr = 0.0f;
    if (rolling_count_ > 0) {
        int rh = 0, rm = 0;
        int n = std::min(rolling_count_, ROLLING_WINDOW);
        for (int i = 0; i < n; i++) {
            rh += rolling_buf_[i].hits;
            rm += rolling_buf_[i].misses;
        }
        int rt = rh + rm;
        rhr    = rt > 0 ? 100.0f * static_cast<float>(rh) / static_cast<float>(rt) : 0.0f;
    }

    GGML_LOG_INFO("[EXPERT-CACHE] hit_rate=%.1f%% rolling=%.1f%% cached=%zu/%d pool=%.1f MB\n", hr, rhr,
                  lookup_map_.size(), n_slots_, pool_size_ / (1024.0 * 1024.0));
}

int ExpertCache::find_empty_slot() const {
    // Called with mutex_ held (shared or unique).
    for (int i = 0; i < n_slots_; i++) {
        if (slots_[i].layer_idx == -1) {
            return i;
        }
    }
    return -1;
}

int64_t ExpertCache::make_key(int layer, int expert) const {
    return static_cast<int64_t>(layer) * MAX_EXPERTS_PER_LAYER + expert;
}

// ---------------------------------------------------------------------------
// PinnedBufferPool implementation
// ---------------------------------------------------------------------------

PinnedBufferPool::~PinnedBufferPool() {
    shutdown();
}

void PinnedBufferPool::init(sycl::queue & q, int device_id, size_t max_experts, size_t act_dim, size_t out_dim) {
    if (is_initialized()) {
        return;
    }

    device_id_   = device_id;
    act_stride_  = act_dim;
    out_stride_  = out_dim;
    max_experts_ = max_experts;

    const size_t act_bytes = max_experts * act_dim * sizeof(float);
    const size_t out_bytes = max_experts * out_dim * sizeof(float);

    // Allocate activation pool via unified_alloc with pinned host constraint
    alloc_request req_act;
    req_act.queue                               = &q;
    req_act.device                              = device_id;
    req_act.size                                = act_bytes;
    req_act.intent.role                         = alloc_role::EXPERT_STAGING;
    req_act.intent.category                     = runtime_category::EXPERT_CACHE;
    req_act.intent.cohort_id                    = "moe_act_pool";
    req_act.intent.constraints.must_host_pinned = true;

    if (!unified_alloc(req_act, &act_alloc_)) {
        GGML_LOG_WARN("[MOE-POOL] Failed to allocate activation pool (%zu bytes)\n", act_bytes);
        return;
    }
    act_pool_ = static_cast<float *>(act_alloc_.ptr);

    // Allocate output pool
    alloc_request req_out    = req_act;
    req_out.size             = out_bytes;
    req_out.intent.cohort_id = "moe_out_pool";

    if (!unified_alloc(req_out, &out_alloc_)) {
        GGML_LOG_WARN("[MOE-POOL] Failed to allocate output pool (%zu bytes)\n", out_bytes);
        unified_free(act_alloc_);
        act_alloc_ = {};
        act_pool_  = nullptr;
        return;
    }
    out_pool_ = static_cast<float *>(out_alloc_.ptr);

    GGML_LOG_INFO("[MOE-POOL] Pinned buffer pool: act=%zu KB, out=%zu KB, max_experts=%zu\n", act_bytes / 1024,
                  out_bytes / 1024, max_experts);
}

void PinnedBufferPool::shutdown() {
    if (act_alloc_.ptr) {
        unified_free(act_alloc_);
        act_alloc_ = {};
        act_pool_  = nullptr;
    }
    if (out_alloc_.ptr) {
        unified_free(out_alloc_);
        out_alloc_ = {};
        out_pool_  = nullptr;
    }
}

PinnedBufferPool::BufferPair PinnedBufferPool::acquire(size_t n_experts) {
    GGML_ASSERT(n_experts <= max_experts_ && "Expert count exceeds pool capacity");
    GGML_ASSERT(act_pool_ && out_pool_ && "Pool not initialized");
    return { act_pool_, out_pool_ };
}

void PinnedBufferPool::release(BufferPair) {
    // Zero output buffer for next use (CPU kernels write partial results).
    if (out_pool_) {
        std::memset(out_pool_, 0, max_experts_ * out_stride_ * sizeof(float));
    }
}

}  // namespace ggml_sycl
