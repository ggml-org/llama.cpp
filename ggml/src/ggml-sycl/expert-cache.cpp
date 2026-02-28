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

    // Read warm-start configuration
    warmup_target_ = get_warmup_token_count();
    warmup_active_ = (warmup_target_ > 0);
    warmup_done_   = !warmup_active_;

    GGML_LOG_INFO("[EXPERT-CACHE] Initialized: pool=%.1f MB device=%d warmup=%d tokens\n",
                  pool_size_ / (1024.0 * 1024.0), device_id_, warmup_target_);
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
    hits_           = 0;
    misses_         = 0;
    global_token_   = 0;
    current_layer_  = -1;
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

float ExpertCache::rolling_hit_rate() const {
    std::shared_lock lock(mutex_);
    if (rolling_count_ == 0) {
        return 0.0f;
    }
    int total_hits   = 0;
    int total_misses = 0;
    int n = std::min(rolling_count_, ROLLING_WINDOW);
    for (int i = 0; i < n; i++) {
        total_hits   += rolling_buf_[i].hits;
        total_misses += rolling_buf_[i].misses;
    }
    int total = total_hits + total_misses;
    return total > 0 ? static_cast<float>(total_hits) / static_cast<float>(total) : 0.0f;
}

bool ExpertCache::is_warmup_phase() const {
    std::shared_lock lock(mutex_);
    return warmup_active_;
}

void ExpertCache::record_access_batch(int current_layer, const int * expert_ids,
                                       int n_experts, uint64_t token_counter) {
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
            int64_t key_j = make_key(current_layer, expert_ids[j]);
            // Encode pair as sorted (min, max) key
            int64_t pair_key = (std::min(key, key_j) << 32) | (std::max(key, key_j) & 0xFFFFFFFF);
            co_activation_[pair_key]++;
        }
    }

    // Update rolling window
    rolling_buf_[rolling_idx_] = { token_hits, token_misses };
    rolling_idx_ = (rolling_idx_ + 1) % ROLLING_WINDOW;
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
    std::vector<std::pair<int64_t, uint32_t>> sorted_freq(warmup_freq_.begin(),
                                                           warmup_freq_.end());
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
        recompute_score(slots_[target_slot], global_token_);

        lookup_map_[key] = target_slot;
        n_loaded++;
    }

    GGML_LOG_INFO("[EXPERT-CACHE] Warm-start complete: loaded %d/%d experts after %d tokens\n",
                  n_loaded, n_slots_, warmup_tokens_);

    warmup_freq_.clear();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void ExpertCache::recompute_score(ExpertSlot & slot, uint64_t current_token) const {
    // Enhanced scoring formula:
    //   score = alpha*frequency + beta*recency + gamma*layer_distance + delta*co_activation
    //
    // alpha = 1.0   : access frequency (higher = more valuable)
    // beta  = -0.01 : recency penalty (older = less valuable)
    // gamma = -0.5  : layer-distance penalty (far from current layer = less valuable)
    // delta = 0.1   : co-activation bonus (frequently paired experts stay together)
    //
    // Higher score = more valuable (less likely to evict)
    constexpr float alpha = 1.0f;
    constexpr float beta  = -0.01f;
    constexpr float gamma = -0.5f;
    constexpr float delta = 0.1f;

    float freq_term    = alpha * static_cast<float>(slot.frequency);
    float recency_term = beta * static_cast<float>(current_token - slot.last_access);

    // Layer-distance penalty: experts from layers far from the current
    // processing layer are less likely to be needed soon.
    float layer_dist_term = 0.0f;
    if (current_layer_ >= 0 && slot.layer_idx >= 0) {
        int dist = std::abs(slot.layer_idx - current_layer_);
        layer_dist_term = gamma * static_cast<float>(dist);
    }

    // Co-activation bonus: check if this expert is frequently activated
    // with any currently-cached expert in the same layer.
    float co_act_term = 0.0f;
    if (!co_activation_.empty()) {
        int64_t slot_key = make_key(slot.layer_idx, slot.expert_idx);
        // O(n) scan over cached experts. Consider per-layer index if n_slots_ > 1000.
        for (const auto & [cached_key, slot_idx] : lookup_map_) {
            if (slot_idx == -1) continue;
            const auto & other = slots_[slot_idx];
            if (other.layer_idx != slot.layer_idx) continue;
            if (other.layer_idx == -1) continue;

            int64_t other_key = make_key(other.layer_idx, other.expert_idx);
            if (other_key == slot_key) continue;

            int64_t pair_key = (std::min(slot_key, other_key) << 32)
                               | (std::max(slot_key, other_key) & 0xFFFFFFFF);
            auto co_it = co_activation_.find(pair_key);
            if (co_it != co_activation_.end()) {
                co_act_term += delta * static_cast<float>(co_it->second);
            }
        }
    }

    slot.score = freq_term + recency_term + layer_dist_term + co_act_term;
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
        rhr = rt > 0 ? 100.0f * static_cast<float>(rh) / static_cast<float>(rt) : 0.0f;
    }

    GGML_LOG_INFO("[EXPERT-CACHE] hit_rate=%.1f%% rolling=%.1f%% cached=%zu/%d pool=%.1f MB\n",
                  hr, rhr, lookup_map_.size(), n_slots_, pool_size_ / (1024.0 * 1024.0));
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

void PinnedBufferPool::init(int max_cpu_experts, int64_t max_K, int64_t max_N,
                            int device, sycl::queue & q) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return;
    }

    if (max_cpu_experts <= 0 || max_K <= 0 || max_N <= 0) {
        GGML_LOG_WARN("[PINNED-POOL] Invalid dimensions: experts=%d K=%ld N=%ld\n",
                      max_cpu_experts, (long) max_K, (long) max_N);
        return;
    }

    device_id_ = device;
    queue_     = &q;

    act_bytes_ = static_cast<size_t>(max_cpu_experts) * static_cast<size_t>(max_K) * sizeof(float);
    out_bytes_ = static_cast<size_t>(max_cpu_experts) * static_cast<size_t>(max_N) * sizeof(float);

    // Track host memory via unified cache budget
    unified_cache_add_runtime_host_bytes(act_bytes_ + out_bytes_);

    // Allocate pinned host buffers
    act_buf_ = static_cast<float *>(ggml_sycl_malloc_host(act_bytes_, q, "pinned_pool_act"));
    out_buf_ = static_cast<float *>(ggml_sycl_malloc_host(out_bytes_, q, "pinned_pool_out"));

    if (!act_buf_ || !out_buf_) {
        GGML_LOG_ERROR("[PINNED-POOL] Failed to allocate pinned buffers "
                       "(act=%zu + out=%zu bytes)\n", act_bytes_, out_bytes_);
        // Clean up partial allocation
        if (act_buf_) {
            sycl::free(act_buf_, q.get_context());
            act_buf_ = nullptr;
        }
        if (out_buf_) {
            sycl::free(out_buf_, q.get_context());
            out_buf_ = nullptr;
        }
        unified_cache_sub_runtime_host_bytes(act_bytes_ + out_bytes_);
        act_bytes_ = 0;
        out_bytes_ = 0;
        return;
    }

    initialized_ = true;
    GGML_LOG_INFO("[PINNED-POOL] Initialized: act=%.1f KB, out=%.1f KB, total=%.1f KB\n",
                  act_bytes_ / 1024.0, out_bytes_ / 1024.0,
                  (act_bytes_ + out_bytes_) / 1024.0);
}

void PinnedBufferPool::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return;
    }

    if (act_buf_ && queue_) {
        sycl::free(act_buf_, queue_->get_context());
        act_buf_ = nullptr;
    }
    if (out_buf_ && queue_) {
        sycl::free(out_buf_, queue_->get_context());
        out_buf_ = nullptr;
    }

    unified_cache_sub_runtime_host_bytes(act_bytes_ + out_bytes_);

    GGML_LOG_INFO("[PINNED-POOL] Shutdown: released %.1f KB, %d acquires total\n",
                  (act_bytes_ + out_bytes_) / 1024.0, acquire_count_);

    act_bytes_   = 0;
    out_bytes_   = 0;
    initialized_ = false;
    in_use_      = false;
}

bool PinnedBufferPool::acquire(int n_cpu, int64_t K, int64_t N,
                               acquired_buffers & out) {
    std::lock_guard<std::mutex> lock(mutex_);
    out = {};

    if (!initialized_) {
        return false;
    }

    const size_t needed_act = static_cast<size_t>(n_cpu) * static_cast<size_t>(K) * sizeof(float);
    const size_t needed_out = static_cast<size_t>(n_cpu) * static_cast<size_t>(N) * sizeof(float);

    if (needed_act > act_bytes_ || needed_out > out_bytes_) {
        GGML_SYCL_DEBUG("[PINNED-POOL] Buffer too small: need act=%zu/%zu, out=%zu/%zu\n",
                        needed_act, act_bytes_, needed_out, out_bytes_);
        return false;
    }

    if (in_use_) {
        // Should not happen in single-device sequential dispatch
        GGML_LOG_WARN("[PINNED-POOL] Pool already in use (concurrent acquire)\n");
        return false;
    }

    in_use_ = true;
    acquire_count_++;
    out.activation = act_buf_;
    out.output     = out_buf_;
    return true;
}

void PinnedBufferPool::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    in_use_ = false;
}

} // namespace ggml_sycl
