//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ggml_sycl {

// Key identifying a unique MoE expert (layer, expert_id pair).
// Used across expert-cache and expert-prefetch subsystems.
struct expert_key {
    int layer;
    int expert_id;

    bool operator==(const expert_key & o) const {
        return layer == o.layer && expert_id == o.expert_id;
    }
};

// Hash function for expert_key.
struct expert_key_hash {
    size_t operator()(const expert_key & k) const {
        return std::hash<int>()(k.layer) ^ (std::hash<int>()(k.expert_id) << 16);
    }
};

// A single slot in the VRAM expert cache pool.
// Each slot holds one expert's weight tensor for one layer.
struct ExpertSlot {
    int      layer_idx   = -1;       // -1 = empty slot
    int      expert_idx  = -1;
    void *   device_ptr  = nullptr;  // Points into contiguous pool
    size_t   size_bytes  = 0;
    uint64_t frequency   = 0;       // Access count
    uint64_t last_access = 0;       // Token counter value at last access
    float    score       = 0.0f;    // Combined eviction score
};

// Result of a cache lookup.
struct ExpertLookup {
    void * device_ptr;  // Non-null if cached in VRAM
    void * host_ptr;    // Always set (host RAM copy)
    bool   is_cached;   // True if device_ptr is valid
};

// Expert-granular VRAM cache for MoE models.
//
// Sits alongside the existing unified_cache. The unified cache manages weight
// placement at tensor granularity; the expert cache manages placement at
// expert granularity within MoE tensors.
//
// Each "expert slot" holds one expert's weight tensor for one layer
// (~4.2 MB at Q4_0 for DeepSeek V3).
//
// The pool is a contiguous sycl::malloc_device allocation, sub-allocated with
// 256-byte alignment per slot.
//
// Thread-safe: reads (lookup) take shared lock, writes (evict/load) take
// exclusive lock.
//
// Budget tracked via unified_cache_add_runtime_bytes() with
// runtime_category::EXPERT_CACHE.
//
// Env var: GGML_SYCL_EXPERT_CACHE_MB=N overrides default budget.
// Default: 50% of remaining VRAM after dense layers.
//
class ExpertCache {
public:
    // Initialize the cache. Must be called after model loading so remaining
    // VRAM budget is known.
    //   device_id:         SYCL device ordinal
    //   vram_budget_bytes: maximum VRAM to allocate for expert pool
    //   q:                 compute queue for H2D copies
    void init(int device_id, size_t vram_budget_bytes, sycl::queue & q);

    // Release all VRAM and clean up.
    void shutdown();

    ~ExpertCache();

    // Non-copyable, non-movable (owns SYCL allocations)
    ExpertCache() = default;
    ExpertCache(const ExpertCache &)             = delete;
    ExpertCache & operator=(const ExpertCache &) = delete;
    ExpertCache(ExpertCache &&)                  = delete;
    ExpertCache & operator=(ExpertCache &&)      = delete;

    // Register all experts at model load time (host pointers).
    // Must be called before lookup/ensure_cached for each expert.
    void register_expert(int layer_idx, int expert_idx,
                         const void * host_ptr, size_t bytes);

    // Fast lookup: is this expert in VRAM? O(1).
    // Returns {device_ptr, host_ptr, is_cached}.
    // Thread-safe (shared lock).
    ExpertLookup lookup(int layer_idx, int expert_idx) const;

    // Load expert into VRAM (evicting if necessary), returns device ptr.
    // Synchronous H2D copy. Returns nullptr only if not registered.
    void * ensure_cached(int layer_idx, int expert_idx, sycl::queue & q);

    // Async prefetch (non-blocking H2D). Returns event for completion.
    sycl::event prefetch_async(int layer_idx, int expert_idx, sycl::queue & q);

    // Evict lowest-score slot and load a new expert into VRAM.
    // Registers the expert if not already registered, then ensures it is cached.
    // Returns device pointer, or nullptr on failure.
    void * evict_and_load(int layer_idx, int expert_idx,
                          const void * host_src, size_t bytes, sycl::queue & q) {
        register_expert(layer_idx, expert_idx, host_src, bytes);
        return ensure_cached(layer_idx, expert_idx, q);
    }

    // Compute default VRAM budget: 50% of remaining VRAM after dense layers.
    // Returns 0 if device info is unavailable.
    static size_t default_budget(int device_id);

    // Update access statistics after use.
    void update_score(int layer_idx, int expert_idx, uint64_t token_counter);

    // Record a batch of expert accesses for the current token/layer.
    // Used to track co-activation patterns and warm-start profiling.
    // current_layer is the layer being processed; expert_ids are the
    // router-selected expert indices for this token.
    void record_access_batch(int current_layer, const int * expert_ids, int n_experts,
                             uint64_t token_counter);

    // Trigger warm-start bulk-load after profiling phase completes.
    // Loads the top-N most-frequent experts into VRAM. Called automatically
    // by record_access_batch() when warmup token count is reached.
    void finish_warmup();

    // Stats
    size_t cached_count() const;
    size_t total_slots() const;
    float  hit_rate() const;         // All-time hits / (hits + misses)
    float  rolling_hit_rate() const; // Rolling window (last 100 tokens)

    size_t vram_budget() const;
    size_t vram_used() const;
    size_t cache_hits() const;
    size_t cache_misses() const;
    size_t entries_count() const;

    // True if init() has been called and pool is allocated.
    bool is_initialized() const { return pool_ != nullptr; }

    // True if warm-start profiling phase is active (collecting stats, not caching yet).
    bool is_warmup_phase() const;

    // -----------------------------------------------------------------
    // Backward-compatible API used by expert_prefetcher (Track C).
    // These delegate to the primary API above.
    // -----------------------------------------------------------------

    // Check if expert is currently cached in VRAM.
    bool is_cached_in_vram(int layer_idx, int expert_idx) const {
        return lookup(layer_idx, expert_idx).is_cached;
    }

    // Synchronous get: ensure cached and return pointer.
    // size parameter is unused (slot_size_ is authoritative).
    void * get_expert(int layer_idx, int expert_idx, size_t /*size*/) {
        if (!queue_) { return nullptr; }
        return ensure_cached(layer_idx, expert_idx, *queue_);
    }

    // Batch prefetch using expert_key vector (compat with prefetcher).
    // expert_size parameter is unused (slot_size_ is authoritative).
    void prefetch(const std::vector<expert_key> & experts, size_t /*expert_size*/) {
        if (!queue_) { return; }
        for (const auto & ek : experts) {
            prefetch_async(ek.layer, ek.expert_id, *queue_);
        }
    }

    // Wait for all async prefetches on the compute queue.
    void wait_prefetch() {
        if (queue_) { queue_->wait(); }
    }

private:
    static constexpr size_t ALIGNMENT = 256;

    // Align size up to ALIGNMENT boundary.
    static size_t align_up(size_t sz) {
        return (sz + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    // Compute eviction score. Lower = evict first.
    // score = alpha*frequency + beta*recency + gamma*layer_distance + delta*co_activation
    void recompute_score(ExpertSlot & slot, uint64_t current_token) const;

    // Find the slot with lowest score (eviction candidate).
    // Returns index into slots_, or -1 if all empty.
    int find_eviction_candidate() const;

    // Make hash key from (layer, expert) pair.
    int64_t make_key(int layer, int expert) const;

    // Log hit rate periodically (every log_interval_ tokens).
    void maybe_log_stats();

    void * pool_      = nullptr;  // sycl::malloc_device contiguous pool
    size_t pool_size_  = 0;
    size_t slot_size_  = 0;       // Max expert size (aligned to 256)
    int    n_slots_    = 0;

    std::vector<ExpertSlot> slots_;

    // Fast lookup: key -> slot index. -1 sentinel not stored.
    std::unordered_map<int64_t, int> lookup_map_;

    // Registered expert host pointers: key -> {host_ptr, size_bytes}.
    struct HostEntry {
        const void * ptr  = nullptr;
        size_t       size = 0;
    };
    std::unordered_map<int64_t, HostEntry> host_entries_;

    mutable std::shared_mutex mutex_;

    // Stats (all-time)
    uint64_t hits_   = 0;
    uint64_t misses_ = 0;

    // Rolling hit rate tracking (window of last ROLLING_WINDOW tokens)
    static constexpr int ROLLING_WINDOW = 100;
    struct RollingEntry {
        int hits   = 0;
        int misses = 0;
    };
    RollingEntry rolling_buf_[ROLLING_WINDOW] = {};
    int          rolling_idx_   = 0;   // Current write index (circular)
    int          rolling_count_ = 0;   // Total entries written (capped at ROLLING_WINDOW)

    // Current layer being processed (set by record_access_batch)
    int current_layer_ = -1;

    // Co-activation tracking: pair(key_a, key_b) -> count.
    // Tracks how often two experts are activated together in the same token.
    // Stored as sorted-pair keys to avoid (a,b)/(b,a) duplication.
    std::unordered_map<int64_t, uint32_t> co_activation_;

    // Warm-start profiling state
    int      warmup_target_   = 32;   // Default: profile first 32 tokens
    int      warmup_tokens_   = 0;    // Tokens seen so far during warmup
    bool     warmup_active_   = true; // True during warmup phase
    bool     warmup_done_     = false; // True after warmup bulk-load completed
    // Expert access frequency during warmup: key -> count
    std::unordered_map<int64_t, uint32_t> warmup_freq_;

    // Periodic logging
    uint64_t log_interval_     = 100;  // Log every N tokens
    uint64_t last_log_token_   = 0;

    // Global token counter for scoring
    uint64_t global_token_ = 0;

    int         device_id_ = -1;
    sycl::queue * queue_   = nullptr;  // Non-owning ptr to compute queue

    // Max expert count for key computation
    static constexpr int MAX_EXPERTS_PER_LAYER = 256;
};

// Backward-compatible type alias used by expert-prefetch (Track C).
using expert_cache = ExpertCache;

} // namespace ggml_sycl
