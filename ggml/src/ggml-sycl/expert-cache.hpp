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

    // Update access statistics after use.
    void update_score(int layer_idx, int expert_idx, uint64_t token_counter);

    // Stats
    size_t cached_count() const;
    size_t total_slots() const;
    float  hit_rate() const;  // Rolling hits / (hits + misses)

    size_t vram_budget() const;
    size_t vram_used() const;
    size_t cache_hits() const;
    size_t cache_misses() const;
    size_t entries_count() const;

    // True if init() has been called and pool is allocated.
    bool is_initialized() const { return pool_ != nullptr; }

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
    // score = alpha * frequency + beta * (token_counter - last_access)
    // alpha = 1.0, beta = -0.01
    void recompute_score(ExpertSlot & slot, uint64_t current_token) const;

    // Find the slot with lowest score (eviction candidate).
    // Returns index into slots_, or -1 if all empty.
    int find_eviction_candidate() const;

    // Make hash key from (layer, expert) pair.
    int64_t make_key(int layer, int expert) const;

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

    // Stats
    uint64_t hits_   = 0;
    uint64_t misses_ = 0;

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
