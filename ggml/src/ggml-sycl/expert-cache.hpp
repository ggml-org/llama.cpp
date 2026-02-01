//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_EXPERT_CACHE_HPP
#define GGML_SYCL_EXPERT_CACHE_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

class pinned_chunk_pool;

// Key identifying a unique MoE expert
struct expert_key {
    int layer;
    int expert_id;

    bool operator==(const expert_key & o) const { return layer == o.layer && expert_id == o.expert_id; }
};

// Hash function for expert_key
struct expert_key_hash {
    size_t operator()(const expert_key & k) const {
        return std::hash<int>()(k.layer) ^ (std::hash<int>()(k.expert_id) << 16);
    }
};

// VRAM cache for MoE expert weights with LRU + frequency-based eviction.
//
// Experts are stored in pinned host memory and dynamically loaded into VRAM
// on demand. When VRAM is full, the cache evicts experts using a combined
// LRU + frequency score (recent access + access count).
//
// Thread-safe: all public methods can be called from multiple threads.
//
// Usage:
//   expert_cache cache(queue, pinned_pool, 2ULL * 1024 * 1024 * 1024);  // 2GB VRAM budget
//   cache.register_expert(layer, expert_id, host_ptr, size);
//   void* vram_ptr = cache.get_expert(layer, expert_id, size);
//
class expert_cache {
  public:
    // Create cache with given VRAM budget.
    // pool: pinned host memory pool containing expert weights
    // vram_budget: maximum VRAM bytes to use for caching
    expert_cache(sycl::queue & compute_queue, pinned_chunk_pool & pool, size_t vram_budget);
    ~expert_cache();

    // Non-copyable, non-movable (owns SYCL allocations)
    expert_cache(const expert_cache &)             = delete;
    expert_cache & operator=(const expert_cache &) = delete;
    expert_cache(expert_cache &&)                  = delete;
    expert_cache & operator=(expert_cache &&)      = delete;

    // Register expert's host memory location.
    // Must be called before get_expert() for each expert.
    void register_expert(int layer, int expert_id, void * host_ptr, size_t size);

    // Get expert data pointer (returns VRAM pointer, loads from host if needed).
    // Returns nullptr only if expert is not registered.
    // May return host pointer if VRAM exhausted (fallback).
    void * get_expert(int layer, int expert_id, size_t size);

    // Check if expert is currently cached in VRAM
    bool is_cached_in_vram(int layer, int expert_id) const;

    // Prefetch experts asynchronously (for compute/transfer overlap).
    // Does not block - transfers run in background.
    void prefetch(const std::vector<expert_key> & experts, size_t expert_size);

    // Wait for all pending prefetch operations to complete
    void wait_prefetch();

    // Statistics
    size_t vram_budget() const { return vram_budget_; }

    size_t vram_used() const { return vram_used_; }

    size_t cache_hits() const { return cache_hits_; }

    size_t cache_misses() const { return cache_misses_; }

    size_t entries_count() const;

  private:
    // Entry for an expert currently in VRAM
    struct vram_entry {
        void *   vram_ptr;
        size_t   size;
        uint64_t last_access;
        uint32_t access_count;
    };

    // Entry for an expert registered in host memory
    struct host_entry {
        void * host_ptr;
        size_t size;
    };

    // Compute eviction score (higher = more likely to keep)
    float compute_score(const vram_entry & e) const;

    // Allocate VRAM, evicting if necessary. Returns nullptr if impossible.
    void * allocate_vram(size_t size);

    // Evict the entry with lowest score
    void evict_lowest_score();

    sycl::queue &       compute_queue_;
    sycl::queue         copy_queue_;  // Separate queue for async prefetch
    pinned_chunk_pool & pinned_pool_;
    int                 device_id_ = -1;

    size_t   vram_budget_;
    size_t   vram_used_ = 0;
    uint64_t time_      = 0;  // Logical clock for LRU

    size_t cache_hits_   = 0;
    size_t cache_misses_ = 0;

    std::unordered_map<expert_key, vram_entry, expert_key_hash> vram_entries_;
    std::unordered_map<expert_key, host_entry, expert_key_hash> host_entries_;
    mutable std::mutex                                          mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_EXPERT_CACHE_HPP
