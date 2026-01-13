//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_UNIFIED_TENSOR_CACHE_HPP
#define GGML_SYCL_UNIFIED_TENSOR_CACHE_HPP

#include "tensor-types.hpp"
#include "vram-pool.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// Result of tensor location query
struct tensor_location {
    void *      ptr;
    memory_tier tier;
};

// Unified tensor cache managing all model weights across memory tiers.
//
// Consolidates previous fragmented caches (unified-cache, expert_cache,
// dense_layer_scheduler, pinned_chunk_pool) into one system.
//
// Design:
// - Owns both VRAM pool and pinned host pool
// - Two-phase loading: set_inventory() then load_tensor_data()
// - Static priority by tensor type + dynamic access-pattern adjustment
// - Auto-enables tiered mode when model exceeds VRAM
//
// Thread-safe: all public methods can be called from multiple threads.
class unified_tensor_cache {
  public:
    // Create cache with VRAM and host budgets
    unified_tensor_cache(sycl::queue & queue, size_t vram_budget, size_t host_budget);
    ~unified_tensor_cache();

    // Non-copyable, non-movable
    unified_tensor_cache(const unified_tensor_cache &)             = delete;
    unified_tensor_cache & operator=(const unified_tensor_cache &) = delete;
    unified_tensor_cache(unified_tensor_cache &&)                  = delete;
    unified_tensor_cache & operator=(unified_tensor_cache &&)      = delete;

    // === Phase 1: Inventory ===

    // Set tensor inventory and compute placement decisions
    // Called once after GGUF parsing, before tensor allocation
    void set_inventory(const tensor_inventory & inventory);

    // Check if tiered mode is enabled (model > VRAM)
    bool is_tiered_enabled() const { return tiered_enabled_; }

    // Get planned tier for a tensor (by inventory index)
    memory_tier get_planned_tier(uint64_t tensor_id) const;

    // === Phase 2: Tensor Access ===

    // Get tensor pointer and current location
    // If tensor not yet loaded, returns nullptr
    tensor_location get_tensor_with_location(uint64_t tensor_id);

    // Load tensor data from source to cache
    // src_ptr: source data (e.g., mmap pointer)
    // Called during model loading
    void load_tensor_data(uint64_t tensor_id, const void * src_ptr);

    // === Dynamic Promotion ===

    // Request async promotion of tensor to VRAM
    // Used when access pattern suggests tensor should be promoted
    void request_promotion(uint64_t tensor_id);

    // Prefetch tensors to VRAM asynchronously
    // Used with router predictions for MoE
    void prefetch(const std::vector<uint64_t> & tensor_ids);

    // Wait for all pending async transfers
    void wait_pending_transfers();

    // === Statistics ===

    size_t vram_budget() const { return vram_budget_; }

    size_t vram_used() const { return vram_pool_.used(); }

    size_t host_budget() const { return host_budget_; }

    size_t host_used() const { return host_used_; }

    size_t cache_hits() const { return cache_hits_.load(); }

    size_t cache_misses() const { return cache_misses_.load(); }

    size_t promotions() const { return promotions_.load(); }

    size_t evictions() const { return evictions_.load(); }

    void print_stats() const;

  private:
    // Entry for a tensor in the cache
    struct tensor_entry {
        tensor_info info;
        void *      host_ptr     = nullptr;  // Pinned host or mmap pointer
        void *      vram_ptr     = nullptr;  // Set if in VRAM
        memory_tier current_tier = memory_tier::MMAP;
        memory_tier planned_tier = memory_tier::MMAP;
        uint64_t    last_access  = 0;
        uint32_t    access_count = 0;
        bool        loaded       = false;
        bool        owns_host    = false;  // True if we allocated host memory
    };

    // Compute placement decisions based on inventory
    void compute_placement();

    // Evict lowest-score tensor from VRAM to make room
    bool evict_one(size_t needed_size);

    // Compute eviction score (higher = more valuable, don't evict)
    float compute_score(const tensor_entry & entry) const;

    // Allocate pinned host memory
    void * allocate_host(size_t size);
    void   free_host(void * ptr, size_t size);

    sycl::queue & queue_;
    sycl::queue   copy_queue_;  // Separate queue for async transfers

    size_t vram_budget_;
    size_t host_budget_;
    size_t host_used_ = 0;

    vram_pool vram_pool_;

    std::unordered_map<uint64_t, tensor_entry> entries_;
    bool                                       tiered_enabled_ = false;
    uint64_t                                   time_           = 0;

    std::atomic<size_t> cache_hits_{ 0 };
    std::atomic<size_t> cache_misses_{ 0 };
    std::atomic<size_t> promotions_{ 0 };
    std::atomic<size_t> evictions_{ 0 };

    mutable std::mutex mutex_;
};

// === Global API ===

// Check if tiered memory should auto-enable
bool should_enable_tiered(const tensor_inventory & inv, size_t vram_available);

}  // namespace ggml_sycl

#endif  // GGML_SYCL_UNIFIED_TENSOR_CACHE_HPP
