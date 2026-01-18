//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// KV Cache Manager with per-attention-head granularity
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-vr5)
//
// Design:
// - Interleaved layout: [K0,V0,K1,V1,...] for independent streaming per head
// - Per-head granularity: Each attention head can be individually streamed
// - Integration with EvictionPolicy: KV heads use P1_ACTIVE_KV / P4_COLD_EXPERT priority
// - Integration with ChunkManager: KV heads are chunked for sub-tensor streaming
//
// Why per-head matters for large models:
// - For 32-head model with 4K context: KV cache ~1GB total
// - Per-head granularity: 32 units of ~32MB each
// - Can evict cold heads (early layers, old context) while keeping hot heads in VRAM
//

#ifndef GGML_SYCL_KV_CACHE_MANAGER_HPP
#define GGML_SYCL_KV_CACHE_MANAGER_HPP

#include "chunk-manager.hpp"
#include "eviction-policy.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// Handle to a KV cache allocation
// Opaque identifier that can be used to access K/V data
struct KVHandle {
    uint64_t id = 0;

    bool operator==(const KVHandle & other) const { return id == other.id; }

    bool operator!=(const KVHandle & other) const { return id != other.id; }
};

// K and V pointer pair for attention computation
struct KVPtrs {
    void * k = nullptr;
    void * v = nullptr;
};

// KV Cache Manager with per-attention-head granularity
//
// Manages KV cache allocations at head level, enabling:
// - Independent eviction of cold attention heads
// - Per-head streaming from host to VRAM
// - Priority-based retention (active heads stay in VRAM)
// - Context extension without full reallocation
class KVCacheManager {
  public:
    KVCacheManager();
    ~KVCacheManager();

    // Non-copyable, non-movable
    KVCacheManager(const KVCacheManager &)             = delete;
    KVCacheManager & operator=(const KVCacheManager &) = delete;
    KVCacheManager(KVCacheManager &&)                  = delete;
    KVCacheManager & operator=(KVCacheManager &&)      = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    // Configure for a model's KV cache requirements
    void configure(uint32_t num_layers, uint32_t num_heads, size_t head_dim, ggml_type kv_type);

    // Getters for configuration
    uint32_t  get_num_layers() const;
    uint32_t  get_num_heads() const;
    size_t    get_head_dim() const;
    ggml_type get_kv_type() const;

    // Bytes per token per head (includes both K and V)
    size_t get_bytes_per_token_per_head() const;

    // =========================================================================
    // Allocation
    // =========================================================================

    // Allocate KV storage for a specific head and sequence range
    // Returns handle for later access
    KVHandle allocate(uint32_t layer, uint32_t head, size_t seq_pos, size_t num_tokens);

    // Extend an existing allocation (context extension)
    // Returns new handle for the extended portion
    KVHandle extend(KVHandle handle, size_t additional_tokens);

    // Check if a handle is valid
    bool is_valid_handle(KVHandle handle) const;

    // Find handle containing a specific token position
    KVHandle find_handle(uint32_t layer, uint32_t head, size_t token_pos) const;

    // Get all handles for a layer
    std::vector<KVHandle> get_handles_for_layer(uint32_t layer) const;

    // =========================================================================
    // Handle Properties
    // =========================================================================

    uint32_t get_layer_id(KVHandle handle) const;
    uint32_t get_head_id(KVHandle handle) const;
    size_t   get_seq_start(KVHandle handle) const;
    size_t   get_seq_len(KVHandle handle) const;
    size_t   get_allocation_size(KVHandle handle) const;

    // Get total tokens allocated for a specific head
    size_t get_total_tokens(uint32_t layer, uint32_t head) const;

    // =========================================================================
    // Data Access
    // =========================================================================

    // Get K data pointer (may trigger streaming from host)
    void * get_k_data(KVHandle handle);

    // Get V data pointer (may trigger streaming from host)
    void * get_v_data(KVHandle handle);

    // Get K/V pointers for multiple heads in a layer (batch access)
    std::vector<KVPtrs> get_layer_kv(uint32_t                      layer,
                                     const std::vector<uint32_t> & heads,
                                     size_t                        seq_start,
                                     size_t                        seq_len);

    // =========================================================================
    // Layout Conversion
    // =========================================================================

    // Convert from grouped [all_K, all_V] to interleaved [K0,V0,K1,V1,...]
    void convert_to_interleaved(const void * grouped_data, void * interleaved_data, size_t seq_len);

    // Convert from interleaved back to grouped (if needed)
    void convert_to_grouped(const void * interleaved_data, void * grouped_data, size_t seq_len);

    // =========================================================================
    // Priority Management
    // =========================================================================

    // Get current priority for a handle
    EvictionPriority get_priority(KVHandle handle) const;

    // Check if handle is pinned (cannot be evicted)
    bool is_pinned(KVHandle handle) const;

    // Mark heads as active (being used in attention computation)
    // Sets P1_ACTIVE_KV priority and pins
    void mark_heads_active(uint32_t layer, const std::vector<uint32_t> & heads);

    // Mark heads as cold (attention computation done)
    // Sets P4_COLD_EXPERT priority and unpins
    void mark_heads_cold(uint32_t layer, const std::vector<uint32_t> & heads);

    // Update LRU timestamp
    void touch(KVHandle handle);

    // Get last access timestamp
    uint64_t get_last_access(KVHandle handle) const;

    // =========================================================================
    // Memory Tier Management
    // =========================================================================

    // Get current memory tier for a handle
    MemoryTier get_tier(KVHandle handle) const;

    // Evict handle from VRAM to HOST
    void evict_to_host(KVHandle handle);

    // Prefetch handle from HOST to VRAM
    void prefetch_to_vram(KVHandle handle);

    // Check if handle is currently being prefetched
    bool is_prefetching(KVHandle handle) const;

    // Prefetch all heads for a layer
    void prefetch_layer(uint32_t layer);

    // =========================================================================
    // Eviction
    // =========================================================================

    // Get candidates for eviction (sorted by priority, then LRU)
    std::vector<KVHandle> get_eviction_candidates(size_t count) const;

    // Evict cold heads to free at least bytes_needed from VRAM
    // Returns actual bytes freed
    size_t evict_cold_heads(size_t bytes_needed);

    // =========================================================================
    // Memory Usage
    // =========================================================================

    // Total VRAM used by KV cache
    size_t get_total_vram_usage() const;

    // Total HOST memory used by KV cache
    size_t get_total_host_usage() const;

  private:
    struct Impl;
    Impl * impl_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_KV_CACHE_MANAGER_HPP
