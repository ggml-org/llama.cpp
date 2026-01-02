//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_KV_OFFLOAD_HPP
#define GGML_SYCL_KV_OFFLOAD_HPP

#include <vector>
#include <unordered_map>
#include <set>
#include <mutex>
#include <cstddef>
#include <cstdint>

#include "dpct/helper.hpp"

// Forward declaration - avoid including full headers
struct ggml_tensor;

namespace ggml_sycl {

// Configuration for KV cache offloading
struct kv_offload_config {
    size_t gpu_kv_budget;         // Maximum GPU bytes for KV cache
    int32_t offload_threshold;    // Context length (tokens) to start offloading
    int block_size = 64;          // Tokens per block (granularity of offload)
    float prefetch_ratio = 1.5f;  // Prefetch this many blocks ahead of attention window
};

// A block of KV cache entries for a specific layer
struct kv_block {
    int layer_id;
    int32_t start_pos;          // Starting token position
    int32_t end_pos;            // Ending token position (exclusive)
    bool on_gpu;                // Currently on GPU?
    void* gpu_ptr;              // GPU memory pointer (nullptr if offloaded)
    void* cpu_ptr;              // Pinned host memory pointer
    size_t size;                // Size of this block in bytes
    int64_t last_access;        // Last access timestamp for LRU
    sycl::event transfer_event; // Event for async transfers
};

// Key for identifying a KV block
struct kv_block_key {
    int layer_id;
    int32_t block_idx;  // Block index within layer

    bool operator==(const kv_block_key& other) const {
        return layer_id == other.layer_id && block_idx == other.block_idx;
    }
};

struct kv_block_key_hash {
    size_t operator()(const kv_block_key& k) const {
        return std::hash<int>()(k.layer_id) ^ (std::hash<int32_t>()(k.block_idx) << 16);
    }
};

// KV layer registration info
struct kv_layer_info {
    int layer_id;
    size_t k_size_per_token;    // Bytes per token for K cache
    size_t v_size_per_token;    // Bytes per token for V cache
    void* k_base_gpu;           // Base GPU pointer for K cache
    void* v_base_gpu;           // Base GPU pointer for V cache
    int32_t max_tokens;         // Maximum context length
};

// KV cache offload manager
// Manages GPU ↔ CPU transfers for KV cache when context exceeds GPU budget
//
// Design:
// - Block-based: KV cache divided into fixed-size blocks (e.g., 64 tokens each)
// - LRU eviction: oldest/least-recently-accessed blocks offloaded first
// - Async transfers: prefetch blocks ahead of attention computation
// - Layer-aware: tracks blocks per layer for efficient batch operations
//
// Integration:
// - register_layer() called during model setup to register KV tensors
// - ensure_on_gpu() called before attention to ensure required blocks are present
// - prefetch_blocks() called to preload blocks for upcoming attention window
// - offload_oldest() called when memory pressure requires freeing GPU memory
class kv_offload_manager {
public:
    // Initialize manager with given SYCL queue and configuration
    kv_offload_manager(sycl::queue& queue, const kv_offload_config& config);
    ~kv_offload_manager();

    // Non-copyable, non-movable
    kv_offload_manager(const kv_offload_manager&) = delete;
    kv_offload_manager& operator=(const kv_offload_manager&) = delete;
    kv_offload_manager(kv_offload_manager&&) = delete;
    kv_offload_manager& operator=(kv_offload_manager&&) = delete;

    // === Layer Registration ===

    // Register KV cache tensors for a layer
    // Called during model/context initialization
    // k_cache, v_cache: ggml tensors for K and V caches
    void register_layer(int layer_id, ggml_tensor* k_cache, ggml_tensor* v_cache);

    // Check if a layer is registered
    bool is_layer_registered(int layer_id) const;

    // Get number of registered layers
    size_t registered_layer_count() const;

    // === Block Management ===

    // Ensure a specific position's KV data is on GPU before attention
    // Returns event that completes when data is ready
    // Blocks if transfer is required (async prefetch should minimize this)
    sycl::event ensure_on_gpu(int layer_id, int32_t pos);

    // Ensure all positions in range [start, end) are on GPU
    // Returns event that completes when all transfers are done
    sycl::event ensure_range_on_gpu(int layer_id, int32_t start_pos, int32_t end_pos);

    // Prefetch blocks for upcoming attention window
    // Non-blocking - schedules async transfers
    // start_pos, end_pos: token position range to prefetch
    void prefetch_blocks(int layer_id, int32_t start_pos, int32_t end_pos);

    // Prefetch for all registered layers at once
    void prefetch_all_layers(int32_t start_pos, int32_t end_pos);

    // === Memory Pressure Handling ===

    // Offload oldest blocks to free GPU memory
    // Returns number of bytes actually freed
    // Called by weight_cache when it needs more GPU memory
    size_t offload_oldest(size_t bytes_needed);

    // Force offload all blocks beyond a certain position
    // Useful for context truncation
    void offload_beyond(int32_t pos);

    // === Context Management ===

    // Notification when new tokens are added to context
    // Triggers offload if context exceeds threshold
    void on_context_extended(int32_t new_length);

    // Get current context length
    int32_t context_length() const { return context_length_; }

    // Check if offloading is active (context > threshold)
    bool is_offloading_active() const;

    // === Memory Stats ===

    // Current GPU memory used by KV cache
    size_t gpu_memory_used() const;

    // Current CPU memory used by offloaded blocks
    size_t cpu_memory_used() const;

    // Number of blocks currently on GPU
    size_t blocks_on_gpu() const;

    // Number of blocks currently on CPU (offloaded)
    size_t blocks_on_cpu() const;

    // === Stats ===

    size_t prefetch_hits() const { return prefetch_hits_; }
    size_t prefetch_misses() const { return prefetch_misses_; }
    float prefetch_hit_rate() const {
        size_t total = prefetch_hits_ + prefetch_misses_;
        return total > 0 ? float(prefetch_hits_) / total : 0.0f;
    }

    void print_stats() const;
    void reset_stats();

private:
    // Get or create block for a position
    kv_block* get_block(int layer_id, int32_t pos);

    // Calculate block index from token position
    int32_t pos_to_block_idx(int32_t pos) const;

    // Transfer block from CPU to GPU
    sycl::event transfer_to_gpu(kv_block* block);

    // Transfer block from GPU to CPU
    sycl::event transfer_to_cpu(kv_block* block);

    // Allocate CPU staging memory for a block
    void* allocate_cpu_block(size_t size);

    // Free CPU staging memory
    void free_cpu_block(void* ptr);

    // Update access timestamp
    void update_access(kv_block* block);

    sycl::queue& queue_;
    kv_offload_config config_;

    // Layer registry
    std::unordered_map<int, kv_layer_info> layers_;

    // Block management: (layer_id, block_idx) -> block
    std::unordered_map<kv_block_key, kv_block, kv_block_key_hash> blocks_;

    // LRU tracking: sorted by last_access (oldest first for eviction)
    // This is rebuilt when needed - not maintained incrementally
    int64_t current_time_ = 0;

    // Context tracking
    int32_t context_length_ = 0;

    // Memory tracking
    size_t gpu_memory_used_ = 0;
    size_t cpu_memory_used_ = 0;

    // Stats
    mutable size_t prefetch_hits_ = 0;
    mutable size_t prefetch_misses_ = 0;

    mutable std::mutex mutex_;
};

} // namespace ggml_sycl

#endif // GGML_SYCL_KV_OFFLOAD_HPP
