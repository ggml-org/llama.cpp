//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_UNIFIED_CACHE_HPP
#define GGML_SYCL_UNIFIED_CACHE_HPP

#include <vector>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <atomic>

#include "dpct/helper.hpp"

namespace ggml_sycl {

// Type of cached entry
enum class cache_entry_type {
    DENSE_WEIGHT,   // Regular weight tensor (attention, FFN, embeddings)
    MOE_EXPERT      // MoE expert weight
};

// Key for identifying a cached entry
struct unified_cache_key {
    cache_entry_type type;
    const void* key_ptr;     // Stable key: tensor->data for dense, mmap ptr for MoE
    int layer_id;            // Layer ID (for expert identification)
    int expert_id;           // Expert ID (-1 for dense weights)

    bool operator==(const unified_cache_key& other) const {
        return type == other.type && key_ptr == other.key_ptr &&
               layer_id == other.layer_id && expert_id == other.expert_id;
    }
};

struct unified_cache_key_hash {
    size_t operator()(const unified_cache_key& k) const {
        // Use key_ptr as primary hash for dense weights (fast lookup)
        // Use layer+expert for MoE experts
        if (k.type == cache_entry_type::DENSE_WEIGHT) {
            return std::hash<const void*>()(k.key_ptr);
        }
        return std::hash<int>()(k.layer_id) ^
               (std::hash<int>()(k.expert_id) << 16) ^
               (std::hash<const void*>()(k.key_ptr) >> 32);
    }
};

// Metadata for a cached entry
struct unified_cache_entry {
    void* device_ptr;        // GPU memory pointer
    const void* src_ptr;     // Source data pointer (for change detection)
    uint64_t content_hash;   // Simple hash of content (first/last bytes)
    size_t size;             // Size in bytes
    cache_entry_type type;   // Dense or MoE
    int layer_id;            // Layer ID
    int expert_id;           // Expert ID (-1 for dense)
    uint32_t access_count;   // Access frequency for LFU
    int64_t last_access;     // Timestamp for recency
    bool pinned;             // Protected from eviction
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here
};

// Unified GPU cache for both dense weights and MoE experts
//
// Design principles:
// 1. Single memory budget - avoids OOM from competing caches
// 2. Type-aware eviction - dense weights and experts have equal priority
// 3. Automatic partitioning - no manual budget splitting needed
// 4. LFU+LRU hybrid eviction - keeps frequently used entries
//
// Usage:
// - call ensure_cached() for any weight (dense or expert)
// - cache automatically evicts lowest-scoring entries when full
// - for MoE: call with layer_id and expert_id
// - for dense: call with layer_id=-1, expert_id=-1
class unified_cache {
public:
    // Initialize with SYCL queue and memory budget
    // budget_bytes: total GPU memory for caching (dense + MoE combined)
    // staging_size: pinned host staging buffer size (default 64MB)
    unified_cache(sycl::queue& queue, size_t budget_bytes,
                  size_t staging_size = 64 * 1024 * 1024);
    ~unified_cache();

    // Non-copyable, non-movable
    unified_cache(const unified_cache&) = delete;
    unified_cache& operator=(const unified_cache&) = delete;
    unified_cache(unified_cache&&) = delete;
    unified_cache& operator=(unified_cache&&) = delete;

    // === Primary API ===

    // Ensure a weight is cached, loading from src_ptr if needed
    // Returns device pointer, or nullptr if cache is full and eviction failed
    //
    // key_ptr: Stable identifier for cache lookup
    //   - For dense weights: tensor->data (device buffer address, stable across set_tensor calls)
    //   - For MoE experts: mmap_ptr (stable across inference)
    // src_ptr: Source data to copy from (may change for same key_ptr)
    // If src_ptr differs from cached entry's src_ptr, data is re-uploaded
    //
    // For dense weights: layer_id=-1, expert_id=-1
    // For MoE experts: layer_id=N, expert_id=M
    void* ensure_cached(const void* key_ptr, const void* src_ptr, size_t size,
                        cache_entry_type type,
                        int layer_id = -1, int expert_id = -1,
                        bool validate_content = false);

    // Check if entry is cached (without loading)
    bool is_cached(const void* mmap_ptr) const;

    // Get device pointer for cached entry (returns nullptr if not cached)
    void* get(const void* key_ptr) const;

    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not in cache

    // === Pinning for Graphs ===

    void pin(const void* mmap_ptr);
    void unpin(const void* mmap_ptr);
    void unpin_all();
    bool is_pinned(const void* mmap_ptr) const;

    // === Memory Management ===

    // Current memory used
    size_t used() const { return used_.load(); }

    // Total budget
    size_t budget() const { return budget_; }

    // Available memory
    size_t available() const { return budget_ - used_.load(); }

    // Force eviction to free at least bytes_needed
    // Returns actual bytes freed
    size_t evict(size_t bytes_needed);

    // === Stats ===

    size_t hits() const { return hits_.load(); }
    size_t misses() const { return misses_.load(); }
    float hit_rate() const {
        size_t total = hits_.load() + misses_.load();
        return total > 0 ? float(hits_.load()) / total : 0.0f;
    }

    size_t dense_count() const;
    size_t expert_count() const;

    void print_stats() const;
    void reset_stats();

private:
    // Evict lowest-scoring entry to make room for new_size bytes
    // Returns true if eviction succeeded, false if all entries are pinned
    bool evict_one(size_t new_size);

    // Compute eviction score: higher = more valuable (keep longer)
    // score = access_count * exp(-decay * age)
    float compute_score(const unified_cache_entry& entry) const;

    // Copy data from mmap to device via staging
    void copy_to_device(void* dst, const void* src, size_t size);

    sycl::queue& queue_;
    size_t budget_;                          // Total GPU memory budget
    std::atomic<size_t> used_{0};            // Current usage
    std::atomic<int64_t> time_{0};           // Monotonic counter

    // Cache storage: mmap_ptr -> entry
    std::unordered_map<unified_cache_key, unified_cache_entry, unified_cache_key_hash> entries_;

    // Fast lookup by mmap_ptr (for dense weights)
    std::unordered_map<const void*, unified_cache_key> ptr_to_key_;

    // Staging buffer for mmap -> device transfers
    void* staging_ = nullptr;
    size_t staging_size_ = 0;

    // Stats
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};

    static constexpr float DECAY_ALPHA = 0.01f;

    mutable std::mutex mutex_;
};

// === Cache Mode ===
// Controls whether cache is shared globally or per-device

enum class unified_cache_mode {
    GLOBAL,      // Single cache on primary device (default for single-GPU)
    PER_DEVICE,  // Separate cache per device (better for multi-GPU)
    AUTO         // Auto-detect: per_device if multiple GPUs, global otherwise
};

// Get current cache mode (from env var or auto-detection)
unified_cache_mode get_unified_cache_mode();

// Set cache mode (call before first cache access)
void set_unified_cache_mode(unified_cache_mode mode);

// === Global API ===

// Get unified cache for the device associated with the given queue
// In GLOBAL mode: returns same cache for all devices
// In PER_DEVICE mode: returns device-specific cache
unified_cache* get_unified_cache(sycl::queue& queue);

// Get unified cache for a specific device ID
// Useful when device ID is known but queue isn't available
unified_cache* get_unified_cache_for_device(int device_id);

// Check if unified cache is enabled (via env var or auto-detection)
bool unified_cache_enabled();

// Set unified cache budget (call before first use)
// In PER_DEVICE mode, this sets budget per device
void set_unified_cache_budget(size_t bytes);

// === MoE Expert Caching API (for compatibility with existing expert_cache usage) ===

// Cache an expert weight tensor via unified cache
// Returns device pointer or nullptr on failure
// mmap_ptr: pointer to expert data in mmap'd file
// expert_size: size of expert in bytes
// layer_id, expert_id: for identification and debugging
void* cache_moe_expert(sycl::queue& queue, const void* mmap_ptr, size_t expert_size,
                       int layer_id, int expert_id);

// Check if expert is cached
bool is_expert_cached(const void* mmap_ptr);

// Get cached expert pointer (returns nullptr if not cached)
void* get_cached_expert(const void* mmap_ptr);

// NOTE: mark_expert_reordered/is_expert_reordered removed
// Reorder state is tracked in tensor->extra->optimized_feature, not in cache

// Pin expert to prevent eviction during graph execution
void pin_expert(const void* mmap_ptr);

// Unpin all experts
void unpin_all_experts();

// === Reorder Callback Support ===

// Callback for SoA weight reordering (called after cache miss)
// data_device: GPU pointer to cached data
// ncols, nrows: tensor dimensions
// size: total size in bytes
// stream: SYCL queue for async operations
using moe_reorder_callback_fn = void (*)(uint8_t* data_device, int ncols, int nrows,
                                          size_t size, sycl::queue* stream);

// Set the global reorder callback (typically set once during initialization)
void set_moe_reorder_callback(moe_reorder_callback_fn callback);

// Cache an expert with optional SoA reorder (for MXFP4 and similar types)
// Returns device pointer or nullptr on failure
// Applies reorder callback if set and entry is newly cached
// NOTE: Caller must track reorder state in tensor->extra->optimized_feature
void* cache_moe_expert_with_reorder(sycl::queue& queue, const void* mmap_ptr, size_t expert_size,
                                     int layer_id, int expert_id, int ncols, int nrows);

// === Shutdown API ===

// Shutdown the unified cache system before SYCL runtime destruction
// Call this during ggml_backend_sycl_free() to avoid static destruction order issues
// After calling this, the cache destructors will skip sycl::free() calls
void shutdown_unified_cache();

} // namespace ggml_sycl

#endif // GGML_SYCL_UNIFIED_CACHE_HPP
