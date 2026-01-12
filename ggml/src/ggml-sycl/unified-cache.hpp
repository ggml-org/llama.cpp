//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_UNIFIED_CACHE_HPP
#define GGML_SYCL_UNIFIED_CACHE_HPP

#include "dpct/helper.hpp"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// Type of cached entry
enum class cache_entry_type {
    DENSE_WEIGHT,  // Regular weight tensor (attention, FFN, embeddings)
    MOE_EXPERT     // MoE expert weight
};

// Cache entry readiness
enum class cache_entry_state {
    READY,
    IN_PROGRESS,
    FAILED,
};

// XMX layout metadata carried with cache entries/results
struct cache_layout_xmx_info {
    int64_t tile_n        = 0;
    int64_t tile_k        = 0;
    int64_t n_tile_groups = 0;
};

// Result status for layout-aware cache API
enum class cache_layout_status {
    READY,
    IN_PROGRESS,
    FAILED,
    INVALID,
};

struct cache_layout_request;
using cache_layout_fill_fn = sycl::event (*)(sycl::queue &                    queue,
                                             void *                           dst,
                                             size_t                           dst_size,
                                             const void *                     src,
                                             size_t                           src_size,
                                             const void *                     ctx,
                                             const std::vector<sycl::event> & deps);

struct cache_layout_request {
    const void *          key_ptr          = nullptr;
    const void *          src_ptr          = nullptr;
    size_t                src_size         = 0;
    size_t                dst_size         = 0;
    cache_entry_type      type             = cache_entry_type::DENSE_WEIGHT;
    int                   layer_id         = -1;
    int                   expert_id        = -1;
    ggml_layout_mode      layout           = GGML_LAYOUT_AOS;
    bool                  validate_content = false;
    bool                  allow_overcommit = false;
    cache_layout_xmx_info xmx_info         = {};
    cache_layout_fill_fn  fill_fn          = nullptr;
    const void *          fill_ctx         = nullptr;
};

struct cache_layout_result {
    void *                device_ptr = nullptr;
    size_t                size       = 0;
    ggml_layout_mode      layout     = GGML_LAYOUT_AOS;
    cache_layout_xmx_info xmx_info   = {};
    sycl::event           event;
    cache_layout_status   status = cache_layout_status::FAILED;
};

// Key for identifying a cached entry
struct unified_cache_key {
    cache_entry_type type;
    const void *     key_ptr;    // Stable key: GGUF identity for dense, mmap ptr for MoE
    int              layer_id;   // Layer ID (for expert identification)
    int              expert_id;  // Expert ID (-1 for dense weights)
    ggml_layout_mode layout;     // Target layout for this entry

    bool operator==(const unified_cache_key & other) const {
        return type == other.type && key_ptr == other.key_ptr && layer_id == other.layer_id &&
               expert_id == other.expert_id && layout == other.layout;
    }
};

struct unified_cache_key_hash {
    size_t operator()(const unified_cache_key & k) const {
        // Use key_ptr as primary hash for dense weights (fast lookup)
        // Use layer+expert for MoE experts
        if (k.type == cache_entry_type::DENSE_WEIGHT) {
            return std::hash<const void *>()(k.key_ptr) ^ (std::hash<int>()(k.layout) << 1);
        }
        return std::hash<int>()(k.layer_id) ^ (std::hash<int>()(k.expert_id) << 16) ^
               (std::hash<const void *>()(k.key_ptr) >> 32) ^ (std::hash<int>()(k.layout) << 1);
    }
};

struct unified_cache_ptr_key {
    const void *     key_ptr;
    ggml_layout_mode layout;

    bool operator==(const unified_cache_ptr_key & other) const {
        return key_ptr == other.key_ptr && layout == other.layout;
    }
};

struct unified_cache_ptr_key_hash {
    size_t operator()(const unified_cache_ptr_key & k) const {
        return std::hash<const void *>()(k.key_ptr) ^ (std::hash<int>()(k.layout) << 1);
    }
};

// Metadata for a cached entry
struct unified_cache_entry {
    void *                device_ptr;       // GPU memory pointer
    const void *          src_ptr;          // Source data pointer (for change detection)
    uint64_t              content_hash;     // Simple hash of content (first/last bytes)
    size_t                size;             // Size in bytes
    cache_entry_type      type;             // Dense or MoE
    int                   layer_id;         // Layer ID
    int                   expert_id;        // Expert ID (-1 for dense)
    ggml_layout_mode      layout;           // Target layout for this entry
    cache_layout_xmx_info xmx_info;         // XMX metadata (when applicable)
    uint32_t              access_count;     // Access frequency for LFU
    int64_t               last_access;      // Timestamp for recency
    bool                  pinned;           // Protected from eviction
    bool                  hot;              // Hot-set hint for MoE experts
    cache_entry_state     state;            // READY vs IN_PROGRESS
    bool                  has_ready_event;  // True if ready_event is valid
    sycl::event           ready_event;      // Completion event for IN_PROGRESS entries
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here
};

// Host cache entry (canonical layouts in host memory)
struct host_cache_entry {
    void *                host_ptr      = nullptr;
    const void *          src_ptr       = nullptr;
    uint64_t              content_hash  = 0;
    size_t                size          = 0;
    cache_entry_type      type          = cache_entry_type::DENSE_WEIGHT;
    int                   layer_id      = -1;
    int                   expert_id     = -1;
    ggml_layout_mode      layout        = GGML_LAYOUT_AOS;
    uint32_t              access_count  = 0;
    int64_t               last_access   = 0;
    bool                  pinned        = false;
    bool                  owns_ptr      = true;
    bool                  pinned_alloc  = false;  // true if allocated via sycl::malloc_host
    cache_layout_xmx_info xmx_info      = {};
};

// Host cache for canonical layouts (pinned-first, pageable fallback)
class host_cache {
  public:
    host_cache(sycl::queue & queue, size_t budget_bytes);
    ~host_cache();

    host_cache(const host_cache &)             = delete;
    host_cache & operator=(const host_cache &) = delete;
    host_cache(host_cache &&)                  = delete;
    host_cache & operator=(host_cache &&)      = delete;

    void * ensure_cached_alloc(const void *     key_ptr,
                               const void *     src_ptr,
                               size_t           src_size,
                               size_t           dst_size,
                               cache_entry_type type,
                               int              layer_id,
                               int              expert_id,
                               ggml_layout_mode layout,
                               bool             validate_content,
                               bool *           needs_fill,
                               bool *           pinned_alloc_out,
                               const cache_layout_xmx_info * xmx_info);

    bool is_cached(const void * key_ptr, ggml_layout_mode layout) const;
    void * get(const void * key_ptr, ggml_layout_mode layout);
    void remove(const void * key_ptr, cache_entry_type type, int layer_id, int expert_id, ggml_layout_mode layout);

    void pin(const void * key_ptr, ggml_layout_mode layout);
    void unpin(const void * key_ptr, ggml_layout_mode layout);
    void unpin_all();

    size_t used() const { return used_.load(); }
    size_t budget() const { return budget_; }
    size_t evict(size_t bytes_needed);

  private:
    size_t evict_one();
    float  compute_score(const host_cache_entry & entry) const;
    void   free_entry(host_cache_entry & entry);

    sycl::queue &        queue_;
    size_t               budget_     = 0;
    std::atomic<size_t>  used_{ 0 };
    std::atomic<int64_t> time_{ 0 };

    std::unordered_map<unified_cache_key, host_cache_entry, unified_cache_key_hash> entries_;
    std::unordered_map<unified_cache_ptr_key, unified_cache_key, unified_cache_ptr_key_hash> ptr_to_key_;

    static constexpr float DECAY_ALPHA = 0.01f;

    mutable std::mutex mutex_;
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
    unified_cache(sycl::queue & queue, size_t budget_bytes, size_t staging_size = 64 * 1024 * 1024);
    ~unified_cache();

    // Non-copyable, non-movable
    unified_cache(const unified_cache &)             = delete;
    unified_cache & operator=(const unified_cache &) = delete;
    unified_cache(unified_cache &&)                  = delete;
    unified_cache & operator=(unified_cache &&)      = delete;

    // === Primary API ===

    // Ensure a weight is cached, loading from src_ptr if needed
    // Returns device pointer, or nullptr if cache is full and eviction failed
    //
    // key_ptr: Stable identifier for cache lookup
    //   - For dense weights: GGUF-based identity pointer (ggml_backend_sycl_get_weight_cache_key)
    //   - For MoE experts: per-expert cache key (cache_uuid + expert_id)
    // src_ptr: Source data to copy from (may change for same key_ptr)
    // If src_ptr differs from cached entry's src_ptr, data is re-uploaded
    //
    // For dense weights: layer_id=-1, expert_id=-1
    // For MoE experts: layer_id=N, expert_id=M
    void * ensure_cached(const void *     key_ptr,
                         const void *     src_ptr,
                         size_t           size,
                         cache_entry_type type,
                         int              layer_id         = -1,
                         int              expert_id        = -1,
                         ggml_layout_mode layout           = GGML_LAYOUT_AOS,
                         bool             validate_content = false);
    // Allocate cache entry without copying data (caller will fill device_ptr).
    // Uses src_ptr/src_size only for content hash and change detection.
    void * ensure_cached_alloc(const void *     key_ptr,
                               const void *     src_ptr,
                               size_t           src_size,
                               size_t           alloc_size,
                               cache_entry_type type,
                               int              layer_id,
                               int              expert_id,
                               ggml_layout_mode layout,
                               bool             validate_content,
                               bool *           needs_fill);

    // Ensure a weight is cached in a specific layout with graph-safe async fill.
    cache_layout_result ensure_cached_layout(const cache_layout_request &     request,
                                             const std::vector<sycl::event> & deps);

    // Check if entry is cached (without loading)
    bool is_cached(const void * key_ptr, ggml_layout_mode layout) const;
    bool is_cached_any(const void * key_ptr) const;

    // Get device pointer for cached entry (returns nullptr if not cached)
    void * get(const void * key_ptr, ggml_layout_mode layout);

    // Remove a cache entry (free device memory).
    void remove(const void * key_ptr, cache_entry_type type, int layer_id, int expert_id, ggml_layout_mode layout);

    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not in cache

    // === Pinning for Graphs ===

    void pin(const void * key_ptr, ggml_layout_mode layout);
    void unpin(const void * key_ptr, ggml_layout_mode layout);
    void unpin_experts();
    void unpin_all();
    bool is_pinned(const void * key_ptr, ggml_layout_mode layout) const;

    // === Memory Management ===

    // Current memory used
    size_t used() const { return used_.load(); }

    // Total budget
    size_t budget() const { return budget_; }

    // Available memory
    size_t available() const {
        const size_t used = used_.load();
        return budget_ > used ? budget_ - used : 0;
    }

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
    size_t used_bytes(cache_entry_type type) const;

    void print_stats() const;
    void reset_stats();

    // Reserve non-cache runtime buffers (compute, KV, etc.)
    void update_reserved_bytes(size_t reserved_bytes);

    // Hot set control (MoE experts)
    void set_hot(const void *     key_ptr,
                 cache_entry_type type,
                 int              layer_id,
                 int              expert_id,
                 ggml_layout_mode layout,
                 bool             hot);
    void clear_hot_experts(int layer_id);

    // Track cache entries pinned for in-flight kernels.
    void unpin_on_event(const void * key_ptr, ggml_layout_mode layout, const sycl::event & event);

  private:
    // Evict lowest-scoring entry to make room for new_size bytes
    // Returns true if eviction succeeded, false if all entries are pinned
    size_t evict_one(size_t new_size);

    struct deferred_free_entry {
        void *      ptr       = nullptr;
        size_t      size      = 0;
        bool        has_event = false;
        sycl::event event;
    };

    // Compute eviction score: higher = more valuable (keep longer)
    // score = access_count * exp(-decay * age)
    float compute_score(const unified_cache_entry & entry) const;

    // Copy data from mmap to device via staging
    void        copy_to_device(void * dst, const void * src, size_t size);
    sycl::event copy_to_device_async(void * dst, const void * src, size_t size, const std::vector<sycl::event> & deps);
    static bool event_complete(const sycl::event & evt);
    sycl::event submit_barrier(const std::vector<sycl::event> & deps);
    sycl::event submit_barrier_all();
    void        process_deferred_frees();
    void        enqueue_deferred_free(void * ptr, size_t size);

    sycl::queue &        queue_;
    size_t               budget_;       // Total GPU memory budget (after reservations)
    size_t               base_budget_;  // Raw cache budget before reservations
    size_t               reserved_;     // Runtime reservation applied to budget_
    std::atomic<size_t>  used_{ 0 };  // Current usage
    std::atomic<int64_t> time_{ 0 };  // Monotonic counter

    // Cache storage: (key_ptr, layout) -> entry
    std::unordered_map<unified_cache_key, unified_cache_entry, unified_cache_key_hash> entries_;

    // Fast lookup by key_ptr + layout
    std::unordered_map<unified_cache_ptr_key, unified_cache_key, unified_cache_ptr_key_hash> ptr_to_key_;

    // Staging buffer for mmap -> device transfers
    void *     staging_      = nullptr;
    size_t     staging_size_ = 0;
    std::mutex staging_mutex_;

    // Deferred frees to avoid releasing buffers while in flight.
    std::vector<deferred_free_entry> deferred_frees_;

    struct inflight_unpin_entry {
        const void *  key_ptr   = nullptr;
        ggml_layout_mode layout = GGML_LAYOUT_AOS;
        bool           has_event = false;
        sycl::event    event;
    };

    // Entries pinned for in-flight kernels.
    std::vector<inflight_unpin_entry> inflight_unpins_;

    // Stats
    mutable std::atomic<size_t> hits_{ 0 };
    mutable std::atomic<size_t> misses_{ 0 };

    static constexpr float DECAY_ALPHA = 0.01f;

    mutable std::mutex              mutex_;
    mutable std::condition_variable entry_cv_;
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
unified_cache * get_unified_cache(sycl::queue & queue);

// Get unified cache for a specific device ID
// Useful when device ID is known but queue isn't available
unified_cache * get_unified_cache_for_device(int device_id);

// Check if unified cache is enabled (via env var or auto-detection)
bool unified_cache_enabled();

// Set unified cache budget (call before first use)
// In PER_DEVICE mode, this sets budget per device
void set_unified_cache_budget(size_t bytes);
// Set unified cache budget as percentage of free VRAM (call before first use)
void set_unified_cache_budget_pct(int pct);
// Set unified host cache budget as percentage of total system RAM (call before first use)
void set_unified_cache_host_budget_pct(int pct);

// Track runtime buffers that must not be evicted from VRAM (compute, KV, etc.)
void   unified_cache_add_runtime_bytes(int device, size_t bytes);
void   unified_cache_sub_runtime_bytes(int device, size_t bytes);
size_t unified_cache_get_runtime_bytes(int device);

// Host cache accessors (canonical layouts in host memory)
host_cache * get_host_cache(sycl::queue & queue);
host_cache * get_host_cache_for_device(int device_id);

// === MoE Expert Caching API (for compatibility with existing expert_cache usage) ===

// Cache an expert weight tensor via unified cache
// Returns device pointer or nullptr on failure
// mmap_ptr: pointer to expert data in mmap'd file
// expert_size: size of expert in bytes
// layer_id, expert_id: for identification and debugging
void * cache_moe_expert(sycl::queue & queue, const void * mmap_ptr, size_t expert_size, int layer_id, int expert_id);

// Check if expert is cached
bool is_expert_cached(const void * mmap_ptr);

// Get cached expert pointer (returns nullptr if not cached)
void * get_cached_expert(const void * mmap_ptr);

// NOTE: mark_expert_reordered/is_expert_reordered removed
// Reorder state is tracked in tensor->extra->optimized_feature, not in cache

// Pin expert to prevent eviction during graph execution
void pin_expert(const void * mmap_ptr);

// Unpin all experts
void unpin_all_experts();

// === Reorder Callback Support ===

// Callback for SoA weight reordering (called after cache miss)
// data_device: GPU pointer to cached data
// ncols, nrows: tensor dimensions
// size: total size in bytes
// stream: SYCL queue for async operations
using moe_reorder_callback_fn =
    void (*)(uint8_t * data_device, int ncols, int nrows, size_t size, sycl::queue * stream);

// Set the global reorder callback (typically set once during initialization)
void set_moe_reorder_callback(moe_reorder_callback_fn callback);

// Cache an expert with optional SoA reorder (for MXFP4 and similar types)
// Returns device pointer or nullptr on failure
// Applies reorder callback if set and entry is newly cached
// NOTE: Caller must track reorder state in tensor->extra->optimized_feature
void * cache_moe_expert_with_reorder(sycl::queue & queue,
                                     const void *  mmap_ptr,
                                     size_t        expert_size,
                                     int           layer_id,
                                     int           expert_id,
                                     int           ncols,
                                     int           nrows);

// === Shutdown API ===

// Shutdown the unified cache system before SYCL runtime destruction
// Call this during ggml_backend_sycl_free() to avoid static destruction order issues
// After calling this, the cache destructors will skip sycl::free() calls
void shutdown_unified_cache();

}  // namespace ggml_sycl

#endif  // GGML_SYCL_UNIFIED_CACHE_HPP
