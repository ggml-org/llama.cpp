//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_UNIFIED_CACHE_HPP
#define GGML_SYCL_UNIFIED_CACHE_HPP

#include "dpct/helper.hpp"
#include "pinned-pool.hpp"
#include "device-pool.hpp"
#include "ggml-sycl.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if !defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace ggml_sycl {

namespace detail {

static constexpr uint64_t k_cache_guard_magic = 0xC0DECA5EC0DECA5EULL;

struct alignas(16) cache_guard_header {
    uint64_t magic        = k_cache_guard_magic;
    size_t   size         = 0;
    size_t   mapping_size = 0;
    void *   mapping_base = nullptr;
};

static inline size_t cache_hash_combine(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

static inline bool cache_id_equal(const ggml_sycl_cache_id & a, const ggml_sycl_cache_id & b) {
    if (a.valid != b.valid || a.model_id != b.model_id || a.has_gguf != b.has_gguf || a.file_idx != b.file_idx ||
        a.file_offs != b.file_offs || a.nbytes != b.nbytes || a.name_hash != b.name_hash || a.type != b.type ||
        a.tp_sharded != b.tp_sharded || a.tp_rank != b.tp_rank || a.tp_world_size != b.tp_world_size ||
        a.aux_id != b.aux_id) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a.ne[i] != b.ne[i] || a.tp_local_ne[i] != b.tp_local_ne[i] || a.tp_offset_ne[i] != b.tp_offset_ne[i]) {
            return false;
        }
    }
    return true;
}

struct cache_id_equal_fn {
    bool operator()(const ggml_sycl_cache_id & a, const ggml_sycl_cache_id & b) const {
        return cache_id_equal(a, b);
    }
};

struct cache_id_hash {
    size_t operator()(const ggml_sycl_cache_id & id) const {
        size_t h = 0;
        h        = cache_hash_combine(h, std::hash<bool>()(id.valid));
        h        = cache_hash_combine(h, std::hash<uint64_t>()(id.model_id));
        h        = cache_hash_combine(h, std::hash<bool>()(id.has_gguf));
        h        = cache_hash_combine(h, std::hash<uint16_t>()(id.file_idx));
        h        = cache_hash_combine(h, std::hash<size_t>()(id.file_offs));
        h        = cache_hash_combine(h, std::hash<size_t>()(id.nbytes));
        h        = cache_hash_combine(h, std::hash<uint64_t>()(id.name_hash));
        h        = cache_hash_combine(h, std::hash<int>()(id.type));
        h        = cache_hash_combine(h, std::hash<bool>()(id.tp_sharded));
        h        = cache_hash_combine(h, std::hash<int>()(id.tp_rank));
        h        = cache_hash_combine(h, std::hash<int>()(id.tp_world_size));
        h        = cache_hash_combine(h, std::hash<uint64_t>()(id.aux_id));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            h = cache_hash_combine(h, std::hash<int64_t>()(id.ne[i]));
            h = cache_hash_combine(h, std::hash<int64_t>()(id.tp_local_ne[i]));
            h = cache_hash_combine(h, std::hash<int64_t>()(id.tp_offset_ne[i]));
        }
        return h;
    }
};

inline bool cache_guard_pages_enabled() {
    const char * env = std::getenv("GGML_SYCL_CACHE_GUARD_PAGES");
    if (!env || env[0] == '\0') {
        env = std::getenv("GGML_SYCL_UNIFIED_CACHE_GUARD_PAGES");
    }
    return env && std::atoi(env) != 0;
}

inline size_t cache_guard_page_size() {
#if !defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
    const long page_size_long = sysconf(_SC_PAGESIZE);
    return page_size_long > 0 ? static_cast<size_t>(page_size_long) : 4096;
#else
    return 4096;
#endif
}

template <typename T>
struct cache_guard_allocator {
    using value_type = T;

    cache_guard_allocator() noexcept = default;
    template <typename U>
    cache_guard_allocator(const cache_guard_allocator<U> &) noexcept {}

    T * allocate(std::size_t n) {
        if (!cache_guard_pages_enabled()) {
            return std::allocator<T>{}.allocate(n);
        }

        const size_t size_bytes = n * sizeof(T);
        if (size_bytes == 0 || (size_bytes % alignof(T)) != 0) {
            return std::allocator<T>{}.allocate(n);
        }

#if !defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
        const size_t page_size = cache_guard_page_size();
        const size_t usable =
            ((sizeof(cache_guard_header) + size_bytes + page_size - 1) / page_size) * page_size;
        const size_t mapping_size = usable + page_size;
        void * mapping = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED || !mapping) {
            throw std::bad_alloc();
        }
        if (mprotect(static_cast<uint8_t *>(mapping) + usable, page_size, PROT_NONE) != 0) {
            munmap(mapping, mapping_size);
            throw std::bad_alloc();
        }

        uint8_t * data_end = static_cast<uint8_t *>(mapping) + usable;
        uint8_t * data     = data_end - size_bytes;
        auto * header      = reinterpret_cast<cache_guard_header *>(data - sizeof(cache_guard_header));
        header->magic        = k_cache_guard_magic;
        header->size         = size_bytes;
        header->mapping_size = mapping_size;
        header->mapping_base = mapping;

        return reinterpret_cast<T *>(data);
#else
        return std::allocator<T>{}.allocate(n);
#endif
    }

    void deallocate(T * ptr, std::size_t n) noexcept {
        if (!ptr) {
            return;
        }
        if (!cache_guard_pages_enabled()) {
            std::allocator<T>{}.deallocate(ptr, n);
            return;
        }

        auto * header = reinterpret_cast<cache_guard_header *>(reinterpret_cast<uint8_t *>(ptr) - sizeof(cache_guard_header));
        if (!header || header->magic != k_cache_guard_magic || header->mapping_size == 0) {
            std::allocator<T>{}.deallocate(ptr, n);
            return;
        }

#if !defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
        if (header->mapping_base && header->mapping_size > 0) {
            munmap(header->mapping_base, header->mapping_size);
        }
#else
        (void) header;
#endif
    }

    template <typename U>
    struct rebind {
        using other = cache_guard_allocator<U>;
    };
};

template <typename T, typename U>
inline bool operator==(const cache_guard_allocator<T> &, const cache_guard_allocator<U> &) noexcept {
    return true;
}
template <typename T, typename U>
inline bool operator!=(const cache_guard_allocator<T> &, const cache_guard_allocator<U> &) noexcept {
    return false;
}

}  // namespace detail

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

// Memory location for cached pointers/buffers
enum class cache_location {
    DEVICE,
    HOST_PINNED,
    HOST_MMAP,
};

// XMX layout metadata carried with cache entries/results
struct cache_layout_xmx_info {
    int64_t tile_n        = 0;
    int64_t tile_k        = 0;
    int64_t n_tile_groups = 0;
};

struct cache_ptr_view {
    void *                ptr      = nullptr;
    size_t                size     = 0;
    ggml_layout_mode      layout   = GGML_LAYOUT_AOS;
    int64_t               onednn_pack_m = 0;
    cache_location        location = cache_location::DEVICE;
    cache_entry_type      type     = cache_entry_type::DENSE_WEIGHT;
    int                   layer_id = -1;
    int                   expert_id = -1;
    cache_layout_xmx_info xmx_info = {};
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
    ggml_sycl_cache_id    key              = {};
    const void *          src_ptr          = nullptr;
    size_t                src_size         = 0;
    size_t                dst_size         = 0;
    cache_entry_type      type             = cache_entry_type::DENSE_WEIGHT;
    int                   layer_id         = -1;
    int                   expert_id        = -1;
    ggml_layout_mode      layout           = GGML_LAYOUT_AOS;
    int64_t               onednn_pack_m    = 0;
    bool                  validate_content = false;
    bool                  prefer_host      = false;
    cache_layout_xmx_info xmx_info         = {};
    cache_layout_fill_fn  fill_fn          = nullptr;
    const void *          fill_ctx         = nullptr;
};

struct cache_layout_result {
    void *                device_ptr = nullptr;
    size_t                size       = 0;
    ggml_layout_mode      layout     = GGML_LAYOUT_AOS;
    int64_t               onednn_pack_m = 0;
    cache_layout_xmx_info xmx_info   = {};
    sycl::event           event;
    cache_layout_status   status        = cache_layout_status::FAILED;
    bool                  host_resident = false;  // true if pointer is in host memory (fallback when VRAM full)
    cache_location        location      = cache_location::DEVICE;
};

// Key for identifying a cached entry
struct unified_cache_key {
    cache_entry_type     type;
    ggml_sycl_cache_id   id;        // Identity for weights/MoE (no layout)
    int                 layer_id;   // Layer ID (for expert identification)
    int                 expert_id;  // Expert ID (-1 for dense weights)

    bool operator==(const unified_cache_key & other) const {
        return type == other.type && detail::cache_id_equal(id, other.id) && layer_id == other.layer_id &&
               expert_id == other.expert_id;
    }
};

struct unified_cache_key_hash {
    size_t operator()(const unified_cache_key & k) const {
        size_t h = 0;
        h        = detail::cache_hash_combine(h, std::hash<int>()(static_cast<int>(k.type)));
        h        = detail::cache_hash_combine(h, detail::cache_id_hash{}(k.id));
        h        = detail::cache_hash_combine(h, std::hash<int>()(k.layer_id));
        h        = detail::cache_hash_combine(h, std::hash<int>()(k.expert_id));
        return h;
    }
};

// Metadata for a cached entry
struct unified_cache_entry {
    void *                device_ptr;       // GPU memory pointer (or host memory if host_resident)
    const void *          src_ptr;          // Source data pointer (for change detection)
    uint64_t              content_hash;     // Simple hash of content (first/last bytes)
    size_t                size;             // Size in bytes
    cache_entry_type      type;             // Dense or MoE
    int                   layer_id;         // Layer ID
    int                   expert_id;        // Expert ID (-1 for dense)
    ggml_layout_mode      layout;           // Target layout for this entry
    int64_t               onednn_pack_m;    // M dimension used for ONEDNN_PACKED/ONEDNN_WOQ (0 when unused)
    cache_layout_xmx_info xmx_info;         // XMX metadata (when applicable)
    uint32_t              access_count;     // Access frequency for LFU
    int64_t               last_access;      // Timestamp for recency
    bool                  pinned;           // Protected from eviction
    bool                  hot;              // Hot-set hint for MoE experts
    cache_entry_state     state;            // READY vs IN_PROGRESS
    bool                  has_ready_event;  // True if ready_event is valid
    sycl::event           ready_event;      // Completion event for IN_PROGRESS entries
    bool                  host_resident;    // Entry lives in host memory, not device (fallback when VRAM full)
    cache_location        location;         // DEVICE/HOST_PINNED/HOST_MMAP
    bool                  pool_allocated;   // True if device_ptr was sub-allocated from layout_pool_
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here
};

// Host cache entry (canonical layouts in host memory)
struct host_cache_entry {
    void *                host_ptr     = nullptr;
    const void *          src_ptr      = nullptr;
    uint64_t              content_hash = 0;
    size_t                size         = 0;
    size_t                guard_size   = 0;
    cache_entry_type      type         = cache_entry_type::DENSE_WEIGHT;
    int                   layer_id     = -1;
    int                   expert_id    = -1;
    ggml_layout_mode      layout       = GGML_LAYOUT_AOS;
    uint32_t              access_count = 0;
    int64_t               last_access  = 0;
    bool                  pinned       = false;
    bool                  owns_ptr     = true;
    bool                  pinned_alloc = false;  // true if allocated via sycl::malloc_host
    cache_location        location     = cache_location::HOST_PINNED;
    cache_layout_xmx_info xmx_info     = {};
};

// Host cache for canonical layouts (pinned-first, mmap alias fallback)
class host_cache {
  public:
    host_cache(sycl::queue & queue, size_t budget_bytes);
    ~host_cache();

    void * allocate_pinned_runtime(size_t size, size_t alignment = 64);
    void   free_pinned_runtime(void * ptr, size_t size);

    host_cache(const host_cache &)             = delete;
    host_cache & operator=(const host_cache &) = delete;
    host_cache(host_cache &&)                  = delete;
    host_cache & operator=(host_cache &&)      = delete;

    void * ensure_cached_alloc(const ggml_sycl_cache_id &     key,
                               const void *                  src_ptr,
                               size_t                        src_size,
                               size_t                        dst_size,
                               cache_entry_type              type,
                               int                           layer_id,
                               int                           expert_id,
                               ggml_layout_mode              layout,
                               bool                          validate_content,
                               bool *                        needs_fill,
                               bool *                        pinned_alloc_out,
                               cache_location *              location_out,
                               const cache_layout_xmx_info * xmx_info);

    bool   is_cached(const ggml_sycl_cache_id & key, cache_entry_type type, int layer_id, int expert_id,
                     ggml_layout_mode layout) const;
    void * get(const ggml_sycl_cache_id & key, cache_entry_type type, int layer_id, int expert_id,
               ggml_layout_mode layout);
    cache_location get_location(const ggml_sycl_cache_id & key, cache_entry_type type, int layer_id, int expert_id,
                                ggml_layout_mode layout) const;
    bool   check_guard(const ggml_sycl_cache_id & key, cache_entry_type type, int layer_id, int expert_id,
                       ggml_layout_mode layout) const;
    bool   check_all_guards(const char * where);
    void   remove(const ggml_sycl_cache_id & key, cache_entry_type type, int layer_id, int expert_id,
                  ggml_layout_mode layout);

    void pin(const ggml_sycl_cache_id & key, cache_entry_type type, int layer_id, int expert_id,
             ggml_layout_mode layout);
    void unpin(const ggml_sycl_cache_id & key, cache_entry_type type, int layer_id, int expert_id,
               ggml_layout_mode layout);
    void unpin_all();

    size_t used() const { return used_.load(); }

    size_t budget() const { return budget_; }

    bool is_budget_exceeded() const { return budget_exceeded_; }

    size_t evict(size_t bytes_needed);

    // Reserve non-cache runtime buffers (compute, KV, etc.)
    void update_reserved_bytes(size_t reserved_bytes);

  private:
    size_t evict_one();
    float  compute_score(const host_cache_entry & entry) const;
    void   free_entry(host_cache_entry & entry);

    // Saturating subtract from used_ to prevent underflow to SIZE_MAX.
    // Logs a warning if underflow is detected, then clamps to 0.
    void saturating_sub_used(size_t bytes) {
        size_t prev = used_.load(std::memory_order_relaxed);
        for (;;) {
            const size_t next = (prev >= bytes) ? (prev - bytes) : 0;
            if (used_.compare_exchange_weak(prev, next, std::memory_order_relaxed)) {
                if (prev < bytes) {
                    fprintf(stderr, "[HOST-CACHE] used_ underflow prevented: prev=%zu sub=%zu clamped to 0\n",
                            prev, bytes);
                }
                return;
            }
        }
    }

    sycl::queue &        queue_;
    size_t               budget_ = 0;
    size_t               base_budget_ = 0;
    size_t               reserved_ = 0;
    bool                 budget_exceeded_ = false;
    std::atomic<size_t>  used_{ 0 };
    std::atomic<int64_t> time_{ 0 };

    // Pinned memory pool (replaces direct malloc_host calls)
    // Uses 8GB chunks to bypass Intel Level Zero's ~11GB per-allocation limit
    std::unique_ptr<pinned_chunk_pool> pinned_pool_;

    std::unordered_map<unified_cache_key,
                       host_cache_entry,
                       unified_cache_key_hash,
                       std::equal_to<unified_cache_key>,
                       detail::cache_guard_allocator<std::pair<const unified_cache_key, host_cache_entry>>>
        entries_;

    static constexpr float DECAY_ALPHA = 0.01f;

    mutable std::mutex mutex_;
};

// Weight set for a transformer layer (for bulk pinning)
// Supports standard dense transformer architecture with attention + FFN blocks.
// For MoE models, use pin/unpin directly with expert_id.
struct layer_weight_set {
    ggml_sycl_cache_id attn_norm;                                    // Attention layer norm
    ggml_sycl_cache_id q_proj, k_proj, v_proj, o_proj;               // Attention projections
    ggml_sycl_cache_id ffn_norm;                                     // FFN layer norm
    ggml_sycl_cache_id gate_proj, up_proj, down_proj;                // FFN projections (SwiGLU)
    // Optional: Some architectures have additional weights
    ggml_sycl_cache_id attn_qkv_proj;                                // Fused QKV for some models
    ggml_sycl_cache_id ffn_gate_up_proj;                             // Fused gate+up for some models
};

// Prefetch priority for async layer prefetch queue.
// Defined locally to avoid circular dependency with unified-kernel.hpp.
enum class prefetch_priority {
    LOW,
    NORMAL,
    HIGH
};

// Result of await_layer - contains device pointers to all layer weights.
// Populated by looking up each weight in the cache after prefetch completes.
struct layer_weight_pointers {
    void * attn_norm        = nullptr;
    void * q_proj           = nullptr;
    void * k_proj           = nullptr;
    void * v_proj           = nullptr;
    void * o_proj           = nullptr;
    void * ffn_norm         = nullptr;
    void * gate_proj        = nullptr;
    void * up_proj          = nullptr;
    void * down_proj        = nullptr;
    void * attn_qkv_proj    = nullptr;  // Fused QKV (optional)
    void * ffn_gate_up_proj = nullptr;  // Fused gate+up (optional)
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
    unified_cache(sycl::queue & queue,
                  size_t       budget_bytes,
                  size_t       staging_size       = 64 * 1024 * 1024,
                  size_t       dma_reserved_bytes = 0);
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
    // key: Stable identifier for cache lookup (no pointers, no layout)
    //   - For dense weights: GGUF data identity + tensor metadata (model_id scoped)
    //   - For MoE experts: model_id + aux_id (cache_uuid) + expert_id/layer_id
    // src_ptr: Source data to copy from (may change for same key)
    // If src_ptr differs from cached entry's src_ptr, data is re-uploaded
    //
    // For dense weights: layer_id=-1, expert_id=-1
    // For MoE experts: layer_id=N, expert_id=M
    void * ensure_cached(const ggml_sycl_cache_id & key,
                         const void *     src_ptr,
                         size_t           size,
                         cache_entry_type type,
                         int              layer_id         = -1,
                         int              expert_id        = -1,
                         ggml_layout_mode layout           = GGML_LAYOUT_AOS,
                         bool             validate_content = false);
    // Allocate cache entry without copying data (caller will fill device_ptr).
    // Uses src_ptr/src_size only for content hash and change detection.
    void * ensure_cached_alloc(const ggml_sycl_cache_id & key,
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
    bool is_cached(const ggml_sycl_cache_id & key, ggml_layout_mode layout) const;
    bool is_cached_any(const ggml_sycl_cache_id & key) const;

    // Get device pointer for cached entry (returns nullptr if not cached)
    void * get(const ggml_sycl_cache_id & key, ggml_layout_mode layout);
    cache_ptr_view get_view(const ggml_sycl_cache_id & key, ggml_layout_mode layout);

    // Get device pointer, waiting for IN_PROGRESS entries to complete.
    // Returns nullptr only if entry doesn't exist at all.
    // This prevents returning mmap'd pointers when cache fill is in flight.
    void * get_or_wait(const ggml_sycl_cache_id & key, ggml_layout_mode layout);

    // Fast path for cache lookup using shared_lock (reader-writer lock).
    // Returns device_ptr if entry exists, is READY, and layout matches.
    // Otherwise returns nullptr. Does not update LRU/LFU stats.
    void * try_get_cached_fast(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout);

    // Fallback lookup by data pointer (aux_id) when primary key lookup fails.
    // Used during graph recording to find entries with aliased tensor names.
    // Only searches entries where src_ptr matches the given data_ptr.
    void * get_by_data_ptr(void * data_ptr, size_t nbytes, ggml_layout_mode layout);

    // Remove a cache entry (free device memory).
    void remove(const ggml_sycl_cache_id & key, cache_entry_type type, int layer_id, int expert_id,
                ggml_layout_mode layout);

    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not in cache

    // === Pinning for Graphs ===

    void pin(const ggml_sycl_cache_id & key, ggml_layout_mode layout);
    void unpin(const ggml_sycl_cache_id & key, ggml_layout_mode layout);
    void unpin_experts();
    void unpin_all();
    bool is_pinned(const ggml_sycl_cache_id & key, ggml_layout_mode layout) const;

    // === Bulk Pinning for Persistent Kernels ===
    // Pin all weights for a layer at once. Returns count of successfully pinned entries.
    // Only pins entries that exist in cache with matching layout.
    int pin_layer_weights(int layer_id, const layer_weight_set & weights, ggml_layout_mode layout);

    // Unpin all weights for a layer. Requires the same weight set used for pinning.
    void unpin_layer_weights(int layer_id, const layer_weight_set & weights, ggml_layout_mode layout);

    // Pin entire model weights (all layers). Returns total count of pinned entries.
    // layers: vector of layer_weight_set for each layer (index = layer_id)
    int pin_model_weights(int n_layers, const std::vector<layer_weight_set> & layers, ggml_layout_mode layout);

    // === Async Layer Prefetch for Persistent Kernels ===
    // Queue a layer for background prefetch (pins weights so they won't be evicted).
    // The prefetch worker thread will pin all weights for the layer and mark it ready.
    // Priority HIGH requests are placed at the front of the queue.
    void queue_layer_prefetch(int                      layer_id,
                              const layer_weight_set & weights,
                              ggml_layout_mode         layout,
                              prefetch_priority        priority = prefetch_priority::NORMAL);

    // Block until the specified layer is prefetched and ready.
    // Returns pointers to all cached weights for the layer.
    layer_weight_pointers await_layer(int layer_id);

    // Check if a layer has been prefetched and is ready (non-blocking).
    bool is_layer_ready(int layer_id) const;

    // Release a prefetched layer (unpins weights, allows eviction).
    void release_layer(int layer_id);

    // === Memory Management ===

    // Current memory used
    size_t used() const { return used_.load(); }

    // Total budget
    size_t budget() const { return budget_; }

    // Check if budget is exceeded (used > budget after eviction)
    bool is_budget_exceeded() const { return budget_exceeded_; }

    // Returns true if any weight has been evicted from device memory.
    // Once true, never resets. Used to disable graph replay with stale baked pointers.
    bool has_evictions() const { return has_evictions_.load(std::memory_order_acquire); }

    // Available memory
    size_t available() const {
        const size_t used = used_.load();
        return budget_ > used ? budget_ - used : 0;
    }

    // Raw VRAM budget before runtime reservations
    size_t base_budget() const { return base_budget_; }

    // Bytes currently occupied by cached weights
    size_t weight_bytes() const { return used_.load(); }

    // Available VRAM for non-weight allocations (KV, compute, staging).
    // This is the budget headroom after weights + current runtime reservations.
    // Higher-level code uses this to size KV cache and compute buffers.
    size_t available_for_compute() const {
        return available();  // budget_ - used_, already accounts for reserved_
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
    // Debug/testing helper: verify internal maps are consistent.
    bool validate() const;

    // Reserve non-cache runtime buffers (compute, KV, etc.)
    void update_reserved_bytes(size_t reserved_bytes);

    // Hot set control (MoE experts)
    void set_hot(const ggml_sycl_cache_id & key,
                 cache_entry_type          type,
                 int                       layer_id,
                 int                       expert_id,
                 ggml_layout_mode          layout,
                 bool                      hot);
    void clear_hot_experts(int layer_id);

    // Track cache entries pinned for in-flight kernels.
    void unpin_on_event(const ggml_sycl_cache_id & key, ggml_layout_mode layout, const sycl::event & event);

    using dma_stream_slice_fn = sycl::event (*)(sycl::queue &                    queue,
                                                void *                           device_slice,
                                                size_t                           slice_bytes,
                                                size_t                           offset_bytes,
                                                const void *                     ctx,
                                                const std::vector<sycl::event> & deps);
    using dma_stream_copy_fn = sycl::event (*)(sycl::queue &                    queue,
                                               void *                           device_slice,
                                               size_t                           slice_bytes,
                                               size_t                           offset_bytes,
                                               const void *                     src_ptr,
                                               size_t                           src_size,
                                               const void *                     ctx,
                                               const std::vector<sycl::event> & deps);

    struct dma_staging_buffers {
        void ** buffers     = nullptr;
        size_t  count       = 0;
        size_t  slice_bytes = 0;
    };

    // Device-resident DMA staging buffer pool (for streaming).
    bool get_dma_staging_buffers(size_t slice_bytes, size_t count, dma_staging_buffers & out);

    struct dma_stream_result {
        sycl::event event;
        bool        ok              = false;
        bool        used_mmap_direct = false;
        bool        mmap_direct_failed = false;
        size_t      slices          = 0;
        size_t      slice_bytes     = 0;
        size_t      buffer_count    = 0;
    };

    dma_stream_result stream_dma(const cache_ptr_view &            src,
                                 size_t                            total_bytes,
                                 size_t                            slice_bytes,
                                 size_t                            buffer_count,
                                 dma_stream_slice_fn               slice_fn,
                                 const void *                      ctx,
                                 const std::vector<sycl::event> &   deps,
                                 dma_stream_copy_fn                copy_fn = nullptr);

    // Defer freeing host allocations until the associated event completes.
    void defer_host_free(void * ptr, size_t size, const sycl::event & event);

    // === OneDNN FP16 Scratch Buffers ===
    // Pre-allocated buffers for dequantized weights and converted activations.
    // These avoid per-op allocations that cause OOM with large KV caches.

    struct onednn_scratch_buffers {
        void * weights     = nullptr;  // Dequantized weights (N*K*2 bytes)
        void * activations = nullptr;  // Converted activations (M*K*2 bytes)
        size_t weights_size     = 0;
        size_t activations_size = 0;
    };

    // Reserve scratch buffers for oneDNN FP16 path.
    // Call once during model load with max dimensions.
    // weights_size: max(N*K*2) across all layers (usually FFN down: 14336*4096*2)
    // activations_size: max(M*K*2) where M=max_batch, K=max_dim
    bool reserve_onednn_scratch(size_t weights_size, size_t activations_size);

    // Get scratch buffers. Returns false if not reserved or sizes exceed reserved.
    // Caller must hold onednn_scratch_mutex_ via lock_onednn_scratch().
    bool get_onednn_scratch(size_t weights_needed, size_t activations_needed,
                            onednn_scratch_buffers & out);

    // Lock/unlock for scratch buffer access (RAII recommended)
    std::unique_lock<std::mutex> lock_onednn_scratch() { return std::unique_lock<std::mutex>(onednn_scratch_mutex_); }

    // Check if scratch is reserved
    bool has_onednn_scratch() const { return onednn_weights_scratch_ != nullptr; }

    // Get reserved sizes
    size_t onednn_weights_scratch_size() const { return onednn_weights_scratch_size_; }
    size_t onednn_activations_scratch_size() const { return onednn_activations_scratch_size_; }

    // === Persistent Scratch Buffers ===
    // Named scratch buffers for persistent kernels (TG optimization).
    // Unlike oneDNN scratch which is shared, these are dedicated per named buffer.
    // Used for: intermediate activations, work counters, temporary storage.

    // Reserve a persistent scratch buffer by name.
    // If buffer already exists with sufficient size, returns true without reallocating.
    // If buffer exists but is too small, it is freed and reallocated.
    // pin: if true, buffer is protected from eviction (default true)
    bool reserve_persistent_scratch(const std::string & buffer_name, size_t size_bytes, bool pin = true);

    // Get pointer to a persistent scratch buffer.
    // Returns nullptr if buffer doesn't exist.
    void * get_persistent_scratch(const std::string & buffer_name);

    // Release (free) a persistent scratch buffer.
    void release_persistent_scratch(const std::string & buffer_name);

    // Check if a persistent scratch buffer exists
    bool has_persistent_scratch(const std::string & buffer_name) const;

    // Get size of a persistent scratch buffer (0 if not found)
    size_t get_persistent_scratch_size(const std::string & buffer_name) const;

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

    struct deferred_host_free_entry {
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
    void        enqueue_deferred_host_free(void * ptr, size_t size, const sycl::event & event);

    // Saturating subtract from used_ to prevent underflow to SIZE_MAX.
    // Logs a warning if underflow is detected, then clamps to 0.
    void saturating_sub_used(size_t bytes) {
        size_t prev = used_.load(std::memory_order_relaxed);
        for (;;) {
            const size_t next = (prev >= bytes) ? (prev - bytes) : 0;
            if (used_.compare_exchange_weak(prev, next, std::memory_order_relaxed)) {
                if (prev < bytes) {
                    fprintf(stderr, "[UNIFIED-CACHE] used_ underflow prevented: prev=%zu sub=%zu clamped to 0\n",
                            prev, bytes);
                }
                return;
            }
        }
    }

    sycl::queue &        queue_;
    size_t               budget_;       // Total GPU memory budget (after reservations)
    size_t               base_budget_;  // Raw cache budget before reservations
    size_t               reserved_;     // Runtime reservation applied to budget_
    bool                 budget_exceeded_ = false;  // Set when used > budget after eviction
    std::atomic<size_t>  used_{ 0 };    // Current usage
    std::atomic<int64_t> time_{ 0 };    // Monotonic counter

    // Cache storage: (identity, type, layer/expert) -> entry
    std::unordered_map<unified_cache_key,
                       unified_cache_entry,
                       unified_cache_key_hash,
                       std::equal_to<unified_cache_key>,
                       detail::cache_guard_allocator<std::pair<const unified_cache_key, unified_cache_entry>>>
        entries_;
    std::unordered_map<ggml_sycl_cache_id,
                       unified_cache_key,
                       detail::cache_id_hash,
                       detail::cache_id_equal_fn>
        id_to_key_;

    // Layout pool: consolidates many individual layout allocations into
    // a few large contiguous chunks to reduce GPU TLB pressure.
    // All layout allocations in ensure_cached_layout() are sub-allocated from this pool.
    // The pool is destroyed (freeing all chunks) in the unified_cache destructor.
    std::unique_ptr<ggml_sycl::sycl_device_pool> layout_pool_;

    // Staging buffer for mmap -> device transfers
    void *     staging_      = nullptr;
    size_t     staging_size_ = 0;
    std::mutex staging_mutex_;

    // Device-resident DMA staging buffers (for streaming).
    std::vector<void *> dma_staging_buffers_;
    size_t              dma_slice_bytes_   = 0;
    size_t              dma_buffer_count_  = 0;
    size_t              dma_reserved_bytes_ = 0;
    std::mutex          dma_staging_mutex_;

    // OneDNN FP16 scratch buffers for prompt processing.
    // Pre-allocated to avoid per-op allocations that cause OOM with large contexts.
    // weights_scratch_: holds dequantized weights (max N*K*2 bytes)
    // activations_scratch_: holds converted activations (max M*K*2 bytes)
    void *     onednn_weights_scratch_     = nullptr;
    void *     onednn_activations_scratch_ = nullptr;
    size_t     onednn_weights_scratch_size_     = 0;
    size_t     onednn_activations_scratch_size_ = 0;
    std::mutex onednn_scratch_mutex_;

    // Persistent scratch buffers for TG optimization (persistent kernels).
    // Keyed by name for flexibility (e.g., "activations", "work_counter", "temp").
    struct persistent_scratch_entry {
        void * device_ptr = nullptr;
        size_t size       = 0;
        bool   pinned     = true;
    };
    std::unordered_map<std::string, persistent_scratch_entry> persistent_scratches_;
    mutable std::mutex                                        persistent_scratch_mutex_;

    // Deferred frees to avoid releasing buffers while in flight.
    std::vector<deferred_free_entry> deferred_frees_;
    std::vector<deferred_host_free_entry> deferred_host_frees_;

    struct inflight_unpin_entry {
        ggml_sycl_cache_id key     = {};
        ggml_layout_mode layout    = GGML_LAYOUT_AOS;
        bool             has_event = false;
        sycl::event      event;
    };

    // Entries pinned for in-flight kernels.
    std::list<inflight_unpin_entry> inflight_unpins_;

    // === Async Layer Prefetch State ===
    // Background worker thread pins layer weights ahead of kernel execution.

    struct prefetch_request {
        int              layer_id;
        layer_weight_set weights;
        ggml_layout_mode layout;
        prefetch_priority priority;
    };

    // Prefetch queue and worker thread
    std::deque<prefetch_request>     prefetch_queue_;
    std::mutex                       prefetch_mutex_;
    std::condition_variable          prefetch_cv_;
    std::thread                      prefetch_worker_;
    std::atomic<bool>                prefetch_shutdown_{ false };
    std::atomic<bool>                prefetch_started_{ false };
    std::mutex                       prefetch_lifecycle_mutex_;  // Guards start/stop of prefetch_worker_

    // Start the background prefetch worker thread (called lazily on first queue_layer_prefetch).
    void start_prefetch_worker();

    // Stop the background prefetch worker thread (called from destructor).
    void stop_prefetch_worker();

    // Per-layer ready tracking
    // Guarded by layer_state_mutex_
    std::unordered_map<int, bool>              layer_ready_;    // layer_id -> ready
    std::unordered_map<int, layer_weight_set>  layer_weights_;  // for release_layer unpin
    std::unordered_map<int, ggml_layout_mode>  layer_layouts_;  // layout used for pinning
    mutable std::mutex                         layer_state_mutex_;
    std::condition_variable                    layer_ready_cv_;

    // The prefetch worker loop (runs on background thread)
    void prefetch_worker_loop();

    // Set to true when any weight has been evicted from device to host.
    // One-way flag (false → true, never reset). Used by graph replay / persistent TG
    // to know that baked pointers may reference freed device memory.
    std::atomic<bool> has_evictions_{false};

    // Stats
    mutable std::atomic<size_t> hits_{ 0 };
    mutable std::atomic<size_t> misses_{ 0 };

    static constexpr float DECAY_ALPHA = 0.01f;

    mutable std::shared_mutex       rw_mutex_;
    mutable std::condition_variable_any entry_cv_;
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

// Classification of VRAM allocations for budget tracking and diagnostics
enum class alloc_hint : uint8_t {
    WEIGHT     = 0,  // Evictable model weights (managed by cache LRU)
    COMPUTE    = 1,  // Per-inference scratch (compute buffers, activation staging)
    EPHEMERAL  = 2,  // Per-graph temporaries (freed within graph_compute)
    PERSISTENT = 3,  // Context-lifetime buffers (persistent kernel state, DAG arrays)
    DEBUG      = 4,  // Debug/profiling allocations (env-gated, not production)
};

// Classification of runtime (non-weight) VRAM reservations for diagnostics.
// Used with unified_cache_add_runtime_bytes() to track where VRAM is consumed.
enum class runtime_category : uint8_t {
    KV_CACHE = 0,  // KV buffer allocations
    COMPUTE  = 1,  // Compute buffer pool + scratch
    STAGING  = 2,  // Expert staging, DMA staging, BLAS fallback
    GRAPH    = 3,  // Persistent TG temp allocs, get_rows_pool
    OTHER    = 4,  // Everything else (default for backward compat)
    COUNT    = 5
};

// Track runtime buffers that must not be evicted from VRAM (compute, KV, etc.)
void   unified_cache_add_runtime_bytes(int device, size_t bytes, runtime_category cat = runtime_category::OTHER);
void   unified_cache_sub_runtime_bytes(int device, size_t bytes, runtime_category cat = runtime_category::OTHER);
size_t unified_cache_get_runtime_bytes(int device);

// Query runtime bytes for a specific category on a device.
size_t unified_cache_get_runtime_bytes_by_category(int device, runtime_category cat);

void   unified_cache_add_runtime_host_bytes(size_t bytes);
void   unified_cache_sub_runtime_host_bytes(size_t bytes);
size_t unified_cache_get_runtime_host_bytes();

// Query available VRAM for non-weight allocations (KV, compute, staging).
size_t unified_cache_available_for_compute(int device);

// Raw VRAM budget before reservations (= free VRAM * budget_pct at init)
size_t unified_cache_total_managed(int device);

// Current weight bytes on device
size_t unified_cache_weight_bytes(int device);

// Log budget summary (weights, runtime, available) for diagnostics
void unified_cache_log_budget_summary(int device);

// Check if the cache budget is exceeded (eviction exhausted but used > budget)
bool unified_cache_is_budget_exceeded(int device);

// Returns true if any unified cache instance has evicted weights from device memory.
// Thread-safe. One-way flag (once true, never resets).
bool unified_cache_has_evictions();

// Budget information exported for external consumers (e.g., llama_params_fit)
struct unified_budget_info {
    int    device_id;
    size_t total_vram;            // Total device memory
    size_t budget_bytes;          // Managed budget (total * pct - headroom)
    size_t weight_bytes;          // Current weight cache usage
    size_t runtime_bytes;         // KV + compute + staging + graph
    size_t available_for_weights; // budget - runtime (what can hold weights)
    int    budget_pct;            // GGML_SYCL_VRAM_BUDGET_PCT value used
    bool   model_exceeds_vram;    // True if model > available_for_weights
    // MoE expert breakdown (non-zero only for MoE models)
    size_t expert_weight_bytes;   // Total bytes for ALL expert tensors
    size_t active_expert_bytes;   // Estimated bytes for active experts only
    int    n_expert_total;        // Total experts per layer (e.g., 8, 128)
    int    n_expert_used;         // Experts per token (e.g., 2, 4)
};

// Get budget info for a device (thread-safe snapshot)
unified_budget_info unified_cache_get_budget_info(int device);

// Get margin in bytes for llama_params_fit (how much free space after weights + runtime)
// Returns 0 if budget is exceeded
size_t unified_cache_get_margin_bytes(int device);

// Check if KV cache should be offloaded to host pinned memory.
// Returns true when VRAM is too tight to hold both weights and KV cache.
// Override: GGML_SYCL_KV_HOST=1 forces host, GGML_SYCL_KV_HOST=0 forces device.
// kv_estimate_bytes: estimated KV cache size (0 = skip margin check, use model_exceeds_vram only)
bool unified_cache_should_offload_kv(int device, size_t kv_estimate_bytes = 0);

// Calculate effective weight bytes accounting for MoE expert sparsity.
// For an 8-expert top-2 model, only ~37.5% (1.5x active ratio) of expert weights are needed.
size_t compute_moe_effective_weight_bytes(size_t total_weight_bytes,
                                          size_t expert_total_bytes,
                                          int n_expert, int n_expert_used);

// Host cache accessors (canonical layouts in host memory)
host_cache * get_host_cache(sycl::queue & queue);
host_cache * get_host_cache_for_device(int device_id);
int host_cache_guard_error_count();
void host_cache_guard_reset();
bool host_cache_guard_check_all(int device_id, const char * where);

// === OneDNN FP16 Scratch Buffer API ===
// Reserve pre-allocated scratch buffers for oneDNN FP16 prompt processing.
// This avoids per-op allocations that cause OOM when KV cache is large.
//
// Call during model load with:
//   weights_size: max(N*K*2) across all matmuls (typically FFN: 14336*4096*2 = 117MB)
//   activations_size: max_batch * max_K * 2 (e.g., 512*14336*2 = 14.6MB)
//
// The buffers are reserved from the unified cache budget and reused across all matmuls.
bool unified_cache_reserve_onednn_scratch(int device_id, size_t weights_size, size_t activations_size);

// Get scratch buffers for oneDNN FP16 path. Returns pointers and acquires lock.
// Caller must call unified_cache_release_onednn_scratch() when done.
// Returns false if scratch not reserved or sizes exceed reserved.
struct onednn_scratch_result {
    void * weights     = nullptr;
    void * activations = nullptr;
    bool   ok          = false;
};
onednn_scratch_result unified_cache_get_onednn_scratch(int device_id, size_t weights_needed, size_t activations_needed);

// Release scratch buffers (unlocks mutex for other threads)
void unified_cache_release_onednn_scratch(int device_id);

// Check if scratch buffers are reserved
bool unified_cache_has_onednn_scratch(int device_id);

// Persistent scratch buffer public API
bool unified_cache_reserve_persistent_scratch(int device_id, const char* buffer_name, size_t size_bytes, bool pin);
void * unified_cache_get_persistent_scratch(int device_id, const char* buffer_name);
void unified_cache_release_persistent_scratch(int device_id, const char* buffer_name);
bool unified_cache_has_persistent_scratch(int device_id, const char* buffer_name);
size_t unified_cache_get_persistent_scratch_size(int device_id, const char* buffer_name);

// === Bulk Weight Pinning API ===
// Pin/unpin all weights for a layer or entire model at once.
// Used by persistent kernels to ensure weights are not evicted during kernel lifetime.

// Pin all weights in a layer_weight_set. Returns count of successfully pinned entries.
// weights: pointer to layer_weight_set (opaque in C API, cast to layer_weight_set* internally)
int unified_cache_pin_layer_weights(int device_id, int layer_id, const layer_weight_set* weights, int layout);

// Unpin all weights in a layer_weight_set.
void unified_cache_unpin_layer_weights(int device_id, int layer_id, const layer_weight_set* weights, int layout);

// Pin entire model weights. Returns total count of pinned entries.
// layers: array of layer_weight_set, n_layers elements
int unified_cache_pin_model_weights(int device_id, int n_layers, const layer_weight_set* layers, int layout);

// Unpin entire model weights.
void unified_cache_unpin_model_weights(int device_id, int n_layers, const layer_weight_set* layers, int layout);

// === MoE Cache Helpers ===
// Unpin all experts
void unpin_all_experts();

// === Routing-Aware Expert Pre-staging ===
// These functions use routing indices from argsort to pre-stage only needed experts

// Result of pre-staging operation
struct prestage_result {
    int  n_staged;   // Number of experts actually staged (not already cached)
    int  n_pinned;   // Number of experts pinned (includes already-cached)
    int  n_unique;   // Number of unique experts in input
    bool success;    // True if all staging/pinning succeeded
};

// Pre-stage only the experts identified by routing indices.
// Deduplicates expert IDs, checks cache hits, stages missing experts, and pins all.
//
// Parameters:
//   queue:           SYCL queue for the device (sycl::queue*)
//   expert_ids:      Routing indices from argsort [n_expert_used * n_tokens]
//   n_expert_used:   Experts per token (typically 4)
//   n_tokens:        Batch size
//   weight_base_ptr: mmap base pointer for expert weights
//   expert_stride:   Bytes between consecutive experts
//   expert_size:     Size of each expert in bytes
//   layer_id:        Layer ID for cache key
//   n_experts_total: Total experts for bounds checking (e.g., 128)
//   device_id:       Device ID for cache lookup
//
// Returns: prestage_result with counts and success status
prestage_result prestage_routed_experts(void *          queue,
                                        const int32_t * expert_ids,
                                        int             n_expert_used,
                                        int             n_tokens,
                                        const void *    weight_base_ptr,
                                        size_t          expert_stride,
                                        size_t          expert_size,
                                        int             layer_id,
                                        int             n_experts_total,
                                        int             device_id,
                                        const char *    tensor_name,
                                        uint64_t        cache_uuid,
                                        uint32_t        model_id);

// Unpin routed experts after MoE computation completes.
// Call this after the MoE kernel finishes to allow eviction of these experts.
//
// Parameters:
//   expert_ids:      Same routing indices used in prestage_routed_experts
//   n_expert_used:   Experts per token
//   n_tokens:        Batch size
//   weight_base_ptr: mmap base pointer for expert weights
//   expert_stride:   Bytes between consecutive experts
//   layer_id:        Layer ID for cache key
//   n_experts_total: Total experts for bounds checking
//   device_id:       Device ID for cache lookup
void unpin_routed_experts(const int32_t * expert_ids,
                          int             n_expert_used,
                          int             n_tokens,
                          const void *    weight_base_ptr,
                          size_t          expert_stride,
                          int             layer_id,
                          int             n_experts_total,
                          int             device_id,
                          const char *    tensor_name,
                          uint64_t        cache_uuid,
                          uint32_t        model_id);

// === Shutdown API ===

// Shutdown the unified cache system before SYCL runtime destruction
// Call this during ggml_backend_sycl_free() to avoid static destruction order issues
// After calling this, the cache destructors will skip sycl::free() calls
void shutdown_unified_cache();

}  // namespace ggml_sycl

// === Cross-module Budget Recalculation ===
// These functions are defined in ggml-sycl.cpp but called from unified-cache.cpp
// to recalculate model placement decisions when VRAM budget changes.

// Recalculate g_model_exceeds_vram based on current effective budget.
// Called when runtime reservations (KV cache, compute buffers) change.
void ggml_sycl_recalc_model_exceeds_vram(size_t effective_budget);

// Get the total model size from tensor inventory (for budget calculations)
size_t ggml_sycl_get_model_size();

// Get MoE expert memory breakdown (for budget calculations)
void ggml_sycl_get_moe_info(size_t * expert_total_bytes, int * n_expert, int * n_expert_used);

#endif  // GGML_SYCL_UNIFIED_CACHE_HPP
