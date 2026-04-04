// Smart handle implementation for SYCL unified memory manager.
// See mem-handle.hpp for design and docs/smart-handle-design.md for architecture.

#include "mem-handle.hpp"

#include "unified-cache.hpp"  // get_unified_cache_for_device, unified_cache

#include <atomic>

namespace ggml_sycl {

// === Global generation counter ===
// File-scoped static — same pattern as g_graph_compute_active in unified-cache.cpp.
// Relaxed ordering: bumps happen under rw_mutex_; resolve_slow acquires the lock
// and sees the consistent cache state regardless of generation ordering.
static std::atomic<uint64_t> g_cache_generation{ 0 };

uint64_t cache_generation() {
    return g_cache_generation.load(std::memory_order_relaxed);
}

void cache_generation_bump() {
    g_cache_generation.fetch_add(1, std::memory_order_relaxed);
}

// === mem_handle factory methods ===

mem_handle mem_handle::from_weight(const unified_cache_key & key, int device) {
    mem_handle h;
    h.kind_   = mem_handle_kind::WEIGHT;
    h.device_ = device;
    h.key_    = key;
    h.gen_    = 0;  // Stale — first resolve() will query the cache
    h.cached_ = {};
    return h;
}

mem_handle mem_handle::from_direct(void * ptr, ggml_layout_mode layout, bool on_device) {
    mem_handle h;
    h.kind_   = mem_handle_kind::DIRECT;
    h.device_ = 0;
    h.key_    = {};
    h.gen_    = 0;
    h.cached_ = { ptr, layout, on_device };
    return h;
}

// === resolve ===

resolved_ptr mem_handle::resolve() const {
    // DIRECT handles are never stale.
    if (kind_ == mem_handle_kind::DIRECT) {
        return cached_;
    }

    // WEIGHT handle: compare cached generation against global.
    const uint64_t current_gen = cache_generation();
    if (gen_ == current_gen && cached_.ptr != nullptr) {
        return cached_;
    }

    return resolve_slow();
}

// === resolve_slow ===
// Re-query the unified cache.  Called ~0-3 times per inference run (only on
// generation mismatch, which means an eviction/promotion just happened).

resolved_ptr mem_handle::resolve_slow() const {
    unified_cache * cache = get_unified_cache_for_device(device_);
    if (!cache) {
        return {};
    }

    // weight_ptr_result has: ptr, layout, on_device
    auto result = cache->get_weight_ptr(key_.id);
    if (!result) {
        return {};
    }

    // Update cached state
    cached_ = { result.ptr, result.layout, result.on_device };
    gen_    = cache_generation();
    return cached_;
}

}  // namespace ggml_sycl
