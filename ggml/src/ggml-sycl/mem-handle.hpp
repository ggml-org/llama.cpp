// Smart handle infrastructure for SYCL unified memory manager.
// Handles cache pointer resolution with generation-based staleness detection.
// P10 of the unified memory manager epic.
//
// See docs/smart-handle-design.md for architecture details.

#pragma once

#include <cstdint>

#include "unified-cache.hpp"  // unified_cache_key, ggml_layout_mode

namespace ggml_sycl {

// === Generation counter ===
// Single global atomic, bumped on eviction/promotion/flush (~0-3 times per
// inference run).  Hot path: compare + return cached ptr (~3 ns).

// Current global generation.  Callers compare their cached generation against
// this to detect staleness.  Relaxed ordering is sufficient — generation bumps
// happen under rw_mutex_ in unified_cache, and any resolve_slow() that follows
// a stale check will acquire the cache lock and see the new state.
uint64_t cache_generation();

// Bump the global generation.  Called from unified_cache when any pointer could
// have moved: evict_one, promote_to_device, finalize_evictions_locked.
void cache_generation_bump();

// === Resolved pointer ===
// The result of resolving a mem_handle.  Contains the current pointer and
// metadata needed by the caller.

struct resolved_ptr {
    void *           ptr       = nullptr;
    ggml_layout_mode layout    = GGML_LAYOUT_AOS;
    bool             on_device = false;

    explicit operator bool() const { return ptr != nullptr; }
};

// === mem_handle ===
// Lightweight handle that caches pointer resolution.  Two kinds:
//
// WEIGHT: cache-managed, gen-checked.  resolve() compares cached generation
//         against global; if stale, calls resolve_slow() which queries the
//         unified cache.  ~3 ns hot path when pointer hasn't moved.
//
// DIRECT: raw pointer wrapper.  resolve() always returns the cached pointer.
//         Used for scratch/KV/staging buffers that are never moved by the cache.

enum class mem_handle_kind : uint8_t {
    WEIGHT = 0,  // Cache-managed, generation-checked
    DIRECT = 1,  // Raw pointer, never stale
};

class mem_handle {
public:
    // Create an invalid handle.
    mem_handle() = default;

    // Create a WEIGHT handle from a cache key + device.
    // The handle starts with gen_ = 0 (stale), so the first resolve() will
    // query the cache.
    static mem_handle from_weight(const unified_cache_key & key, int device);

    // Create a DIRECT handle from a raw pointer.
    // resolve() always returns this pointer without checking the cache.
    static mem_handle from_direct(void * ptr, ggml_layout_mode layout, bool on_device);

    // Resolve the current pointer.  Hot path (~3 ns):
    //   if (kind == DIRECT || gen_ == cache_generation())
    //       return cached resolved_ptr
    //   else
    //       return resolve_slow()
    resolved_ptr resolve() const;

    // True if this handle has ever been successfully resolved.
    bool valid() const { return cached_.ptr != nullptr || kind_ == mem_handle_kind::DIRECT; }

    // Access the cache key (only meaningful for WEIGHT handles).
    const unified_cache_key & key() const { return key_; }

    // Access the device ID.
    int device() const { return device_; }

    // Access the handle kind.
    mem_handle_kind kind() const { return kind_; }

private:
    // Slow path: re-query the unified cache for the current pointer.
    resolved_ptr resolve_slow() const;

    mem_handle_kind    kind_   = mem_handle_kind::DIRECT;
    int                device_ = 0;
    unified_cache_key  key_    = {};

    // Mutable because resolve() is logically const (returns the current
    // pointer) but updates the cache as a side effect.
    mutable uint64_t     gen_    = 0;
    mutable resolved_ptr cached_ = {};
};

}  // namespace ggml_sycl
