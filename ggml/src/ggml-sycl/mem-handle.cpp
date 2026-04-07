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

mem_handle mem_handle::from_cache_id(const ggml_sycl_cache_id & id, int device) {
    unified_cache_key key;
    key.type       = cache_entry_type::DENSE_WEIGHT;
    key.id         = id;
    key.layer_id   = -1;
    key.expert_id  = -1;
    return from_weight(key, device);
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

mem_handle mem_handle::from_arena_zone(int zone_id, size_t offset, size_t size,
                                       int device_id, uint64_t generation) {
    mem_handle h;
    // Map zone_id to the appropriate arena handle kind.
    // vram_zone_id: KV=0, WEIGHT=1, ONEDNN=2, RUNTIME=3, SCRATCH=4
    switch (zone_id) {
        case static_cast<int>(vram_zone_id::RUNTIME): h.kind_ = mem_handle_kind::ARENA_RUNTIME; break;
        case static_cast<int>(vram_zone_id::SCRATCH): h.kind_ = mem_handle_kind::ARENA_SCRATCH; break;
        case static_cast<int>(vram_zone_id::ONEDNN):  h.kind_ = mem_handle_kind::ARENA_ONEDNN;  break;
        default:
            GGML_LOG_WARN("[MEM-HANDLE] from_arena_zone: unexpected zone_id %d, defaulting to ARENA_RUNTIME\n", zone_id);
            h.kind_ = mem_handle_kind::ARENA_RUNTIME;
            break;
    }
    h.device_    = device_id;
    h.zone_id_   = zone_id;
    h.offset_    = offset;
    h.size_      = size;
    h.arena_gen_ = generation;
    h.gen_       = 0;  // Force first resolve
    h.cached_    = {};
    return h;
}

// === resolve ===

resolved_ptr mem_handle::resolve() const {
    // DIRECT handles are never stale.
    if (kind_ == mem_handle_kind::DIRECT) {
        return cached_;
    }

    // Arena handles: check arena generation, then resolve base + offset.
    if (kind_ >= mem_handle_kind::ARENA_RUNTIME &&
        kind_ <= mem_handle_kind::ARENA_ONEDNN) {
        // If we have a cached pointer and the generation hasn't changed,
        // return immediately.
        if (cached_.ptr != nullptr && gen_ == arena_gen_) {
            return cached_;
        }
        return resolve_arena();
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

    // get_weight_ptr returns weight_ptr_result with: ptr, layout, on_device
    auto result = cache->get_weight_ptr(key_.id);
    if (!result) {
        return {};
    }

    // Update cached state
    cached_ = { result.ptr, result.layout, result.on_device };
    gen_    = cache_generation();
    return cached_;
}

// === resolve_arena ===
// Resolve an arena handle by querying the arena base pointer from unified_cache,
// then adding the zone start + offset.  Returns nullptr if the arena has been
// recreated (generation mismatch).

resolved_ptr mem_handle::resolve_arena() const {
    // Device arena: query unified_cache for arena methods.
    unified_cache * cache = get_unified_cache_for_device(device_);
    if (!cache) {
        return {};
    }

    if (!cache->arena_active()) {
        return {};
    }

    // Check generation: if the arena was destroyed and recreated, our handle
    // is stale.
    uint64_t current_gen = cache->arena_generation();
    if (arena_gen_ != current_gen) {
        return {};
    }

    // Resolve: zone_alloc returned an offset within the arena, but our offset_
    // is the raw arena offset (base-relative).  Use offset_to_ptr directly.
    void * ptr = cache->offset_to_ptr(offset_);
    if (!ptr) {
        return {};
    }

    // Cache the resolved pointer.  Arena handles are always on-device with
    // AOS layout (arena zones hold raw allocations, not cache-managed weights).
    cached_ = { ptr, GGML_LAYOUT_AOS, true };
    gen_    = arena_gen_;
    return cached_;
}

// === layer_weight_handles ===

bool layer_weight_handles::resolve_all(layer_weight_pointers & out) const {
    auto r_attn_norm = attn_norm.resolve();
    auto r_q_proj    = q_proj.resolve();
    auto r_k_proj    = k_proj.resolve();
    auto r_v_proj    = v_proj.resolve();
    auto r_o_proj    = o_proj.resolve();
    auto r_ffn_norm  = ffn_norm.resolve();
    auto r_gate_proj = gate_proj.resolve();
    auto r_up_proj   = up_proj.resolve();
    auto r_down_proj = down_proj.resolve();

    if (!r_attn_norm || !r_q_proj || !r_k_proj || !r_v_proj || !r_o_proj ||
        !r_ffn_norm  || !r_gate_proj || !r_up_proj || !r_down_proj) {
        return false;
    }

    out.attn_norm = r_attn_norm.ptr;
    out.q_proj    = r_q_proj.ptr;
    out.k_proj    = r_k_proj.ptr;
    out.v_proj    = r_v_proj.ptr;
    out.o_proj    = r_o_proj.ptr;
    out.ffn_norm  = r_ffn_norm.ptr;
    out.gate_proj = r_gate_proj.ptr;
    out.up_proj   = r_up_proj.ptr;
    out.down_proj = r_down_proj.ptr;

    // Optional fused weights — resolve if handle is valid
    auto r_qkv     = attn_qkv_proj.resolve();
    auto r_gate_up = ffn_gate_up_proj.resolve();
    out.attn_qkv_proj    = r_qkv ? r_qkv.ptr : nullptr;
    out.ffn_gate_up_proj = r_gate_up ? r_gate_up.ptr : nullptr;

    return true;
}

layer_weight_handles layer_weight_handles::from_weight_set(const layer_weight_set & ws, int device) {
    layer_weight_handles h;
    h.attn_norm = mem_handle::from_cache_id(ws.attn_norm, device);
    h.q_proj    = mem_handle::from_cache_id(ws.q_proj, device);
    h.k_proj    = mem_handle::from_cache_id(ws.k_proj, device);
    h.v_proj    = mem_handle::from_cache_id(ws.v_proj, device);
    h.o_proj    = mem_handle::from_cache_id(ws.o_proj, device);
    h.ffn_norm  = mem_handle::from_cache_id(ws.ffn_norm, device);
    h.gate_proj = mem_handle::from_cache_id(ws.gate_proj, device);
    h.up_proj   = mem_handle::from_cache_id(ws.up_proj, device);
    h.down_proj = mem_handle::from_cache_id(ws.down_proj, device);

    // Optional fused weights
    if (ws.attn_qkv_proj.valid) {
        h.attn_qkv_proj = mem_handle::from_cache_id(ws.attn_qkv_proj, device);
    }
    if (ws.ffn_gate_up_proj.valid) {
        h.ffn_gate_up_proj = mem_handle::from_cache_id(ws.ffn_gate_up_proj, device);
    }

    return h;
}

// === build_layer_handles ===

bool build_layer_handles(int device, int layer_id, layer_weight_handles & out) {
    unified_cache * cache = get_unified_cache_for_device(device);
    if (!cache) {
        return false;
    }

    layer_weight_set ws;
    if (!cache->get_layer_weight_set(layer_id, ws)) {
        return false;
    }

    out = layer_weight_handles::from_weight_set(ws, device);
    return true;
}

}  // namespace ggml_sycl
