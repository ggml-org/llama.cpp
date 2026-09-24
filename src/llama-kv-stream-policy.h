#pragma once

#include "../ggml/src/ggml-kv-stream.h"

#include <vector>

struct llama_kv_stream_policy_cache {
    uint64_t id = 0;
    uint32_t layers = 0;
};

struct llama_kv_stream_policy_config {
    ggml_kv_stream_shape shape;
    ggml_kv_stream_capabilities capabilities;
    size_t pool_bytes = 0;
    // Full-attention layers in execution order, not all model blocks.
    uint32_t layers = 0;
    // Empty for legacy single-cache callers. Otherwise, cache-local layers are concatenated in execution order.
    std::vector<llama_kv_stream_policy_cache> caches;
    // Zero uses the reference's automatic (up to eight slots) startup hint.
    uint32_t initial_ring_slots = 0;
    bool fixed_ring = false;
    double overlap_ratio = 1.10;
    uint32_t grow_evaluations = 3, shrink_evaluations = 8, cooldown_evaluations = 64;
};

struct llama_kv_stream_policy_budget {
    ggml_kv_stream_shape shape;
    ggml_kv_stream_execution page;
    size_t pool_bytes = 0, conversion_offset = 0, unused_bytes = 0;
    uint32_t pages = 0, layers = 0, minimum_ring_slots = 0;
    std::vector<llama_kv_stream_policy_cache> caches;
};

struct llama_kv_stream_feedback {
    bool available = false;
    // Caller changes this when runtime/content feedback loses continuity; not an arena generation.
    uint64_t epoch = 0, samples = 0, misses = 0;
    double copy_busy_ratio = 0;
    uint32_t peak_slots = 0;
    // Complete-layer readiness is sampled before any consumer wait; upload counters remain diagnostic.
    uint64_t layer_samples = 0, layer_misses = 0;
};

struct llama_kv_stream_policy_state {
    llama_kv_stream_policy_budget budget;
    uint32_t resident_pages_per_layer = 0, ring_slots = 0;
    // Zero means uniform placement. Nonzero selects concentrated decode placement.
    uint32_t decode_active_pages = 0;
    // Deadline pressure spreads the streamed footprint across all layers so additional ring slots
    // become prefetch headroom instead of being consumed by fewer, larger streamed spans.
    bool spread_streaming = false;
    uint32_t starved = 0, overprovisioned = 0, evaluations_since_repartition = UINT32_MAX;
    bool feedback_initialized = false;
    uint64_t feedback_epoch = 0, samples = 0, misses = 0;
    uint64_t layer_samples = 0, layer_misses = 0;
};

struct llama_kv_stream_policy_observation {
    size_t active_tokens = 0;
    uint32_t query_tokens = 1;
    llama_kv_stream_feedback feedback;
    bool uniform_prefill = false;
};

struct llama_kv_stream_policy_decision {
    llama_kv_stream_policy_state next;
    uint32_t target_resident_pages = 0;
    bool partition_changed = false;
    // Address-layout changes only; active extent, contents, and capture eligibility have separate lifecycles.
    bool layout_changed = false;
    // A fresh valid delta was accepted; this does not imply that a partition change was needed.
    bool feedback_used = false, feedback_reset = false;
    bool layer_feedback_available = false, layer_feedback_used = false;
};

struct llama_kv_stream_policy_layer {
    uint64_t cache_id = 0;
    uint32_t cache_layer = 0;
    uint32_t capacity_pages = 0, resident_live_pages = 0, streamed_pages = 0, waves = 0;
    size_t offset = 0;
    ggml_kv_stream_layout planes;
};
struct llama_kv_stream_policy_layout {
    ggml_kv_stream_layout ring;
    std::vector<llama_kv_stream_policy_layer> layers;
    size_t conversion_offset = 0, conversion_bytes = 0, unused_bytes = 0;
};

struct llama_kv_stream_policy_rebind {
    llama_kv_stream_policy_config config;
    llama_kv_stream_policy_state state;
    llama_kv_stream_policy_layout layout;
};

enum class llama_kv_stream_policy_status {
    success, invalid_config, geometry_error, invalid_budget, invalid_state, invalid_observation,
    overflow, allocation_failed,
};
struct llama_kv_stream_policy_result {
    llama_kv_stream_policy_status status = llama_kv_stream_policy_status::success;
    ggml_kv_stream_result geometry;
};

// Return the smallest complete pool containing one resident page per layer, one ring page, and conversion scratch.
// The input pool_bytes is ignored; failure leaves output unchanged.
llama_kv_stream_policy_result llama_kv_stream_policy_minimum_pool_bytes(
        const llama_kv_stream_policy_config & config, size_t & output);
// Recompute a larger pool at an existing committed frontier. This does not retain or move storage.
// Failure leaves output unchanged.
llama_kv_stream_policy_result llama_kv_stream_policy_grow(
        const llama_kv_stream_policy_config & current,
        size_t active_tokens,
        bool decode,
        size_t pool_bytes,
        llama_kv_stream_policy_rebind & output);
// Recompute a smaller pool at an existing committed frontier. Failure leaves output unchanged.
llama_kv_stream_policy_result llama_kv_stream_policy_shrink(
        const llama_kv_stream_policy_config & current,
        size_t active_tokens,
        bool decode,
        size_t pool_bytes,
        llama_kv_stream_policy_rebind & output);
// All APIs are pure metadata calculations; failure leaves output unchanged. No allocation of KV/device storage.
llama_kv_stream_policy_result llama_kv_stream_policy_initialize(
        const llama_kv_stream_policy_config & config, llama_kv_stream_policy_state & output);
// Propose state only. The adapter must publish it only after the runtime accepts the corresponding layout.
llama_kv_stream_policy_result llama_kv_stream_policy_step(
        const llama_kv_stream_policy_config & config, const llama_kv_stream_policy_state & previous,
        const llama_kv_stream_policy_observation & observation, llama_kv_stream_policy_decision & output);
// Materialize offsets only when needed; step does not allocate per-layer vectors on the steady-state path.
llama_kv_stream_policy_result llama_kv_stream_policy_layout_make(
        const llama_kv_stream_policy_config & config, const llama_kv_stream_policy_state & state,
        size_t active_tokens, llama_kv_stream_policy_layout & output);
