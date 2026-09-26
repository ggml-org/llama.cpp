#pragma once

#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_moe_cache_config {
    size_t budget_bytes;
    size_t reserve_bytes;
    size_t minimum_slab_bytes;
    size_t min_expert_bytes;
    // 0 lets each selected device raise or lower the admission floor.
    int32_t min_expert_explicit;
    int32_t max_batch;
    int32_t min_compute_capability;
    // minimum number of selected devices the session needs; 1 for every mode,
    // automatic mode still enforces its 1 GiB slab floor per device
    int32_t min_devices;
    // -1 selects the provider policy; 0..8 is a fixed total per node.
    int32_t overlap_cpu_rows;
};

struct ggml_moe_cache_device_caps {
    int32_t logical_device;
    int32_t physical_device;
    int32_t compute_capability;
    // Effective admission floor for this device and configuration.
    size_t min_expert_bytes;
};

struct ggml_moe_cache_shape_caps {
    size_t scratch_bytes;
    size_t pool_bytes;
    size_t minimum_bytes;
};

struct ggml_moe_cache_tensor_desc {
    const char * name;
    const void * data;
    size_t expert_size;
    int64_t n_in;
    int64_t n_out;
    int64_t n_expert;
    int32_t type;
};

enum ggml_moe_cache_mode {
    GGML_MOE_CACHE_MODE_UNSPECIFIED = -1,
    GGML_MOE_CACHE_MODE_OFF = 0,
    GGML_MOE_CACHE_MODE_AUTO = 1,
    GGML_MOE_CACHE_MODE_ON = 2,
};

struct ggml_moe_cache_api {
    const void * owner;

    int (*query_config)(int automatic, size_t budget_mib, struct ggml_moe_cache_config * config);
    int (*query_device)(void * device, const struct ggml_moe_cache_config * config, struct ggml_moe_cache_device_caps * caps);
    int (*query_shape)(int wtype, int64_t n_in, int64_t n_out, int64_t n_expert, size_t expert_size, struct ggml_moe_cache_shape_caps * caps);
    // Fused SwiGLU path: 1 when this provider can fuse up * GLU(gate) for the
    // weight type. CUDA implements this and covers the canonical list exactly;
    // providers without a fused path leave it NULL.
    int (*query_fused)(int wtype);

    // The scheduler owns one cache session. backends contains the scheduler's actual backend set, so the provider can use only selected CUDA devices.
    void * (*session_create)(void * const * backends, int n_backends, const struct ggml_moe_cache_config * config);
    void   (*session_destroy)(void * session);
    // NULL and dormant sessions still create a suppressing thread-local scope.
    void   (*session_enter)(void * session);
    void   (*session_leave)(void * session);

    // Begin one CPU MUL_MAT_ID node. Returns an opaque plan, or NULL when the stock CPU path should handle the complete node.
    void * (*begin)(const char * tensor_name, const void * host_base, size_t expert_size,
                    int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
                    int64_t n_tokens, int64_t n_rows);

    // Mark cache hits and enqueue bounded demand fills for misses. A nonnegative slot index means that the row may be omitted from CPU work only if dispatch subsequently succeeds.
    int (*plan)(void * node, const int32_t * ids, int n_ids, int32_t * slot_idx);

    // Dispatch all planned hit rows. Returns 1 only after the complete GPU operation has been accepted.
    // On 0, the caller must restore every row to the normal CPU mapping before worker threads start.
    int (*dispatch)(void * node, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                    const int32_t * slot_idx, const float * const * act_rows);

    // Copy GPU results into dst_rows. On 0, the caller must recompute every skipped row on the CPU.
    int (*collect)(void * node, int n_hits, float * const * dst_rows, int64_t n_out);

    // Releases slot pins and all per-node ownership. Must be called exactly once for every non-NULL begin result, on every success or failure path.
    void (*end)(void * node);

    // Dispatch one fused up * GLU(gate) operation over experts resident for both tensors. This is used only after the CPU backend proves that the two MUL_MAT_ID nodes and GLU form an elidable subgraph. ids and act_rows contain n_rows flattened token-major routed rows. Returns a regular node accepted by collect/end and marks the skipped logical rows in hit_mask.
    void * (*fused_begin)(const struct ggml_moe_cache_tensor_desc * up,
                          const struct ggml_moe_cache_tensor_desc * gate,
                          int glu_op, float up_min, float up_max,
                          float gate_min, float gate_max,
                          const int32_t * ids, int n_rows, int64_t n_tokens,
                          const float * const * act_rows, uint64_t * hit_mask);

    // Host buffer mutation or teardown notification. Sessions cancel or finish any fill that still reads the supplied range before this call returns.
    void (*invalidate)(const void * base, size_t size);
};

// Canonical expert weight type set for the MoE cache. Every provider
// (CUDA/HIP, Vulkan, Metal) must accept exactly this set in query_shape and
// begin; a provider that silently accepts less produces silent partial
// coverage. Keep in sync with the CUDA kernel case tables in
// ggml-cuda/mmvq.cu: the plain dispatch tables
// (ggml_cuda_moe_cache_mmv_supported) and the fused SwiGLU tables
// (ggml_cuda_moe_cache_mmv_fused_supported) must both equal this list - a
// canonical type missing from the fused tables silently skips the fusion
// optimization. moe-cache.cu asserts both directions at registration time.
// enum tag form: ggml.h declares the type without a typedef, so bare
// ggml_type is invalid in C (ggml-cpu.c includes this header as C).
static const enum ggml_type ggml_moe_cache_types[] = {
    GGML_TYPE_Q1_0,
    GGML_TYPE_Q2_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_MXFP4,
    GGML_TYPE_NVFP4,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
};

static inline int ggml_moe_cache_wtype_count(void) {
    return (int) (sizeof(ggml_moe_cache_types) / sizeof(ggml_moe_cache_types[0]));
}

// Single source of truth for cacheable weight types. Providers call this from
// both query_shape and begin so the two gates cannot drift apart.
static inline int ggml_moe_cache_wtype_supported(int wtype) {
    for (int i = 0; i < ggml_moe_cache_wtype_count(); i++) {
        if ((int) ggml_moe_cache_types[i] == wtype) {
            return 1;
        }
    }
    return 0;
}

GGML_API void ggml_moe_cache_register(const struct ggml_moe_cache_api * api);
GGML_API void ggml_moe_cache_unregister(const void * owner);
// Returns a copy of the provider table registered for owner, zeroed if absent.
// The scheduler keeps this copy with its session so later registration changes
// cannot alter the session's callbacks.
GGML_API struct ggml_moe_cache_api ggml_moe_cache_get(const void * owner);
// Active provider table for the current thread: set by the scheduler while it
// executes a graph with a cache session; falls back to the first registered
// provider for direct CPU users.
GGML_API struct ggml_moe_cache_api ggml_moe_cache_active(void);
GGML_API void ggml_backend_sched_set_moe_cache(
        ggml_backend_sched_t sched, enum ggml_moe_cache_mode mode, size_t budget_mib);

#ifdef __cplusplus
}
#endif
