#pragma once
#include "kv-stream-dispatch.cuh"
#include "fattn-tile-spans.h"
#include "../ggml-kv-stream-device.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define KV_TILE_DECLARE(K) \
    fattn_kernel_t ggml_cuda_kv_stream_tile_kernel_##K(ggml_type value, uint32_t ncols1, uint32_t ncols2); \
    fattn_kernel_t ggml_cuda_kv_stream_tile_kernel_##K##_D512(ggml_type value, uint32_t ncols1, uint32_t ncols2);
KV_TILE_DECLARE(F32)
KV_TILE_DECLARE(F16)
KV_TILE_DECLARE(BF16)
KV_TILE_DECLARE(Q4_0)
KV_TILE_DECLARE(Q4_1)
KV_TILE_DECLARE(Q5_0)
KV_TILE_DECLARE(Q5_1)
KV_TILE_DECLARE(Q8_0)
#undef KV_TILE_DECLARE

static inline fattn_kernel_t ggml_cuda_kv_stream_tile_kernel(ggml_type key, ggml_type value,
        uint32_t columns, uint32_t groups, uint32_t head_dim_k = 256, uint32_t head_dim_v = 256) {
    if (head_dim_k != head_dim_v || (head_dim_k != 256 && head_dim_k != 512)) return nullptr;
    switch (key) {
#define KV_TILE_CASE(K) case GGML_TYPE_##K: return head_dim_k == 512 ? \
    ggml_cuda_kv_stream_tile_kernel_##K##_D512(value,columns,groups) : ggml_cuda_kv_stream_tile_kernel_##K(value,columns,groups);
        KV_TILE_CASE(F32) KV_TILE_CASE(F16) KV_TILE_CASE(BF16) KV_TILE_CASE(Q4_0)
        KV_TILE_CASE(Q4_1) KV_TILE_CASE(Q5_0) KV_TILE_CASE(Q5_1) KV_TILE_CASE(Q8_0)
#undef KV_TILE_CASE
        default: return nullptr;
    }
}

bool ggml_cuda_kv_stream_tile_plan(ggml_backend_cuda_context & ctx, ggml_type key, ggml_type value,
        uint32_t heads, uint32_t kv_heads, uint32_t queries, size_t tokens, ggml_kv_stream_resume_plan & output);
bool ggml_cuda_kv_stream_tile_layout(ggml_backend_cuda_context & ctx, ggml_type key, ggml_type value,
        const ggml_kv_stream_resume_plan & plan, ggml_cuda_fattn_tile_resume_layout & output, size_t * shared_bytes = nullptr);
// Enqueue on the caller's ordered stream; grants and scratch remain alive through its read fence.
bool ggml_cuda_kv_stream_tile_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * op,
        const ggml_kv_stream_span_plan_view & view, ggml_backend_buffer_t workspace,
        const ggml_kv_stream_resume_plan & plan, size_t first, size_t end);
#endif
