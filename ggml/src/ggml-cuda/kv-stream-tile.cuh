#pragma once
#include "kv-stream-dispatch.cuh"
#include "fattn-tile-resume.h"
#include "../ggml-kv-stream-device.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define KV_TILE_DECLARE(K) fattn_kernel_t ggml_cuda_kv_stream_tile_kernel_##K(ggml_type value, uint32_t ncols1, uint32_t ncols2);
KV_TILE_DECLARE(F16)
KV_TILE_DECLARE(BF16)
KV_TILE_DECLARE(Q4_0)
KV_TILE_DECLARE(Q4_1)
KV_TILE_DECLARE(Q5_0)
KV_TILE_DECLARE(Q5_1)
KV_TILE_DECLARE(Q8_0)
#undef KV_TILE_DECLARE

bool ggml_cuda_kv_stream_tile_plan(ggml_backend_cuda_context & ctx, ggml_type key, ggml_type value,
        uint32_t heads, uint32_t kv_heads, uint32_t queries, size_t tokens, ggml_kv_stream_resume_plan & output);
bool ggml_cuda_kv_stream_tile_layout(ggml_backend_cuda_context & ctx, ggml_type key, ggml_type value,
        const ggml_kv_stream_resume_plan & plan, ggml_cuda_fattn_tile_resume_layout & output, size_t * shared_bytes = nullptr);
// Enqueue on the caller's ordered stream; grants and scratch remain alive through its read fence.
bool ggml_cuda_kv_stream_tile_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * op,
        const ggml_kv_stream_span_plan_view & view, ggml_backend_buffer_t workspace,
        const ggml_kv_stream_resume_plan & plan, size_t first, size_t end);
#endif
