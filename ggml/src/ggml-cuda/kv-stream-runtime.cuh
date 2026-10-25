#pragma once

#include "kv-stream-dispatch.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
using ggml_kv_mma_runtime_kernel_t = void (*)(
        const char *, const char *, const char *, const char *, const char *, const int *, float *, float2 *,
        float, float, float, float, uint32_t, float,
        int32_t, uint3, int32_t, int32_t, int32_t, int32_t, int32_t,
        int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t,
        int32_t, int32_t, int64_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t,
        const ggml_cuda_kv_span *, int, int, uint32_t, ggml_cuda_fattn_mma_resume_control, ggml_type, ggml_type);

ggml_kv_mma_runtime_kernel_t ggml_cuda_kv_stream_mma_runtime_kernel(
        ggml_type key, ggml_type value, uint32_t columns, uint32_t groups, bool resume, bool dense);
#endif
