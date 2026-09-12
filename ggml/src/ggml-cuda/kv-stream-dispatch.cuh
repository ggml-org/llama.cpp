#pragma once
#include "fattn-common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define KV_STREAM_DECLARE(K) fattn_kernel_t ggml_cuda_kv_stream_kernel_##K(ggml_type value);
KV_STREAM_DECLARE(F16)
KV_STREAM_DECLARE(BF16)
KV_STREAM_DECLARE(Q4_0)
KV_STREAM_DECLARE(Q4_1)
KV_STREAM_DECLARE(Q5_0)
KV_STREAM_DECLARE(Q5_1)
KV_STREAM_DECLARE(Q8_0)
#undef KV_STREAM_DECLARE

// Null means this build does not contain the native pair; it may still permit bounded conversion.
static inline fattn_kernel_t ggml_cuda_kv_stream_kernel(ggml_type key, ggml_type value) {
    switch (key) {
#define KV_STREAM_CASE(K) case GGML_TYPE_##K: return ggml_cuda_kv_stream_kernel_##K(value);
        KV_STREAM_CASE(F16)
        KV_STREAM_CASE(BF16)
        KV_STREAM_CASE(Q4_0)
        KV_STREAM_CASE(Q4_1)
        KV_STREAM_CASE(Q5_0)
        KV_STREAM_CASE(Q5_1)
        KV_STREAM_CASE(Q8_0)
#undef KV_STREAM_CASE
        default: return nullptr;
    }
}
#endif
