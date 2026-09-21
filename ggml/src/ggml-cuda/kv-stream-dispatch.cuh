#pragma once
#include "fattn-common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

using ggml_kv_resume_kernel_t = void (*)(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33, float *, int, int, bool, bool);
#define KV_RESUME_DECLARE(K) ggml_kv_resume_kernel_t ggml_cuda_kv_stream_resume_kernel_##K(ggml_type value, uint32_t queries);
KV_RESUME_DECLARE(F16)
KV_RESUME_DECLARE(BF16)
KV_RESUME_DECLARE(Q4_0)
KV_RESUME_DECLARE(Q4_1)
KV_RESUME_DECLARE(Q5_0)
KV_RESUME_DECLARE(Q5_1)
KV_RESUME_DECLARE(Q8_0)
#undef KV_RESUME_DECLARE
#define KV_VECTOR_DECLARE(K) fattn_kernel_t ggml_cuda_kv_stream_vector_kernel_##K(ggml_type value, uint32_t queries);
KV_VECTOR_DECLARE(F16)
KV_VECTOR_DECLARE(BF16)
KV_VECTOR_DECLARE(Q4_0)
KV_VECTOR_DECLARE(Q4_1)
KV_VECTOR_DECLARE(Q5_0)
KV_VECTOR_DECLARE(Q5_1)
KV_VECTOR_DECLARE(Q8_0)
#undef KV_VECTOR_DECLARE
static inline fattn_kernel_t ggml_cuda_kv_stream_vector_kernel(ggml_type key, ggml_type value, uint32_t queries) {
    switch (key) {
#define KV_VECTOR_CASE(K) case GGML_TYPE_##K: return ggml_cuda_kv_stream_vector_kernel_##K(value, queries);
        KV_VECTOR_CASE(F16)
        KV_VECTOR_CASE(BF16)
        KV_VECTOR_CASE(Q4_0)
        KV_VECTOR_CASE(Q4_1)
        KV_VECTOR_CASE(Q5_0)
        KV_VECTOR_CASE(Q5_1)
        KV_VECTOR_CASE(Q8_0)
#undef KV_VECTOR_CASE
        default: return nullptr;
    }
}
static inline ggml_kv_resume_kernel_t ggml_cuda_kv_stream_resume_kernel(ggml_type key, ggml_type value, uint32_t queries) {
    switch (key) {
#define KV_RESUME_CASE(K) case GGML_TYPE_##K: return ggml_cuda_kv_stream_resume_kernel_##K(value, queries);
        KV_RESUME_CASE(F16)
        KV_RESUME_CASE(BF16)
        KV_RESUME_CASE(Q4_0)
        KV_RESUME_CASE(Q4_1)
        KV_RESUME_CASE(Q5_0)
        KV_RESUME_CASE(Q5_1)
        KV_RESUME_CASE(Q8_0)
#undef KV_RESUME_CASE
        default: return nullptr;
    }
}

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
