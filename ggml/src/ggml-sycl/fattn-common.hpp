//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_COMMON_HPP
#define GGML_SYCL_FATTN_COMMON_HPP

#include "common.hpp"
#include "presets.hpp"

// Flash Attention constants
#define FATTN_KQ_STRIDE       256
#define SOFTMAX_FTZ_THRESHOLD -20.0f  // Softmax exp. values smaller than this are flushed to zero

// Default thread configuration for flash attention vector kernel
#define FATTN_VEC_NTHREADS    128

// Supported head dimensions
constexpr bool fattn_vec_supports_head_dim(int D) {
    return D == 64 || D == 128 || D == 256;
}

// Note: get_alibi_slope is defined in common.hpp

// Compute dot product of Q and K vectors for F16 type
template <int D, int nthreads>
static inline float vec_dot_fattn_vec_KQ_f16(
    const sycl::half * K_h, const sycl::half2 * Q_h2, int tid) {

    float sum = 0.0f;

    // Each thread processes D/(2*nthreads) half2 elements
    #pragma unroll
    for (int i = tid; i < D/2; i += nthreads) {
        sycl::half2 k_val = *reinterpret_cast<const sycl::half2*>(&K_h[i * 2]);
        sycl::half2 q_val = Q_h2[i / nthreads];

        // Dot product: sum += k.x * q.x + k.y * q.y
        sum += static_cast<float>(k_val.x()) * static_cast<float>(q_val.x());
        sum += static_cast<float>(k_val.y()) * static_cast<float>(q_val.y());
    }

    return sum;
}

// Dequantize V values from F16 to float2
template <int V_rows_per_thread>
static inline void dequantize_V_f16(
    const sycl::half * V_h, sycl::float2 * dst, int base_idx) {

    #pragma unroll
    for (int i = 0; i < V_rows_per_thread / 2; ++i) {
        int idx = base_idx + i * 2;
        dst[i].x() = static_cast<float>(V_h[idx]);
        dst[i].y() = static_cast<float>(V_h[idx + 1]);
    }
}

// Flash attention parameters structure
struct fattn_params {
    const char * Q;
    const char * K;
    const char * V;
    const char * mask;
    const char * sinks;  // Attention sinks tensor (src[4])
    float * dst;

    float scale;
    float max_bias;
    float m0;
    float m1;
    uint32_t n_head_log2;
    float logit_softcap;

    // Q dimensions: [ne03, ne02, ne01, ne00] = [batch, n_heads, n_queries, head_dim]
    int32_t ne00, ne01, ne02, ne03;
    int32_t nb01, nb02, nb03;

    // K dimensions: [ne13, ne12, ne11, ne10] = [batch, n_kv_heads, n_kv, head_dim]
    int32_t ne10, ne11, ne12, ne13;
    int32_t nb11, nb12;
    int64_t nb13;

    // V strides
    int32_t nb21, nb22;
    int64_t nb23;

    // Mask dimensions and strides: [ne33, ne32, ne31, ne30] = [batch, heads, n_tokens_padded, n_kv]
    int32_t ne30, ne31, ne32, ne33;
    int32_t nb31, nb32;
    int64_t nb33;
};

#endif // GGML_SYCL_FATTN_COMMON_HPP
