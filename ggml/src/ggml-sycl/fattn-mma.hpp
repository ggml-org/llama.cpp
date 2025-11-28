//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_MMA_HPP
#define GGML_SYCL_FATTN_MMA_HPP

#include "fattn-common.hpp"
#include <sycl/sycl.hpp>
#include <cfloat>

// Check for joint_matrix support - use __has_include since version macro may not be defined
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#define SYCL_JOINT_MATRIX_AVAILABLE 1
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#define SYCL_JOINT_MATRIX_AVAILABLE 0
#endif

#if SYCL_JOINT_MATRIX_AVAILABLE

namespace sycl_mma = sycl::ext::oneapi::experimental::matrix;

// MMA tile configuration for flash attention
// Intel XMX supports: M=1,2,4,8  N=8,16  K=16 for fp16
template <int D>
struct fattn_mma_config;

// Configuration for D=64
template <>
struct fattn_mma_config<64> {
    static constexpr int TM = 8;      // Tile M dimension
    static constexpr int TN = 16;     // Tile N dimension
    static constexpr int TK = 16;     // Tile K dimension for fp16
    static constexpr int BATCH_KV = 32;  // K/V positions processed per iteration
    static constexpr int SG_SIZE = WARP_SIZE;  // Use same sub-group size as vec kernel
};

// Configuration for D=128
template <>
struct fattn_mma_config<128> {
    static constexpr int TM = 8;
    static constexpr int TN = 16;
    static constexpr int TK = 16;
    static constexpr int BATCH_KV = 32;
    static constexpr int SG_SIZE = WARP_SIZE;
};

// Configuration for D=256
template <>
struct fattn_mma_config<256> {
    static constexpr int TM = 8;
    static constexpr int TN = 16;
    static constexpr int TK = 16;
    static constexpr int BATCH_KV = 16;  // Smaller batch due to larger D
    static constexpr int SG_SIZE = WARP_SIZE;
};

// Flash Attention MMA Kernel for SYCL using joint_matrix
// Implements tiled matrix multiply for Q*K^T and softmax(QK)*V

// Debug flag - set to 1 to enable debug output
#define FATTN_MMA_DEBUG 0

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
static void flash_attn_mma_ext_f16(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    const char * __restrict__ mask,
    float * __restrict__ dst,
    float scale,
    float max_bias,
    float m0,
    float m1,
    uint32_t n_head_log2,
    float logit_softcap,
    int ne00, int ne01, int ne02, int ne03,
    int nb01, int nb02, int nb03,
    int ne10, int ne11, int ne12, int ne13,
    int nb11, int nb12, int64_t nb13,
    int nb21, int nb22, int64_t nb23,
    int ne30, int ne31, int ne32, int ne33,
    int nb31, int nb32, int64_t nb33,
    const sycl::nd_item<3> & item,
    sycl::half * shared_mem) {

    using config = fattn_mma_config<D>;

    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_linear_id();
    const int lane_id = sg.get_local_linear_id();
    const int n_sg = item.get_local_range(2) / config::SG_SIZE;

    // Work-group indices
    const int ic0 = item.get_group(2) * ncols;  // Query column block
    const int sequence = item.get_group(0) / ne02;  // Batch index
    const int head = item.get_group(0) % ne02;  // Head index
    const int gqa_ratio = ne02 / ne12;

    // DEBUG: Skip heads >= 32 to test if first 32 work correctly
    // if (head >= 32) return;

    // Pointers to this work-group's data (byte pointers for stride arithmetic)
    const char * Q_base = Q + nb03 * sequence + nb02 * head;
    const int kv_head = head / gqa_ratio;
    const char * K_base = K + nb13 * sequence + nb12 * kv_head;
    const char * V_base = V + nb23 * sequence + nb22 * kv_head;

    // Mask setup
    const sycl::half * maskh = mask ?
        reinterpret_cast<const sycl::half*>(mask + nb33 * (sequence % ne33) + nb31 * ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Shared memory layout for tiles
    // tile_Q: [ncols][D] in half precision
    // tile_K: [BATCH_KV][D] in half precision
    // tile_V: [BATCH_KV][D] in half precision
    // KQ_tile: [ncols][BATCH_KV] for attention scores
    sycl::half * tile_Q = shared_mem;
    sycl::half * tile_K = tile_Q + ncols * D;
    sycl::half * tile_V = tile_K + config::BATCH_KV * D;
    float * KQ_tile = reinterpret_cast<float*>(tile_V + config::BATCH_KV * D);
    float * reduce_buf = KQ_tile + ncols * config::BATCH_KV;

    const int tid = item.get_local_linear_id();
    constexpr int nthreads = 128;  // Fixed thread count

    // Compile-time constants for array sizes
    constexpr int D_per_thread = (D + nthreads - 1) / nthreads;

    // Load Q into shared memory (convert to half if needed)
    for (int idx = tid; idx < ncols * D; idx += nthreads) {
        const int j = idx / D;
        const int d = idx % D;
        if (ic0 + j < ne01) {
            const Q_type * Q_ptr = reinterpret_cast<const Q_type*>(Q_base + nb01 * (ic0 + j));
            tile_Q[j * D + d] = static_cast<sycl::half>(static_cast<float>(Q_ptr[d]) * scale);
        } else {
            tile_Q[j * D + d] = sycl::half(0.0f);
        }
    }
    sycl::group_barrier(item.get_group());

    // Accumulators for online softmax (per query column)
    float VKQ[ncols][D_per_thread];
    float KQ_max[ncols];
    float KQ_sum[ncols];

    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX / 2.0f;
        KQ_sum[j] = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_per_thread; ++i) {
            VKQ[j][i] = 0.0f;
        }
    }

    // Process K/V in batches
    for (int kv_start = 0; kv_start < ne11; kv_start += config::BATCH_KV) {
        const int kv_end = sycl::min(kv_start + config::BATCH_KV, ne11);
        const int kv_count = kv_end - kv_start;

        // Load K tile into shared memory
        for (int idx = tid; idx < kv_count * D; idx += nthreads) {
            const int k = idx / D;
            const int d = idx % D;
            const sycl::half * K_row = reinterpret_cast<const sycl::half*>(K_base + nb11 * (kv_start + k));
            tile_K[k * D + d] = K_row[d];
        }
        // Zero-pad remaining K positions
        for (int idx = tid + kv_count * D; idx < config::BATCH_KV * D; idx += nthreads) {
            tile_K[idx] = sycl::half(0.0f);
        }
        sycl::group_barrier(item.get_group());

        // Compute Q * K^T using joint_matrix MMA
        // Result: KQ_tile[ncols][BATCH_KV]
        // Q is [ncols][D], K^T is [D][BATCH_KV]

        // For now, use a simple tiled approach without joint_matrix
        // (joint_matrix requires specific memory layouts that need careful setup)
        // Each thread computes partial dot products

        // Zero KQ_tile
        for (int idx = tid; idx < ncols * config::BATCH_KV; idx += nthreads) {
            KQ_tile[idx] = 0.0f;
        }
        sycl::group_barrier(item.get_group());

        // Compute Q * K^T
        // Each sub-group handles a portion of the computation
        for (int q_idx = sg_id; q_idx < ncols; q_idx += n_sg) {
            if (ic0 + q_idx >= ne01) continue;

            for (int k_idx = 0; k_idx < kv_count; ++k_idx) {
                float dot = 0.0f;

                // Each lane handles part of the D dimension
                for (int d = lane_id; d < D; d += config::SG_SIZE) {
                    dot += static_cast<float>(tile_Q[q_idx * D + d]) *
                           static_cast<float>(tile_K[k_idx * D + d]);
                }

                // Reduce within sub-group
                #pragma unroll
                for (int offset = config::SG_SIZE / 2; offset > 0; offset >>= 1) {
                    dot += sycl::shift_group_left(sg, dot, offset);
                }

                // Lane 0 writes result
                if (lane_id == 0) {
                    float val = dot;
                    if (use_logit_softcap) {
                        val = logit_softcap * sycl::tanh(val);
                    }
                    // Apply mask - mask is at base + nb31*ic0, so we access [q_idx][kv_idx]
                    // maskh points to mask + nb31*ic0, stride is ne11 (K sequence length)
                    float mask_val = 0.0f;
                    if (maskh) {
                        mask_val = static_cast<float>(maskh[q_idx * ne11 + kv_start + k_idx]);
                        val += slope * mask_val;
                    }
#if FATTN_MMA_DEBUG
                    // Debug: print mask values for last query in prompt (should see multiple valid K)
                    // Only print for layer 0 (first head=0), q >= 4 (last token of 5-token prompt), first few K
                    if (head == 0 && kv_start == 0 && k_idx < 8 && ic0 + q_idx >= 4 && ne01 >= 5) {
                        sycl::ext::oneapi::experimental::printf(
                            "[FA-MASK-L] h=%d q=%d k=%d dot=%.3f mask=%.2f val=%.3f\n",
                            head, ic0 + q_idx, kv_start + k_idx, dot, mask_val, val);
                    }
#endif
                    KQ_tile[q_idx * config::BATCH_KV + k_idx] = val;
                }
            }
        }
        sycl::group_barrier(item.get_group());

        // Load V tile into shared memory
        for (int idx = tid; idx < kv_count * D; idx += nthreads) {
            const int k = idx / D;
            const int d = idx % D;
            const sycl::half * V_row = reinterpret_cast<const sycl::half*>(V_base + nb21 * (kv_start + k));
            tile_V[k * D + d] = V_row[d];
        }
        // Zero-pad remaining V positions
        for (int idx = tid + kv_count * D; idx < config::BATCH_KV * D; idx += nthreads) {
            tile_V[idx] = sycl::half(0.0f);
        }
        sycl::group_barrier(item.get_group());

        // Online softmax and V accumulation
        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j >= ne01) continue;

            // Find max in this batch
            float batch_max = -FLT_MAX;
            for (int k = 0; k < kv_count; ++k) {
                batch_max = sycl::fmax(batch_max, KQ_tile[j * config::BATCH_KV + k]);
            }

            // Update running max
            const float new_max = sycl::fmax(KQ_max[j], batch_max);
            const float scale_old = sycl::exp(KQ_max[j] - new_max);
            KQ_max[j] = new_max;

            // Scale previous accumulator
            #pragma unroll
            for (int i = 0; i < D_per_thread; ++i) {
                VKQ[j][i] *= scale_old;
            }
            KQ_sum[j] *= scale_old;

            // Compute softmax weights and accumulate V
            for (int k = 0; k < kv_count; ++k) {
                const float kq_val = KQ_tile[j * config::BATCH_KV + k];
                const float w = sycl::exp(kq_val - KQ_max[j]);
                KQ_sum[j] += w;

                // Accumulate weighted V
                #pragma unroll
                for (int i = 0; i < D_per_thread; ++i) {
                    const int d_idx = tid + i * nthreads;
                    if (d_idx < D) {
                        VKQ[j][i] += w * static_cast<float>(tile_V[k * D + d_idx]);
                    }
                }
            }
        }
        sycl::group_barrier(item.get_group());
    }

    // Write output
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (ic0 + j >= ne01) continue;

        const float inv_sum = (KQ_sum[j] > 0.0f) ? (1.0f / KQ_sum[j]) : 0.0f;
        float * dst_row = dst + ((sequence * ne01 + (ic0 + j)) * ne02 + head) * D;

#if FATTN_MMA_DEBUG
        // Debug: print softmax statistics for first head, last query during prompt eval
        if (tid == 0 && head == 0 && ic0 + j >= 4 && ne01 >= 5) {
            sycl::ext::oneapi::experimental::printf(
                "[FA-MMA-OUT] h=%d q=%d KQ_max=%f KQ_sum=%f inv_sum=%f D=%d ne01=%d ne11=%d\n",
                head, ic0 + j, KQ_max[j], KQ_sum[j], inv_sum, D, ne01, ne11);
        }
#endif

        // DEBUG: Test output indexing by writing head index
        // Uncomment to verify output layout is correct
        // #define DEBUG_OUTPUT_HEAD_INDEX
        #ifdef DEBUG_OUTPUT_HEAD_INDEX
        #pragma unroll
        for (int i = 0; i < D_per_thread; ++i) {
            const int d_idx = tid + i * nthreads;
            if (d_idx < D) {
                dst_row[d_idx] = static_cast<float>(head);
            }
        }
        #else
        #pragma unroll
        for (int i = 0; i < D_per_thread; ++i) {
            const int d_idx = tid + i * nthreads;
            if (d_idx < D) {
                float val = VKQ[j][i] * inv_sum;
                dst_row[d_idx] = sycl::isfinite(val) ? val : 0.0f;
            }
        }
        #endif
    }
}

#endif // SYCL_JOINT_MATRIX_AVAILABLE

// Launch function for MMA-based flash attention
template <int D, int ncols, bool use_logit_softcap, typename Q_type>
void launch_fattn_mma(
    const fattn_params & params,
    dpct::queue_ptr stream) {

#if SYCL_JOINT_MATRIX_AVAILABLE
    using config = fattn_mma_config<D>;

    // Thread configuration - use sub-group size of 16 for Intel XMX
    constexpr int nthreads = 128;  // Multiple of SG_SIZE

    // Shared memory size calculation
    // tile_Q: ncols * D * sizeof(half)
    // tile_K: BATCH_KV * D * sizeof(half)
    // tile_V: BATCH_KV * D * sizeof(half)
    // KQ_tile: ncols * BATCH_KV * sizeof(float)
    // reduce_buf: nthreads * sizeof(float)
    constexpr size_t shared_mem_half = ncols * D + config::BATCH_KV * D * 2;
    constexpr size_t shared_mem_float = ncols * config::BATCH_KV + nthreads;
    constexpr size_t shared_mem_size = shared_mem_half + shared_mem_float * 2;  // floats are 2x half size

    const int n_query_blocks = (params.ne01 + ncols - 1) / ncols;
    sycl::range<3> grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, nthreads);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> shared_acc(sycl::range<1>(shared_mem_size), cgh);

        const char * Q_ptr = params.Q;
        const char * K_ptr = params.K;
        const char * V_ptr = params.V;
        const char * mask_ptr = params.mask;
        float * dst_ptr = params.dst;
        const float scale_val = params.scale;
        const float max_bias_val = params.max_bias;
        const float m0_val = params.m0;
        const float m1_val = params.m1;
        const uint32_t n_head_log2_val = params.n_head_log2;
        const float logit_softcap_val = params.logit_softcap;
        const int ne00 = params.ne00, ne01 = params.ne01, ne02 = params.ne02, ne03 = params.ne03;
        const int nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
        const int ne10 = params.ne10, ne11 = params.ne11, ne12 = params.ne12, ne13 = params.ne13;
        const int nb11 = params.nb11, nb12 = params.nb12;
        const int64_t nb13 = params.nb13;
        const int nb21 = params.nb21, nb22 = params.nb22;
        const int64_t nb23 = params.nb23;
        const int ne30 = params.ne30, ne31 = params.ne31, ne32 = params.ne32, ne33 = params.ne33;
        const int nb31 = params.nb31, nb32 = params.nb32;
        const int64_t nb33 = params.nb33;

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                sycl::half * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                flash_attn_mma_ext_f16<D, ncols, use_logit_softcap, Q_type>(
                    Q_ptr, K_ptr, V_ptr, mask_ptr, dst_ptr,
                    scale_val, max_bias_val, m0_val, m1_val, n_head_log2_val, logit_softcap_val,
                    ne00, ne01, ne02, ne03,
                    nb01, nb02, nb03,
                    ne10, ne11, ne12, ne13,
                    nb11, nb12, nb13,
                    nb21, nb22, nb23,
                    ne30, ne31, ne32, ne33,
                    nb31, nb32, nb33,
                    item, shared);
            });
    });
#else
    // Fallback: joint_matrix not available, should not reach here
    GGML_UNUSED(params);
    GGML_UNUSED(stream);
    GGML_ASSERT(false && "SYCL joint_matrix not available");
#endif
}

// Check if MMA-based FA is available
inline bool fattn_mma_available() {
#if SYCL_JOINT_MATRIX_AVAILABLE
    return true;
#else
    return false;
#endif
}

#endif // GGML_SYCL_FATTN_MMA_HPP
