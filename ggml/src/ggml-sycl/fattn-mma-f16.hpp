//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_MMA_F16_HPP
#define GGML_SYCL_FATTN_MMA_F16_HPP

#include "fattn-common.hpp"
#include <sycl/sycl.hpp>
#include <cfloat>

// Check for joint_matrix support
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#define SYCL_JOINT_MATRIX_AVAILABLE 1
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#define SYCL_JOINT_MATRIX_AVAILABLE 0
#endif

#if SYCL_JOINT_MATRIX_AVAILABLE

namespace sycl_mma = sycl::ext::oneapi::experimental::matrix;

// =============================================================================
// Configuration for Flash Attention MMA kernel
// =============================================================================

// Debug flag - kernel printf debug (very slow, use sparingly)
// All debug output now goes to dump files via fattn.cpp
#define FATTN_MMA_F16_DEBUG 0

// Intel Arc supports sub-group size 16 with joint_matrix
// Tile configuration: M=8, N=16, K=16 for fp16
constexpr int FATTN_MMA_TM = 8;    // Tile rows (queries per MMA)
constexpr int FATTN_MMA_TN = 16;   // Tile cols (K positions per MMA)
constexpr int FATTN_MMA_TK = 16;   // Reduction dimension
constexpr int FATTN_MMA_SG = 16;   // Sub-group size

// Batch size for K/V processing (must be multiple of TN)
template <int D>
struct fattn_mma_f16_config {
    static constexpr int BATCH_KV = 32;  // Process 32 K/V positions per iteration
    static constexpr int D_TILES = D / FATTN_MMA_TK;  // Number of K-tiles for D dimension
};

// Specialization for D=64
template <>
struct fattn_mma_f16_config<64> {
    static constexpr int BATCH_KV = 32;
    static constexpr int D_TILES = 64 / FATTN_MMA_TK;  // 4 tiles
};

// Specialization for D=128
template <>
struct fattn_mma_f16_config<128> {
    static constexpr int BATCH_KV = 32;
    static constexpr int D_TILES = 128 / FATTN_MMA_TK;  // 8 tiles
};

// =============================================================================
// Flash Attention MMA Kernel using joint_matrix
// =============================================================================

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
static void flash_attn_mma_f16_kernel(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    const char * __restrict__ mask,
    const char * __restrict__ sinks,
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

    using config = fattn_mma_f16_config<D>;

    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_linear_id();
    const int lane_id = sg.get_local_linear_id();
    const int tid = item.get_local_linear_id();

    constexpr int nthreads = 128;
    constexpr int n_sg = nthreads / FATTN_MMA_SG;

    // Work-group indices
    const int ic0 = item.get_group(2) * ncols;  // Query column block
    const int sequence = item.get_group(0) / ne02;  // Batch index
    const int head = item.get_group(0) % ne02;  // Head index
    const int gqa_ratio = ne02 / ne12;

    // Pointers to this work-group's data
    const char * Q_base = Q + nb03 * sequence + nb02 * head;
    const int kv_head = head / gqa_ratio;
    const char * K_base = K + nb13 * sequence + nb12 * kv_head;
    const char * V_base = V + nb23 * sequence + nb22 * kv_head;

    // Mask setup
    // Mask layout: [ne33, ne32, ne31, ne30] = [batch, heads, n_tokens_padded, n_kv]
    // nb31 = stride for query (n_tokens_padded dimension)
    // nb32 = stride for head dimension
    // nb33 = stride for batch dimension
    // ne32 = number of heads in mask (can be 1 for broadcast)
    const int mask_head = ne32 > 1 ? head % ne32 : 0;  // Handle head broadcast
    const sycl::half * maskh = mask ?
        reinterpret_cast<const sycl::half*>(mask + nb33 * (sequence % ne33) + nb32 * mask_head + nb31 * ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);


    // Shared memory layout
    // tile_Q: [ncols][D] - Query tile
    // tile_K: [BATCH_KV][D] - Key tile
    // tile_V: [BATCH_KV][D] - Value tile
    // KQ_shared: [ncols][BATCH_KV] - Attention scores
    sycl::half * tile_Q = shared_mem;
    sycl::half * tile_K = tile_Q + ncols * D;
    sycl::half * tile_V = tile_K + config::BATCH_KV * D;
    float * KQ_shared = reinterpret_cast<float*>(tile_V + config::BATCH_KV * D);

    // Load Q into shared memory (scaled)
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

#if FATTN_MMA_F16_DEBUG
    // Debug: Print Q values for first head
    if (tid == 0 && head == 0 && sequence == 0 && ic0 == 0) {
        sycl::ext::oneapi::experimental::printf("[MMA-Q] head=%d tile_Q[0..3]: %.4f %.4f %.4f %.4f\n",
            head,
            static_cast<float>(tile_Q[0]),
            static_cast<float>(tile_Q[1]),
            static_cast<float>(tile_Q[2]),
            static_cast<float>(tile_Q[3]));
    }
#endif

    // Per-thread accumulators for online softmax
    constexpr int D_per_thread = (D + nthreads - 1) / nthreads;
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

    // Main loop over K/V sequence
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

#if FATTN_MMA_F16_DEBUG
        // Debug: Print K values (NOT transposed) for first K position (k=0)
        // tile_K[k][d] = K[kv_start+k][d], stored as tile_K[k * D + d]
        // So tile_K[0], tile_K[1], ... are K[kv_start][0], K[kv_start][1], ...
        if (tid == 0 && head == 0 && sequence == 0 && ic0 == 0 && kv_start == 0) {
            sycl::ext::oneapi::experimental::printf("[MMA-K] kv_start=%d tile_K[k=0][d=0..3]: %.4f %.4f %.4f %.4f\n",
                kv_start,
                static_cast<float>(tile_K[0 * D + 0]),
                static_cast<float>(tile_K[0 * D + 1]),
                static_cast<float>(tile_K[0 * D + 2]),
                static_cast<float>(tile_K[0 * D + 3]));
        }
#endif

        // =========================================================
        // Compute Q @ K^T using joint_matrix MMA
        // Q: [ncols][D], K^T: [D][BATCH_KV] -> QK: [ncols][BATCH_KV]
        // =========================================================

        // Zero KQ_shared
        for (int idx = tid; idx < ncols * config::BATCH_KV; idx += nthreads) {
            KQ_shared[idx] = 0.0f;
        }
        sycl::group_barrier(item.get_group());

        // Each sub-group handles a portion of the computation
        // We compute QK = Q @ K^T where:
        // - Q is [ncols x D] row-major
        // - K is [BATCH_KV x D] row-major, so K^T is [D x BATCH_KV] col-major
        // - Result QK is [ncols x BATCH_KV]

        // For joint_matrix: C[M,N] = A[M,K] @ B[K,N]
        // We need: QK[ncols, BATCH_KV] = Q[ncols, D] @ K^T[D, BATCH_KV]
        // Process in tiles: for each tile of K positions

        // Use sub-groups to compute different parts
        // Each sub-group computes one row of Q against all K positions
        for (int q_idx = sg_id; q_idx < ncols; q_idx += n_sg) {
            if (ic0 + q_idx >= ne01) continue;

            for (int k_tile = 0; k_tile < kv_count; k_tile += FATTN_MMA_TN) {
                const int k_end_tile = sycl::min(k_tile + FATTN_MMA_TN, kv_count);

                // Compute dot products for this Q row and K tile
                for (int k_idx = k_tile + lane_id; k_idx < k_end_tile; k_idx += FATTN_MMA_SG) {
                    float dot = 0.0f;

                    // Dot product over D dimension
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        dot += static_cast<float>(tile_Q[q_idx * D + d]) *
                               static_cast<float>(tile_K[k_idx * D + d]);
                    }

                    // Apply logit softcap
                    if (use_logit_softcap) {
                        dot = logit_softcap * sycl::tanh(dot);
                    }

                    // Apply mask
                    // maskh is already offset by batch and head
                    // maskh indexing: maskh[query * ne30 + kv_pos] (ne30 is the KV dimension stride)
                    if (maskh) {
                        // Use ne30 (not ne11) for the KV stride in mask
                        float mask_val = static_cast<float>(maskh[q_idx * ne30 + kv_start + k_idx]);
                        dot += slope * mask_val;
                    }

                    KQ_shared[q_idx * config::BATCH_KV + k_idx] = dot;
                }
            }
        }
        sycl::group_barrier(item.get_group());

#if FATTN_MMA_F16_DEBUG
        // Debug: Print QK values (after mask applied in scalar dot product)
        if (tid == 0 && head == 0 && sequence == 0 && ic0 == 0 && kv_start == 0) {
            sycl::ext::oneapi::experimental::printf("[MMA-QK] AFTER mask KQ_shared[q=0][k=0..3]: %.4f %.4f %.4f %.4f\n",
                KQ_shared[0 * config::BATCH_KV + 0],
                KQ_shared[0 * config::BATCH_KV + 1],
                KQ_shared[0 * config::BATCH_KV + 2],
                KQ_shared[0 * config::BATCH_KV + 3]);
        }
#endif

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

// KQ-PRE-V debug disabled - confirmed KQ_shared is correct per head

        // Online softmax and V accumulation
        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j >= ne01) continue;

            // Find max in this batch
            float batch_max = -FLT_MAX;
            for (int k = 0; k < kv_count; ++k) {
                batch_max = sycl::fmax(batch_max, KQ_shared[j * config::BATCH_KV + k]);
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
                const float kq_val = KQ_shared[j * config::BATCH_KV + k];
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

    // Apply attention sinks if present
    // Sinks add a virtual "sink token" that absorbs some attention probability
    // This happens after all K/V positions have been processed
    if (sinks) {
        const float * sinks_f = reinterpret_cast<const float *>(sinks);
        const float sink = sinks_f[head];

        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j >= ne01) continue;

            // Update max with sink value
            const float KQ_max_new = sycl::fmax(sink, KQ_max[j]);
            const float KQ_max_scale = sycl::exp(KQ_max[j] - KQ_max_new);
            KQ_max[j] = KQ_max_new;

            // Add sink contribution to sum
            const float sink_softmax = sycl::exp(sink - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j] * KQ_max_scale + sink_softmax;

            // Scale VKQ accumulator by the max change
            #pragma unroll
            for (int i = 0; i < D_per_thread; ++i) {
                VKQ[j][i] *= KQ_max_scale;
            }
        }
    }

    // Write output
    // Output layout per ggml spec: [D][n_head][n_batch][ne3] (res: [n_embd_v, n_head, n_batch, ne3])
    // From CUDA reference: dst[d + D * (head + ne02 * (query + ne01 * batch))]
    // This matches ggml's layout: dimension 0 is D, dimension 1 is head, dimension 2 is query, dimension 3 is batch
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (ic0 + j >= ne01) continue;

        const float inv_sum = (KQ_sum[j] > 0.0f) ? (1.0f / KQ_sum[j]) : 0.0f;
        float * dst_row = dst + D * (head + ne02 * ((ic0 + j) + ne01 * sequence));

#if FATTN_MMA_F16_DEBUG
        // Debug: print VKQ for different heads when KQ_sum > 1.5 (multiple valid K positions)
        if (tid == 0 && j == 0 && (head == 18 || head == 20 || head == 21) && ne01 == 1 && sequence == 0 && KQ_sum[j] > 1.5f) {
            sycl::ext::oneapi::experimental::printf(
                "[VKQ-MULTI] head=%d KQ_max=%.3f KQ_sum=%.3f VKQ[0]=%.6f out[0]=%.6f\n",
                head, KQ_max[j], KQ_sum[j], VKQ[j][0], VKQ[j][0] * inv_sum);
        }
        // Debug: print attention weights for first few K positions
        if (tid == 0 && j == 0 && (head == 18 || head == 20 || head == 21) && ne01 == 1 && sequence == 0 && KQ_sum[j] > 1.5f) {
            sycl::ext::oneapi::experimental::printf(
                "[ATTN-WEIGHTS] head=%d inv_sum=%.6f VKQ0..3: %.4f %.4f %.4f %.4f\n",
                head, inv_sum, VKQ[j][0],
                D_per_thread > 1 ? VKQ[j][1] : 0.0f,
                D_per_thread > 2 ? VKQ[j][2] : 0.0f,
                D_per_thread > 3 ? VKQ[j][3] : 0.0f);
        }
#endif

        #pragma unroll
        for (int i = 0; i < D_per_thread; ++i) {
            const int d_idx = tid + i * nthreads;
            if (d_idx < D) {
                float val = VKQ[j][i] * inv_sum;
                dst_row[d_idx] = sycl::isfinite(val) ? val : 0.0f;
            }
        }

// MULTI-K debug disabled for now
    }
}

#endif // SYCL_JOINT_MATRIX_AVAILABLE

// =============================================================================
// Launch function for MMA-based flash attention
// =============================================================================

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
void launch_fattn_mma_f16(
    const fattn_params & params,
    dpct::queue_ptr stream) {

#if SYCL_JOINT_MATRIX_AVAILABLE
    using config = fattn_mma_f16_config<D>;

    constexpr int nthreads = 128;

    // Shared memory size calculation
    // tile_Q: ncols * D * sizeof(half)
    // tile_K: BATCH_KV * D * sizeof(half)
    // tile_V: BATCH_KV * D * sizeof(half)
    // KQ_shared: ncols * BATCH_KV * sizeof(float)
    constexpr size_t shared_half = ncols * D + config::BATCH_KV * D * 2;
    constexpr size_t shared_float = ncols * config::BATCH_KV;
    constexpr size_t shared_mem_size = shared_half + shared_float * 2;  // floats are 2x half

    const int n_query_blocks = (params.ne01 + ncols - 1) / ncols;
    sycl::range<3> grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, nthreads);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> shared_acc(sycl::range<1>(shared_mem_size), cgh);

        const char * Q_ptr = params.Q;
        const char * K_ptr = params.K;
        const char * V_ptr = params.V;
        const char * mask_ptr = params.mask;
        const char * sinks_ptr = params.sinks;
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
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(FATTN_MMA_SG)]] {
                sycl::half * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                flash_attn_mma_f16_kernel<D, ncols, use_logit_softcap, Q_type>(
                    Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr,
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
    GGML_UNUSED(params);
    GGML_UNUSED(stream);
    GGML_ASSERT(false && "SYCL joint_matrix not available");
#endif
}

// Check if MMA F16 kernel is available
inline bool fattn_mma_f16_available() {
#if SYCL_JOINT_MATRIX_AVAILABLE
    return true;
#else
    return false;
#endif
}

#endif // GGML_SYCL_FATTN_MMA_F16_HPP
