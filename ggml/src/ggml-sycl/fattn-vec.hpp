//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

#include "fattn-common.hpp"
#include <cfloat>

// Flash Attention Vector Kernel for SYCL
// Simplified implementation matching CUDA algorithm flow

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
static void flash_attn_vec_ext_f16(
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
    float * shared_mem) {

    constexpr int nthreads = FATTN_VEC_NTHREADS;  // 128

    // Linear thread ID within work-group
    const int tid = item.get_local_linear_id();

    // Block indices
    const int ic0 = item.get_group(2) * ncols;  // Query column block
    const int sequence = item.get_group(0) / ne02;  // Batch index
    const int head = item.get_group(0) % ne02;  // Head index
    const int gqa_ratio = ne02 / ne12;  // Q heads per KV head

    // Pointers to this work-group's data
    const char * Q_base = Q + nb03 * sequence + nb02 * head;
    const int kv_head = head / gqa_ratio;
    const char * K_base = K + nb13 * sequence + nb12 * kv_head;
    const char * V_base = V + nb23 * sequence + nb22 * kv_head;

    // Mask setup (matching CUDA exactly)
    const sycl::half * maskh = mask ?
        reinterpret_cast<const sycl::half*>(mask + nb33 * (sequence % ne33) + nb31 * ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    auto sg = item.get_sub_group();
    const int lane_id = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();
    // Use runtime SG size to be safe
    const int sg_size = sg.get_local_range()[0];
    const int num_sgs = nthreads / sg_size;

    float * KQ_shared = shared_mem;
    float * reduce_shared = shared_mem + ncols;  // [num_sgs * ncols] for reduction

    // Each thread handles multiple D elements
    constexpr int D_per_thread = (D + nthreads - 1) / nthreads;

    // Load Q into registers (scaled)
    float Q_reg[ncols][D_per_thread];
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const Q_type * Q_ptr = (ic0 + j < ne01) ?
            reinterpret_cast<const Q_type*>(Q_base + nb01 * (ic0 + j)) : nullptr;
        #pragma unroll
        for (int i = 0; i < D_per_thread; ++i) {
            const int d_idx = tid + i * nthreads;
            Q_reg[j][i] = (Q_ptr && d_idx < D) ? static_cast<float>(Q_ptr[d_idx]) * scale : 0.0f;
        }
    }

    // Accumulators for online softmax
    float VKQ[ncols][D_per_thread];  // Weighted V accumulator
    float KQ_max[ncols];  // Running max of KQ values
    float KQ_sum[ncols];  // Running sum of exp(KQ - KQ_max)

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
    for (int k_idx = 0; k_idx < ne11; ++k_idx) {
        const sycl::half * K_h = reinterpret_cast<const sycl::half*>(K_base + nb11 * k_idx);

        // Compute Q·K for all query columns
        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float sum = 0.0f;

            // Dot product - each thread computes partial sum
            #pragma unroll
            for (int i = 0; i < D_per_thread; ++i) {
                const int d_idx = tid + i * nthreads;
                if (d_idx < D) {
                    sum += Q_reg[j][i] * static_cast<float>(K_h[d_idx]);
                }
            }

            // Subgroup reduction
            for (int offset = sg_size / 2; offset > 0; offset >>= 1) {
                sum += sycl::shift_group_left(sg, sum, offset);
            }

            // Store partial result from each subgroup
            if (lane_id == 0) {
                reduce_shared[j * num_sgs + sg_id] = sum;
            }
        }

        sycl::group_barrier(item.get_group());

        // Final reduction across subgroups and apply mask
        if (tid < ncols) {
            float total = 0.0f;
            #pragma unroll
            for (int s = 0; s < num_sgs; ++s) {
                total += reduce_shared[tid * num_sgs + s];
            }

            if (use_logit_softcap) {
                total = logit_softcap * sycl::tanh(total);
            }

            // Apply mask
            if (maskh && ic0 + tid < ne01) {
                total += slope * static_cast<float>(maskh[tid * ne11 + k_idx]);
            }

            KQ_shared[tid] = (ic0 + tid < ne01) ? total : -FLT_MAX;
        }

        sycl::group_barrier(item.get_group());

        // Update accumulators with V using online softmax
        const sycl::half * V_h = reinterpret_cast<const sycl::half*>(V_base + nb21 * k_idx);

        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float KQ_val = KQ_shared[j];

            // Online softmax update
            const float KQ_max_new = sycl::fmax(KQ_max[j], KQ_val);
            const float KQ_max_scale = sycl::exp(KQ_max[j] - KQ_max_new);
            KQ_max[j] = KQ_max_new;

            const float KQ_softmax = sycl::exp(KQ_val - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j] * KQ_max_scale + KQ_softmax;

            // Update V accumulator
            #pragma unroll
            for (int i = 0; i < D_per_thread; ++i) {
                VKQ[j][i] *= KQ_max_scale;
                const int d_idx = tid + i * nthreads;
                if (d_idx < D) {
                    VKQ[j][i] += KQ_softmax * static_cast<float>(V_h[d_idx]);
                }
            }
        }

        sycl::group_barrier(item.get_group());
    }

    // Apply attention sinks if present
    // Sinks add a virtual "sink token" that absorbs some attention probability
    // This happens after all K/V positions have been processed
    if (sinks) {
        // Get per-head sink value
        const float * sinks_f = reinterpret_cast<const float *>(sinks);
        const float sink = sinks_f[head];

        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j >= ne01) continue;

            // Update max with sink value
            const float KQ_max_new = sycl::fmax(sink, KQ_max[j]);
            const float KQ_max_scale = sycl::exp(KQ_max[j] - KQ_max_new);
            KQ_max[j] = KQ_max_new;

            // Add sink contribution to sum - ALL threads need to do this
            // because each thread uses its own KQ_sum[j] for normalization
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
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (ic0 + j >= ne01) continue;

        const float inv_sum = (KQ_sum[j] > 0.0f) ? (1.0f / KQ_sum[j]) : 0.0f;

        // Output layout: dst[((sequence * ne01 + query) * ne02 + head) * D + d]
        float * dst_row = dst + ((sequence * ne01 + (ic0 + j)) * ne02 + head) * D;

        #pragma unroll
        for (int i = 0; i < D_per_thread; ++i) {
            const int d_idx = tid + i * nthreads;
            if (d_idx < D) {
                float val = VKQ[j][i] * inv_sum;
                dst_row[d_idx] = sycl::isfinite(val) ? val : 0.0f;
            }
        }
    }
}

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
void launch_fattn_vec(
    const fattn_params & params,
    dpct::queue_ptr stream) {

    constexpr int nthreads = FATTN_VEC_NTHREADS;
    constexpr int num_sgs = nthreads / 16; // Allocate for worst case SG=16
    // Shared mem: ncols for KQ values + num_sgs * ncols for reduction
    constexpr int shared_mem_size = ncols + num_sgs * ncols;

    const int n_query_blocks = (params.ne01 + ncols - 1) / ncols;
    sycl::range<3> grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, nthreads);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_acc(sycl::range<1>(shared_mem_size), cgh);

        const char * Q = params.Q;
        const char * K = params.K;
        const char * V = params.V;
        const char * mask = params.mask;
        const char * sinks = params.sinks;
        float * dst = params.dst;
        const float scale = params.scale;
        const float max_bias = params.max_bias;
        const float m0 = params.m0;
        const float m1 = params.m1;
        const uint32_t n_head_log2 = params.n_head_log2;
        const float logit_softcap = params.logit_softcap;
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
                float * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                flash_attn_vec_ext_f16<D, ncols, use_logit_softcap, Q_type>(
                    Q, K, V, mask, sinks, dst,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    ne00, ne01, ne02, ne03,
                    nb01, nb02, nb03,
                    ne10, ne11, ne12, ne13,
                    nb11, nb12, nb13,
                    nb21, nb22, nb23,
                    ne30, ne31, ne32, ne33,
                    nb31, nb32, nb33,
                    item, shared
                );
            }
        );
    });
}

#endif // GGML_SYCL_FATTN_VEC_HPP
