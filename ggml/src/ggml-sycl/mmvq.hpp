//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#include "common.hpp"


void ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// MoE-aware MUL_MAT_ID dispatch: GPU-side expert routing without host sync
// This is compatible with SYCL command graph recording
// Returns true if the operation was handled, false to fall back to host-side routing
bool ggml_sycl_mul_mat_id_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, const ggml_tensor *ids, ggml_tensor *dst,
    const layout_mode * forced_layout = nullptr);

#ifdef GGML_SYCL_GRAPH
// Pre-allocate Q8_1 buffers for MoE graph recording
// Must be called before graph recording starts (during decode phase)
void ggml_sycl_moe_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph);
#endif

// Batched MMVQ dispatch for MoE hybrid GPU path.
// Replaces per-expert ggml_sycl_mul_mat() loop with a single kernel submission.
// expert_ptrs_device: device pointer table (expert_id -> device weight ptr)
// n_experts: total number of experts in the model
// gpu_expert_ids: host array of expert IDs that have cached device pointers
// gpu_iid1s: host array of token indices (parallel to gpu_expert_ids)
// gpu_ids: host array of expert slot indices (parallel to gpu_expert_ids)
// n_gpu_entries: number of GPU-dispatched entries
// Returns true if handled, false to fall back to per-expert loop.
bool mmvq_moe_batched_dispatch(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *         src0,
    const ggml_tensor *         src1,
    ggml_tensor *               dst,
    const void * const *        expert_ptrs_device,
    const int32_t *             gpu_expert_ids,
    const int64_t *             gpu_iid1s,
    const int64_t *             gpu_ids,
    int                         n_gpu_entries,
    int                         n_experts,
    int64_t                     n_ids);

// Convert reordered tensor to coalesced layout for better memory bandwidth
// Call this AFTER reorder_qw and BEFORE graph recording (at model load time)
// Returns true if conversion was performed, false if not enabled or wrong type
bool ggml_sycl_convert_to_coalesced_q4_0(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_q8_0(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_q6_k(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_mxfp4(const ggml_tensor * tensor, dpct::queue_ptr stream);

// =============================================================================
// Direct MMVQ kernel submission for persistent TG micro-graph
// =============================================================================
// These wrappers submit standalone MMVQ kernels to a queue without requiring
// ggml_tensor metadata.  They are used by the micro-graph persistent TG path
// to replace the generic dp4a compute_matmul_tile with the optimized MMVQ
// kernels that use pre-quantized Q8_1 activations.
//
// Weights must be in SOA (Structure-of-Arrays) layout.
// Activations (vy) must be in SOA Q8_1 format (quants at [0,K), ds at [K,...)).
// Use quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa> to produce vy.

// Q4_0 SOA MMVQ: weights in SOA layout, activations in SOA Q8_1
void mmvq_submit_q4_0_soa(sycl::queue & q,
                           const void * weights_soa,
                           const void * y_q8_soa,
                           float *      dst,
                           int          ncols,
                           int          nrows,
                           int          total_nrows,
                           int          row_low);

// Q6_K SOA MMVQ: weights in SOA layout, activations in SOA Q8_1
void mmvq_submit_q6_k_soa(sycl::queue & q,
                            const void * weights_soa,
                            const void * y_q8_soa,
                            float *      dst,
                            int          ncols,
                            int          nrows,
                            int          total_nrows,
                            int          row_low);

void mmvq_submit_mxfp4_soa(sycl::queue & q,
                            const void * weights_soa,
                            const void * y_q8_soa,
                            float *      dst,
                            int          ncols,
                            int          nrows,
                            int          total_nrows,
                            int          row_low);

// Coalesced MXFP4 MMVQ kernel submission
// vx: coalesced-layout weights (quants tiled word-major, then exponents)
// vy: SOA Q8_1 activations (quants[ncols] then ds[ncols/QK8_1])
void mmvq_submit_mxfp4_coalesced(sycl::queue & q,
                                  const void * vx,
                                  const void * vy,
                                  float *      dst,
                                  int          ncols,
                                  int          nrows);

// Float-to-Q8_1 SOA quantization kernel submission for micro-graph
// Input:  x[ncols] float activations
// Output: y_q8[ncols + ncols/QK8_1 * 4] SOA Q8_1
void mmvq_submit_quantize_q8_1_soa(sycl::queue & q,
                                     const float * x,
                                     void *        y_q8_soa,
                                     int           ncols);

// Compute SOA Q8_1 buffer size in bytes for a given K dimension
inline size_t mmvq_q8_1_soa_size(int K) {
    // quants: K bytes, ds: (K/QK8_1) * sizeof(half2) = K/32 * 4
    // Q6_K SOA kernel reads ds[iby + bq8_offset + 2*i] with max index = K/QK8_1 + 7
    // Add 8 half2 entries of overflow padding to prevent out-of-bounds read
    constexpr size_t Q6K_DS_OVERFLOW_PAD = 8 * sizeof(sycl::half2);  // 32 bytes
    return static_cast<size_t>(K) + (K / QK8_1) * sizeof(sycl::half2) + Q6K_DS_OVERFLOW_PAD;
}

#endif // GGML_SYCL_MMVQ_HPP
