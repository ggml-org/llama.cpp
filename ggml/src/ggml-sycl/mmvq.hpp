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

// Fused MoE mul_mat_vec_q for one token: takes all expert weights and a
// device-side ids tensor, and dispatches all n_experts_used expert matmuls
// in a single kernel launch. Supports all quantized types the per-expert
// MMVQ kernels support: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
// Q5_K, Q6_K. Weights must be in the standard (non-reorder) block layout.
// - vx_base / dst_base: start of the full weight / output tensors.
// - vy: pre-quantized src1 buffer (Q8_1 AOS).
// - ids_dev: device pointer to int32 tensor of length n_experts_used with
//   the expert index for each slot.
// - expert_weight_stride: byte stride between consecutive experts in vx_base.
// - dst_row_stride: byte stride between consecutive dst rows.
// - src1_row_stride: 0 for a shared src1 (gate/up proj pattern), else
//   per-expert byte stride into vy (down proj pattern).
// Returns true if the type was handled, false if the caller should fall back.
bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,   // 0 = shared src1, else per-expert stride in bytes
    dpct::queue_ptr    stream);

#endif // GGML_SYCL_MMVQ_HPP
