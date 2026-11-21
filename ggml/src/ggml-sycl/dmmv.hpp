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

#ifndef GGML_SYCL_DMMV_HPP
#define GGML_SYCL_DMMV_HPP

#include "common.hpp"


void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
bool ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    enum ggml_type src0_type, enum ggml_glu_op glu_op,
    const void * vx, const void * vgate, const float * y,
    float * dst, int ncols, int nrows, dpct::queue_ptr stream);

// Routed-expert mat-vec of one token on reordered expert slices (Q3_K), on the ESIMD DMMV kernel: reads the
// f32 activation, so no q8_1 quantize. Only defined when GGML_SYCL_DMMV_HAS_ESIMD; false if the type is unhandled.
bool ggml_sycl_mul_mat_vec_q_id_reorder_esimd(
    enum ggml_type src0_type, const void * vx_base, const float * y, const int32_t * ids_dev, float * dst_base,
    int ncols, int nrows, int n_experts_used, size_t expert_weight_stride, size_t dst_row_stride,
    size_t src1_row_stride, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
