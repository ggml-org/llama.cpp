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

// Q6_K DMMV kernel dispatch functions (for unit testing)
void dequantize_mul_mat_vec_q6_K_sycl(const void *vx, const float *y,
                                      float *dst, const int ncols,
                                      const int nrows,
                                      dpct::queue_ptr stream);

void dequantize_mul_mat_vec_q6_K_sycl_soa(const void *vx, const float *y,
                                          float *dst, const int ncols,
                                          const int nrows, const int64_t ne01,
                                          const int row_low,
                                          dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
