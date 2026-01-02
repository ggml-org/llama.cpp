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
    const ggml_tensor *src0, const ggml_tensor *src1, const ggml_tensor *ids, ggml_tensor *dst);

#ifdef GGML_SYCL_GRAPH
// Pre-allocate Q8_1 buffers for MoE graph recording
// Must be called before graph recording starts (during decode phase)
void ggml_sycl_moe_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph);
#endif

// Convert reordered tensor to coalesced layout for better memory bandwidth
// Call this AFTER reorder_qw and BEFORE graph recording (at model load time)
// Returns true if conversion was performed, false if not enabled or wrong type
bool ggml_sycl_convert_to_coalesced_q4_0(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_q8_0(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_q6_k(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_mxfp4(const ggml_tensor * tensor, dpct::queue_ptr stream);

#endif // GGML_SYCL_MMVQ_HPP
