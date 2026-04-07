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

#include "conv.hpp"

static  void conv_transpose_1d_kernel(
        const int s0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0, const int dst_ne0,
        const float * src0, const float * src1,  float * dst,
        const sycl::nd_item<3> &item_ct1) {
    int global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value =  src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_sycl(
    const int s0, const int output_size,
    const int src0_ne0, const int src0_ne1, const int src0_ne2,
    const int src1_ne0, const int dst_ne0,
    const float *src0, const float *src1, float *dst,
    const queue_ptr& stream) {

    const int num_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_size,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/2);
    auto src0 = dst.src(0);
    auto src1 = dst.src(1);
    const float * src0_d = src0.resolve_as<const float>();
    const float * src1_d = src1.resolve_as<const float>();

    float * dst_d = dst.resolve_as<float>();
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0.type() == GGML_TYPE_F32);
    GGML_ASSERT(dst.type() == GGML_TYPE_F32);

    GGML_ASSERT(src0.is_contiguous());
    GGML_ASSERT(src1.is_contiguous());

    const int32_t * opts = static_cast<const int32_t *>(dst.op_params());

    const int s0 = opts[0];

    const int64_t output_size = dst.nelements();

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0.ne(0), src0.ne(1), src0.ne(2),
        src1.ne(0), dst.ne(0),
        src0_d, src1_d, dst_d, stream);
}
