//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

//#include "common.hpp"
#include "pad.hpp"

static void pad_f32(const float * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    float * dst,
                    const int lp0, const int rp0, const int lp1, const int rp1,
                    const int lp2, const int rp2, const int lp3, const int rp3,
                    const int ne0, const int ne1, const int ne2, const int ne3,
                    sycl::nd_item<3> item_ct1) {
    int i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int i1 = item_ct1.get_group(1);
    int i2 = item_ct1.get_group(0) % ne2;
    int i3 = item_ct1.get_group(0) / ne2;
    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    if ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        const int64_t i00 = i0 - lp0;
        const int64_t i01 = i1 - lp1;
        const int64_t i02 = i2 - lp2;
        const int64_t i03 = i3 - lp3;

        const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } else {
        dst[dst_idx] = 0.0f;
    }
}

static void pad_f32_sycl(const float * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         float * dst, const int lp0, const int rp0, const int lp1, const int rp1,
                         const int lp2, const int rp2, const int lp3, const int rp3,
                         const int ne0, const int ne1, const int ne2, const int ne3,
                         dpct::queue_ptr stream) {
    int num_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> grid(ne2 * ne3, ne1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, item_ct1);
        });
}

void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const size_t ts = ggml_type_size(src0->type);
    const size_t s00 = src0->nb[0] / ts;
    const size_t s01 = src0->nb[1] / ts;
    const size_t s02 = src0->nb[2] / ts;
    const size_t s03 = src0->nb[3] / ts;

    const int32_t lp0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t rp0 = ((const int32_t *)(dst->op_params))[1];
    const int32_t lp1 = ((const int32_t *)(dst->op_params))[2];
    const int32_t rp1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t lp2 = ((const int32_t *)(dst->op_params))[4];
    const int32_t rp2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t lp3 = ((const int32_t *)(dst->op_params))[6];
    const int32_t rp3 = ((const int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

void ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
