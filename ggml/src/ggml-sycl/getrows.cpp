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

#include "ggml-impl.h"
#include "common.hpp"
#include "dequantize.hpp"
#include "getrows.hpp"

// dispatch by int tag: a function pointer NTTP embeds the helper mangled name into the
// kernel image name, and IGC in the driver fails to compile such images
enum dequantize_tag {
    DEQUANT_Q1_0,
    DEQUANT_MXFP4,
    DEQUANT_NVFP4,
    DEQUANT_IQ2_XXS,
    DEQUANT_IQ2_XS,
    DEQUANT_IQ2_S,
    DEQUANT_IQ3_XXS,
    DEQUANT_IQ1_S,
    DEQUANT_IQ1_M,
    DEQUANT_IQ3_S,
    DEQUANT_IQ4_NL,
    DEQUANT_IQ4_XS,
    DEQUANT_Q3_K,
    DEQUANT_Q4_0,
    DEQUANT_Q4_1,
    DEQUANT_Q5_0,
    DEQUANT_Q5_1,
    DEQUANT_Q6_K,
    DEQUANT_Q8_0,
};

enum dequantize_f32_tag {
    DEQUANT_Q2_K_F32,
    DEQUANT_Q4_K_F32,
    DEQUANT_Q5_K_F32,
};

template<int qk, int qr, dequantize_tag deq, typename dst_t>
static void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    switch (deq) {
        case DEQUANT_Q1_0:    dequantize_q1_0(src0_row, ib, iqs, v); break;
        case DEQUANT_MXFP4:   dequantize_mxfp4(src0_row, ib, iqs, v); break;
        case DEQUANT_NVFP4:   dequantize_nvfp4(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ2_XXS: dequantize_iq2_xxs(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ2_XS:  dequantize_iq2_xs(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ2_S:   dequantize_iq2_s(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ3_XXS: dequantize_iq3_xxs(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ1_S:   dequantize_iq1_s(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ1_M:   dequantize_iq1_m(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ3_S:   dequantize_iq3_s(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ4_NL:  dequantize_iq4_nl(src0_row, ib, iqs, v); break;
        case DEQUANT_IQ4_XS:  dequantize_iq4_xs(src0_row, ib, iqs, v); break;
        case DEQUANT_Q3_K:    dequantize_q3_K(src0_row, ib, iqs, v); break;
        case DEQUANT_Q4_0:    dequantize_q4_0(src0_row, ib, iqs, v); break;
        case DEQUANT_Q4_1:    dequantize_q4_1(src0_row, ib, iqs, v); break;
        case DEQUANT_Q5_0:    dequantize_q5_0(src0_row, ib, iqs, v); break;
        case DEQUANT_Q5_1:    dequantize_q5_1(src0_row, ib, iqs, v); break;
        case DEQUANT_Q6_K:    dequantize_q6_K(src0_row, ib, iqs, v); break;
        case DEQUANT_Q8_0:    dequantize_q8_0(src0_row, ib, iqs, v); break;
    }

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

template<int qk, int qr, dequantize_f32_tag deq, typename dst_t>
static void k_get_rows_f32(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk;
    const int iqs = (i00%qk)/qr;
    const int iybs = i00 - i00%qk;
    const int y_offset = qr == 1 ? 1 : qk/2;

    float v0;
    float v1;
    switch (deq) {
        case DEQUANT_Q2_K_F32: dequantize_q2_K_f32(src0_row, ib, iqs, v0, v1); break;
        case DEQUANT_Q4_K_F32: dequantize_q4_K_f32(src0_row, ib, iqs, v0, v1); break;
        case DEQUANT_Q5_K_F32: dequantize_q5_K_f32(src0_row, ib, iqs, v0, v1); break;
    }

    dst_row[iybs + iqs + 0] = (dst_t) v0;
    dst_row[iybs + iqs + y_offset] = (dst_t) v1;
}

template<typename src0_t, typename dst_t>
static void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = src0_row[i00];
}

template <int qk, int qr, dequantize_tag dq>
static void get_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <int qk, int qr, dequantize_f32_tag dq>
static void get_rows_sycl_f32(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const void *src0_dd,
                              const int32_t *src1_dd, float *dst_dd,
                              queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_f32<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <typename src0_t, typename dst_t>
static void get_rows_sycl_float(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *src0_dd, const int32_t *src1_dd,
                                dst_t *dst_dd, queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_get_rows_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
            });
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <typename src0_t>
static void k_get_rows_back_float(const src0_t * src0, const int32_t * src1, float * dst,
                                  const int64_t ncols, const int64_t nrows_grad_10, const int64_t nrows_grad_11, const int64_t nrows_dst,
                                  const size_t s01, const size_t s02,
                                  const size_t s10, const size_t s11,
                                  const size_t s1,
                                  const int64_t block_num_y,
                                  const sycl::nd_item<3> & item_ct1) {
    const int64_t col = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (col >= ncols) {
        return;
    }

    // block_num_y is clamped, so stride over destination rows like CUDA k_get_rows_back_float
    for (int64_t dst_row = item_ct1.get_group(1); dst_row < nrows_dst; dst_row += block_num_y) {
        float sum = 0.0f;

        const int64_t nrows_grad_total = nrows_grad_10 * nrows_grad_11;
        for (int64_t i = 0; i < nrows_grad_total; ++i) {
            const int64_t i10 = i % nrows_grad_10;
            const int64_t i11 = i / nrows_grad_10;
            if (src1[i10*s10 + i11*s11] != dst_row) {
                continue;
            }
            sum += (float) src0[col + i10*s01 + i11*s02];
        }

        dst[col + dst_row*s1] = sum;
    }
}

template <typename src0_t>
static void get_rows_back_sycl_float(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                     const ggml_tensor * src1, ggml_tensor * dst,
                                     const src0_t * src0_dd, const int32_t * src1_dd,
                                     float * dst_dd, queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    const int64_t ncols = ne00;
    const int64_t nrows_grad_10 = ne10;
    const int64_t nrows_grad_11 = ne11;
    const int64_t nrows_dst = ne1;

    const size_t s01 = nb01 / sizeof(src0_t);
    const size_t s02 = nb02 / sizeof(src0_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);

    const size_t s1 = nb1 / sizeof(float);

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int64_t block_num_x = (ncols + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const int64_t block_num_y = std::min<int64_t>(nrows_dst, (int64_t) UINT16_MAX);
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_back_float(src0_dd, src1_dd, dst_dd,
                                                   ncols, nrows_grad_10, nrows_grad_11, nrows_dst,
                                                   s01, s02, s10, s11, s1,
                                                   block_num_y, item_ct1);
                         });

    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32 );

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_size(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_size(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) dst->src[1]->data;
    /* TODO: Refactor and remove duplicates */
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::half *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_BF16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::ext::oneapi::bfloat16 *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_I32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const int32_t *)dst->src[0]->data,
            src1_i32, (int32_t *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q1_0:
            get_rows_sycl<QK1_0, 1, DEQUANT_Q1_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_MXFP4:
            get_rows_sycl<QK_MXFP4, 2, DEQUANT_MXFP4>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_NVFP4:
            get_rows_sycl<QK_NVFP4, 1, DEQUANT_NVFP4>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_sycl<QK_K, 1, DEQUANT_IQ2_XXS>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_sycl<QK_K, 1, DEQUANT_IQ2_XS>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_sycl<QK_K, 1, DEQUANT_IQ2_S>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_sycl<QK_K, 1, DEQUANT_IQ3_XXS>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_sycl<QK_K, 1, DEQUANT_IQ1_S>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_sycl<QK_K, 1, DEQUANT_IQ1_M>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_sycl<QK_K, 1, DEQUANT_IQ3_S>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_sycl<QK4_NL, 1, DEQUANT_IQ4_NL>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_sycl<QK_K, 1, DEQUANT_IQ4_XS>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q2_K:
            get_rows_sycl_f32<QK_K, 1, DEQUANT_Q2_K_F32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q3_K:
            get_rows_sycl<QK_K, 1, DEQUANT_Q3_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl<QK4_0, QR4_0, DEQUANT_Q4_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, DEQUANT_Q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_K:
            get_rows_sycl_f32<QK_K, 1, DEQUANT_Q4_K_F32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, DEQUANT_Q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, DEQUANT_Q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_K:
            get_rows_sycl_f32<QK_K, 1, DEQUANT_Q5_K_F32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q6_K:
            get_rows_sycl<QK_K, 1, DEQUANT_Q6_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl<QK8_0, QR8_0, DEQUANT_Q8_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        default:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

void ggml_sycl_op_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(dst));

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_back_sycl_float(ctx, src0, src1, dst, (const sycl::half *) src0->data,
                                     (const int32_t *) src1->data, (float *) dst->data,
                                     ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_back_sycl_float(ctx, src0, src1, dst, (const float *) src0->data,
                                     (const int32_t *) src1->data, (float *) dst->data,
                                     ctx.stream());
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0->type));
            break;
    }
}
