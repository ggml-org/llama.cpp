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


template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
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
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

template<int qk, int qr, dequantize_kernel_f32_t dequantize_kernel, typename dst_t>
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
    dequantize_kernel(src0_row, ib, iqs, v0, v1);

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

template <int qk, int qr, dequantize_kernel_t dq>
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

template <int qk, int qr, dequantize_kernel_f32_t dq>
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
            get_rows_sycl<QK1_0, 1, dequantize_q1_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_MXFP4:
            get_rows_sycl<QK_MXFP4, 2, dequantize_mxfp4>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_NVFP4:
            get_rows_sycl<QK_NVFP4, 1, dequantize_nvfp4>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_sycl<QK_K, 1, dequantize_iq2_xxs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_sycl<QK_K, 1, dequantize_iq2_xs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_sycl<QK_K, 1, dequantize_iq2_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_sycl<QK_K, 1, dequantize_iq3_xxs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_sycl<QK_K, 1, dequantize_iq1_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_sycl<QK_K, 1, dequantize_iq1_m>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_sycl<QK_K, 1, dequantize_iq3_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_sycl<QK4_NL, 1, dequantize_iq4_nl>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_sycl<QK_K, 1, dequantize_iq4_xs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q2_K:
            get_rows_sycl_f32<QK_K, 1, dequantize_q2_K_f32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q3_K:
            get_rows_sycl<QK_K, 1, dequantize_q3_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_K:
            get_rows_sycl_f32<QK_K, 1, dequantize_q4_K_f32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_K:
            get_rows_sycl_f32<QK_K, 1, dequantize_q5_K_f32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q6_K:
            get_rows_sycl<QK_K, 1, dequantize_q6_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        default:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

bool ggml_sycl_qsa_gather_shape(const ggml_cgraph * cgraph, int i) {
    if (i + 3 >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * cont_in  = cgraph->nodes[i];
    ggml_tensor * rows     = cgraph->nodes[i + 1];
    ggml_tensor * perm_out = cgraph->nodes[i + 2];
    ggml_tensor * cont_out = cgraph->nodes[i + 3];

    if (cont_in->op != GGML_OP_CONT || rows->op != GGML_OP_GET_ROWS || perm_out->op != GGML_OP_PERMUTE ||
        cont_out->op != GGML_OP_CONT) {
        return false;
    }
    for (const ggml_tensor * node : { cont_in, rows, perm_out, cont_out }) {
        if (node->type != GGML_TYPE_F32 || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    if (ggml_is_empty(cont_in) || ggml_is_empty(rows) || ggml_is_empty(cont_out)) {
        return false;
    }
    if (cont_in->view_src || rows->view_src || cont_out->view_src) {
        return false;
    }
    if ((cont_in->flags | rows->flags | perm_out->flags) & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1 ||
        ggml_node_get_use_count(cgraph, i + 2) != 1) {
        return false;
    }
    if (rows->src[0] != cont_in || perm_out->src[0] != rows || perm_out->view_src != rows ||
        perm_out->view_offs != 0 || cont_out->src[0] != perm_out) {
        return false;
    }

    const ggml_tensor * src = cont_in->src[0];
    const ggml_tensor * idx = rows->src[1];
    if (!src || !idx || src->type != GGML_TYPE_F32 || idx->type != GGML_TYPE_I32) {
        return false;
    }
    if (!ggml_are_same_shape(src, cont_in) || src->ne[2] != 1 || src->ne[3] != 1) {
        return false;
    }
    if (idx->ne[1] != 1 || idx->ne[2] != 1 || idx->ne[3] != 1 || idx->nb[0] != ggml_type_size(GGML_TYPE_I32)) {
        return false;
    }
    const size_t type_size = ggml_type_size(GGML_TYPE_F32);
    if (src->nb[0] % type_size != 0 || src->nb[1] % type_size != 0) {
        return false;
    }
    if (rows->ne[0] != cont_in->ne[0] || rows->ne[1] != idx->ne[0] || rows->ne[2] != 1 || rows->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(cont_in) || !ggml_is_contiguous(rows) || !ggml_is_contiguous(cont_out)) {
        return false;
    }
    if (perm_out->ne[0] != rows->ne[1] || perm_out->ne[1] != rows->ne[0] || perm_out->ne[2] != 1 ||
        perm_out->ne[3] != 1 || perm_out->nb[0] != rows->nb[1] || perm_out->nb[1] != rows->nb[0]) {
        return false;
    }
    return ggml_are_same_shape(cont_out, perm_out);
}

bool ggml_sycl_can_fuse_qsa_gather(const ggml_cgraph * cgraph, int i) {
    return g_ggml_sycl_enable_fusion && ggml_sycl_qsa_gather_shape(cgraph, i);
}

template <int width>
static void k_qsa_gather(const char * src, const int32_t * idx, float * dst,
                         int64_t n_idx, size_t nb0, size_t nb1, const sycl::nd_item<2> & item) {
    const int64_t t = item.get_global_id(0);
    const int64_t c = (int64_t) item.get_global_id(1) * width;
    if (c >= n_idx) {
        return;
    }

    const char * row = src + t * nb0;
    float * out = dst + t * n_idx + c;

    if constexpr (width > 1) {
        sycl::vec<float, width> values;
#pragma unroll
        for (int j = 0; j < width; ++j) {
            values[j] = *(const float *) (row + (int64_t) idx[c + j] * nb1);
        }
        *(sycl::vec<float, width> *) out = values;
    } else {
        *out = *(const float *) (row + (int64_t) idx[c] * nb1);
    }
}

int ggml_sycl_fuse_qsa_gather(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!ggml_sycl_can_fuse_qsa_gather(cgraph, i)) {
        return 0;
    }

    const ggml_tensor * src = cgraph->nodes[i]->src[0];
    const ggml_tensor * idx = cgraph->nodes[i + 1]->src[1];
    ggml_tensor * cont_out = cgraph->nodes[i + 3];

    const int64_t n_idx = cont_out->ne[0];
    const int64_t n_cols = cont_out->ne[1];
    const int width = n_idx % 4 == 0 && (uintptr_t) cont_out->data % 16 == 0 ? 4 : 1;

    constexpr int block = 256;
    const int64_t items = (n_idx + width - 1) / width;
    const sycl::range<2> local(1, block);
    const sycl::range<2> global(n_cols, ((items + block - 1) / block) * block);

    const char * src_data = (const char *) src->data;
    const int32_t * idx_data = (const int32_t *) idx->data;
    float * dst_data = (float *) cont_out->data;
    const size_t nb0 = src->nb[0];
    const size_t nb1 = src->nb[1];
    GGML_ASSERT(src_data && idx_data && dst_data);

    auto launch = [&](auto w) {
        ctx.stream()->parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            k_qsa_gather<decltype(w)::value>(src_data, idx_data, dst_data, n_idx, nb0, nb1, item);
        });
    };
    if (width == 4) {
        launch(std::integral_constant<int, 4>{});
    } else {
        launch(std::integral_constant<int, 1>{});
    }
    return 3;
}
