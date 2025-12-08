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

template <typename src0_t>
static void get_rows_sycl_float(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *src0_dd, const int32_t *src1_dd,
                                float *dst_dd, queue_ptr stream) {

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
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_size(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_size(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    // Use device-specific pointers for TP mode (KV cache is allocated per-device)
    const int device = ctx.device;
    const void * src0_d = ggml_sycl_get_data_ptr(dst->src[0], device);
    const int32_t * src1_i32 = (const int32_t *) ggml_sycl_get_data_ptr(dst->src[1], device);
    float * dst_d = (float *) ggml_sycl_get_data_ptr(dst, device);

    // DEBUG: Check token embedding lookup for batch=1 (single row)
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int getrows_b1_dbg = 0;
    bool is_tok_embd = dst->src[0]->name && strstr(dst->src[0]->name, "token_embd");
    int64_t n_rows = dst->src[1]->ne[0];  // Number of rows to fetch
    if (g_ggml_sycl_tp_debug && is_tok_embd && n_rows == 1 && getrows_b1_dbg++ < 5) {
        // Read token ID
        int32_t tok_id;
        ctx.stream()->memcpy(&tok_id, src1_i32, sizeof(int32_t)).wait();
        fprintf(stderr, "TP DEBUG GET_ROWS tok_embd batch=1: device=%d, tok_id=%d, src0_d=%p, src1_i32=%p, dst_d=%p\n",
                device, tok_id, src0_d, (void*)src1_i32, (void*)dst_d);

        // Check if embedding table has per-device pointers (extra->data_device)
        ggml_tensor * emb_tensor = dst->src[0];
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) emb_tensor->extra;
        if (extra) {
            fprintf(stderr, "TP DEBUG GET_ROWS: emb extra=%p, data_device[0]=%p, data_device[1]=%p\n",
                    extra, extra->data_device[0], extra->data_device[1]);
        } else {
            fprintf(stderr, "TP DEBUG GET_ROWS: emb extra=NULL (no per-device pointers)\n");
        }

        // Check embedding table values at tok_id position
        // Embedding table has shape [vocab_size, n_embd] = [32000, 4096]
        // For F32: row_offset = tok_id * n_embd * sizeof(float)
        // For F16: row_offset = tok_id * n_embd * sizeof(sycl::half)
        int64_t n_embd = dst->src[0]->ne[0];
        size_t elem_size = ggml_type_size(dst->src[0]->type);
        size_t row_offset = tok_id * n_embd * elem_size;

        fprintf(stderr, "TP DEBUG GET_ROWS embd table: ne[0]=%lld (n_embd), ne[1]=%lld (vocab), type=%s, elem_size=%zu\n",
                (long long)dst->src[0]->ne[0], (long long)dst->src[0]->ne[1],
                ggml_type_name(dst->src[0]->type), elem_size);

        // Read Q4_0 block at tok_id row (first block)
        // Q4_0 block: 2 bytes scale (fp16) + 16 bytes quants (32 4-bit values)
        // Block offset = tok_id * blocks_per_row * block_size = tok_id * (4096/32) * 18
        struct block_q4_0_t {
            sycl::half d;
            uint8_t qs[16];
        };
        int64_t blocks_per_row = n_embd / 32;  // 4096/32 = 128 blocks per row
        size_t q4_row_offset = tok_id * blocks_per_row * sizeof(block_q4_0_t);

        block_q4_0_t blk;
        const char * q4_ptr = (const char*)src0_d + q4_row_offset;
        ctx.stream()->memcpy(&blk, q4_ptr, sizeof(blk)).wait();

        float d_val = (float)blk.d;
        // Dequantize first 4 values
        int v0 = (blk.qs[0] & 0xF) - 8;
        int v1 = (blk.qs[0] >> 4) - 8;
        int v2 = (blk.qs[1] & 0xF) - 8;
        int v3 = (blk.qs[1] >> 4) - 8;
        fprintf(stderr, "TP DEBUG GET_ROWS Q4_0[tok=%d]: ptr=%p, d=%.6f, qs[0-1]=0x%02x%02x, deq=[%.6f,%.6f,%.6f,%.6f]\n",
                tok_id, q4_ptr, d_val, blk.qs[0], blk.qs[1], v0*d_val, v1*d_val, v2*d_val, v3*d_val);

        // Also check token 0 and token 1 to verify embedding table has data
        block_q4_0_t blk0, blk1;
        ctx.stream()->memcpy(&blk0, src0_d, sizeof(blk0)).wait();  // Token 0, block 0
        ctx.stream()->memcpy(&blk1, (const char*)src0_d + blocks_per_row * sizeof(block_q4_0_t), sizeof(blk1)).wait();  // Token 1, block 0
        fprintf(stderr, "TP DEBUG GET_ROWS tok0: d=%.6f, qs[0]=0x%02x | tok1: d=%.6f, qs[0]=0x%02x\n",
                (float)blk0.d, blk0.qs[0], (float)blk1.d, blk1.qs[0]);

        // Check which device the queue is actually on
        sycl::device queue_dev = ctx.stream()->get_device();
        fprintf(stderr, "TP DEBUG GET_ROWS: ctx.device=%d, queue_device='%s'\n",
                device, queue_dev.get_info<sycl::info::device::name>().c_str());

        // Also check tensor->data vs resolved pointer
        fprintf(stderr, "TP DEBUG GET_ROWS: tensor->data=%p, resolved src0_d=%p (match=%d)\n",
                dst->src[0]->data, src0_d, (dst->src[0]->data == src0_d));
    }

    // DEBUG: Check F32 get_rows for inp_out_ids reduction (attention output to FFN)
    // This is triggered when src0 is F32 and batch reduces from >1 to 1
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int f32_getrows_dbg = 0;
    bool is_f32 = dst->src[0]->type == GGML_TYPE_F32;
    bool is_reduction = dst->src[0]->ne[1] > 1 && n_rows == 1;  // Batch >1 reduced to 1
    if (g_ggml_sycl_tp_debug && is_f32 && is_reduction && f32_getrows_dbg++ < 10) {
        const char * name = dst->src[0]->name ? dst->src[0]->name : "?";
        int32_t row_idx;
        ctx.stream()->memcpy(&row_idx, src1_i32, sizeof(int32_t)).wait();

        // Read values at the row being extracted
        int64_t ne0 = dst->src[0]->ne[0];  // Row width
        size_t row_offset = row_idx * ne0 * sizeof(float);
        float src_vals[4];
        const float * row_ptr = (const float*)((const char*)src0_d + row_offset);
        ctx.stream()->memcpy(src_vals, row_ptr, 4*sizeof(float)).wait();

        fprintf(stderr, "TP DEBUG GET_ROWS F32 reduction: src0=%s ne=[%lldx%lld], extracting row %d, values=[%f,%f,%f,%f]\n",
                name, (long long)ne0, (long long)dst->src[0]->ne[1], row_idx,
                src_vals[0], src_vals[1], src_vals[2], src_vals[3]);

        // Check if any values are NaN
        bool has_nan = false;
        for (int i = 0; i < 4; i++) if (std::isnan(src_vals[i])) has_nan = true;
        if (has_nan) {
            fprintf(stderr, "TP DEBUG GET_ROWS F32: WARNING - NaN found in source row %d!\n", row_idx);
        }
    }

    /* TODO: Refactor and remove duplicates */
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::half *)src0_d,
                                src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        default:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }

    // DEBUG: Check output after kernel for token embedding batch=1
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int getrows_out_dbg = 0;
    bool is_tok_embd_out = dst->src[0]->name && strstr(dst->src[0]->name, "token_embd");
    int64_t n_rows_out = dst->src[1]->ne[0];
    if (g_ggml_sycl_tp_debug && is_tok_embd_out && n_rows_out == 1 && getrows_out_dbg++ < 5) {
        ctx.stream()->wait();
        float out_vals[8];
        ctx.stream()->memcpy(out_vals, dst_d, std::min((size_t)8*sizeof(float), dst->ne[0]*sizeof(float))).wait();

        // Check for zeros
        int zero_count = 0;
        for (int i = 0; i < 8; i++) {
            if (out_vals[i] == 0.0f) zero_count++;
        }

        fprintf(stderr, "TP DEBUG GET_ROWS tok_embd OUTPUT batch=1: device=%d, dst[0..7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f zeros=%d/8\n",
                device, out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                out_vals[4], out_vals[5], out_vals[6], out_vals[7], zero_count);
    }
}
