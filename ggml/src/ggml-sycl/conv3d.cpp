//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "conv3d.hpp"
#include "im2col.hpp"
#include "quantize.hpp"
#include "convert.hpp"
#include <cstring>
#include "gemm.hpp"
extern int g_ggml_sycl_disable_dnn;

static inline int64_t ggml_sycl_conv3d_calc_patch_total(const ggml_tensor * dst, int32_t n) {
    return (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

static inline int64_t ggml_sycl_conv3d_calc_knl_n_total(const ggml_tensor * src0, int32_t c) {
    return (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

static inline void ggml_sycl_conv3d_write_output(
        const ggml_tensor * dst,
        const float * src, float * dst_data,
        int64_t patch_total, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    const int64_t dst_nb0 = dst->nb[0];
    const int64_t dst_nb1 = dst->nb[1];
    const int64_t dst_nb2 = dst->nb[2];
    const int64_t dst_nb3 = dst->nb[3];
    const int64_t total = patch_total * oc;
    const int64_t block_size = 256;
    const int64_t num_work_items = ((total + block_size - 1) / block_size) * block_size;

    stream->parallel_for(sycl::range<1>(num_work_items), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        if (i >= total) {
            return;
        }

        const int64_t patch_idx = i / oc;
        const int64_t out_ch = i % oc;
        const int64_t p_in_batch = patch_idx % (dst_w * dst_h * dst_d);
        const int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        const int64_t dst_z = p_in_batch / (dst_w * dst_h);
        const int64_t dst_y = (p_in_batch % (dst_w * dst_h)) / dst_w;
        const int64_t dst_x = p_in_batch % dst_w;
        const int64_t ocn_idx = batch_idx * oc + out_ch;

        const int64_t dst_offset = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        const int64_t src_index = patch_idx + out_ch * patch_total;
        const float value = src[src_index];
        *(float *)((char *)dst_data + dst_offset) = value;
    });
}

void ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int32_t s0 = opts[0];
    const int32_t s1 = opts[1];
    const int32_t s2 = opts[2];
    const int32_t p0 = opts[3];
    const int32_t p1 = opts[4];
    const int32_t p2 = opts[5];
    const int32_t d0 = opts[6];
    const int32_t d1 = opts[7];
    const int32_t d2 = opts[8];
    const int32_t c  = opts[9];
    const int32_t n  = opts[10];
    const int32_t oc = opts[11];

    const int64_t knl_w = src0->ne[0];
    const int64_t knl_h = src0->ne[1];
    const int64_t knl_d = src0->ne[2];

    const int64_t patch_total = ggml_sycl_conv3d_calc_patch_total(dst, n);
    const int64_t knl_n_total = ggml_sycl_conv3d_calc_knl_n_total(src0, c);

    const size_t kernel_type_size = ggml_element_size(src0);
    const int64_t tmp_elements = knl_n_total * patch_total;

    ggml_sycl_pool_alloc<char> tmp_alloc(ctx.pool());
    tmp_alloc.alloc((size_t) tmp_elements * kernel_type_size);

    ggml_tensor tmp = {};
    tmp.type = src0->type;
    tmp.ne[0] = knl_w;
    tmp.ne[1] = knl_h;
    tmp.ne[2] = knl_d;
    tmp.ne[3] = patch_total * c;
    tmp.nb[0] = kernel_type_size;
    tmp.nb[1] = tmp.nb[0] * tmp.ne[0];
    tmp.nb[2] = tmp.nb[1] * tmp.ne[1];
    tmp.nb[3] = tmp.nb[2] * tmp.ne[2];
    tmp.data = tmp_alloc.get();
    tmp.buffer = dst->buffer;
    tmp.extra = dst->extra;
    tmp.src[0] = const_cast<ggml_tensor *>(src0);
    tmp.src[1] = const_cast<ggml_tensor *>(src1);
    int32_t tmp_op_params[10] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, c };
    memcpy(tmp.op_params, tmp_op_params, sizeof(tmp_op_params));

    // Inline im2col_3d implementation (avoid calling ggml_sycl_op_im2col_3d)
    {
        dpct::queue_ptr stream = ctx.stream();
        const int64_t total_elems = tmp_elements; // knl_n_total * patch_total
        const int64_t block_size = 256;
        const int64_t num_work_items = ((total_elems + block_size - 1) / block_size) * block_size;

        const char * src_base = (const char *) src1->data;
        const int64_t src_nb0 = src1->nb[0];
        const int64_t src_nb1 = src1->nb[1];
        const int64_t src_nb2 = src1->nb[2];
        const int64_t src_nb3 = src1->nb[3];

        const int64_t KW = knl_w;
        const int64_t KH = knl_h;
        const int64_t KD = knl_d;
        const int64_t KN = knl_n_total; // == c * KD*KH*KW
        const int64_t PW = dst->ne[0];
        const int64_t PH = dst->ne[1];
        const int64_t PD = dst->ne[2];

        stream->parallel_for(sycl::range<1>(num_work_items), [=](sycl::id<1> id) {
            const int64_t idx = id[0];
            if (idx >= total_elems) return;

            const int64_t k_index = idx % KN;
            const int64_t patch_idx = idx / KN;

            const int64_t ic = k_index / (KD * KH * KW);
            const int64_t rem = k_index - ic * (KD * KH * KW);
            const int64_t kz = rem / (KH * KW);
            const int64_t rem2 = rem - kz * (KH * KW);
            const int64_t ky = rem2 / KW;
            const int64_t kx = rem2 % KW;

            const int64_t p_in_batch = patch_idx % (PW * PH * PD);
            const int64_t batch_idx = patch_idx / (PW * PH * PD);
            const int64_t dst_z = p_in_batch / (PW * PH);
            const int64_t dst_y = (p_in_batch % (PW * PH)) / PW;
            const int64_t dst_x = p_in_batch % PW;

            const int64_t sx = dst_x * s0 + kx * d0 - p0;
            const int64_t sy = dst_y * s1 + ky * d1 - p1;
            const int64_t sz = dst_z * s2 + kz * d2 - p2;

            float val = 0.0f;
            if (sx >= 0 && sx < src1->ne[0] && sy >= 0 && sy < src1->ne[1] && sz >= 0 && sz < src1->ne[2]) {
                const int64_t channel_idx = batch_idx * c + ic;
                const char * ptr = src_base + sx * src_nb0 + sy * src_nb1 + sz * src_nb2 + channel_idx * src_nb3;
                val = *(const float *) ptr;
            }

            if (tmp.type == GGML_TYPE_F32) {
                float * dstf = (float *) tmp.data;
                dstf[idx] = val;
            } else {
                sycl::half * dsth = (sycl::half *) tmp.data;
                dsth[idx] = sycl::half(val);
            }
        });
    }

    ggml_tensor src0_mat = {};
    src0_mat.type = src0->type;
    src0_mat.ne[0] = knl_n_total;
    src0_mat.ne[1] = oc;
    src0_mat.ne[2] = 1;
    src0_mat.ne[3] = 1;
    src0_mat.nb[0] = kernel_type_size;
    src0_mat.nb[1] = src0_mat.nb[0] * src0_mat.ne[0];
    src0_mat.nb[2] = src0_mat.nb[1];
    src0_mat.nb[3] = src0_mat.nb[2];
    src0_mat.data = src0->data;
    src0_mat.buffer = src0->buffer;
    src0_mat.extra = src0->extra;

    ggml_tensor src1_mat = {};
    src1_mat.type = src0->type;
    src1_mat.ne[0] = knl_n_total;
    src1_mat.ne[1] = patch_total;
    src1_mat.ne[2] = 1;
    src1_mat.ne[3] = 1;
    src1_mat.nb[0] = kernel_type_size;
    src1_mat.nb[1] = src1_mat.nb[0] * src1_mat.ne[0];
    src1_mat.nb[2] = src1_mat.nb[1];
    src1_mat.nb[3] = src1_mat.nb[2];
    src1_mat.data = tmp.data;
    src1_mat.buffer = dst->buffer;
    src1_mat.extra = dst->extra;

    ggml_sycl_pool_alloc<float> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_total * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_total;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = sizeof(float);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.data = gemm_output.get();
    dst_mat.buffer = dst->buffer;
    dst_mat.extra = dst->extra;

    // Fallback: pack src1_mat and src0_mat into contiguous column-major float buffers, then call GEMM
    dpct::queue_ptr stream = ctx.stream();
    {

        const int m = (int) patch_total; // rows of C
        const int n = (int) oc; // cols of C
        const int k = (int) knl_n_total;

        // allocate packed arrays: A_packed (k x m), B_packed (k x n)
        ggml_sycl_pool_alloc<float> A_packed_alloc(ctx.pool());
        ggml_sycl_pool_alloc<float> B_packed_alloc(ctx.pool());
        A_packed_alloc.alloc((size_t) k * m * sizeof(float));
        B_packed_alloc.alloc((size_t) k * n * sizeof(float));

        float * A_packed = A_packed_alloc.get();
        float * B_packed = B_packed_alloc.get();

        // pack src1_mat into A_packed (k x m) in column-major: A_packed[row + col*k]
        const int64_t src1_nb0 = src1_mat.nb[0];
        const int64_t src1_nb1 = src1_mat.nb[1];
        const int64_t src0_nb0 = src0_mat.nb[0];
        const int64_t src0_nb1 = src0_mat.nb[1];

        // pack A (src1)
        stream->parallel_for(sycl::range<1>((size_t)k * m), [=](sycl::id<1> id) {
            const int64_t t = id[0];
            const int64_t row = t % k;
            const int64_t col = t / k;
            const char * src_ptr = (const char *) src1_mat.data + row * src1_nb0 + col * src1_nb1;
            float v;
            if (src1_mat.type == GGML_TYPE_F32) {
                v = *(const float *) src_ptr;
            } else {
                v = sycl::vec<sycl::half, 1>(*(const sycl::half *) src_ptr).convert<float, sycl::rounding_mode::automatic>()[0];
            }
            A_packed[row + col * (int64_t)k] = v;
        });

        // pack B (src0)
        stream->parallel_for(sycl::range<1>((size_t)k * n), [=](sycl::id<1> id) {
            const int64_t t = id[0];
            const int64_t row = t % k;
            const int64_t col = t / k;
            const char * src_ptr = (const char *) src0_mat.data + row * src0_nb0 + col * src0_nb1;
            float v;
            if (src0_mat.type == GGML_TYPE_F32) {
                v = *(const float *) src_ptr;
            } else {
                v = sycl::vec<sycl::half, 1>(*(const sycl::half *) src_ptr).convert<float, sycl::rounding_mode::automatic>()[0];
            }
            B_packed[row + col * (int64_t)k] = v;
        });

        // call GEMM: dst = A^T * B  where A is (k x m), so A^T is (m x k); B is (k x n)
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        const int lda = k;
        const int ldb = k;
        const int ldc = m;

        SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
            *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
            m, n, k,
            dpct::get_value(&alpha, *stream),
            (const float *) A_packed, lda,
            (const float *) B_packed, ldb,
            dpct::get_value(&beta, *stream),
            (float *) dst_mat.data, ldc)));
    }

    const float * gemm_data = (const float *) dst_mat.data;
    float * dst_data = (float *) dst->data;

    ggml_sycl_conv3d_write_output(dst, gemm_data, dst_data, patch_total, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
