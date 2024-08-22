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

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>

#include <sycl/sycl.hpp>
#include <sycl/half_type.hpp>

#include "ggml-sycl.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/gemm.hpp"

bool   ggml_sycl_loaded(void);
void   ggml_sycl_free_data(struct ggml_tensor * tensor);
void   ggml_sycl_copy_to_device(struct ggml_tensor * tensor);
void   ggml_sycl_set_main_device(int main_device);
void   ggml_sycl_set_mul_mat_q(bool mul_mat_q);
void   ggml_sycl_get_device_description(int device, char * description, size_t description_size);
bool   ggml_backend_is_sycl(ggml_backend_t backend);
int    ggml_backend_sycl_get_device(ggml_backend_t backend);
static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer);
static inline int get_sycl_env(const char *env_name, int default_val);


void dev2dev_memcpy(sycl::queue &q_dst, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {
    char *host_buf = (char *)malloc(size);
    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
    free(host_buf);
}

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);
typedef void (*ggml_sycl_func_t)(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream);
typedef void (*ggml_sycl_op_flatten_t)(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst, const float *src0_dd,
                                       const float *src1_dd, float *dst_dd,
                                       const queue_ptr &main_stream);

static __dpct_inline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __dpct_inline__ float op_add(const float a, const float b) {
    return a + b;
}

static __dpct_inline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __dpct_inline__ float op_div(const float a, const float b) {
    return a / b;
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s10,*/ int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {
    const int i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    const int i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    const int i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 = i3*s3 + i2*s2 + i1*s1;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  = i_src0;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_unravel(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s10,*/ int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {

    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 = i3*s3 + i2*s2 + i1*s1;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  = i_src0;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
}

static void acc_f32(const float * x, const float * y, float * dst, const int ne,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, int offset, const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= ne) {
        return;
    }
    int src1_idx = i - offset;
    int oz = src1_idx / nb2;
    int oy = (src1_idx - (oz * nb2)) / nb1;
    int ox = src1_idx % nb1;
    if (src1_idx >= 0 && ox < ne10 && oy < ne11 && oz < ne12) {
        dst[i] = x[i] + y[ox + oy * ne10 + oz * ne10 * ne11];
    } else {
        dst[i] = x[i];
    }
}

static void gelu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = 0.5f * xi *
             (1.0f +
              sycl::tanh(SQRT_2_OVER_PI * xi * (1.0f + GELU_COEF_A * xi * xi)));
}

static void silu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + sycl::native::exp(-x[i]));
}

static void gelu_quick_f32(const float *x, float *dst, int k,
                           const sycl::nd_item<3> &item_ct1) {
    const float GELU_QUICK_COEF = -1.702f;
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (1.0f / (1.0f + sycl::native::exp(GELU_QUICK_COEF * x[i])));
}

static void tanh_f32(const float *x, float *dst, int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::tanh((float)(x[i]));
}

static void relu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmax((float)(x[i]), (float)0);
}

static void hardsigmoid_f32(const float * x, float * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmin(1.0f, sycl::fmax(0.0f, (x[i] + 3.0f) / 6.0f));
}

static void hardswish_f32(const float * x, float * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * sycl::fmin(1.0f, sycl::fmax(0.0f, (x[i] + 3.0f) / 6.0f));
}

static void leaky_relu_f32(const float *x, float *dst, const int k, const float negative_slope,
                           const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmax((float)(x[i]), (float)0) +
             sycl::fmin((float)(x[i]), 0.0f) * negative_slope;
}

static void sqr_f32(const float * x, float * dst, const int k,
                    const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

static void upscale_f32(const float  *x, float *dst, const int nb00, const int nb01,
                        const int nb02, const int nb03, const int ne10, const int ne11,
                        const int ne12, const int ne13, const float sf0, const float sf1,
                        const float sf2, const float sf3, const sycl::nd_item<1> &item_ct1) {
    int index = item_ct1.get_local_id(0) +
               item_ct1.get_group(0) * item_ct1.get_local_range(0);
    if (index >= ne10 * ne11 * ne12 * ne13) {
        return;
    }
    // operation
    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    int i00 = i10 / sf0;
    int i01 = i11 / sf1;
    int i02 = i12 / sf2;
    int i03 = i13 / sf3;

    dst[index] = *(float *)((char *)x + i03 * nb03 + i02 * nb02 + i01 * nb01 + i00 * nb00);
}

static void pad_f32(const float  *x, float *dst, const int ne0, const int ne00, const int ne01, const int ne02,
                    const sycl::nd_item<3> &item_ct1) {
    int nidx = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (nidx < ne00 && item_ct1.get_group(1) < ne01 &&
        item_ct1.get_group(0) < ne02) {
        int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                         item_ct1.get_group(0) * ne00 * ne01;
            dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = 0.0f;
    }
}

template<int QUANT_BLOCK_TILE>
static void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded,
                          const sycl::nd_item<3> &item_ct1) {
    const int ix = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2)) * QUANT_BLOCK_TILE;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                   item_ct1.get_local_id(1);

    const int i_padded = iy*kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i_padded / QK8_1; // block index
    const int iqs = i_padded % QK8_1; // quant index
    typedef  sycl::vec<float, QUANT_BLOCK_TILE> TC;
    typedef  sycl::vec<int8_t, QUANT_BLOCK_TILE> TQ;
    TC zeros;
    TQ qzeros;
#pragma unroll
    for (int i = 0; i < QUANT_BLOCK_TILE; i++)
    {
        zeros[i] = 0.f;
        qzeros[i] = 0;
    }
    const TC xi = ix < kx ? *(TC *)&x[iy * kx + ix] : zeros;
    float sum = xi[0];
    float amax = sycl::fabs(xi[0]);
#pragma unroll
    for (int i = 1; i < QUANT_BLOCK_TILE; i++)
    {
        sum += xi[i];
        amax = sycl::fmax(sycl::fabs(xi[i]), amax);
    }
    sum = warp_reduce_sum(sum, item_ct1);
    amax = warp_reduce_max(amax, item_ct1);

    const float d = amax / 127;
    TQ q = qzeros;
    if (amax != 0.0f)
    {
#pragma unroll
        for (int i = 0; i < QUANT_BLOCK_TILE; i++) {
            q[i] = sycl::round(xi[i] / d);
        }
    }

    *(TQ *)&y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<sycl::half &>(y[ib].ds.x()) = d;
    reinterpret_cast<sycl::half &>(y[ib].ds.y()) = sum;
}

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

static void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel*nrows_y + row_y;

        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    sycl::half *dsti = (sycl::half *)cdsti;

    *dsti = sycl::vec<float, 1>(*xi)
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const sycl::half *xi = (const sycl::half *)cxi;
    sycl::half *dsti = (sycl::half *)cdsti;

    *dsti = *xi;
}

static void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const sycl::half *xi = (const sycl::half *)cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i16_i16(const char * cxi, char * cdsti) {
    const int16_t *xi = (const int16_t *)cxi;
    int16_t *dsti = (int16_t *)cdsti;

    *dsti = *xi;
}

static void cpy_1_i32_i32(const char * cxi, char * cdsti) {
    const int32_t *xi = (const int32_t *)cxi;
    int32_t *dsti = (int32_t *)cdsti;

    *dsti = *xi;
}

template <cpy_kernel_t cpy_1>
static void cpy_f32_f16(const char * cx, char * cdst, const int ne,
                        const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                        const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                        const int nb12, const int nb13, const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= ne) {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i03 = i/(ne00 * ne01 * ne02);
    const int i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int i13 = i/(ne10 * ne11 * ne12);
    const int i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

static void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q8_0 * dsti = (block_q8_0 *) cdsti;

    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = xi[j];
        amax = sycl::fmax(amax, sycl::fabs((float)v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = xi[j]*id;

        dsti->qs[j] = sycl::round((float)x0);
    }
}

static void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_0 * dsti = (block_q4_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        if (amax < sycl::fabs((float)v)) {
            amax = sycl::fabs((float)v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK4_0/2 + j]*id;

        const uint8_t xi0 = dpct::min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t)(x1 + 8.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_1 * dsti = (block_q4_1 *) cdsti;

    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = xi[j];

        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (xi[0       + j] - vmin)*id;
        const float x1 = (xi[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = dpct::min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t)(x1 + 0.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_f32_q(const char * cx, char * cdst, const int ne,
                      const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                      const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                      const int nb12, const int nb13, const sycl::nd_item<3> &item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                   item_ct1.get_local_id(2)) *
                  qk;

    if (i >= ne) {
        return;
    }

    const int i03 = i/(ne00 * ne01 * ne02);
    const int i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int i13 = i/(ne10 * ne11 * ne12);
    const int i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols,
                           const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void
k_argsort_f32_i32(const float *x, int *dst, const int ncols, int ncols_pad,
                  const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
    // bitonic sort
    int col = item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    auto dst_row = (int *)dpct_local;

    // initialize indices
    dst_row[col] = col;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            /*
            DPCT1118:1: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}


static void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past,
                              const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void scale_f32(const float * x, float * dst, const float scale, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i];
}

static void clamp_f32(const float * x, float * dst, const float min, const float max, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
}

template <typename Ti, typename To>
static  void pool2d_nchw_kernel(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if (idx >= parallel_elements) {
            return;
        }

        const int I_HW = ih * iw;
        const int O_HW = oh * ow;
        const int nc = idx / O_HW;
        const int cur_oh = idx % O_HW / ow;
        const int cur_ow = idx % O_HW % ow;
        const Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        const int start_h = cur_oh * sh - ph;
        const int bh = sycl::max(0, start_h);
        const int eh = sycl::min(ih, start_h + kh);
        const int start_w = cur_ow * sw - pw;
        const int bw = sycl::max(0, start_w);
        const int ew = sycl::min(iw, start_w + kw);

        To res = 0;

        switch (op) {
            case GGML_OP_POOL_AVG: res = 0; break;
            case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        }

        for (int i = bh; i < eh; i += 1) {
            for (int j = bw; j < ew; j += 1) {
#if DPCT_COMPATIBILITY_TEMP >= 350
                /*
                DPCT1098:106: The '*' expression is used instead of the __ldg
                call. These two expressions do not provide the exact same
                functionality. Check the generated code for potential precision
                and/or performance issues.
                */
                Ti cur = *(i_ptr + i * iw + j);
#else
                Ti cur = i_ptr[i * iw + j];
#endif
                switch (op) {
                    case GGML_OP_POOL_AVG: res += (cur / (kh * kw)); break;
                    case GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); break;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
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

    (void) dst;
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

    (void) dst;
}

template<float (*bin_op)(const float, const float)>
struct bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    void operator()(ggml_backend_sycl_context & ctx,
                    const struct ggml_tensor *src0,
                    const struct ggml_tensor *src1, struct ggml_tensor *dst,
                    const src0_t *src0_dd, const src1_t *src1_dd, dst_t *dst_dd,
                    queue_ptr stream) {

        GGML_TENSOR_BINARY_OP_LOCALS

        int nr0 = ne10/ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne0[] = {ne0, ne1, ne2, ne3};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb0[] = {nb0, nb1, nb2, nb3};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        for (int i = 0; i < 4; i++) {
            if (nr[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse(cne0);
                collapse(cne1);
            }
        }
        {
            int64_t ne0 = cne0[0];
            int64_t ne1 = cne0[1];
            int64_t ne2 = cne0[2];
            int64_t ne3 = cne0[3];

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb0[0];
            size_t nb1 = cnb0[1];
            size_t nb2 = cnb0[2];
            size_t nb3 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned int>(hne0, block_size);
            block_dims[1] = std::min<unsigned int>(
                ne1, block_size / (unsigned int)block_dims[2]);
            block_dims[0] = std::min(
                std::min<unsigned int>(
                    ne2 * ne3, block_size / (unsigned int)block_dims[2] /
                                   (unsigned int)block_dims[1]),
                64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            if (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                int block_num = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_num) *
                                              sycl::range<3>(1, 1, block_size),
                                          sycl::range<3>(1, 1, block_size)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s11, s12,
                                s13, item_ct1);
                        });
                }
            } else {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1,
                                            ne2, ne3, ne10, ne11, ne12, ne13,
                                            s1, s2, s3, s11, s12, s13,
                                            item_ct1);
                    });
            }
        }
    }
};

static void acc_f32_sycl(const float *x, const float *y, float *dst,
                         const int n_elements, const int ne10, const int ne11,
                         const int ne12, const int nb1, const int nb2,
                         const int offset, queue_ptr stream) {
    int num_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            acc_f32(x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset,
                    item_ct1);
        });
}

static void gelu_f32_sycl(const float *x, float *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_GELU_BLOCK_SIZE - 1) / SYCL_GELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_f32(x, dst, k, item_ct1);
        });
}

static void silu_f32_sycl(const float *x, float *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_SILU_BLOCK_SIZE - 1) / SYCL_SILU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SILU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SILU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            silu_f32(x, dst, k, item_ct1);
        });
}

static void gelu_quick_f32_sycl(const float *x, float *dst, const int k,
                                queue_ptr stream) {
    const int num_blocks = (k + SYCL_GELU_BLOCK_SIZE - 1) / SYCL_GELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_quick_f32(x, dst, k, item_ct1);
        });
}

static void tanh_f32_sycl(const float *x, float *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_TANH_BLOCK_SIZE - 1) / SYCL_TANH_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_TANH_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_TANH_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            tanh_f32(x, dst, k, item_ct1);
        });
}

static void relu_f32_sycl(const float *x, float *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_RELU_BLOCK_SIZE - 1) / SYCL_RELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            relu_f32(x, dst, k, item_ct1);
        });
}

static void hardsigmoid_f32_sycl(const float *x, float *dst, const int k,
                                 queue_ptr stream) {
    const int num_blocks = (k + SYCL_HARDSIGMOID_BLOCK_SIZE - 1) / SYCL_HARDSIGMOID_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_HARDSIGMOID_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_HARDSIGMOID_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            hardsigmoid_f32(x, dst, k, item_ct1);
        });
}

static void hardswish_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_HARDSWISH_BLOCK_SIZE - 1) / SYCL_HARDSWISH_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_HARDSWISH_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_HARDSWISH_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            hardswish_f32(x, dst, k, item_ct1);
        });
}

static void leaky_relu_f32_sycl(const float *x, float *dst, const int k,
                                const float negative_slope,
                                queue_ptr stream) {
    const int num_blocks = (k + SYCL_RELU_BLOCK_SIZE - 1) / SYCL_RELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            leaky_relu_f32(x, dst, k, negative_slope, item_ct1);
        });
}

static void sqr_f32_sycl(const float *x, float *dst, const int k,
                         queue_ptr stream) {
    const int num_blocks = (k + SYCL_SQR_BLOCK_SIZE - 1) / SYCL_SQR_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SQR_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SQR_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sqr_f32(x, dst, k, item_ct1);
        });
}

static void upscale_f32_sycl(const float *x, float *dst, const int nb00, const int nb01,
                             const int nb02, const int nb03, const int ne10, const int ne11,
                             const int ne12, const int ne13, const float sf0, const float sf1,
                             const float sf2, const float sf3, queue_ptr stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;
    sycl::range<1> gridDim(num_blocks * SYCL_UPSCALE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(gridDim, sycl::range<1>(SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3, item_ct1);
        });
}

static void pad_f32_sycl(const float *x, float *dst, const int ne00,
                         const int ne01, const int ne02, const int ne0,
                         const int ne1, const int ne2, queue_ptr stream) {
    int num_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(x, dst, ne0, ne00, ne01, ne02, item_ct1);
        });
}

static void quantize_row_q8_1_sycl(const float *x, void *vy, const int kx,
                                   const int ky, const int kx_padded,
                                   queue_ptr stream) {
    const int block_num_x = (kx_padded + SYCL_QUANTIZE_BLOCK_SIZE - 1) / SYCL_QUANTIZE_BLOCK_SIZE;
    const sycl::range<3> num_blocks(1, ky, block_num_x);
    int constexpr QUANT_BLOCK_TILE = QK8_1 / WARP_SIZE;
    static_assert(QK8_1 % WARP_SIZE == 0);
    const sycl::range<3> block_size(1, 1, SYCL_QUANTIZE_BLOCK_SIZE / QUANT_BLOCK_TILE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(num_blocks * block_size, block_size),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                quantize_q8_1<QUANT_BLOCK_TILE>(x, vy, kx, kx_padded, item_ct1);
            });
    }
}

static void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y,
                                           float *dst, const int ncols_x,
                                           const int nrows_x,
                                           const int nchannels_x,
                                           const int nchannels_y,
                                           queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(
    const void *vx, const float *y, float *dst, const int ncols_x,
    const int nrows_x, const int row_stride_x, const int nchannels_x,
    const int nchannels_y, const int channel_stride_x, queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}

static void
ggml_cpy_f16_f32_sycl(const char *cx, char *cdst, const int ne, const int ne00,
                      const int ne01, const int ne02, const int nb00,
                      const int nb01, const int nb02, const int nb03,
                      const int ne10, const int ne11, const int ne12,
                      const int nb10, const int nb11, const int nb12,
                      const int nb13, queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                           nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_f32_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int ne02, const int nb00,
                                  const int nb01, const int nb02,
                                  const int nb03, const int ne10,
                                  const int ne11, const int ne12,
                                  const int nb10, const int nb11,
                                  const int nb12, const int nb13,
                                  queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                           nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                           item_ct1);
            });
    }
}

static void ggml_cpy_f32_f16_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int ne02, const int nb00,
                                  const int nb01, const int nb02,
                                  const int nb03, const int ne10,
                                  const int ne11, const int ne12,
                                  const int nb10, const int nb11,
                                  const int nb12, const int nb13,
                                  queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                           nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                           item_ct1);
            });
    }
}

static void ggml_cpy_f32_q8_0_sycl(const char *cx, char *cdst, const int ne,
                                   const int ne00, const int ne01,
                                   const int ne02, const int nb00,
                                   const int nb01, const int nb02,
                                   const int nb03, const int ne10,
                                   const int ne11, const int ne12,
                                   const int nb10, const int nb11,
                                   const int nb12, const int nb13,
                                   queue_ptr stream) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks),
                                           sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(
                                 cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                 nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                 item_ct1);
                         });
}

static void ggml_cpy_f32_q4_0_sycl(const char *cx, char *cdst, const int ne,
                                   const int ne00, const int ne01,
                                   const int ne02, const int nb00,
                                   const int nb01, const int nb02,
                                   const int nb03, const int ne10,
                                   const int ne11, const int ne12,
                                   const int nb10, const int nb11,
                                   const int nb12, const int nb13,
                                   queue_ptr stream) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks),
                                           sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(
                                 cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                 nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                 item_ct1);
                         });
}

static void ggml_cpy_f32_q4_1_sycl(const char *cx, char *cdst, const int ne,
                                   const int ne00, const int ne01,
                                   const int ne02, const int nb00,
                                   const int nb01, const int nb02,
                                   const int nb03, const int ne10,
                                   const int ne11, const int ne12,
                                   const int nb10, const int nb11,
                                   const int nb12, const int nb13,
                                   queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks),
                                           sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(
                                 cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                 nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                 item_ct1);
                         });
}

static void ggml_cpy_f16_f16_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int ne02, const int nb00,
                                  const int nb01, const int nb02,
                                  const int nb03, const int ne10,
                                  const int ne11, const int ne12,
                                  const int nb10, const int nb11,
                                  const int nb12, const int nb13,
                                  queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                           nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                           item_ct1);
            });
    }
}

static void ggml_cpy_i16_i16_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int ne02, const int nb00,
                                  const int nb01, const int nb02,
                                  const int nb03, const int ne10,
                                  const int ne11, const int ne12,
                                  const int nb10, const int nb11,
                                  const int nb12, const int nb13,
                                  queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                           nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                           item_ct1);
            });
    }
}

static void ggml_cpy_i32_i32_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int ne02, const int nb00,
                                  const int nb01, const int nb02,
                                  const int nb03, const int ne10,
                                  const int ne11, const int ne12,
                                  const int nb10, const int nb11,
                                  const int nb12, const int nb13,
                                  queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                           nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                           item_ct1);
            });
    }
}

static void scale_f32_sycl(const float *x, float *dst, const float scale,
                           const int k, queue_ptr stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, k, item_ct1);
        });
}

static void clamp_f32_sycl(const float *x, float *dst, const float min,
                           const float max, const int k,
                           queue_ptr stream) {
    const int num_blocks = (k + SYCL_CLAMP_BLOCK_SIZE - 1) / SYCL_CLAMP_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_CLAMP_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CLAMP_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            clamp_f32(x, dst, min, max, k, item_ct1);
        });
}

static void sum_rows_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const sycl::range<3> block_dims(1, 1, ncols_pad);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                        x, dst, ncols, ncols_pad, item_ct1,
                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                        x, dst, ncols, ncols_pad, item_ct1,
                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

static void diag_mask_inf_f32_sycl(const float *x, float *dst,
                                   const int ncols_x, const int nrows_x,
                                   const int rows_per_channel, const int n_past,
                                   queue_ptr stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

static bool g_sycl_loaded = false;

bool ggml_sycl_loaded(void) {
    return g_sycl_loaded;
}

void print_device_detail(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;

    fprintf(stderr, "|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    fprintf(stderr, "found %d SYCL devices:\n", device_count);
    fprintf(stderr, "|  |                   |                                       |       |Max    |        |Max  |Global |                     |\n");
    fprintf(stderr, "|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |\n");
    fprintf(stderr, "|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|\n");
    fprintf(stderr, "|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|\n");
    for (int id = 0; id < device_count; ++id) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        sycl::backend backend = device.get_backend();
        std::string backend_type = get_device_backend_and_type(device);
        int type_id=DeviceNums[backend_type]++;
        std::stringstream device_type;
        device_type << "[" <<  backend_type << ":" << std::to_string(type_id) << "]";
        print_device_detail(id, device, device_type.str());
    }
}

static inline int get_sycl_env(const char *env_name, int default_val) {
    char *user_device_string = getenv(env_name);
    int user_number = default_val;

    unsigned n;
    if (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (int)n;
    } else {
        user_number = default_val;
    }
    return user_number;
}

static void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        fprintf(stderr, "[SYCL] call ggml_check_sycl\n");
        g_ggml_sycl_debug = get_sycl_env("GGML_SYCL_DEBUG", 0);

        fprintf(stderr, "%s: GGML_SYCL_DEBUG: %d\n", __func__, g_ggml_sycl_debug);

#if defined(GGML_SYCL_F16)
        fprintf(stderr, "%s: GGML_SYCL_F16: yes\n", __func__);
#else
        fprintf(stderr, "%s: GGML_SYCL_F16: no\n", __func__);
#endif

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/

        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                            dpct::dev_mgr::instance().device_count()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);
        ggml_backend_sycl_print_sycl_devices();
        initialized = true;
        g_sycl_loaded = true;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static ggml_sycl_device_info ggml_sycl_init() {
    ggml_sycl_device_info info = {};

    info.device_count = dpct::dev_mgr::instance().device_count();
    if (info.device_count == 0) {
        fprintf(stderr, "%s: failed to initialize " GGML_SYCL_NAME ": %s\n", __func__);
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
#if defined(GGML_SYCL_FORCE_MMQ)
    fprintf(stderr, "%s: GGML_SYCL_FORCE_MMQ:   yes\n", __func__);
#else
    fprintf(stderr, "%s: GGML_SYCL_FORCE_MMQ:   no\n", __func__);
#endif
#if defined(SYCL_USE_XMX)
    fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
    fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
    fprintf(stderr, "%s: found %d " GGML_SYCL_NAME " devices:\n", __func__, info.device_count);

    for (int i = 0; i < info.device_count; ++i) {
        info.devices[i].vmm = 0;
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(i))));

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }
    return info;
}

const ggml_sycl_device_info & ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
inline void check_allow_gpu_index(const int device_index) {
  if (device_index >= ggml_sycl_info().device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_count - 1);
    fprintf(stderr, "%s\n", error_buf);
    assert(false);
  }
}

// buffer pool for sycl (legacy)
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;

    int device;
    queue_ptr qptr;
    struct ggml_sycl_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, int device_) :
        qptr(qptr_),
        device(device_) {
    }

    ~ggml_sycl_pool_leg() {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(
            CHECK_TRY_ERROR(ptr = (void *)sycl::malloc_device(
                                look_ahead_size, *qptr)));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

    #ifdef DEBUG_SYCL_MALLOC
        fprintf(stderr, "%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
    #endif
        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%lu, return %p\n", look_ahead_size, ptr);
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        fprintf(stderr, "WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, *qptr)));
        pool_size -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, int device) {
    // TBD: NO VMM support
    // if (ggml_sycl_info().devices[device].vmm) {
    //     return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(device));
    // }
   return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}

// TBD pool with virtual memory management
// struct ggml_sycl_pool_vmm : public ggml_sycl_pool

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst,
                                          const struct ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          queue_ptr stream) try {

    dpct::memcpy_direction kind;
    char * src_ptr;
    if (src->backend == GGML_BACKEND_TYPE_CPU) {
        kind = dpct::host_to_device;
        src_ptr = (char *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } else if (src->backend == GGML_BACKEND_TYPE_GPU || src->backend == GGML_BACKEND_TYPE_GPU_SPLIT) {
        GGML_ASSERT(src->backend != GGML_BACKEND_TYPE_GPU_SPLIT || (i1_low == 0 && i1_high == src->ne[1]));
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    char * dst_ptr = (char *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) return r;
        }
        return 0;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                  const ggml_tensor *src1, ggml_tensor *dst,
                                  const float *src0_d, const float *src1_d,
                                  float *dst_d, const queue_ptr &stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) src1_d;

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, src0, src1, dst, (const sycl::half *)src0_d,
                                src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        default:
            // TODO: k-quants
            fprintf(stderr, "%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
            GGML_ABORT("fatal error");
            break;
    }
}

template <class op>
inline void ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd,
                                   const queue_ptr &main_stream) {

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()(ctx, src0, src1, dst, (const sycl::half *)src0_dd, src1_dd,
             (sycl::half *)dst_dd, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        op()(ctx, src0, src1, dst, (const sycl::half *)src0_dd, src1_dd, dst_dd,
             main_stream);
    } else if (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()(ctx, src0, src1, dst, (const int32_t *)src0_dd, (const int32_t *)src1_dd, (int32_t *)dst_dd,
             main_stream);
    } else if (src0->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()(ctx, src0, src1, dst, (const int16_t *)src0_dd, (const int16_t *)src1_dd, (int16_t *)dst_dd,
             main_stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

static void ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const float *src0_d, const float *src1_d,
                                float *dst_d,
                                const queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, src0, dst, nullptr, src0_d, dst_d, main_stream);

    (void) src1;
    (void) src1_d;
}

inline void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[3] == 1); // just 3D tensors supported

    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    acc_f32_sycl(src0_dd, src1_dd, dst_dd, ggml_nelements(dst), src1->ne[0], src1->ne[1], src1->ne[2], nb1, nb2, offset, main_stream);

    (void) dst;
}

inline void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_quick_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    tanh_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    relu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

static void ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1, ggml_tensor *dst,
                                     const float *src0_dd, const float *src1_dd,
                                     float *dst_dd,
                                     const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardsigmoid_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

static void ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardswish_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    leaky_relu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), negative_slope, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqr_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const float *src0_dd, const float *src1_dd,
                                 float *dst_dd,
                                 const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float sf0 = (float)dst->ne[0]/src0->ne[0];
    const float sf1 = (float)dst->ne[1]/src0->ne[1];
    const float sf2 = (float)dst->ne[2]/src0->ne[2];
    const float sf3 = (float)dst->ne[3]/src0->ne[3];

    upscale_f32_sycl(src0_dd, dst_dd, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                     dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3,
                     main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    pad_f32_sycl(src0_dd, dst_dd,
        src0->ne[0], src0->ne[1], src0->ne[2],
        dst->ne[0], dst->ne[1], dst->ne[2], main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (tensor_split[i] < (i + 1 < ggml_sycl_info().device_count ? tensor_split[i + 1] : 1.0f)) {
            if (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            if (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ABORT("fatal error");
    }

}

inline void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = id == ctx.device ? ne0 : row_diff;

#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = false;
#endif
    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        use_fp16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        dst->op_params[0] == GGML_PREC_DEFAULT) {

        // GGML_SYCL_DEBUG("ggml_sycl_op_mul_mat_sycl - fp16 path\n");
        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1->data + src1_padded_row_size
                                         : src1_as_f16.get();
        ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool(), row_diff * src1_ncols);

        const sycl::half alpha_f16 = 1.0f;
        const sycl::half beta_f16 = 0.0f;
#if !GGML_SYCL_DNNL
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
            *stream, oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
            &alpha_f16, src0_ptr, dpct::library_data_t::real_half, ne00,
            src1_ptr, dpct::library_data_t::real_half, ne10, &beta_f16,
            dst_f16.get(), dpct::library_data_t::real_half, ldc,
            dpct::library_data_t::real_half)));
        const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16);
        to_fp32_sycl(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
#else
        auto dnnl_stream = ctx.stream_dnnl(stream);
        DnnlGemmWrapper::row_gemm(dnnl_stream, false, true, src1_ncols, row_diff, ne10, src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
            src0_ptr, DnnlGemmWrapper::to_dt<sycl::half>(), dst_f16.get(), DnnlGemmWrapper::to_dt<sycl::half>());
        const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16);
        to_fp32_sycl(dst_f16.get(), dst_dd_i, row_diff* src1_ncols, stream);
#endif
    }
    else {
        // GGML_SYCL_DEBUG("ggml_sycl_op_mul_mat_sycl - fp32 path\n");
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;
#if !GGML_SYCL_DNNL
        SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
            *stream, oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
            dpct::get_value(&alpha, *stream), src0_ddf_i, ne00,
            src1_ddf1_i, ne10, dpct::get_value(&beta, *stream),
            dst_dd_i, ldc)));
#else
        auto dnnl_stream = ctx.stream_dnnl(stream);
         DnnlGemmWrapper::row_gemm(dnnl_stream, false, true, src1_ncols, row_diff, ne10, src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
            src0_ddf_i, DnnlGemmWrapper::to_dt<float>(), dst_dd_i, DnnlGemmWrapper::to_dt<float>());
#endif
    }
    (void) dst;
    (void) src1_ddq_i;
    (void) src1_padded_row_size;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const float *src0_dd, const float *src1_dd,
                                float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    const int64_t IH = src0->ne[1];
    const int64_t IW = src0->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = N * OC * OH * OW;
    const int num_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, num_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               item_ct1);
        });

    (void) src1;
    (void) src1_dd;
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                  const ggml_tensor *src1, ggml_tensor *dst,
                                  const float *src0_dd, const float *src1_dd,
                                  float *dst_dd,
                                  const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const float *src0_dd, const float *src1_dd,
                                 float *dst_dd,
                                 const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst, const float *src0_dd,
                                       const float *src1_dd, float *dst_dd,
                                       const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int nrows0 = ggml_nrows(src0);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                               ggml_tensor *dst, const float *src0_dd,
                               const float *src1_dd, float *dst_dd,
                               const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    scale_f32_sycl(src0_dd, dst_dd, scale, ggml_nelements(src0), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                               ggml_tensor *dst, const float *src0_dd,
                               const float *src1_dd, float *dst_dd,
                               const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    clamp_f32_sycl(src0_dd, dst_dd, min, max, ggml_nelements(src0), main_stream);
    /*
    DPCT1010:88: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

static void ggml_sycl_op_flatten(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const ggml_sycl_op_flatten_t op) try {
    const int64_t nrows0 = ggml_nrows(src0);

    const bool use_src1 = src1 != nullptr;
    const int64_t nrows1 = use_src1 ? ggml_nrows(src1) : 1;

    GGML_ASSERT(!use_src1 || src1->backend != GGML_BACKEND_TYPE_GPU_SPLIT);
    GGML_ASSERT(              dst->backend != GGML_BACKEND_TYPE_GPU_SPLIT);

    ggml_tensor_extra_gpu * src0_extra =            (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = use_src1 ? (ggml_tensor_extra_gpu *) src1->extra : nullptr;
    ggml_tensor_extra_gpu * dst_extra  =            (ggml_tensor_extra_gpu *)  dst->extra;

    // dd = data device
    float * src0_ddf = (float *) src0->data;
    float * src1_ddf = use_src1 ? (float *) src1->data : nullptr;
    float *  dst_ddf = (float *) dst->data;

    ggml_sycl_pool_alloc<float> src0_f(ctx.pool());
    ggml_sycl_pool_alloc<float> src1_f(ctx.pool());
    ggml_sycl_pool_alloc<float>  dst_f(ctx.pool());

    ggml_sycl_set_device(ctx.device);
    queue_ptr main_stream = ctx.stream();
    // GGML_SYCL_DEBUG("ctx.device=%d, main_stream=%p src0_on_device=%d, src1_on_device=%d, dst_on_device=%d\n",
        // ctx.device, main_stream, src0_on_device, src1_on_device, dst_on_device);

    // do the computation
    op(ctx, src0, src1, dst, src0_ddf, src1_ddf, dst_ddf, main_stream);
    // print_ggml_tensor("tensor", dst);
}
catch (sycl::exception const &exc) {

  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        for (int id_other = 0; id_other < ggml_sycl_info().device_count; ++id_other) {
            if (i == id_other) {
                continue;
            }
            if (i != main_device && id_other != main_device) {
                continue;
            }

            // int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op,
                                 const bool convert_src1_to_q8_1) try {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(dst->backend != GGML_BACKEND_TYPE_GPU_SPLIT);
    GGML_ASSERT(src1->backend != GGML_BACKEND_TYPE_GPU_SPLIT);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    ggml_tensor_extra_gpu *  dst_extra = (ggml_tensor_extra_gpu *)  dst->extra;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = src0->backend == GGML_BACKEND_TYPE_GPU_SPLIT;
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char> src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char> src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *src0_dd = nullptr;
        float *src1_ddf = nullptr; // float
        char *src1_ddq = nullptr;  // q8_1
        float *dst_dd = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low  = ne01*tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high  = ne01*tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            dev[i].src0_dd = (char *) src0->data;
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (float *) src1->data;
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if (convert_src1_to_q8_1) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                quantize_row_q8_1_sycl(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                /*
                DPCT1010:90: SYCL uses exceptions to report errors and does not
                use the error codes. The call was replaced with 0. You need to
                rewrite this code.
                */
                SYCL_CHECK(0);
            }
        }

        if (dst_on_device) {
            dev[i].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        /*
        DPCT1024:91: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool src1_on_device = i == ctx.device;
            const bool  dst_on_device = i == ctx.device;
            const int64_t row_diff = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                /*
                DPCT1009:163: SYCL uses exceptions to report errors and does not
                use the error codes. The original code was commented out and a
                warning string was inserted. You need to rewrite this code.
                */
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = dev[i].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if (convert_src1_to_q8_1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                          SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
                                src1_ddq_i, src1_ddq_i_source,
                                src1_ncols * src1_padded_col_size * q8_1_ts /
                                    q8_1_bs).wait()));
                        } else {

                            float * src1_ddf_i_source = (float *) src1_extra->data_device[ctx.device];
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;

                            SYCL_CHECK(CHECK_TRY_ERROR(dev2dev_memcpy(*stream, *main_stream,
                                src1_ddf_i, src1_ddf_i_source,
                                src1_ncols * ne10 * sizeof(float))));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(
                                   src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ABORT("fatal error");
                }

                if (convert_src1_to_q8_1 && !src1_is_contiguous) {
                    quantize_row_q8_1_sycl(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, src1_padded_col_size, stream);
                    /*
                    DPCT1010:92: SYCL uses exceptions to report errors and does
                    not use the error codes. The call was replaced with 0. You
                    need to rewrite this code.
                    */
                    SYCL_CHECK(0);
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[i].row_low, dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)));
                /*
                DPCT1010:93: SYCL uses exceptions to report errors and does not
                use the error codes. The call was replaced with 0. You need to
                rewrite this code.
                */
                SYCL_CHECK(0);

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i,
                            row_diff * sizeof(float), row_diff * sizeof(float),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    /*
                    DPCT1024:94: The original code returned the error code that
                    was further consumed by the program logic. This original
                    code was replaced with 0. You may need to rewrite the
                    program logic consuming the error code.
                    */
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_repeat);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_get_rows);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_add(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_add);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_acc(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_acc);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_mul(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_mul);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_div(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_div);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_gelu);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_silu(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_silu);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_gelu_quick);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_tanh);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_relu(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_relu);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_hardsigmoid);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_hardswish);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_leaky_relu);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_sqr);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_norm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_norm);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_group_norm);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_upscale);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_pad(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_pad);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}


static void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_rms_norm);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->backend != GGML_BACKEND_TYPE_GPU_SPLIT);
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1,
                                     ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->backend != GGML_BACKEND_TYPE_GPU_SPLIT);
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half *src0_as_f16,
                                   const sycl::half *src1_as_f16, char *dst,
                                   const void **ptrs_src, void **ptrs_dst,
                                   int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12,
                                   size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3,
                                   const sycl::nd_item<3> &item_ct1) {
    int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                  item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    int64_t i03 = i13 / r3;
    int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx,
                                             const ggml_tensor *src0,
                                             const ggml_tensor *src1,
                                             ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(src0->backend != GGML_BACKEND_TYPE_GPU_SPLIT);
    GGML_ASSERT(src0->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();;

    void * src0_ddq = src0->data;
    sycl::half *src0_as_f16 = (sycl::half *)src0_ddq;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf = (float *) dst->data;

    // convert src1 to fp16
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());
    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type);
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_ddf, src1_f16_alloc.get(), ne_src1, main_stream);
    }
    sycl::half *src1_f16 = src1->type == GGML_TYPE_F16 ? (sycl::half *)src1_ddf
                                                       : src1_f16_alloc.get();

    char * dst_t;

    dpct::library_data_t cu_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t cu_data_type = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    const void * alpha = &alpha_f32;
    const void * beta  = &beta_f32;

    dst_t = (char *) dst_ddf;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    if (r2 == 1 && r3 == 1 && ggml_is_contiguous_2(src0) && ggml_is_contiguous_2(src1)) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
            *main_stream, oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
            (const char *)src0_as_f16, dpct::library_data_t::real_half,
            nb01 / nb00, nb02 / nb00,
            (const char *)src1_f16, dpct::library_data_t::real_half,
            nb11 / nb10, nb12 / nb10, beta,
            (char *)dst_t, cu_data_type, ne01, nb2 / nb0,
            ne12 * ne13, cu_compute_type)));
    } else {
        const int ne23 = ne12*ne13;

        ggml_sycl_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_sycl_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        sycl::range<3> block_dims(1, ne12, ne13);
        /*
        DPCT1049:47: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(main_stream->get_device(),
                                         {sycl::aspect::fp16});

            main_stream->submit([&](sycl::handler &cgh) {
                const void **ptrs_src_get = ptrs_src.get();
                void **ptrs_dst_get = ptrs_dst.get();
                size_t nb12_scaled = src1->type == GGML_TYPE_F16 ? nb12 : nb12 / 2;
                size_t nb13_scaled = src1->type == GGML_TYPE_F16 ? nb13 : nb13 / 2;
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     k_compute_batched_ptrs(
                                         src0_as_f16, src1_f16,
                                         dst_t, ptrs_src_get,
                                         ptrs_dst_get, ne12, ne13, ne23,
                                         nb02, nb03, nb12_scaled, nb13_scaled,
                                         nbd2, nbd3, r2, r3, item_ct1);
                                 });
            });
        }
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
            *main_stream, oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
            (const void **)(ptrs_src.get() + 0 * ne23),
            dpct::library_data_t::real_half, nb01 / nb00,
            (const void **)(ptrs_src.get() + 1 * ne23),
            dpct::library_data_t::real_half, nb11 / nb10, beta,
            (void **)(ptrs_dst.get() + 0 * ne23), cu_data_type, ne01, ne23,
            cu_compute_type)));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // TODO: accuracy issues in MMQ
    return false;
}

bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
            return true;
        default:
            return false;
    }
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_sycl_info().device_count; ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_count ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } else {
        min_compute_capability    = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = ggml_sycl_supports_dmmv(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % GGML_SYCL_DMMV_X == 0 && src1->ne[1] == 1;

    bool use_mul_mat_vec_q =  ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE
        && (ctx.stream()->get_backend() == sycl::backend::ext_oneapi_cuda || src1->ne[1] > MMVQ_MIN_BATCH_SIZE);

    bool use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggerganov/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // SYCL_USE_XMX

    // mmvq path is faster in the CUDA backend.
    if (ctx.stream()->get_backend() == sycl::backend::ext_oneapi_cuda)
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;

    if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // KQ single-batch
        ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        ggml_sycl_op_mul_mat(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec, false);
    } else if (use_mul_mat_vec_q) {
        ggml_sycl_op_mul_mat(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q, true);
    } else if (use_mul_mat_q) {
        ggml_sycl_op_mul_mat(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q, true);
    } else {
        ggml_sycl_op_mul_mat(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl, false);
    }
}


struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

__dpct_inline__ static void k_copy_src1_to_contiguous(
    const char *__restrict__ src1_original, char *__restrict__ src1_contiguous,
    int *__restrict__ cur_src1_row, mmid_row_mapping *__restrict__ row_mapping,
    const char *__restrict ids, int64_t i02, size_t ids_nb1, size_t ids_nb0,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    const sycl::nd_item<3> &item_ct1, int &src1_row) {
    int32_t iid1 = item_ct1.get_group(2);
    int32_t id = item_ct1.get_group(1);

    const int32_t row_id_i = *(const int32_t *) (ids + iid1*ids_nb1 + id*ids_nb0);

    if (row_id_i != i02) {
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    if (item_ct1.get_local_id(2) == 0) {
        src1_row =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                cur_src1_row, 1);
        row_mapping[src1_row] = {id, iid1};
    }
    /*
    DPCT1065:194: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(
    char *__restrict__ dst_original, const char *__restrict__ dst_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1,
                                 ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char *src0_original = (char *)src0->data;
    char *src1_original = (char *)src1->data;
    char *dst_original = (char *)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1   + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);

                    GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

                    if (row_id_i != i02) {
                        continue;
                    }

                    num_src1_rows++;
                }
            }

            if (num_src1_rows == 0) {
                continue;
            }


            ggml_sycl_pool_alloc<int> dev_cur_src1_row(ctx.pool(), 1);
            ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), num_src1_rows);
            SYCL_CHECK(CHECK_TRY_ERROR(
                stream->memset(dev_cur_src1_row.get(), 0, sizeof(int))));

            {
                sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne10, 768u));
                sycl::range<3> grid_dims(1, n_ids, ids->ne[1]);
                stream->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<int, 0> src1_row_acc(cgh);

                    char *__restrict src1_contiguous_get =
                        src1_contiguous.get();
                    int *__restrict dev_cur_src1_row_get =
                        dev_cur_src1_row.get();
                    mmid_row_mapping *__restrict dev_row_mapping_get =
                        dev_row_mapping.get();
                    size_t ids_nb_ct6 = ids->nb[1];
                    size_t ids_nb_ct7 = ids->nb[0];

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_copy_src1_to_contiguous(
                                src1_original, src1_contiguous_get,
                                dev_cur_src1_row_get,
                                dev_row_mapping_get, ids_dev, i02,
                                ids_nb_ct6, ids_nb_ct7, ne11, ne10, nb11, nb12,
                                item_ct1, src1_row_acc);
                        });
                });
            }

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);

            {
                sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, 768u));
                sycl::range<3> grid_dims(1, 1, num_src1_rows);
                stream->submit([&](sycl::handler &cgh) {
                    const char *__restrict dst_contiguous_get =
                        dst_contiguous.get();
                    const mmid_row_mapping *__restrict dev_row_mapping_get =
                        dev_row_mapping.get();

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_copy_dst_from_contiguous(dst_original,
                                                       dst_contiguous_get,
                                                       dev_row_mapping_get,
                                                       ne0, nb1, nb2, item_ct1);
                        });
                });
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_scale(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_scale);
}

static void ggml_sycl_clamp(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_clamp);
}

static void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst) try {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
    GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);

    GGML_TENSOR_BINARY_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else {
        fprintf(stderr, "%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }

    (void) dst;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_dup(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // TODO: why do we pass dst as src1 here?
    ggml_sycl_cpy(ctx, src0, dst, nullptr);
    (void) src1;
}

static void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_diag_mask_inf);
}

static void ggml_sycl_soft_max(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_soft_max);
}

static void ggml_sycl_rope(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0)); // TODO: this restriction is temporary until non-cont support is implemented
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_rope);
}

static void ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_pool2d);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_im2col);
}

static void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_sum_rows);
}

static void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    ggml_sycl_op_flatten(ctx, src0, src1, dst, ggml_sycl_op_argsort);
}

static void ggml_sycl_nop(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

void ggml_sycl_set_main_device(const int main_device) try {
    if (dpct::get_current_device_id() == main_device) return;
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        fprintf(stderr, "Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * tensor) {
    if (!g_sycl_loaded) return false;

    ggml_sycl_func_t func;

    switch (tensor->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            func = ggml_sycl_op_conv_transpose_1d;
            break;
        case GGML_OP_REPEAT:
            func = ggml_sycl_repeat;
            break;
        case GGML_OP_GET_ROWS:
            func = ggml_sycl_get_rows;
            break;
        case GGML_OP_DUP:
            func = ggml_sycl_dup;
            break;
        case GGML_OP_ADD:
            func = ggml_sycl_add;
            break;
        case GGML_OP_ACC:
            func = ggml_sycl_acc;
            break;
        case GGML_OP_MUL:
            func = ggml_sycl_mul;
            break;
        case GGML_OP_DIV:
            func = ggml_sycl_div;
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    func = ggml_sycl_gelu;
                    break;
                case GGML_UNARY_OP_SILU:
                    func = ggml_sycl_silu;
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    func = ggml_sycl_gelu_quick;
                    break;
                case GGML_UNARY_OP_TANH:
                    func = ggml_sycl_tanh;
                    break;
                case GGML_UNARY_OP_RELU:
                    func = ggml_sycl_relu;
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    func = ggml_sycl_hardsigmoid;
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    func = ggml_sycl_hardswish;
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            func = ggml_sycl_norm;
            break;
        case GGML_OP_GROUP_NORM:
            func = ggml_sycl_group_norm;
            break;
        case GGML_OP_CONCAT:
            func = ggml_sycl_op_concat;
            break;
        case GGML_OP_UPSCALE:
            func = ggml_sycl_upscale;
            break;
        case GGML_OP_PAD:
            func = ggml_sycl_pad;
            break;
        case GGML_OP_LEAKY_RELU:
            func = ggml_sycl_leaky_relu;
            break;
        case GGML_OP_RMS_NORM:
            func = ggml_sycl_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            if (tensor->src[0]->ne[3] != tensor->src[1]->ne[3]) {
                return false;
            }
            func = ggml_sycl_mul_mat;
            break;
        case GGML_OP_MUL_MAT_ID:
            if (tensor->src[0]->ne[3] != tensor->src[1]->ne[3]) {
                return false;
            }
            func = ggml_sycl_mul_mat_id;
            break;
        case GGML_OP_SCALE:
            func = ggml_sycl_scale;
            break;
        case GGML_OP_SQR:
            func = ggml_sycl_sqr;
            break;
        case GGML_OP_CLAMP:
            func = ggml_sycl_clamp;
            break;
        case GGML_OP_CPY:
            func = ggml_sycl_cpy;
            break;
        case GGML_OP_CONT:
            func = ggml_sycl_dup;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            func = ggml_sycl_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            func = ggml_sycl_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            func = ggml_sycl_soft_max;
            break;
        case GGML_OP_ROPE:
            func = ggml_sycl_rope;
            break;
        case GGML_OP_IM2COL:
            func = ggml_sycl_im2col;
            break;
        case GGML_OP_POOL_2D:
            func = ggml_sycl_pool2d;
            break;
        case GGML_OP_SUM_ROWS:
            func = ggml_sycl_sum_rows;
            break;
        case GGML_OP_ARGSORT:
            func = ggml_sycl_argsort;
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            func = ggml_sycl_op_timestep_embedding;
            break;
        default:
            return false;
    }

    if (tensor->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(tensor->src[0]->buffer)) {
        ggml_sycl_set_peer_access(tensor->src[1]->ne[1], ctx.device);
    }

    func(ctx, tensor->src[0], tensor->src[1], tensor);
    return true;
}

GGML_API GGML_CALL void   ggml_sycl_get_gpu_list(int *id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_get_gpu_list\n");
    for(int i=0;i<max_len;i++) id_list[i] = -1;

    for (int i=0;i< ggml_sycl_info().device_count;i++){
        if (i>=max_len) break;
        id_list[i] = i;
    }
    return;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int ggml_sycl_get_device_count() try {
    int device_count;
    if (CHECK_TRY_ERROR(device_count =
                             dpct::dev_mgr::instance().device_count()) != 0) {
        return 0;
    }
    return device_count;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_API GGML_CALL void ggml_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL void ggml_backend_sycl_get_device_memory(int device, size_t *free,
                                                   size_t *total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ggml_sycl_set_device(device);

    /*
    DPCT1009:218: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1106:217: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::dev_mgr::instance().get_device(device).get_memory_info(*free, *total)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

#define UNUSED GGML_UNUSED

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;

     ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream) :
        device(device), dev_ptr(dev_ptr), stream(stream) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
        }


    ~ggml_backend_sycl_buffer_context() {
        if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(dev_ptr, *stream)));
        }
    }
};

GGML_CALL static const char * ggml_backend_sycl_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_sycl_buffer_get_name;
}

static void
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) try {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        tensor->backend = tensor->view_src->backend;
        tensor->extra = tensor->view_src->extra;
        return;
    }


    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (char *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) try {

    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
    char* host_buf = (char*)malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(
        CHECK_TRY_ERROR((*stream).memcpy((char *)tensor->data + offset, host_buf, size)
                             .wait()));
    free(host_buf);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) try {

    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (const char *)tensor->data + offset, size)
            .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL static bool
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst) try {
    if (ggml_backend_buffer_is_sycl(src->buffer)) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(*stream_dst, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#if 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (char *)dst->data, (const char *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        return true;
    }
    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) try {
     ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    SYCL_CHECK(CHECK_TRY_ERROR((*stream)
                                    .memset(ctx->dev_ptr, value, buffer->size)
                                    .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .get_name        = */ ggml_backend_sycl_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ NULL,
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

GGML_CALL static const char * ggml_backend_sycl_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}
GGML_CALL static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0

    void * dev_ptr;
    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_device(
                                    size, *stream)));
    ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream);
    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return dpct::get_current_device().get_max_mem_alloc_size();

    UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .is_host          = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    if (device>=ggml_sycl_info().device_count or device<0) {
        printf("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count-1);
        GGML_ASSERT(device<ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    int device = ctx->device;
    if (device>=ggml_sycl_info().device_count or device<0) {
        printf("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count-1);
        GGML_ASSERT(device<ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer type
static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;
    if (id == ggml_sycl_info().device_count - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

struct ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() try {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
                for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
                    if (extra->events[i][is] != nullptr) {
                        /*
                        DPCT1009:206: SYCL uses exceptions to report errors and
                        does not use the error codes. The original code was
                        commented out and a warning string was inserted. You
                        need to rewrite this code.
                        */
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            dpct::destroy_event(extra->events[i][is])));
                    }
                }
                if (extra->data_device[i] != nullptr) {
                    /*
                    DPCT1009:207: SYCL uses exceptions to report errors and does
                    not use the error codes. The original code was commented out
                    and a warning string was inserted. You need to rewrite this
                    code.
                    */
                    ggml_sycl_set_device(i);
                    SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(
                        extra->data_device[i], *(streams[i]))));
                }
            }
            delete extra;
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

GGML_CALL static const char * ggml_backend_sycl_split_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_SYCL_NAME "_Split";

    UNUSED(buffer);
}

static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   return buffer->iface.get_name == ggml_backend_sycl_split_buffer_get_name;
}

GGML_CALL static void ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    UNUSED(buffer);
}

GGML_CALL static void
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) try {
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
        ctx->streams.push_back(&(dpct::get_current_device().default_queue()));

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if cudaMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char * buf;
        /*
        DPCT1009:208: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)sycl::malloc_device(
                                        size, *stream)));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_size, 0, size - original_size)
                    .wait()));
        }

        extra->data_device[i] = buf;

        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = new sycl::event()));
        }
    }
    tensor->backend = GGML_BACKEND_TYPE_GPU_SPLIT;
    tensor->extra = extra;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL static void
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, const void *data,
                                          size_t offset, size_t size) try {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL static void
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          const ggml_tensor *tensor, void *data,
                                          size_t offset, size_t size) try {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL static void ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    UNUSED(buffer);
    UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_sycl_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ NULL,
};

GGML_CALL static const char * ggml_backend_sycl_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Split";

    UNUSED(buft);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = new ggml_backend_sycl_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

GGML_CALL static bool ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_sycl_split_buffer_type_is_host,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");
    ggml_check_sycl();
    // FIXME: this is not thread safe
    static std::map<std::array<float, GGML_SYCL_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .context = */ new ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

// host buffer type

GGML_CALL static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_sycl_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_SYCL_NAME "_Host";

    UNUSED(buffer);
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_sycl_host_free(buffer->context);
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_sycl_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_sycl_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type\n");
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_sycl_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // TODO: return device.maxBufferLength
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_sycl_buffer_type_host;
}

// backend

GGML_CALL static const char * ggml_backend_sycl_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    return sycl_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}


GGML_CALL static ggml_backend_buffer_type_t ggml_backend_sycl_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    return ggml_backend_sycl_buffer_type(sycl_ctx->device);
}

GGML_CALL static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        (char *)tensor->data + offset, data, size).wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size).wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                                         const ggml_tensor *src,
                                                         ggml_tensor *dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    if (dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && ggml_backend_buffer_is_sycl(src->buffer)) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst)).wait()));
        return true;
    }

    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

GGML_CALL static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_sycl_set_main_device(sycl_ctx->device);


    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            fprintf(stderr, "%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_sycl_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                ggml_type a_type = a->type;
                if (a_type == GGML_TYPE_IQ4_NL  || a_type == GGML_TYPE_IQ4_XS ||
                    a_type == GGML_TYPE_IQ3_XXS || a_type == GGML_TYPE_IQ3_S  ||
                    a_type == GGML_TYPE_IQ2_XXS || a_type == GGML_TYPE_IQ2_XS || a_type == GGML_TYPE_IQ2_S ||
                    a_type == GGML_TYPE_IQ1_S || a_type == GGML_TYPE_IQ1_M
                    ) {
                    if (b->ne[1] == 1 && ggml_nrows(b) > 1) {
                        return false;
                    }
                }
                ggml_type src0_type = op->src[0]->type;
                if (src0_type == GGML_TYPE_BF16) {
                    return false;
                }
                return true;
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                int dim = op->op_params[0];
                return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16 && dim == 2;
            } break;
        case GGML_OP_DUP:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_REPEAT:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_ROPE:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
            return true;
        default:
            return false;
    }

    UNUSED(backend);
}

GGML_CALL static bool ggml_backend_sycl_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    const int min_batch_size = 32;
    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS && op->op != GGML_OP_MUL_MAT_ID;
    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_sycl_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    return buft_ctx->device == sycl_ctx->device;
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .get_default_buffer_type = */ ggml_backend_sycl_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL, //ggml_backend_sycl_cpy_tensor_async, // TODO: update for the new interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .supports_op             = */ ggml_backend_sycl_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_sycl_init(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to allocate context\n", __func__);
        return nullptr;
    };

    ggml_backend_t sycl_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_sycl_guid(),
        /* .interface = */ ggml_backend_sycl_interface,
        /* .context   = */ ctx
    };

    return sycl_backend;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

GGML_CALL int ggml_backend_sycl_get_device_count() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_count\n");
    return ggml_sycl_info().device_count;
}

GGML_CALL static ggml_backend_t ggml_backend_reg_sycl_init(const char * params, void * user_data) {
    ggml_backend_t sycl_backend = ggml_backend_sycl_init((int) (intptr_t) user_data);
    return sycl_backend;

    UNUSED(params);
}

extern "C" int ggml_backend_sycl_reg_devices();

int ggml_backend_sycl_reg_devices() {
    assert(ggml_sycl_info().device_count>0);
    for (int i = 0; i < ggml_sycl_info().device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_SYCL_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_sycl_init, ggml_backend_sycl_buffer_type(i), (void *) (intptr_t) i);
    }
    return ggml_sycl_info().device_count;
}
