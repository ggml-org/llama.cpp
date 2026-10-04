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

#ifndef GGML_SYCL_WDECOMP_HPP
#define GGML_SYCL_WDECOMP_HPP

#include "common.hpp"

// Integer type of the packed weights handed to oneDNN weight decompression.
enum ggml_sycl_wdecomp_int {
    GGML_SYCL_WDECOMP_S8,
    GGML_SYCL_WDECOMP_S4,
    GGML_SYCL_WDECOMP_U4,
};

// How a quantized type is expressed for oneDNN weight decompression: every weight is
// scale * q (+ bias), with q of type wt and one scale (and bias) per group of `group` weights along K.
struct ggml_sycl_wdecomp_fmt {
    ggml_sycl_wdecomp_int wt;
    int                   group;
    bool                  has_bias;
};

// Fills fmt and returns true if weights of this type, in the reorder layout when reordered is set or the
// standard layout otherwise, can be converted by ggml_sycl_dequantize_to_int.
bool ggml_sycl_wdecomp_format(ggml_type type, bool reordered, ggml_sycl_wdecomp_fmt & fmt);

// True if the weight-decompression path is expected to beat fp16 dequantize + GEMM for an M x K src1
// on Arc Pro B70. Its int GEMM loses ground to the fp16 GEMM as M grows while the dequantize saving is
// fixed per weight, and formats with a bias also pay an M x N bias GEMM. Measured over N, K in {4096,
// 14336} and M in {64 .. 2048}: formats without a bias won 1.5-54% while 2 * M < K and lost up to 8% at
// M = 2048, K = 4096. Formats with a bias won while M * 256 <= K * group and lost up to 25% beyond it.
bool ggml_sycl_wdecomp_pays(const ggml_sycl_wdecomp_fmt & fmt, int64_t M, int64_t K);

// Converts nrows x ncols weights to packed integers w[nrows][ncols], scales[ncols / group][nrows] and, for
// formats with a bias, bias[ncols / group][nrows], following ggml_sycl_wdecomp_format(type, reordered).
void ggml_sycl_dequantize_to_int(ggml_type type, bool reordered, const void * vx, void * w, sycl::half * scales,
                                 sycl::half * bias, int64_t nrows, int64_t ncols, dpct::queue_ptr stream);

// Converts k f32 values to fp16 and writes the sum of every `group` consecutive values (16 or 32) to gsum.
void ggml_sycl_f32_to_f16_gsum(const float * x, sycl::half * y, sycl::half * gsum, int64_t k, int group,
                               dpct::queue_ptr stream);

#endif  // GGML_SYCL_WDECOMP_HPP
