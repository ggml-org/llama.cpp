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

#ifndef GGML_SYCL_NORM_HPP
#define GGML_SYCL_NORM_HPP

#include "common.hpp"

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst, const float* src0_dd,
    const float* src1_dd, float* dst_dd,
    const queue_ptr& main_stream);

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst,
    const float* src0_dd, const float* src1_dd,
    float* dst_dd,
    const queue_ptr& main_stream);

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst,
    const float* src0_dd, const float* src1_dd,
    float* dst_dd,
    const queue_ptr& main_stream);

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst,
    const float* src0_dd, const float* src1_dd,
    float* dst_dd,
    const queue_ptr& main_stream);

#endif // GGML_SYCL_NORM_HPP
