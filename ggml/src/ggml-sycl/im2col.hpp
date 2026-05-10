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

#ifndef GGML_SYCL_IM2COL_HPP
#define GGML_SYCL_IM2COL_HPP

#include "common.hpp"

#define SYCL_IM2COL_BLOCK_SIZE 256

void ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_IM2COL_HPP
