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

#ifndef GGML_SYCL_ROPE_HPP
#define GGML_SYCL_ROPE_HPP

#include "common.hpp"

#define SYCL_ROPE_BLOCK_SIZE 256

void ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

void ggml_sycl_rope_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_rope_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

#endif // GGML_SYCL_ROPE_HPP
