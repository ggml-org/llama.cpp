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

#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "common.hpp"

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// Scratch that flash attention needs beyond the output tensor
struct ggml_sycl_fattn_extra {
    uintptr_t K_buffer_ptr     = 0;   // F16 copy of the K cache
    uintptr_t V_buffer_ptr     = 0;   // F16 copy of the V cache
    uintptr_t Q_buffer_ptr     = 0;   // dense F16 copy of Q, oneDNN only
    uintptr_t scale_buffer_ptr = 0;   // the softmax scale as an F16 scalar, oneDNN only
    uintptr_t out_buffer_ptr   = 0;   // F16 SDPA output before conversion to F32, oneDNN only
    uintptr_t end              = 0;   // one past the last reserved byte; sizes the allocation
};

// ggml_sycl_fattn_get_extra() is the single source of truth for the layout: it both sizes
// the reservation and hands out the pointers, so the two cannot disagree.
// Each field is the address of one reserved block, or 0 if that block was not reserved,
// in which case the caller allocates from the scratch pool instead.
ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(const ggml_tensor * dst);

size_t ggml_sycl_flash_attn_ext_get_alloc_size(const ggml_tensor * dst);

void ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HPP
