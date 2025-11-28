//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "common.hpp"

// Check if flash attention is supported for the given tensor configuration
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst);

// Execute flash attention operation
void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HPP
