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

// Pre-allocate V2 partition buffers before SYCL graph recording.
// This ensures V2 dispatch works during graph recording (malloc/free forbidden during recording).
// Should be called before graph recording starts, similar to graph_pre_reorder_all.
void ggml_sycl_v2_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph);

#endif // GGML_SYCL_FATTN_HPP
