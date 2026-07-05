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

#ifndef GGML_SYCL_FATTN_HYBRID_HPP
#define GGML_SYCL_FATTN_HYBRID_HPP

#include "common.hpp"

// Check whether this FA call qualifies for the hybrid path:
// dequantize K/V to f16, then route through oneDNN's fused SDPA kernel.
bool ggml_sycl_flash_attn_ext_hybrid_supported(const ggml_tensor * dst);

// Run FA: dequant K/V → oneDNN SDPA → permute to ggml dst.
// Falls back to MKL GEMM on failure.
void ggml_sycl_flash_attn_ext_hybrid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HYBRID_HPP
