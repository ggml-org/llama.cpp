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

#ifndef GGML_SYCL_FATTN_ONEDNN_HPP
#define GGML_SYCL_FATTN_ONEDNN_HPP

#include "common.hpp"

// Static-only check: oneDNN Graph SDPA path for native f16 KV FA.
// (f16 KV, mask required, no softcap/ALiBi, single stream, prefill-sized q.)
bool ggml_sycl_flash_attn_ext_onednn_supported(const ggml_tensor * dst);

// Run flash attention through oneDNN's fused XMX SDPA kernel.
// Falls back to the TILE kernel on any failure.
void ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#if GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

// Shared SDPA partition type and builder — used by both the native-F16
// onednn path and the hybrid dequant-then-SDPA path.
struct sdpa_partition {
    dnnl::graph::compiled_partition          cp;
    std::vector<dnnl::graph::logical_tensor> ins;
    dnnl::graph::logical_tensor              out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    bool   ok = false;
};

sdpa_partition build_sdpa(const dnnl::engine & eng, int H, int Hkv, int q, int seq, int d);

// Permute SDPA output (f16 [mb,H,q,d]) → ggml dst (f32 [d,H,q,mb]).
// Uses actual dst strides for interleaved head layout support.
void permute_sdpa_out_sycl(const sycl::half * out, float * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d,
        size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream);
#endif

#endif // GGML_SYCL_FATTN_ONEDNN_HPP
