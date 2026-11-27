#pragma once

#include "common.hpp"

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst);

#define GGML_SYCL_CONV_WINDOW_MAX_SLOTS 8

// Match the recurrent window, rollback cache copies, convolution and SiLU.
struct ggml_sycl_conv_window_match {
    const ggml_tensor * concat   = nullptr;
    ggml_tensor *       conv     = nullptr;
    ggml_tensor *       add      = nullptr;
    ggml_tensor *       silu     = nullptr;
    int                 conv_idx = -1;
    int                 last_idx = -1;  // the silu node: the last node the kernel writes
    int                 n_slots  = 0;
    const ggml_tensor * cont[GGML_SYCL_CONV_WINDOW_MAX_SLOTS] = {};
    const ggml_tensor * cpy[GGML_SYCL_CONV_WINDOW_MAX_SLOTS]  = {};
};

// Purely structural, so graph_optimize and the compute loop see the same answer.
bool ggml_sycl_match_conv_window(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_conv_window_match & m);
void ggml_sycl_ssm_conv_window(ggml_backend_sycl_context & ctx, const ggml_sycl_conv_window_match & m);
