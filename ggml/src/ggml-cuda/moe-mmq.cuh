#pragma once

#include "common.cuh"

struct ggml_cuda_moe_mmq_args {
    const ggml_tensor * gate_up;
    const ggml_tensor * input;
    const ggml_tensor * ids;
    ggml_tensor *       gate_up_dst;
    const ggml_tensor * gate_up_bias;
    ggml_tensor *       gate_up_biased;
    ggml_tensor *       activation;
    const ggml_tensor * down;
    ggml_tensor *       down_dst;
    const ggml_tensor * down_bias;
    ggml_tensor *       down_biased;
    const ggml_tensor * weights;
    ggml_tensor *       weighted;
    ggml_tensor *       dst;
};

struct ggml_cuda_moe_cutlass_nvfp4_args {
    const ggml_tensor * gate;
    const ggml_tensor * up;
    const ggml_tensor * down;
    const ggml_tensor * input;
    const ggml_tensor * ids;
    const ggml_tensor * gate_scale;
    const ggml_tensor * up_scale;
    const ggml_tensor * down_scale;
    const ggml_tensor * weights;
    ggml_tensor *       dst;
};

bool ggml_cuda_moe_mmq(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_mmq_args & args);

bool ggml_cuda_moe_cutlass_nvfp4(ggml_backend_cuda_context &              ctx,
                                 const ggml_cuda_moe_cutlass_nvfp4_args & args);

bool ggml_cuda_moe_cutlass_prefill_requested();
