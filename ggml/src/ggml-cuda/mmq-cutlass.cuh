#pragma once

#include "repack-cutlass-w4a4.cuh"

bool ggml_cuda_cutlass_compiled();
bool ggml_cuda_cutlass_enabled();

bool ggml_cuda_cutlass_mul_mat_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

bool ggml_cuda_cutlass_mul_mat_id_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

struct ggml_cuda_cutlass_ffn_args {
    const ggml_tensor * gate;
    const ggml_tensor * up;
    const ggml_tensor * down;
    const ggml_tensor * input;
    const ggml_tensor * gate_scale;
    const ggml_tensor * up_scale;
    const ggml_tensor * down_scale;
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

bool ggml_cuda_cutlass_mul_mat(ggml_backend_cuda_context & ctx,
                               const ggml_tensor *         src0,
                               const ggml_tensor *         src1,
                               ggml_tensor *               dst);

bool ggml_cuda_cutlass_mul_mat_id(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor *         src0,
                                  const ggml_tensor *         src1,
                                  const ggml_tensor *         ids,
                                  ggml_tensor *               dst);

bool ggml_cuda_cutlass_ffn(ggml_backend_cuda_context & ctx, const ggml_cuda_cutlass_ffn_args & args);

bool ggml_cuda_moe_cutlass_nvfp4(ggml_backend_cuda_context &              ctx,
                                 const ggml_cuda_moe_cutlass_nvfp4_args & args);
