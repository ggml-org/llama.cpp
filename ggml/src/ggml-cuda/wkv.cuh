#include "common.cuh"

#define CUDA_WKV_BLOCK_SIZE 64

void ggml_cuda_op_rwkv_wkv6(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rwkv_wkv7(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rwkv7_output_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * norm,
        const ggml_tensor * norm_weight,
        const ggml_tensor * norm_bias,
        const ggml_tensor * k,
        const ggml_tensor * r,
        const ggml_tensor * r_k,
        const ggml_tensor * v,
        const ggml_tensor * gate,
        ggml_tensor * dst);
