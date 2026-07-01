#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

void ggml_cuda_op_lerp_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * x_prev,
    const ggml_tensor * cur,
    const ggml_tensor * weight,
    ggml_tensor * dst);

void ggml_cuda_op_mul_sub_add_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * base,
    const ggml_tensor * scale,
    const ggml_tensor * value,
    ggml_tensor * dst);
