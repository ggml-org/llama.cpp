#include "common.cuh"

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

void ggml_cuda_op_add_mul_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * scale,
    ggml_tensor * dst);

void ggml_cuda_op_rwkv_rk_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * cur,
    const ggml_tensor * k,
    const ggml_tensor * r,
    const ggml_tensor * v,
    const ggml_tensor * r_k,
    ggml_tensor * dst);
