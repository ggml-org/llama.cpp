#include "common.cuh"

void ggml_cuda_op_lerp_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * x_prev,
    const ggml_tensor * cur,
    const ggml_tensor * weight,
    ggml_tensor * dst);
