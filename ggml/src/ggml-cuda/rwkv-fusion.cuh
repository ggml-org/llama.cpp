#include "common.cuh"

void ggml_cuda_op_key_adjust_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * k,
    const ggml_tensor * a,
    const ggml_tensor * k_a,
    ggml_tensor * dst);
