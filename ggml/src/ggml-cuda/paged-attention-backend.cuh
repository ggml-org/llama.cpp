#include "common.cuh"

void ggml_cuda_op_paged_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_can_paged_attention(const ggml_tensor * dst);
