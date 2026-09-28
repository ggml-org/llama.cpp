#include "common.cuh"

void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_flash_attn_sparse_supported(int device, const ggml_tensor * dst);
