#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// the kernel that would run dst writes a per-row log-sum-exp when asked (op_params GGML_FA_PARAM_LSE)
bool ggml_cuda_flash_attn_ext_writes_lse(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
