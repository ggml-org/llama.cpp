#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_can_fuse_top_k_mask(int device, const ggml_tensor * top_k_mask, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_top_k_mask_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * top_k_mask, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
