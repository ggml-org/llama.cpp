#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// the kernel that would run dst writes a per-row log-sum-exp when asked (op_params GGML_FA_PARAM_LSE)
bool ggml_cuda_flash_attn_ext_writes_lse(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// writes n_rows rows of n_cols columns to out: 0 before col0, then mask's columns (a mask narrower than its K/V widened)
void ggml_cuda_fattn_widen_mask(const ggml_tensor * mask, int64_t n_rows, half * out, int64_t n_cols, int64_t col0,
        cudaStream_t stream);
