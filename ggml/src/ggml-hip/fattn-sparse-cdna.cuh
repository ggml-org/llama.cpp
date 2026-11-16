#pragma once

#include "ggml-cuda/common.cuh"

// Sparse (QSA) FLASH_ATTN_EXT for 1-8 queries with 12 Q heads per K/V head on CDNA, uses MFMA.
bool ggml_hip_flash_attn_ext_sparse_cdna_supported(int cc, const ggml_tensor * dst);

void ggml_hip_flash_attn_ext_sparse_cdna(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
