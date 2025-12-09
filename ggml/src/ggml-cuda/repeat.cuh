#pragma once

#include "common.cuh"

#define CUDA_REPEAT_BLOCK_SIZE 256

/**
 * Check if the repeat operation is supported for MLA tensor shapes on CUDA.
 * 
 * MLA-specific repeat patterns:
 * - k_pe: [qk_rope_head_dim, 1, n_tokens] -> [qk_rope_head_dim, n_head, n_tokens]
 * - This is a 1 -> n_head broadcast along dimension 1
 * 
 * Supported data types:
 * - GGML_TYPE_F32 (float32)
 * - GGML_TYPE_F16 (float16)
 * 
 * @param dst The destination tensor (output of repeat)
 * @return true if the configuration is supported on CUDA
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 2.1, 2.4**
 */
bool ggml_cuda_repeat_mla_supported(const struct ggml_tensor * dst);

/**
 * Execute the optimized CUDA repeat kernel for MLA tensor shapes.
 * 
 * This kernel is optimized for the specific repeat pattern used in MLA:
 * - Broadcasting from [dim0, 1, dim2] to [dim0, n_head, dim2]
 * - Coalesced memory access for optimal performance
 * - Maintains tensor device consistency (output on same device as input)
 * 
 * @param ctx The CUDA backend context
 * @param dst The destination tensor
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 2.1, 2.2, 2.3**
 */
void ggml_cuda_op_repeat_mla(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
