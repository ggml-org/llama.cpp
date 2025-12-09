/**
 * CUDA Concat Kernel for MLA Tensor Shapes
 * 
 * This file provides optimized CUDA kernels for the concat operation
 * specifically designed for MLA (Multi-head Latent Attention) tensor patterns.
 * 
 * MLA uses concat to combine k_nope and k_pe_repeated:
 * - k_nope: [qk_nope_head_dim, n_head, n_tokens] = [128, 64, n_tokens]
 * - k_pe_repeated: [qk_rope_head_dim, n_head, n_tokens] = [64, 64, n_tokens]
 * - K (output): [n_embd_head_k_mla, n_head, n_tokens] = [192, 64, n_tokens]
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 3.1, 3.2**
 */

#pragma once

#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

/**
 * Check if the concat operation is supported for MLA tensor shapes on CUDA.
 * 
 * MLA-specific concat patterns:
 * - Concatenating k_nope [128, n_head, n_tokens] with k_pe_repeated [64, n_head, n_tokens]
 * - Result: K [192, n_head, n_tokens]
 * - Concat along dimension 0
 * 
 * Supported data types:
 * - GGML_TYPE_F32 (float32)
 * - GGML_TYPE_F16 (float16)
 * 
 * @param dst The destination tensor (output of concat)
 * @return true if the configuration is supported on CUDA
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 3.1, 3.4**
 */
bool ggml_cuda_concat_mla_supported(const struct ggml_tensor * dst);

/**
 * Execute the standard CUDA concat operation.
 * 
 * @param ctx The CUDA backend context
 * @param dst The destination tensor
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 3.1, 3.2, 3.3**
 */
void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
