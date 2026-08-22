#pragma once

#include "common.cuh"

// Prototype: dequantize-to-float matvec for K-quant weights (decode, n=1).
// Skips the q8_1 activation quantization pass that mul_mat_vec_q requires.
// On by default where arch_default is true (RDNA3.5); GGML_CUDA_DQ_MMV overrides
// (unset = arch default, 0 = force off, non-zero = force on).

bool ggml_cuda_dq_mmv_enabled(bool arch_default);

// Q6_K dequant-float matvec (bandwidth-bound, ~neutral vs mmvq). Same override
// semantics via GGML_CUDA_DQ_Q6K (unset = arch default, 0 = off, non-zero = on).
bool ggml_cuda_dq_q6k_enabled(bool arch_default);

void ggml_cuda_mul_mat_vec_dq_q4_K(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_dq_q5_K(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_dq_q6_K(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Fused gate+up SwiGLU dequant matvec (up = src0, gate = separate weight, shared
// activation src1). Writes silu(gate)*up. Handles Q4_K/Q5_K/Q6_K (dispatch on up->type).
void ggml_cuda_mul_mat_vec_dq_glu(
    ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
    const ggml_tensor * src1, ggml_tensor * dst);
