#pragma once

#include "common.cuh"

// Fused TQ weight mul_mat: handles ne[1]=1 (decode) and ne[1]>1 (prefill/speculative)
void ggml_cuda_mul_mat_tq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Legacy single-token alias
inline void ggml_cuda_mul_mat_vec_tq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_mul_mat_tq(ctx, src0, src1, dst);
}

// Large prefill: runtime TQ4_1S → q8_0 scratch + cuBLAS
void ggml_cuda_mul_mat_tq4_1s_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Load-time conversion: TQ4_1S → q8_0 in VRAM (dequant + requantize)
void ggml_cuda_convert_tq4_1s_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream);
