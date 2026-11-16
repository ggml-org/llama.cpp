#pragma once

#include "ggml-cuda/common.cuh"

// Q8_0 x f32 matrix-vector product for 1 to a few src1 columns on CDNA and RDNA3/4.
// fusion: gate + GLU for 1 src1 column, as in ggml_cuda_mul_mat_vec_q.
bool ggml_hip_should_use_mmvq_q8_cdna(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc,
        const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_hip_mul_mat_vec_q8_cdna(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion = nullptr);
