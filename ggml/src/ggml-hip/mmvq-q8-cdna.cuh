#pragma once

#include "ggml-cuda/common.cuh"

// Q8_0 x f32 matrix-vector product for 1 to a few src1 columns on CDNA1.
bool ggml_hip_should_use_mmvq_q8_cdna(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_hip_mul_mat_vec_q8_cdna(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
