#pragma once

#include "common.cuh"

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_fp8_fallback(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
