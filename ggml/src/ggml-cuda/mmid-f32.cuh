#pragma once
#include "common.cuh"

bool ggml_cuda_mul_mat_id_f32_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_id_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
