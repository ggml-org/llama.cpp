#pragma once

#include "common.cuh"

void ggml_cuda_op_kapsl_kv_write(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kapsl_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
