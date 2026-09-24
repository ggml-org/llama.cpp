#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_should_fuse_set_rows_pair(const ggml_tensor * a, const ggml_tensor * b);

void ggml_cuda_op_set_rows_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst0, ggml_tensor * dst1);
