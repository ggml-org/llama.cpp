#pragma once
#include "common.cuh"

#define CUDA_CONV3D_BLOCK_SIZE 256
void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
