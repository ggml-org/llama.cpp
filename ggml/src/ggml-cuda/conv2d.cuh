#pragma once
#include "common.cuh"

#define BS_OC     32
#define BS_ICKHKW 16
#define BS_NOHOW  32

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CUDA_CONV2D_BLOCK_SIZE 128

void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
