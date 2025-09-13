#pragma once
#include "common.cuh"

#define BS_OC     16
#define BS_ICKHKW 16
#define BS_NOHOW  128

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
