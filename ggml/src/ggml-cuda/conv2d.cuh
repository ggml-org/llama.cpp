#pragma once
#include "common.cuh"

constexpr int BS_OC     = 128;
constexpr int BS_ICKHKW = 16;
constexpr int BS_NOHOW  = 128;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
