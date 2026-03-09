#pragma once

#include "common.cuh"

static __global__ void ggml_cuda_scale_f32_kernel(float * dst, int64_t n, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] *= scale;
    }
}

static inline void ggml_cuda_scale_f32(float * dst, int64_t n, float scale, cudaStream_t stream) {
    if (!(scale != 1.0f) || n <= 0) {
        return;
    }

    const int block_size = 256;
    const int64_t nblocks = (n + block_size - 1) / block_size;
    ggml_cuda_scale_f32_kernel<<<nblocks, block_size, 0, stream>>>(dst, n, scale);
}
