#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define GGML_COMMON_DECL_CUDA
#include "../ggml-common.h"
#undef GGML_COMMON_DECL_CUDA

#ifndef QK_TQ3_0
#define QK_TQ3_0 32
#endif

static __device__ __forceinline__ float ggml_cuda_tq3_sign(const int i) {
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

static __device__ __forceinline__ float ggml_cuda_tq3_centroid(const uint8_t idx) {
    switch (idx & 7) {
        case 0: return -2.1519f;
        case 1: return -1.3439f;
        case 2: return -0.7560f;
        case 3: return -0.2451f;
        case 4: return  0.2451f;
        case 5: return  0.7560f;
        case 6: return  1.3439f;
        default: return 2.1519f;
    }
}

static __device__ __forceinline__ uint8_t ggml_cuda_tq3_unpack_idx(const uint32_t packed, const int r) {
    return (packed >> (3 * r)) & 7;
}

// Native TQ3_0 x q8_0 block dot: one warp handles one full QK_TQ3_0 block
// without materializing a temporary dequant buffer.
static __device__ __forceinline__ float vec_dot_tq3_0_q8_0_native_block(
    const block_tq3_0 * __restrict__ bq,
    const block_q8_0  * __restrict__ bq8_0) {

    const int lane = threadIdx.x;
    const int g = lane / 8;
    const int r = lane % 8;
    const int leader = g * 8;

    uint32_t packed = 0;
    float rms = 0.0f;
    if (r == 0) {
        const uint8_t * qp = bq->qs + g * 3;
        packed = (uint32_t) qp[0] | ((uint32_t) qp[1] << 8) | ((uint32_t) qp[2] << 16);
        rms = __half2float(bq->d);
    }

    packed = __shfl_sync(0xFFFFFFFF, packed, leader);
    rms    = __shfl_sync(0xFFFFFFFF, rms, leader);

    float val = ggml_cuda_tq3_centroid(ggml_cuda_tq3_unpack_idx(packed, r));

#pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }

    float contrib = val * ggml_cuda_tq3_sign(lane) * (float) bq8_0->qs[lane];
    contrib *= (rms * __half2float(bq8_0->d)) / sqrtf((float) QK_TQ3_0);

#pragma unroll
    for (int step = 16; step > 0; step >>= 1) {
        contrib += __shfl_xor_sync(0xFFFFFFFF, contrib, step);
    }

    return contrib;
}

__global__ void ggml_cuda_native_tq3_dot_kernel(
        const block_tq3_0 * __restrict__ in,
        const block_q8_0  * __restrict__ act,
        float * __restrict__ out,
        int nblocks);

// Rotate activations in-place (declaration — implementation in tq3-native.cu)
void ggml_cuda_tq3_rotate_act(float * x, int64_t n, cudaStream_t stream);
