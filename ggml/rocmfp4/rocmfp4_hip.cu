#include "rocmfp4.h"

#include <hip/hip_runtime.h>

#include "rocmfp4_hip_scale.cuh"

// Standalone ROCm/HIP dequant kernel for integration tests and future fused
// paths. One lane owns one packed byte and writes the matching low/high
// half-block values, so each byte is read once.
extern "C" __global__ void rocmfp4_dequantize_q4_0_f32_kernel(
        const block_rocmfp4 * __restrict__ x,
        float               * __restrict__ y,
        int64_t                          k) {
    const int64_t packed_idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t nblocks = (k + QK_ROCMFP4 - 1) / QK_ROCMFP4;
    const int64_t packed_count = nblocks * (QK_ROCMFP4/2);

    if (packed_idx >= packed_count) {
        return;
    }

    const int64_t ib = packed_idx / (QK_ROCMFP4/2);
    const int tid = packed_idx - ib*(QK_ROCMFP4/2);
    const int64_t base = ib*QK_ROCMFP4;
    const uint8_t packed = x[ib].qs[tid];
    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e[0]);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e[1]);

    if (base + tid < k) {
        y[base + tid] = (float) rocmfp4_decode_i8(packed & 0x0f) * d0;
    }
    if (base + tid + QK_ROCMFP4/2 < k) {
        y[base + tid + QK_ROCMFP4/2] = (float) rocmfp4_decode_i8(packed >> 4) * d1;
    }
}

extern "C" __global__ void rocmfp4_dequantize_q4_0_fast_f32_kernel(
        const block_rocmfp4_fast * __restrict__ x,
        float                    * __restrict__ y,
        int64_t                               k) {
    const int64_t packed_idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t nblocks = (k + QK_ROCMFP4 - 1) / QK_ROCMFP4;
    const int64_t packed_count = nblocks * (QK_ROCMFP4/2);

    if (packed_idx >= packed_count) {
        return;
    }

    const int64_t ib = packed_idx / (QK_ROCMFP4/2);
    const int tid = packed_idx - ib*(QK_ROCMFP4/2);
    const int64_t base = ib*QK_ROCMFP4;
    const uint8_t packed = x[ib].qs[tid];
    const float d = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e);

    if (base + tid < k) {
        y[base + tid] = (float) rocmfp4_decode_i8(packed & 0x0f) * d;
    }
    if (base + tid + QK_ROCMFP4/2 < k) {
        y[base + tid + QK_ROCMFP4/2] = (float) rocmfp4_decode_i8(packed >> 4) * d;
    }
}

extern "C" void rocmfp4_hip_dequantize_q4_0_to_f32(
        const void  * src,
        float       * dst,
        int64_t       k,
        hipStream_t   stream) {
    const int64_t nblocks = (k + QK_ROCMFP4 - 1) / QK_ROCMFP4;
    const int64_t packed_count = nblocks * (QK_ROCMFP4/2);
    const dim3 block(256);
    const dim3 grid((unsigned int) ((packed_count + block.x - 1) / block.x));
    rocmfp4_dequantize_q4_0_f32_kernel<<<grid, block, 0, stream>>>((const block_rocmfp4 *) src, dst, k);
}

extern "C" void rocmfp4_hip_dequantize_q4_0_fast_to_f32(
        const void  * src,
        float       * dst,
        int64_t       k,
        hipStream_t   stream) {
    const int64_t nblocks = (k + QK_ROCMFP4 - 1) / QK_ROCMFP4;
    const int64_t packed_count = nblocks * (QK_ROCMFP4/2);
    const dim3 block(256);
    const dim3 grid((unsigned int) ((packed_count + block.x - 1) / block.x));
    rocmfp4_dequantize_q4_0_fast_f32_kernel<<<grid, block, 0, stream>>>((const block_rocmfp4_fast *) src, dst, k);
}
