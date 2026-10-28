#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// FP4 E2M1 block (32 elements)
struct block_fp4_e2m1 {
    uint8_t qs[16];  // 32 elements packed 2 per byte
    float scale;     // FP32 scale per block
};

// E2M1 quantization values
static const int8_t KVALUES_FP4_E2M1[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,   // positive
    0, -1, -2, -3, -4, -6, -8, -12  // negative
};

// Dequantization kernels
__global__ void dequantize_fp4_e2m1_kernel_f16(const block_fp4_e2m1* __restrict__ src,
                                               half* __restrict__ dst,
                                               int64_t num_blocks);

// Fallback: FP4 E2M1 -> FP16 -> F16 MMA
// Uses existing highly optimized F16 MMA kernel
void ggml_cuda_flash_attn_fp4_e2m1_via_mma_f16(const ggml_cuda_context & ctx, ggml_tensor * dst);

// Native FP4 MMA kernel (requires MXFP4 format, not yet implemented)
void ggml_cuda_flash_attn_fp4_e2m1_mma(const ggml_cuda_context & ctx, ggml_tensor * dst);