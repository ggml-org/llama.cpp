//******************************************************************************
// ET Quantization Library
// Provides quantization block structures, constants, and dequantization
// functions for bare metal kernels
//******************************************************************************

#ifndef QUANTS_H
#define QUANTS_H

#include "math_fp.h"

#include <stdint.h>

//******************************************************************************
// Quantization Block Size Constants
//******************************************************************************

// Q4_0 quantization: 32 4-bit values per block + 1 fp16 scale
#define QK4_0 32

// Q8_0 quantization: 32 int8 values per block + 1 fp16 scale
#define QK8_0 32

// F16 block size: 32 f16 values per block (64 bytes = 1 cache line)
#define QK_F16 32

// F32 block size: 16 f32 values per block (64 bytes = 1 cache line)
#define QK_F32 16

// Q4_K super-block size: 256 values per super-block (8 groups of 32)
#define QK_K         256
#define K_SCALE_SIZE 12

//******************************************************************************
// Q4_0 Quantization Block Structure
//******************************************************************************

// Q4_0 quantization block (matches GGML definition)
// Each block contains 32 quantized 4-bit values (packed in 16 bytes) + 1 fp16 scale factor
// Total size: 2 bytes (scale) + 16 bytes (nibbles) = 18 bytes
typedef struct {
    uint16_t d;              // Scale factor (delta) as fp16 - 2 bytes
    uint8_t  qs[QK4_0 / 2];  // Quantized values (32 x 4-bit packed) - 16 bytes
} block_q4_0;

//******************************************************************************
// Q8_0 Quantization Block Structure
//******************************************************************************

// Q8_0 quantization block (matches GGML definition)
// Each block contains 32 quantized int8 values + 1 fp16 scale factor
// Total size: 2 bytes (scale) + 32 bytes (values) = 34 bytes
typedef struct {
    uint16_t d;          // Scale factor (delta) as fp16 - 2 bytes
    int8_t   qs[QK8_0];  // Quantized values (32 x int8) - 32 bytes
} block_q8_0;

//******************************************************************************
// Q4_K Quantization Block Structure
//******************************************************************************

// Q4_K quantization super-block (matches GGML definition)
// 8 groups of 32 elements each, weight = a * q + b
// Total size: 2 bytes (d) + 2 bytes (dmin) + 12 bytes (scales) + 128 bytes (qs) = 144 bytes
typedef struct {
    uint16_t d;                     // Super-block scale for quantized scales (fp16)
    uint16_t dmin;                  // Super-block scale for quantized mins (fp16)
    uint8_t  scales[K_SCALE_SIZE];  // Scales and mins, quantized with 6 bits
    uint8_t  qs[QK_K / 2];          // 4-bit quants (256 values packed in 128 bytes)
} block_q4_K;

//******************************************************************************
// Dequantization Functions
//******************************************************************************

// Dequantize a Q8_0 block to F32 values
// Converts 32 quantized int8 values to float using the block's scale factor
static inline void dequantize_q8_0_block(const block_q8_0 * block, float * dst) {
    // Convert fp16 scale to fp32
    const float scale = fp16_to_fp32(block->d);

    // Convert each quantized int8 value to float using scale
    for (int i = 0; i < QK8_0; i++) {
        dst[i] = scale * (float) block->qs[i];
    }
}

// Dequantize a Q4_0 block to F32 values
// Low nibbles fill the first half (dst[0..15]), high nibbles fill the second half (dst[16..31])
static inline void dequantize_q4_0_block(const block_q4_0 * block, float * dst) {
    const float scale = fp16_to_fp32(block->d);

    for (int i = 0; i < QK4_0 / 2; i++) {
        const uint8_t byte = block->qs[i];
        dst[i]             = scale * (float) ((int) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (float) ((int) (byte >> 4) - 8);
    }
}

// Extract 6-bit scale and min from the packed scales array of Q4_K
// For groups 0-3: simple 6-bit mask from scales[j] and scales[j+4]
// For groups 4-7: reconstructed from split high bits
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantize a Q4_K super-block (256 elements) to F32 values
static inline void dequantize_q4_K_block(const block_q4_K * block, float * dst) {
    const uint8_t * q   = block->qs;
    const float     d   = fp16_to_fp32(block->d);
    const float     min = fp16_to_fp32(block->dmin);

    int     is = 0;
    uint8_t sc, m;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
