//******************************************************************************
// ET Quantization Library
// Provides quantization block structures, constants, and dequantization
// functions for bare metal kernels
//******************************************************************************

#ifndef QUANTS_H
#define QUANTS_H

#include <stdint.h>
#include "math_fp.h"

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

//******************************************************************************
// Q4_0 Quantization Block Structure
//******************************************************************************

// Q4_0 quantization block (matches GGML definition)
// Each block contains 32 quantized 4-bit values (packed in 16 bytes) + 1 fp16 scale factor
// Total size: 2 bytes (scale) + 16 bytes (nibbles) = 18 bytes
typedef struct {
    uint16_t d;              // Scale factor (delta) as fp16 - 2 bytes
    uint8_t qs[QK4_0 / 2];  // Quantized values (32 x 4-bit packed) - 16 bytes
} block_q4_0;

//******************************************************************************
// Q8_0 Quantization Block Structure
//******************************************************************************

// Q8_0 quantization block (matches GGML definition)
// Each block contains 32 quantized int8 values + 1 fp16 scale factor
// Total size: 2 bytes (scale) + 32 bytes (values) = 34 bytes
typedef struct {
    uint16_t d;           // Scale factor (delta) as fp16 - 2 bytes
    int8_t qs[QK8_0];     // Quantized values (32 x int8) - 32 bytes
} block_q8_0;

//******************************************************************************
// Dequantization Functions
//******************************************************************************

// Dequantize a Q8_0 block to F32 values
// Converts 32 quantized int8 values to float using the block's scale factor
static inline void dequantize_q8_0_block(const block_q8_0* block, float* dst) {
    // Convert fp16 scale to fp32
    const float scale = fp16_to_fp32(block->d);

    // Convert each quantized int8 value to float using scale
    for (int i = 0; i < QK8_0; i++) {
        dst[i] = scale * (float)block->qs[i];
    }
}

// Dequantize a Q4_0 block to F32 values
// Low nibbles fill the first half (dst[0..15]), high nibbles fill the second half (dst[16..31])
static inline void dequantize_q4_0_block(const block_q4_0* block, float* dst) {
    const float scale = fp16_to_fp32(block->d);

    for (int i = 0; i < QK4_0 / 2; i++) {
        const uint8_t byte = block->qs[i];
        dst[i]              = scale * (float)((int)(byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (float)((int)(byte >> 4)  - 8);
    }
}

#endif // QUANTS_H
