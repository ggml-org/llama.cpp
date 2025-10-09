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

// Q8_0 quantization: 32 int8 values per block + 1 fp16 scale
#define QK8_0 32

// F16 block size: 32 f16 values per block (64 bytes = 1 cache line)
#define QK_F16 32

// F32 block size: 16 f32 values per block (64 bytes = 1 cache line)
#define QK_F32 16

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

#endif // QUANTS_H
