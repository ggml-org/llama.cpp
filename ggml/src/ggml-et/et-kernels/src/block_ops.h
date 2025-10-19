//******************************************************************************
// ET Vectorized Block Operations Library
// Provides optimized block-level operations using ET hardware vector instructions
//******************************************************************************

#ifndef BLOCK_OPS_H
#define BLOCK_OPS_H

#include <stdint.h>
#include "math_fp.h"
#include "quants.h"

//******************************************************************************
// Block Dot Product Operations
//******************************************************************************

// Compute dot product between dequantized q8_0 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 int8 values (QK8_0)
static inline float compute_block_dot_product_q8_0(const block_q8_0* a_block, const float* b_col_start) {
    const float scale = fp16_to_fp32(a_block->d);

    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator vector

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Process 32 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk * 8;

        // Vectorized int8->float conversion + multiply-accumulate
        // Using gather pattern for byte loading and vector conversion
        static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

        __asm__ volatile(
            "flw.ps f10, %[acc]\n"                   // Load current accumulator (8 floats)
            "flw.ps f31, %[gather]\n"                // Load gather pattern into f31
            "fgb.ps f11, f31(%[a_ptr])\n"            // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"                  // Convert int8 vector to float vector
            "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (floats)
            "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
            "fsw.ps f10, %[result]\n"                // Store back to accumulator

            : [result] "=m"(*(float(*)[8])acc_vec)
            : [acc] "m"(*(const float(*)[8])acc_vec),
              [a_ptr] "r"(&a_block->qs[offset]),
              [b_vec] "m"(*(const float(*)[8])&b_col_start[offset]),
              [gather] "m"(*(const int32_t(*)[8])gather_pattern)
            : "f10", "f11", "f12", "f31"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    return final_sum * scale;
}

// Compute dot product between f16 block and f32 column vector (NAIVE VERSION)
// Scalar implementation for debugging - no vectorization
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16_naive(const uint16_t* a_block, const float* b_col_start) {
    float sum = 0.0f;

    // Simple scalar loop - convert each f16 to f32 and multiply
    for (int i = 0; i < QK_F16; i++) {
        float a_val = fp16_to_fp32(a_block[i]);
        float b_val = b_col_start[i];
        sum += a_val * b_val;
    }

    return sum;
}

// Compute dot product between f16 block and f32 column vector
// SCALAR implementation for partial blocks
// Block size: up to 32 f16 values (can handle partial blocks for misaligned K)
static inline float compute_block_dot_product_f16_partial(const uint16_t* a_block, const float* b_col_start, int elements) {
    // This matches compute_block_dot_product_f16_naive behavior
    float sum = 0.0f;

    for (int i = 0; i < elements; i++) {
        float a_val = fp16_to_fp32(a_block[i]);
        float b_val = b_col_start[i];
        sum += a_val * b_val;
    }

    return sum;
}

// Compute dot product between f16 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16(const uint16_t* a_block, const float* b_col_start) {
    return compute_block_dot_product_f16_partial(a_block, b_col_start, QK_F16);
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: up to 16 f32 values (can handle partial blocks for misaligned K)
static inline float compute_block_dot_product_f32_partial(const float* a_block, const float* b_col_start, int elements) {
    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator vector

    // Calculate how many full 8-element chunks we can process
    int vec_end = (elements / 8) * 8;

    if (vec_end > 0) {
        // Set mask register to enable all 8 vector elements
        unsigned long temp_mask;
        __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
        __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

        // Process full 8-element chunks
        for (int i = 0; i < vec_end; i += 8) {
            // Vectorized f32 multiply-accumulate
            __asm__ volatile(
                "flw.ps f10, %[acc]\n"                   // Load current accumulator (8 floats)
                "flw.ps f11, %[a_vec]\n"                 // Load 8 A values (f32)
                "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (f32)
                "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
                "fsw.ps f10, %[result]\n"                // Store back to accumulator

                : [result] "=m"(*(float(*)[8])acc_vec)
                : [acc] "m"(*(const float(*)[8])acc_vec),
                  [a_vec] "m"(*(const float(*)[8])(a_block + i)),
                  [b_vec] "m"(*(const float(*)[8])(b_col_start + i))
                : "f10", "f11", "f12"
            );
        }

        // Restore original mask
        __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
    }

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    // Handle remaining elements (< 8) with scalar operations
    for (int i = vec_end; i < elements; i++) {
        final_sum += a_block[i] * b_col_start[i];
    }

    return final_sum;
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 16 f32 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f32(const float* a_block, const float* b_col_start) {
    return compute_block_dot_product_f32_partial(a_block, b_col_start, QK_F32);
}

#endif // BLOCK_OPS_H
