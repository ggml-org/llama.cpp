#ifndef GGML_STFMA_ADAPTER_H
#define GGML_STFMA_ADAPTER_H

#include "ggml-common.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Configuration                                                              */
/* ========================================================================== */

#ifndef GGML_STFMA_THRESHOLD
#define GGML_STFMA_THRESHOLD 1024
#endif

/* ========================================================================== */
/* Encoding Conversion Functions                                             */
/* ========================================================================== */

/**
 * Convert a single byte from TQ2_0 encoding to sparse-ternary-fma encoding.
 * 
 * TQ2_0:  00 (-1), 01 (0), 10 (+1), 11 (invalid)
 * STFMA:  10 (-1), 00 (0), 01 (+1), 11 (invalid)
 */
uint8_t convert_tq2_to_stfma_byte(uint8_t b);

/**
 * Convert an array of TQ2_0 encoded bytes to sparse-ternary-fma encoding.
 */
void convert_tq2_to_stfma_array(
    const uint8_t* tq2_packed,
    uint8_t* stfma_packed,
    size_t num_bytes
);

/* ========================================================================== */
/* Type Conversion Functions                                                  */
/* ========================================================================== */

/**
 * Convert Q8_K int8 values to int32 for sparse-ternary-fma.
 */
void convert_q8k_to_int32(
    const int8_t* q8_values,
    int32_t* int32_buffer,
    size_t n
);

/* ========================================================================== */
/* Sparse Ternary FMA Operations (int32 variants)                            */
/* ========================================================================== */

/**
 * Scalar implementation of sparse ternary FMA.
 */
void sparse_ternary_fma_int32_scalar(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
);

#if defined(__AVX2__)
/**
 * AVX2 implementation of sparse ternary FMA.
 */
void sparse_ternary_fma_int32_avx2(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
);
#endif

#if defined(__AVX512F__)
/**
 * AVX-512 implementation of sparse ternary FMA.
 */
void sparse_ternary_fma_int32_avx512(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
);
#endif

/* ========================================================================== */
/* High-Level Integration Function                                           */
/* ========================================================================== */

/**
 * Compute dot product of TQ2_0 weights and Q8_K activations using sparse-ternary-fma.
 * 
 * This function handles:
 * - Encoding conversion (TQ2_0 -> sparse-ternary-fma)
 * - Type conversion (Q8_K int8 -> int32)
 * - Sparse ternary FMA computation
 * - Scaling and accumulation
 */
void ggml_vec_dot_tq2_0_q8_K_stfma(
    int n,
    float* s,
    size_t bs,
    const void* vx,
    size_t bx,
    const void* vy,
    size_t by,
    int nrc
);

#ifdef __cplusplus
}
#endif

#endif // GGML_STFMA_ADAPTER_H
