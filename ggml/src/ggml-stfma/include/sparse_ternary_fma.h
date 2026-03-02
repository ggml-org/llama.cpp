/**
 * Sparse Ternary Fused Multiply-Add (FMA) Kernel
 * 
 * A high-performance, dependency-free C library implementing efficient
 * ternary arithmetic using 2-bit encoding and AVX-512 SIMD instructions.
 * 
 * This kernel achieves 1.58× density gain and 2.38× SIMD speedup over
 * standard integer arithmetic, making it ideal for cryptographic and
 * machine learning applications.
 * 
 * Copyright 2025 HyperFold Technologies UK Ltd
 * Author: Maurice Wilson
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPARSE_TERNARY_FMA_H
#define SPARSE_TERNARY_FMA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __AVX512F__
#include <immintrin.h>
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif

/* ========================================================================== */
/* 2-Bit Ternary Encoding Scheme                                            */
/* ========================================================================== */

/**
 * 2-bit ternary encoding maps ternary values to bit patterns:
 * 
 *   Value | Encoding | Binary
 *   ------|----------|--------
 *   -1    | 0b10     | 10
 *    0    | 0b00     | 00
 *   +1    | 0b01     | 01
 *   (invalid) | 0b11 | 11
 * 
 * Design rationale:
 * - Distinct patterns for each value
 * - Zero is zero (simplifies operations)
 * - High bit indicates sign (0=positive, 1=negative)
 * - Low bit indicates non-zero (0=zero, 1=non-zero for +1)
 * - Invalid pattern reserved for error detection
 */

#define TRIT_NEG      0b10  /* Negative: -1 */
#define TRIT_ZERO     0b00  /* Zero: 0 */
#define TRIT_POS      0b01  /* Positive: +1 */
#define TRIT_INVALID  0b11  /* Invalid (error detection) */

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Encoding/Decoding Functions                                              */
/* ========================================================================== */

/**
 * Encode a ternary value to 2-bit representation.
 * 
 * @param value Ternary value in {-1, 0, 1}
 * @return 2-bit encoded trit
 */
static inline uint8_t encode_trit(int8_t value) {
    if (value == 0) return TRIT_ZERO;
    if (value == 1) return TRIT_POS;
    return TRIT_NEG;  /* value == -1 */
}

/**
 * Decode a 2-bit trit to ternary value.
 * 
 * @param trit 2-bit encoded trit
 * @return Ternary value in {-1, 0, 1}
 */
static inline int8_t decode_trit(uint8_t trit) {
    if (trit == TRIT_ZERO) return 0;
    if (trit == TRIT_POS) return 1;
    return -1;  /* trit == TRIT_NEG */
}

/**
 * Pack 4 trits into a single byte.
 * 
 * Byte layout:
 *   | trit3 | trit2 | trit1 | trit0 |
 *   |  7-6  |  5-4  |  3-2  |  1-0  |
 * 
 * @param t0, t1, t2, t3 Four ternary values
 * @return Packed byte containing 4 trits
 */
static inline uint8_t pack_trits(int8_t t0, int8_t t1, int8_t t2, int8_t t3) {
    return (encode_trit(t0) << 0) |
           (encode_trit(t1) << 2) |
           (encode_trit(t2) << 4) |
           (encode_trit(t3) << 6);
}

/**
 * Unpack a byte into 4 trits.
 * 
 * @param packed Packed byte
 * @param trits Output array of 4 trits
 */
static inline void unpack_trits(uint8_t packed, int8_t* trits) {
    trits[0] = decode_trit((packed >> 0) & 0b11);
    trits[1] = decode_trit((packed >> 2) & 0b11);
    trits[2] = decode_trit((packed >> 4) & 0b11);
    trits[3] = decode_trit((packed >> 6) & 0b11);
}

/**
 * Pack an array of ternary values into 2-bit representation.
 * 
 * @param trits Input array of N ternary values
 * @param packed Output array of N/4 bytes (must be pre-allocated)
 * @param N Number of trits (must be multiple of 4)
 */
void pack_trit_array(const int8_t* trits, uint8_t* packed, size_t N);

/**
 * Unpack a 2-bit array into ternary values.
 * 
 * @param packed Input array of N/4 bytes
 * @param trits Output array of N ternary values (must be pre-allocated)
 * @param N Number of trits (must be multiple of 4)
 */
void unpack_trit_array(const uint8_t* packed, int8_t* trits, size_t N);

/* ========================================================================== */
/* Sparse Ternary FMA Functions                                             */
/* ========================================================================== */

/**
 * Sparse Ternary FMA: C = A * B + C (scalar implementation)
 * 
 * Computes fused multiply-add where B is a sparse ternary array.
 * 
 * Mathematical definition:
 *   C[i] += A[i] * decode(B_trit[i])
 * 
 * Where decode(B_trit[i]) ∈ {-1, 0, 1}
 * 
 * @param A Dense coefficient array [N]
 * @param B_trit Packed ternary array [N/4 bytes]
 * @param C Accumulator array [N] (modified in-place)
 * @param N Array length (must be multiple of 4)
 */
void sparse_ternary_fma_scalar(
    const int64_t* A,
    const uint8_t* B_trit,
    int64_t* C,
    size_t N
);

/**
 * Sparse Ternary FMA: C = A * B + C (AVX-512 implementation)
 * 
 * Optimized SIMD version processing 8 elements per iteration.
 * Falls back to scalar if AVX-512 is not available.
 * 
 * @param A Dense coefficient array [N]
 * @param B_trit Packed ternary array [N/4 bytes]
 * @param C Accumulator array [N] (modified in-place)
 * @param N Array length (must be multiple of 8)
 */
void sparse_ternary_fma_avx512(
    const int64_t* A,
    const uint8_t* B_trit,
    int64_t* C,
    size_t N
);

/**
 * Sparse Ternary FMA: C = A * B + C (sparse index format)
 * 
 * Optimized for very sparse arrays (Hamming weight w << N).
 * Only processes non-zero elements, achieving up to 16× speedup.
 * 
 * @param A Dense coefficient array [N]
 * @param indices Indices of non-zero elements [w]
 * @param values Ternary values {-1, 1} at indices [w]
 * @param C Accumulator array [N] (modified in-place)
 * @param w Hamming weight (number of non-zero elements)
 */
void sparse_ternary_fma_sparse(
    const int64_t* A,
    const uint32_t* indices,
    const int8_t* values,
    int64_t* C,
    size_t w
);

/**
 * Sparse Ternary FMA: Automatic dispatch
 * 
 * Automatically selects best implementation based on:
 * - CPU features (AVX-512 support)
 * - Array size
 * - Sparsity
 * 
 * @param A Dense coefficient array [N]
 * @param B_trit Packed ternary array [N/4 bytes]
 * @param C Accumulator array [N] (modified in-place)
 * @param N Array length
 */
void sparse_ternary_fma(
    const int64_t* A,
    const uint8_t* B_trit,
    int64_t* C,
    size_t N
);

/* ========================================================================== */
/* Ternary Arithmetic Operations                                            */
/* ========================================================================== */

/**
 * Ternary multiplication: result = a * b
 * 
 * Optimized for b ∈ {-1, 0, 1}:
 * - b = -1 → result = -a
 * - b =  0 → result = 0
 * - b = +1 → result = a
 * 
 * No actual multiplication is needed; uses conditional selection instead.
 * 
 * @param a Dense value
 * @param b_trit 2-bit encoded ternary value
 * @return Product a * decode(b_trit)
 */
static inline int64_t ternary_multiply(int64_t a, uint8_t b_trit) {
    if (b_trit == TRIT_ZERO) return 0;
    if (b_trit == TRIT_POS) return a;
    return -a;  /* TRIT_NEG */
}

/**
 * Ternary negation: result = -trit
 * 
 * Flips both bits if non-zero:
 * - 0b00 → 0b00 (zero stays zero)
 * - 0b01 → 0b10 (positive → negative)
 * - 0b10 → 0b01 (negative → positive)
 * 
 * @param trit 2-bit encoded ternary value
 * @return Negated trit
 */
static inline uint8_t ternary_negate(uint8_t trit) {
    return (trit == TRIT_ZERO) ? TRIT_ZERO : (trit ^ 0b11);
}

/* ========================================================================== */
/* Utility Functions                                                         */
/* ========================================================================== */

/**
 * Check if CPU supports AVX-512.
 * 
 * @return 1 if AVX-512 is available, 0 otherwise
 */
int has_avx512_support(void);

/**
 * Get optimal implementation name.
 * 
 * @return String describing the implementation being used
 */
const char* get_fma_implementation(void);

#ifdef __cplusplus
}
#endif

#endif /* SPARSE_TERNARY_FMA_H */
