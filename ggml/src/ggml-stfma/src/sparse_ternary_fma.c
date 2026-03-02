/**
 * Sparse Ternary Fused Multiply-Add (FMA) Kernel - Implementation
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

#include "../include/sparse_ternary_fma.h"
#include <string.h>
#include <stdio.h>

#ifdef __x86_64__
#include <cpuid.h>
#endif

/* ========================================================================== */
/* Packing/Unpacking Functions                                              */
/* ========================================================================== */

void pack_trit_array(const int8_t* trits, uint8_t* packed, size_t N) {
    for (size_t i = 0; i < N; i += 4) {
        packed[i / 4] = pack_trits(
            trits[i],
            trits[i + 1],
            trits[i + 2],
            trits[i + 3]
        );
    }
}

void unpack_trit_array(const uint8_t* packed, int8_t* trits, size_t N) {
    for (size_t i = 0; i < N; i += 4) {
        unpack_trits(packed[i / 4], &trits[i]);
    }
}

/* ========================================================================== */
/* Scalar Implementation                                                     */
/* ========================================================================== */

void sparse_ternary_fma_scalar(
    const int64_t* A,
    const uint8_t* B_trit,
    int64_t* C,
    size_t N
) {
    for (size_t i = 0; i < N; i++) {
        /* Extract 2-bit trit from packed array */
        size_t byte_idx = i / 4;
        size_t trit_offset = (i % 4) * 2;
        uint8_t trit = (B_trit[byte_idx] >> trit_offset) & 0b11;
        
        /* Decode and accumulate */
        if (trit == TRIT_POS) {
            C[i] += A[i];
        } else if (trit == TRIT_NEG) {
            C[i] -= A[i];
        }
        /* else: trit == TRIT_ZERO, skip (no contribution) */
    }
}

/* ========================================================================== */
/* AVX-512 Implementation                                                    */
/* ========================================================================== */

#if HAS_AVX512

void sparse_ternary_fma_avx512(
    const int64_t* A,
    const uint8_t* B_trit,
    int64_t* C,
    size_t N
) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask_low = _mm512_set1_epi64(1);
    
    for (size_t i = 0; i < N; i += 8) {
        /* Load 8 coefficients (64-bit each) */
        __m512i a_vec = _mm512_loadu_si512(&A[i]);
        
        /* Load 8 accumulators */
        __m512i c_vec = _mm512_loadu_si512(&C[i]);
        
        /* Load 2 bytes containing 8 trits (8 × 2 bits = 16 bits) */
        /* Each byte contains 4 trits, so 8 trits = 2 bytes */
        size_t byte_idx = i / 4;
        uint16_t trit_packed = ((uint16_t)B_trit[byte_idx + 1] << 8) | 
                                B_trit[byte_idx];
        
        /* Extract 8 trits into array */
        uint64_t trits[8];
        for (int j = 0; j < 8; j++) {
            trits[j] = (trit_packed >> (j * 2)) & 0b11;
        }
        
        /* Load trits into 512-bit vector */
        __m512i trit_vec = _mm512_set_epi64(
            trits[7], trits[6], trits[5], trits[4],
            trits[3], trits[2], trits[1], trits[0]
        );
        
        /* Create nonzero mask: true if trit != 0b00 */
        /* This correctly handles both +1 (0b01) and -1 (0b10) */
        __mmask8 nonzero_mask = _mm512_cmpneq_epi64_mask(trit_vec, zero);
        
        /* Extract sign bit (high bit) for negative detection */
        /* sign=1 only for -1 (0b10), sign=0 for +1 (0b01) and 0 (0b00) */
        __m512i sign = _mm512_srli_epi64(trit_vec, 1);
        sign = _mm512_and_si512(sign, mask_low);
        __mmask8 sign_mask = _mm512_cmpneq_epi64_mask(sign, zero);
        
        /* Compute contribution: 0 if trit=0, A if trit!=0 */
        __m512i contribution = _mm512_maskz_mov_epi64(nonzero_mask, a_vec);
        
        /* Conditionally negate if sign=1 (i.e., trit=0b10=-1) */
        __m512i negated = _mm512_sub_epi64(zero, contribution);
        contribution = _mm512_mask_blend_epi64(sign_mask, contribution, negated);
        
        /* FMA: C += contribution (update accumulator) */
        c_vec = _mm512_add_epi64(c_vec, contribution);
        
        /* Store result */
        _mm512_storeu_si512(&C[i], c_vec);
    }
}

#else

/* Fallback to scalar if AVX-512 not available */
void sparse_ternary_fma_avx512(
    const int64_t* A,
    const uint8_t* B_trit,
    int64_t* C,
    size_t N
) {
    sparse_ternary_fma_scalar(A, B_trit, C, N);
}

#endif

/* ========================================================================== */
/* Sparse Implementation                                                     */
/* ========================================================================== */

void sparse_ternary_fma_sparse(
    const int64_t* A,
    const uint32_t* indices,
    const int8_t* values,
    int64_t* C,
    size_t w
) {
    for (size_t i = 0; i < w; i++) {
        uint32_t idx = indices[i];
        int8_t value = values[i];
        
        if (value == 1) {
            C[idx] += A[idx];
        } else {  /* value == -1 */
            C[idx] -= A[idx];
        }
    }
}

/* ========================================================================== */
/* Automatic Dispatch                                                        */
/* ========================================================================== */

void sparse_ternary_fma(
    const int64_t* A,
    const uint8_t* B_trit,
    int64_t* C,
    size_t N
) {
#if HAS_AVX512
    if (has_avx512_support() && N >= 8 && N % 8 == 0) {
        sparse_ternary_fma_avx512(A, B_trit, C, N);
    } else {
        sparse_ternary_fma_scalar(A, B_trit, C, N);
    }
#else
    sparse_ternary_fma_scalar(A, B_trit, C, N);
#endif
}

/* ========================================================================== */
/* Utility Functions                                                         */
/* ========================================================================== */

int has_avx512_support(void) {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    /* Check for AVX-512 Foundation (AVX512F) */
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;  /* Bit 16 = AVX512F */
    }
#endif
    
    return 0;
}

const char* get_fma_implementation(void) {
#if HAS_AVX512
    if (has_avx512_support()) {
        return "AVX-512 (SIMD)";
    }
#endif
    return "Scalar (Reference)";
}
