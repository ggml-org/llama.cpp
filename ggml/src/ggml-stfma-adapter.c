#include "ggml-stfma-adapter.h"
#include "ggml-stfma/include/sparse_ternary_fma.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include <string.h>
#include <stdlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ========================================================================== */
/* Thread-Local Buffer Management                                            */
/* ========================================================================== */

static _Thread_local struct stfma_thread_buffers {
    uint8_t* encoding_buffer;
    int32_t* int32_buffer;
    int32_t* accumulator_buffer;
    size_t buffer_size;
} tl_buffers = {NULL, NULL, NULL, 0};

static void ensure_buffer_size(size_t required_size) {
    if (tl_buffers.buffer_size < required_size) {
        free(tl_buffers.encoding_buffer);
        free(tl_buffers.int32_buffer);
        free(tl_buffers.accumulator_buffer);
        
        tl_buffers.encoding_buffer = (uint8_t*)malloc(required_size);
        tl_buffers.int32_buffer = (int32_t*)malloc(required_size * 4);
        tl_buffers.accumulator_buffer = (int32_t*)malloc(required_size * 4);
        tl_buffers.buffer_size = required_size;
    }
}

/* ========================================================================== */
/* Encoding Conversion Functions                                             */
/* ========================================================================== */

uint8_t convert_tq2_to_stfma_byte(uint8_t b) {
    // TQ2_0:  00 (-1), 01 (0), 10 (+1), 11 (invalid)
    // STFMA:  10 (-1), 00 (0), 01 (+1), 11 (invalid)
    //
    // Formula:
    //   out_low  = in_high
    //   out_high = ~(in_high XOR in_low)
    
    uint8_t low_bits = b & 0x55;
    uint8_t high_bits = b & 0xAA;
    uint8_t out_low = (high_bits >> 1);
    uint8_t high_bits_shifted = (high_bits >> 1);
    uint8_t xor_result = high_bits_shifted ^ low_bits;
    uint8_t out_high = (~xor_result) & 0x55;
    out_high = out_high << 1;
    return out_high | out_low;
}

void convert_tq2_to_stfma_array(
    const uint8_t* tq2_packed,
    uint8_t* stfma_packed,
    size_t num_bytes
) {
    for (size_t i = 0; i < num_bytes; i++) {
        stfma_packed[i] = convert_tq2_to_stfma_byte(tq2_packed[i]);
    }
}

/* ========================================================================== */
/* Type Conversion Functions                                                  */
/* ========================================================================== */

void convert_q8k_to_int32(
    const int8_t* q8_values,
    int32_t* int32_buffer,
    size_t n
) {
#if defined(__AVX2__)
    // Vectorized conversion using AVX2
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        // Load 32 int8 values
        __m256i q8_vec = _mm256_loadu_si256((const __m256i*)&q8_values[i]);
        
        // Split into two 128-bit halves
        __m128i q8_low = _mm256_castsi256_si128(q8_vec);
        __m128i q8_high = _mm256_extracti128_si256(q8_vec, 1);
        
        // Sign-extend to int32 (8 elements at a time)
        __m256i int32_0 = _mm256_cvtepi8_epi32(q8_low);
        __m256i int32_1 = _mm256_cvtepi8_epi32(_mm_shuffle_epi32(q8_low, 0x39));
        __m256i int32_2 = _mm256_cvtepi8_epi32(q8_high);
        __m256i int32_3 = _mm256_cvtepi8_epi32(_mm_shuffle_epi32(q8_high, 0x39));
        
        // Store results
        _mm256_storeu_si256((__m256i*)&int32_buffer[i], int32_0);
        _mm256_storeu_si256((__m256i*)&int32_buffer[i + 8], int32_1);
        _mm256_storeu_si256((__m256i*)&int32_buffer[i + 16], int32_2);
        _mm256_storeu_si256((__m256i*)&int32_buffer[i + 24], int32_3);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        int32_buffer[i] = (int32_t)q8_values[i];
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < n; i++) {
        int32_buffer[i] = (int32_t)q8_values[i];
    }
#endif
}

/* ========================================================================== */
/* Sparse Ternary FMA Operations (int32 variants)                            */
/* ========================================================================== */

void sparse_ternary_fma_int32_scalar(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
) {
    for (size_t i = 0; i < N; i++) {
        uint8_t trit_byte = B_trit[i / 4];
        uint8_t trit = (trit_byte >> ((i % 4) * 2)) & 0b11;
        
        if (trit == 0b01) {
            C[i] += A[i];
        } else if (trit == 0b10) {
            C[i] -= A[i];
        }
    }
}

#if defined(__AVX2__)

void sparse_ternary_fma_int32_avx2(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    
    size_t i = 0;
    
    for (; i + 8 <= N; i += 8) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)&A[i]);
        __m256i c_vec = _mm256_loadu_si256((const __m256i*)&C[i]);
        
        size_t byte_idx = i / 4;
        uint16_t trit_packed = ((uint16_t)B_trit[byte_idx + 1] << 8) | B_trit[byte_idx];
        
        __m256i packed_vec = _mm256_set1_epi32(trit_packed);
        __m256i shift_amounts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        __m256i shifted = _mm256_srlv_epi32(packed_vec, shift_amounts);
        __m256i mask_2bits = _mm256_set1_epi32(0b11);
        __m256i trit_vec = _mm256_and_si256(shifted, mask_2bits);
        
        __m256i nonzero_cmp = _mm256_cmpgt_epi32(trit_vec, zero);
        __m256i is_plus_one = _mm256_cmpeq_epi32(trit_vec, one);
        __m256i is_minus_one = _mm256_andnot_si256(is_plus_one, nonzero_cmp);
        
        __m256i add_val = _mm256_and_si256(is_plus_one, a_vec);
        __m256i sub_val = _mm256_and_si256(is_minus_one, a_vec);
        
        c_vec = _mm256_add_epi32(c_vec, add_val);
        c_vec = _mm256_sub_epi32(c_vec, sub_val);
        
        _mm256_storeu_si256((__m256i*)&C[i], c_vec);
    }
    
    for (; i < N; i++) {
        uint8_t trit_byte = B_trit[i / 4];
        uint8_t trit = (trit_byte >> ((i % 4) * 2)) & 0b11;
        
        if (trit == 0b01) {
            C[i] += A[i];
        } else if (trit == 0b10) {
            C[i] -= A[i];
        }
    }
}

#endif

#if defined(__AVX512F__)

void sparse_ternary_fma_int32_avx512(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i one = _mm512_set1_epi32(1);
    
    size_t i = 0;
    
    for (; i + 16 <= N; i += 16) {
        __m512i a_vec = _mm512_loadu_si512(&A[i]);
        __m512i c_vec = _mm512_loadu_si512(&C[i]);
        
        size_t byte_idx = i / 4;
        uint32_t trit_packed = *(uint32_t*)&B_trit[byte_idx];
        
        __m512i packed_vec = _mm512_set1_epi32(trit_packed);
        __m512i shift_amounts = _mm512_setr_epi32(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30
        );
        __m512i shifted = _mm512_srlv_epi32(packed_vec, shift_amounts);
        __m512i mask_2bits = _mm512_set1_epi32(0b11);
        __m512i trit_vec = _mm512_and_si512(shifted, mask_2bits);
        
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(trit_vec, zero);
        __mmask16 is_plus_one = _mm512_cmpeq_epi32_mask(trit_vec, one);
        __mmask16 is_minus_one = nonzero_mask & ~is_plus_one;
        
        c_vec = _mm512_mask_add_epi32(c_vec, is_plus_one, c_vec, a_vec);
        c_vec = _mm512_mask_sub_epi32(c_vec, is_minus_one, c_vec, a_vec);
        
        _mm512_storeu_si512(&C[i], c_vec);
    }
    
    for (; i < N; i++) {
        uint8_t trit_byte = B_trit[i / 4];
        uint8_t trit = (trit_byte >> ((i % 4) * 2)) & 0b11;
        
        if (trit == 0b01) {
            C[i] += A[i];
        } else if (trit == 0b10) {
            C[i] -= A[i];
        }
    }
}

#endif

/* ========================================================================== */
/* High-Level Integration Function                                           */
/* ========================================================================== */

void ggml_vec_dot_tq2_0_q8_K_stfma(
    int n,
    float* s,
    size_t bs,
    const void* vx,
    size_t bx,
    const void* vy,
    size_t by,
    int nrc
) {
    (void)nrc;
    (void)bx;
    (void)by;
    (void)bs;
    
    const block_tq2_0* x = (const block_tq2_0*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;
    
    const int nb = n / QK_K;
    
    // Ensure buffers are large enough
    ensure_buffer_size(QK_K / 4);
    
    float sumf = 0.0f;
    
    for (int i = 0; i < nb; i++) {
        // Convert TQ2_0 encoding to sparse-ternary-fma encoding
        convert_tq2_to_stfma_array(x[i].qs, tl_buffers.encoding_buffer, QK_K / 4);
        
        // Convert Q8_K to int32
        convert_q8k_to_int32(y[i].qs, tl_buffers.int32_buffer, QK_K);
        
        // Initialize accumulator to zero
        memset(tl_buffers.accumulator_buffer, 0, QK_K * sizeof(int32_t));
        
        // Perform sparse ternary FMA
#if defined(__AVX512F__)
        sparse_ternary_fma_int32_avx512(
            tl_buffers.int32_buffer,
            tl_buffers.encoding_buffer,
            tl_buffers.accumulator_buffer,
            QK_K
        );
#elif defined(__AVX2__)
        sparse_ternary_fma_int32_avx2(
            tl_buffers.int32_buffer,
            tl_buffers.encoding_buffer,
            tl_buffers.accumulator_buffer,
            QK_K
        );
#else
        sparse_ternary_fma_int32_scalar(
            tl_buffers.int32_buffer,
            tl_buffers.encoding_buffer,
            tl_buffers.accumulator_buffer,
            QK_K
        );
#endif
        
        // Sum the accumulator
        int32_t sumi = 0;
        for (size_t j = 0; j < QK_K; j++) {
            sumi += tl_buffers.accumulator_buffer[j];
        }
        
        // Apply scale factors
        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        sumf += (float)sumi * d;
    }
    
    *s = sumf;
}
