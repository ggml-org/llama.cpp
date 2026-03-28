// TurboQuant SIMD Implementation
// AVX2/AVX-512 vectorized operations

#include "turboquant_simd.h"
#include <cstring>
#include <iostream>

// SIMD intrinsics
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace turboquant {
namespace simd {

// Runtime CPU feature detection
bool has_avx2() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

bool has_avx512() {
#if defined(__AVX512F__)
    return true;
#else
    return false;
#endif
}

// Scalar (non-vectorized) implementation
void apply_rotation_scalar(const float* Q, const float* input, float* output, int dim) {
    for (int i = 0; i < dim; ++i) {
        float val = 0.0f;
        const float* Q_row = &Q[i * dim];
        for (int j = 0; j < dim; ++j) {
            val += Q_row[j] * input[j];
        }
        output[i] = val;
    }
}

#ifdef __AVX2__
// AVX2 implementation - processes 8 floats at a time
void apply_rotation_avx2(const float* Q, const float* input, float* output, int dim) {
    for (int i = 0; i < dim; ++i) {
        const float* Q_row = &Q[i * dim];
        
        // Process 8 elements at a time with AVX2
        int j = 0;
        __m256 acc = _mm256_setzero_ps();
        
        for (; j + 7 < dim; j += 8) {
            __m256 q = _mm256_loadu_ps(&Q_row[j]);
            __m256 inp = _mm256_loadu_ps(&input[j]);
            acc = _mm256_fmadd_ps(q, inp, acc);
        }
        
        // Horizontal sum
        float sum[8];
        _mm256_storeu_ps(sum, acc);
        float val = sum[0] + sum[1] + sum[2] + sum[3] + 
                    sum[4] + sum[5] + sum[6] + sum[7];
        
        // Handle remainder
        for (; j < dim; ++j) {
            val += Q_row[j] * input[j];
        }
        
        output[i] = val;
    }
}
#endif

#ifdef __AVX512F__
// AVX-512 implementation - processes 16 floats at a time
void apply_rotation_avx512(const float* Q, const float* input, float* output, int dim) {
    for (int i = 0; i < dim; ++i) {
        const float* Q_row = &Q[i * dim];
        
        // Process 16 elements at a time with AVX-512
        int j = 0;
        __m512 acc = _mm512_setzero_ps();
        
        for (; j + 15 < dim; j += 16) {
            __m512 q = _mm512_loadu_ps(&Q_row[j]);
            __m512 inp = _mm512_loadu_ps(&input[j]);
            acc = _mm512_fmadd_ps(q, inp, acc);
        }
        
        // Horizontal sum
        float sum[16];
        _mm512_storeu_ps(sum, acc);
        float val = 0.0f;
        for (int k = 0; k < 16; ++k) {
            val += sum[k];
        }
        
        // Handle remainder
        for (; j < dim; ++j) {
            val += Q_row[j] * input[j];
        }
        
        output[i] = val;
    }
}
#endif

// Auto-dispatch implementation
void apply_rotation_auto(const float* Q, const float* input, float* output, int dim) {
#ifdef __AVX512F__
    if (has_avx512()) {
        apply_rotation_avx512(Q, input, output, dim);
        return;
    }
#endif

#ifdef __AVX2__
    if (has_avx2()) {
        apply_rotation_avx2(Q, input, output, dim);
        return;
    }
#endif

    // Fallback to scalar
    apply_rotation_scalar(Q, input, output, dim);
}

} // namespace simd
} // namespace turboquant
