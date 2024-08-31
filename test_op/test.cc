#include "string.h"
#include <type_traits>
#include <stdio.h>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
typedef float16_t float_type;
#else
#include <stdint.h>
typedef float float_type;
#endif

#if defined __AVX2__
#define extract_low_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
#define extract_high_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))
#define extract_low_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
#define extract_high_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))
#endif

#define BM 128
#define BK 64
#define M 1536
#define K 4096

int32_t partial_max(int k, void* lut_scales_, void* b_) {
    float_type* lut_scales = (float_type*)lut_scales_;
    float_type* b = (float_type*)b_;
#if defined __AVX2__
    __m256 max_vec = _mm256_set1_ps(0.f);
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < k / 8; i++) {
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);
        max_vec = _mm256_max_ps(vec_babs, max_vec);
    }
            // float* tmp0 = reinterpret_cast<float*>(&(max_vec));
            // for (int cc = 0; cc < 8; cc++) {    
            //     printf("%f ", tmp0[cc]);
            // }
            // printf("\n");
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));
    float scales = _mm_cvtss_f32(max1) / 127;
    *lut_scales = std::max(*lut_scales, scales);
#endif

    return 0;
}

int32_t partial_max_reset(void* lut_scales_) {
    float_type* lut_scales = (float_type*)lut_scales_;
    *lut_scales = 0.0;
    return 0;
}

inline int32_t lut_ctor_g4_int8_k0_b2(int32_t act_k, int8_t* qlut, float_type* b, float_type* lut_scales, float_type* lut_biases) {
#if defined __AVX2__
    __m256 vec_lut[16];
    float biases = 0.0;
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);
    float scales = *lut_scales;
    float t_scales = scales ? 1.0f / scales : 0.0f;
#pragma unroll
    for (int k = 0; k < act_k / 16; ++k) {
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 16 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 16 + 1, vec_bi, 1);

        vec_lut[15] = _mm256_setzero_ps();

        vec_lut[14] = _mm256_setzero_ps();

        vec_lut[13] = _mm256_setzero_ps();

        vec_lut[12] = _mm256_setzero_ps();

        vec_lut[11] = _mm256_setzero_ps();

        vec_lut[10] = _mm256_setzero_ps();

        vec_lut[9] = _mm256_setzero_ps();

        // 1 1
        vec_lut[8] = vec_b0;
        vec_lut[8] = _mm256_add_ps(vec_lut[8], vec_b1);

        // 1 0
        vec_lut[7] = vec_b0;

        // 1 -1
        vec_lut[6] = vec_b0;
        vec_lut[6] = _mm256_sub_ps(vec_lut[6], vec_b1);

        // 0 1
        vec_lut[5] = vec_b1;

        // 0 0
        vec_lut[4] = _mm256_setzero_ps();

        // 0 -1
        vec_lut[3] = _mm256_setzero_ps();
        vec_lut[3] = _mm256_sub_ps(vec_lut[3], vec_b1);

        // -1 1
        vec_lut[2] = _mm256_setzero_ps();
        vec_lut[2] = _mm256_sub_ps(vec_lut[2], vec_b0);
        vec_lut[2] = _mm256_add_ps(vec_lut[2], vec_b1);

        // -1 0
        vec_lut[1] = _mm256_setzero_ps();
        vec_lut[1] = _mm256_sub_ps(vec_lut[1], vec_b0);

        // -1 -1
        vec_lut[0] = _mm256_setzero_ps();
        vec_lut[0] = _mm256_sub_ps(vec_lut[0], vec_b0);
        vec_lut[0] = _mm256_sub_ps(vec_lut[0], vec_b1);


#pragma unroll
        for (int g = 0; g < 9; ++g) {
            vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(t_scales));
        }

        __m256i ix[16];

        for (int g = 0; g < 16; ++g) {
            ix[g] = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        for (int g = 0; g < 8; ++g) {
            ix[g * 2] = _mm256_packs_epi32(ix[g * 2], ix[g * 2 + 1]);
        }

            // printf("i016\n");
            // int16_t* tmp0 = reinterpret_cast<int16_t*>(&(ix[2]));
            // for (int cc = 0; cc < 16; cc++) {    
            //     printf("%d ", tmp0[cc]);
            // }
            // printf("\n");

        // i0 = _mm256_packs_epi16(i0, i2);
        // i4 = _mm256_packs_epi16(i4, i6);
        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);

        // first half 128i
        qlut_i8[k * 256 + 0 * 32 + 0] = _mm256_extract_epi8(ix[0], 0);
        qlut_i8[k * 256 + 0 * 32 + 1] = _mm256_extract_epi8(ix[0], 8);
        qlut_i8[k * 256 + 0 * 32 + 2] = _mm256_extract_epi8(ix[2], 0);
        qlut_i8[k * 256 + 0 * 32 + 3] = _mm256_extract_epi8(ix[2], 8);
        qlut_i8[k * 256 + 0 * 32 + 4] = _mm256_extract_epi8(ix[4], 0);
        qlut_i8[k * 256 + 0 * 32 + 5] = _mm256_extract_epi8(ix[4], 8);
        qlut_i8[k * 256 + 0 * 32 + 6] = _mm256_extract_epi8(ix[6], 0);
        qlut_i8[k * 256 + 0 * 32 + 7] = _mm256_extract_epi8(ix[6], 8);
        qlut_i8[k * 256 + 0 * 32 + 8] = _mm256_extract_epi8(ix[8], 0);
        qlut_i8[k * 256 + 0 * 32 + 9] = _mm256_extract_epi8(ix[8], 8);
        qlut_i8[k * 256 + 0 * 32 + 10] = _mm256_extract_epi8(ix[10], 0);
        qlut_i8[k * 256 + 0 * 32 + 11] = _mm256_extract_epi8(ix[10], 8);
        qlut_i8[k * 256 + 0 * 32 + 12] = _mm256_extract_epi8(ix[12], 0);
        qlut_i8[k * 256 + 0 * 32 + 13] = _mm256_extract_epi8(ix[12], 8);
        qlut_i8[k * 256 + 0 * 32 + 14] = _mm256_extract_epi8(ix[14], 0);
        qlut_i8[k * 256 + 0 * 32 + 15] = _mm256_extract_epi8(ix[14], 8);

        // second half 128i
        qlut_i8[k * 256 + 0 * 32 + 16] = _mm256_extract_epi8(ix[0], 1);
        qlut_i8[k * 256 + 0 * 32 + 17] = _mm256_extract_epi8(ix[0], 9);
        qlut_i8[k * 256 + 0 * 32 + 18] = _mm256_extract_epi8(ix[2], 1);
        qlut_i8[k * 256 + 0 * 32 + 19] = _mm256_extract_epi8(ix[2], 9);
        qlut_i8[k * 256 + 0 * 32 + 20] = _mm256_extract_epi8(ix[4], 1);
        qlut_i8[k * 256 + 0 * 32 + 21] = _mm256_extract_epi8(ix[4], 9);
        qlut_i8[k * 256 + 0 * 32 + 22] = _mm256_extract_epi8(ix[6], 1);
        qlut_i8[k * 256 + 0 * 32 + 23] = _mm256_extract_epi8(ix[6], 9);
        qlut_i8[k * 256 + 0 * 32 + 24] = _mm256_extract_epi8(ix[8], 1);
        qlut_i8[k * 256 + 0 * 32 + 25] = _mm256_extract_epi8(ix[8], 9);
        qlut_i8[k * 256 + 0 * 32 + 26] = _mm256_extract_epi8(ix[10], 1);
        qlut_i8[k * 256 + 0 * 32 + 27] = _mm256_extract_epi8(ix[10], 9);
        qlut_i8[k * 256 + 0 * 32 + 28] = _mm256_extract_epi8(ix[12], 1);
        qlut_i8[k * 256 + 0 * 32 + 29] = _mm256_extract_epi8(ix[12], 9);
        qlut_i8[k * 256 + 0 * 32 + 30] = _mm256_extract_epi8(ix[14], 1);
        qlut_i8[k * 256 + 0 * 32 + 31] = _mm256_extract_epi8(ix[14], 9);

        qlut_i8[k * 256 + 1 * 32 + 0] = _mm256_extract_epi8(ix[0], 2);
        qlut_i8[k * 256 + 1 * 32 + 1] = _mm256_extract_epi8(ix[0], 10);
        qlut_i8[k * 256 + 1 * 32 + 2] = _mm256_extract_epi8(ix[2], 2);
        qlut_i8[k * 256 + 1 * 32 + 3] = _mm256_extract_epi8(ix[2], 10);
        qlut_i8[k * 256 + 1 * 32 + 4] = _mm256_extract_epi8(ix[4], 2);
        qlut_i8[k * 256 + 1 * 32 + 5] = _mm256_extract_epi8(ix[4], 10);
        qlut_i8[k * 256 + 1 * 32 + 6] = _mm256_extract_epi8(ix[6], 2);
        qlut_i8[k * 256 + 1 * 32 + 7] = _mm256_extract_epi8(ix[6], 10);
        qlut_i8[k * 256 + 1 * 32 + 8] = _mm256_extract_epi8(ix[8], 2);
        qlut_i8[k * 256 + 1 * 32 + 9] = _mm256_extract_epi8(ix[8], 10);
        qlut_i8[k * 256 + 1 * 32 + 10] = _mm256_extract_epi8(ix[10], 2);
        qlut_i8[k * 256 + 1 * 32 + 11] = _mm256_extract_epi8(ix[10], 10);
        qlut_i8[k * 256 + 1 * 32 + 12] = _mm256_extract_epi8(ix[12], 2);
        qlut_i8[k * 256 + 1 * 32 + 13] = _mm256_extract_epi8(ix[12], 10);
        qlut_i8[k * 256 + 1 * 32 + 14] = _mm256_extract_epi8(ix[14], 2);
        qlut_i8[k * 256 + 1 * 32 + 15] = _mm256_extract_epi8(ix[14], 10);

        qlut_i8[k * 256 + 1 * 32 + 16] = _mm256_extract_epi8(ix[0], 3);
        qlut_i8[k * 256 + 1 * 32 + 17] = _mm256_extract_epi8(ix[0], 11);
        qlut_i8[k * 256 + 1 * 32 + 18] = _mm256_extract_epi8(ix[2], 3);
        qlut_i8[k * 256 + 1 * 32 + 19] = _mm256_extract_epi8(ix[2], 11);
        qlut_i8[k * 256 + 1 * 32 + 20] = _mm256_extract_epi8(ix[4], 3);
        qlut_i8[k * 256 + 1 * 32 + 21] = _mm256_extract_epi8(ix[4], 11);
        qlut_i8[k * 256 + 1 * 32 + 22] = _mm256_extract_epi8(ix[6], 3);
        qlut_i8[k * 256 + 1 * 32 + 23] = _mm256_extract_epi8(ix[6], 11);
        qlut_i8[k * 256 + 1 * 32 + 24] = _mm256_extract_epi8(ix[8], 3);
        qlut_i8[k * 256 + 1 * 32 + 25] = _mm256_extract_epi8(ix[8], 11);
        qlut_i8[k * 256 + 1 * 32 + 26] = _mm256_extract_epi8(ix[10], 3);
        qlut_i8[k * 256 + 1 * 32 + 27] = _mm256_extract_epi8(ix[10], 11);
        qlut_i8[k * 256 + 1 * 32 + 28] = _mm256_extract_epi8(ix[12], 3);
        qlut_i8[k * 256 + 1 * 32 + 29] = _mm256_extract_epi8(ix[12], 11);
        qlut_i8[k * 256 + 1 * 32 + 30] = _mm256_extract_epi8(ix[14], 3);
        qlut_i8[k * 256 + 1 * 32 + 31] = _mm256_extract_epi8(ix[14], 11);

        // printf("res\n");
        // for (int i=0; i<32; i++) {
        //     printf("%d ", qlut_i8[k * 256 + 0 * 32 + 0]);
        // }
        // printf("\n");

        qlut_i8[k * 256 + 2 * 32 + 0] = _mm256_extract_epi8(ix[0], 4);
        qlut_i8[k * 256 + 2 * 32 + 1] = _mm256_extract_epi8(ix[0], 12);
        qlut_i8[k * 256 + 2 * 32 + 2] = _mm256_extract_epi8(ix[2], 4);
        qlut_i8[k * 256 + 2 * 32 + 3] = _mm256_extract_epi8(ix[2], 12);
        qlut_i8[k * 256 + 2 * 32 + 4] = _mm256_extract_epi8(ix[4], 4);
        qlut_i8[k * 256 + 2 * 32 + 5] = _mm256_extract_epi8(ix[4], 12);
        qlut_i8[k * 256 + 2 * 32 + 6] = _mm256_extract_epi8(ix[6], 4);
        qlut_i8[k * 256 + 2 * 32 + 7] = _mm256_extract_epi8(ix[6], 12);
        qlut_i8[k * 256 + 2 * 32 + 8] = _mm256_extract_epi8(ix[8], 4);
        qlut_i8[k * 256 + 2 * 32 + 9] = _mm256_extract_epi8(ix[8], 12);
        qlut_i8[k * 256 + 2 * 32 + 10] = _mm256_extract_epi8(ix[10], 4);
        qlut_i8[k * 256 + 2 * 32 + 11] = _mm256_extract_epi8(ix[10], 12);
        qlut_i8[k * 256 + 2 * 32 + 12] = _mm256_extract_epi8(ix[12], 4);
        qlut_i8[k * 256 + 2 * 32 + 13] = _mm256_extract_epi8(ix[12], 12);
        qlut_i8[k * 256 + 2 * 32 + 14] = _mm256_extract_epi8(ix[14], 4);
        qlut_i8[k * 256 + 2 * 32 + 15] = _mm256_extract_epi8(ix[14], 12);

        qlut_i8[k * 256 + 2 * 32 + 16] = _mm256_extract_epi8(ix[0], 5);
        qlut_i8[k * 256 + 2 * 32 + 17] = _mm256_extract_epi8(ix[0], 13);
        qlut_i8[k * 256 + 2 * 32 + 18] = _mm256_extract_epi8(ix[2], 5);
        qlut_i8[k * 256 + 2 * 32 + 19] = _mm256_extract_epi8(ix[2], 13);
        qlut_i8[k * 256 + 2 * 32 + 20] = _mm256_extract_epi8(ix[4], 5);
        qlut_i8[k * 256 + 2 * 32 + 21] = _mm256_extract_epi8(ix[4], 13);
        qlut_i8[k * 256 + 2 * 32 + 22] = _mm256_extract_epi8(ix[6], 5);
        qlut_i8[k * 256 + 2 * 32 + 23] = _mm256_extract_epi8(ix[6], 13);
        qlut_i8[k * 256 + 2 * 32 + 24] = _mm256_extract_epi8(ix[8], 5);
        qlut_i8[k * 256 + 2 * 32 + 25] = _mm256_extract_epi8(ix[8], 13);
        qlut_i8[k * 256 + 2 * 32 + 26] = _mm256_extract_epi8(ix[10], 5);
        qlut_i8[k * 256 + 2 * 32 + 27] = _mm256_extract_epi8(ix[10], 13);
        qlut_i8[k * 256 + 2 * 32 + 28] = _mm256_extract_epi8(ix[12], 5);
        qlut_i8[k * 256 + 2 * 32 + 29] = _mm256_extract_epi8(ix[12], 13);
        qlut_i8[k * 256 + 2 * 32 + 30] = _mm256_extract_epi8(ix[14], 5);
        qlut_i8[k * 256 + 2 * 32 + 31] = _mm256_extract_epi8(ix[14], 13);

        qlut_i8[k * 256 + 3 * 32 + 0] = _mm256_extract_epi8(ix[0], 6);
        qlut_i8[k * 256 + 3 * 32 + 1] = _mm256_extract_epi8(ix[0], 14);
        qlut_i8[k * 256 + 3 * 32 + 2] = _mm256_extract_epi8(ix[2], 6);
        qlut_i8[k * 256 + 3 * 32 + 3] = _mm256_extract_epi8(ix[2], 14);
        qlut_i8[k * 256 + 3 * 32 + 4] = _mm256_extract_epi8(ix[4], 6);
        qlut_i8[k * 256 + 3 * 32 + 5] = _mm256_extract_epi8(ix[4], 14);
        qlut_i8[k * 256 + 3 * 32 + 6] = _mm256_extract_epi8(ix[6], 6);
        qlut_i8[k * 256 + 3 * 32 + 7] = _mm256_extract_epi8(ix[6], 14);
        qlut_i8[k * 256 + 3 * 32 + 8] = _mm256_extract_epi8(ix[8], 6);
        qlut_i8[k * 256 + 3 * 32 + 9] = _mm256_extract_epi8(ix[8], 14);
        qlut_i8[k * 256 + 3 * 32 + 10] = _mm256_extract_epi8(ix[10], 6);
        qlut_i8[k * 256 + 3 * 32 + 11] = _mm256_extract_epi8(ix[10], 14);
        qlut_i8[k * 256 + 3 * 32 + 12] = _mm256_extract_epi8(ix[12], 6);
        qlut_i8[k * 256 + 3 * 32 + 13] = _mm256_extract_epi8(ix[12], 14);
        qlut_i8[k * 256 + 3 * 32 + 14] = _mm256_extract_epi8(ix[14], 6);
        qlut_i8[k * 256 + 3 * 32 + 15] = _mm256_extract_epi8(ix[14], 14);

        qlut_i8[k * 256 + 3 * 32 + 16] = _mm256_extract_epi8(ix[0], 7);
        qlut_i8[k * 256 + 3 * 32 + 17] = _mm256_extract_epi8(ix[0], 15);
        qlut_i8[k * 256 + 3 * 32 + 18] = _mm256_extract_epi8(ix[2], 7);
        qlut_i8[k * 256 + 3 * 32 + 19] = _mm256_extract_epi8(ix[2], 15);
        qlut_i8[k * 256 + 3 * 32 + 20] = _mm256_extract_epi8(ix[4], 7);
        qlut_i8[k * 256 + 3 * 32 + 21] = _mm256_extract_epi8(ix[4], 15);
        qlut_i8[k * 256 + 3 * 32 + 22] = _mm256_extract_epi8(ix[6], 7);
        qlut_i8[k * 256 + 3 * 32 + 23] = _mm256_extract_epi8(ix[6], 15);
        qlut_i8[k * 256 + 3 * 32 + 24] = _mm256_extract_epi8(ix[8], 7);
        qlut_i8[k * 256 + 3 * 32 + 25] = _mm256_extract_epi8(ix[8], 15);
        qlut_i8[k * 256 + 3 * 32 + 26] = _mm256_extract_epi8(ix[10], 7);
        qlut_i8[k * 256 + 3 * 32 + 27] = _mm256_extract_epi8(ix[10], 15);
        qlut_i8[k * 256 + 3 * 32 + 28] = _mm256_extract_epi8(ix[12], 7);
        qlut_i8[k * 256 + 3 * 32 + 29] = _mm256_extract_epi8(ix[12], 15);
        qlut_i8[k * 256 + 3 * 32 + 30] = _mm256_extract_epi8(ix[14], 7);
        qlut_i8[k * 256 + 3 * 32 + 31] = _mm256_extract_epi8(ix[14], 15);

        qlut_i8[k * 256 + 4 * 32 + 0] = _mm256_extract_epi8(ix[0], 16);
        qlut_i8[k * 256 + 4 * 32 + 1] = _mm256_extract_epi8(ix[0], 24);
        qlut_i8[k * 256 + 4 * 32 + 2] = _mm256_extract_epi8(ix[2], 16);
        qlut_i8[k * 256 + 4 * 32 + 3] = _mm256_extract_epi8(ix[2], 24);
        qlut_i8[k * 256 + 4 * 32 + 4] = _mm256_extract_epi8(ix[4], 16);
        qlut_i8[k * 256 + 4 * 32 + 5] = _mm256_extract_epi8(ix[4], 24);
        qlut_i8[k * 256 + 4 * 32 + 6] = _mm256_extract_epi8(ix[6], 16);
        qlut_i8[k * 256 + 4 * 32 + 7] = _mm256_extract_epi8(ix[6], 24);
        qlut_i8[k * 256 + 4 * 32 + 8] = _mm256_extract_epi8(ix[8], 16);
        qlut_i8[k * 256 + 4 * 32 + 9] = _mm256_extract_epi8(ix[8], 24);
        qlut_i8[k * 256 + 4 * 32 + 10] = _mm256_extract_epi8(ix[10], 16);
        qlut_i8[k * 256 + 4 * 32 + 11] = _mm256_extract_epi8(ix[10], 24);
        qlut_i8[k * 256 + 4 * 32 + 12] = _mm256_extract_epi8(ix[12], 16);
        qlut_i8[k * 256 + 4 * 32 + 13] = _mm256_extract_epi8(ix[12], 24);
        qlut_i8[k * 256 + 4 * 32 + 14] = _mm256_extract_epi8(ix[14], 16);
        qlut_i8[k * 256 + 4 * 32 + 15] = _mm256_extract_epi8(ix[14], 24);

        qlut_i8[k * 256 + 4 * 32 + 16] = _mm256_extract_epi8(ix[0], 17);
        qlut_i8[k * 256 + 4 * 32 + 17] = _mm256_extract_epi8(ix[0], 25);
        qlut_i8[k * 256 + 4 * 32 + 18] = _mm256_extract_epi8(ix[2], 17);
        qlut_i8[k * 256 + 4 * 32 + 19] = _mm256_extract_epi8(ix[2], 25);
        qlut_i8[k * 256 + 4 * 32 + 20] = _mm256_extract_epi8(ix[4], 17);
        qlut_i8[k * 256 + 4 * 32 + 21] = _mm256_extract_epi8(ix[4], 25);
        qlut_i8[k * 256 + 4 * 32 + 22] = _mm256_extract_epi8(ix[6], 17);
        qlut_i8[k * 256 + 4 * 32 + 23] = _mm256_extract_epi8(ix[6], 25);
        qlut_i8[k * 256 + 4 * 32 + 24] = _mm256_extract_epi8(ix[8], 17);
        qlut_i8[k * 256 + 4 * 32 + 25] = _mm256_extract_epi8(ix[8], 25);
        qlut_i8[k * 256 + 4 * 32 + 26] = _mm256_extract_epi8(ix[10], 17);
        qlut_i8[k * 256 + 4 * 32 + 27] = _mm256_extract_epi8(ix[10], 25);
        qlut_i8[k * 256 + 4 * 32 + 28] = _mm256_extract_epi8(ix[12], 17);
        qlut_i8[k * 256 + 4 * 32 + 29] = _mm256_extract_epi8(ix[12], 25);
        qlut_i8[k * 256 + 4 * 32 + 30] = _mm256_extract_epi8(ix[14], 17);
        qlut_i8[k * 256 + 4 * 32 + 31] = _mm256_extract_epi8(ix[14], 25);

        qlut_i8[k * 256 + 5 * 32 + 0] = _mm256_extract_epi8(ix[0], 18);
        qlut_i8[k * 256 + 5 * 32 + 1] = _mm256_extract_epi8(ix[0], 26);
        qlut_i8[k * 256 + 5 * 32 + 2] = _mm256_extract_epi8(ix[2], 18);
        qlut_i8[k * 256 + 5 * 32 + 3] = _mm256_extract_epi8(ix[2], 26);
        qlut_i8[k * 256 + 5 * 32 + 4] = _mm256_extract_epi8(ix[4], 18);
        qlut_i8[k * 256 + 5 * 32 + 5] = _mm256_extract_epi8(ix[4], 26);
        qlut_i8[k * 256 + 5 * 32 + 6] = _mm256_extract_epi8(ix[6], 18);
        qlut_i8[k * 256 + 5 * 32 + 7] = _mm256_extract_epi8(ix[6], 26);
        qlut_i8[k * 256 + 5 * 32 + 8] = _mm256_extract_epi8(ix[8], 18);
        qlut_i8[k * 256 + 5 * 32 + 9] = _mm256_extract_epi8(ix[8], 26);
        qlut_i8[k * 256 + 5 * 32 + 10] = _mm256_extract_epi8(ix[10], 18);
        qlut_i8[k * 256 + 5 * 32 + 11] = _mm256_extract_epi8(ix[10], 26);
        qlut_i8[k * 256 + 5 * 32 + 12] = _mm256_extract_epi8(ix[12], 18);
        qlut_i8[k * 256 + 5 * 32 + 13] = _mm256_extract_epi8(ix[12], 26);
        qlut_i8[k * 256 + 5 * 32 + 14] = _mm256_extract_epi8(ix[14], 18);
        qlut_i8[k * 256 + 5 * 32 + 15] = _mm256_extract_epi8(ix[14], 26);

        qlut_i8[k * 256 + 5 * 32 + 16] = _mm256_extract_epi8(ix[0], 19);
        qlut_i8[k * 256 + 5 * 32 + 17] = _mm256_extract_epi8(ix[0], 27);
        qlut_i8[k * 256 + 5 * 32 + 18] = _mm256_extract_epi8(ix[2], 19);
        qlut_i8[k * 256 + 5 * 32 + 19] = _mm256_extract_epi8(ix[2], 27);
        qlut_i8[k * 256 + 5 * 32 + 20] = _mm256_extract_epi8(ix[4], 19);
        qlut_i8[k * 256 + 5 * 32 + 21] = _mm256_extract_epi8(ix[4], 27);
        qlut_i8[k * 256 + 5 * 32 + 22] = _mm256_extract_epi8(ix[6], 19);
        qlut_i8[k * 256 + 5 * 32 + 23] = _mm256_extract_epi8(ix[6], 27);
        qlut_i8[k * 256 + 5 * 32 + 24] = _mm256_extract_epi8(ix[8], 19);
        qlut_i8[k * 256 + 5 * 32 + 25] = _mm256_extract_epi8(ix[8], 27);
        qlut_i8[k * 256 + 5 * 32 + 26] = _mm256_extract_epi8(ix[10], 19);
        qlut_i8[k * 256 + 5 * 32 + 27] = _mm256_extract_epi8(ix[10], 27);
        qlut_i8[k * 256 + 5 * 32 + 28] = _mm256_extract_epi8(ix[12], 19);
        qlut_i8[k * 256 + 5 * 32 + 29] = _mm256_extract_epi8(ix[12], 27);
        qlut_i8[k * 256 + 5 * 32 + 30] = _mm256_extract_epi8(ix[14], 19);
        qlut_i8[k * 256 + 5 * 32 + 31] = _mm256_extract_epi8(ix[14], 27);

        qlut_i8[k * 256 + 6 * 32 + 0] = _mm256_extract_epi8(ix[0], 20);
        qlut_i8[k * 256 + 6 * 32 + 1] = _mm256_extract_epi8(ix[0], 28);
        qlut_i8[k * 256 + 6 * 32 + 2] = _mm256_extract_epi8(ix[2], 20);
        qlut_i8[k * 256 + 6 * 32 + 3] = _mm256_extract_epi8(ix[2], 28);
        qlut_i8[k * 256 + 6 * 32 + 4] = _mm256_extract_epi8(ix[4], 20);
        qlut_i8[k * 256 + 6 * 32 + 5] = _mm256_extract_epi8(ix[4], 28);
        qlut_i8[k * 256 + 6 * 32 + 6] = _mm256_extract_epi8(ix[6], 20);
        qlut_i8[k * 256 + 6 * 32 + 7] = _mm256_extract_epi8(ix[6], 28);
        qlut_i8[k * 256 + 6 * 32 + 8] = _mm256_extract_epi8(ix[8], 20);
        qlut_i8[k * 256 + 6 * 32 + 9] = _mm256_extract_epi8(ix[8], 28);
        qlut_i8[k * 256 + 6 * 32 + 10] = _mm256_extract_epi8(ix[10], 20);
        qlut_i8[k * 256 + 6 * 32 + 11] = _mm256_extract_epi8(ix[10], 28);
        qlut_i8[k * 256 + 6 * 32 + 12] = _mm256_extract_epi8(ix[12], 20);
        qlut_i8[k * 256 + 6 * 32 + 13] = _mm256_extract_epi8(ix[12], 28);
        qlut_i8[k * 256 + 6 * 32 + 14] = _mm256_extract_epi8(ix[14], 20);
        qlut_i8[k * 256 + 6 * 32 + 15] = _mm256_extract_epi8(ix[14], 28);

        qlut_i8[k * 256 + 6 * 32 + 16] = _mm256_extract_epi8(ix[0], 21);
        qlut_i8[k * 256 + 6 * 32 + 17] = _mm256_extract_epi8(ix[0], 29);
        qlut_i8[k * 256 + 6 * 32 + 18] = _mm256_extract_epi8(ix[2], 21);
        qlut_i8[k * 256 + 6 * 32 + 19] = _mm256_extract_epi8(ix[2], 29);
        qlut_i8[k * 256 + 6 * 32 + 20] = _mm256_extract_epi8(ix[4], 21);
        qlut_i8[k * 256 + 6 * 32 + 21] = _mm256_extract_epi8(ix[4], 29);
        qlut_i8[k * 256 + 6 * 32 + 22] = _mm256_extract_epi8(ix[6], 21);
        qlut_i8[k * 256 + 6 * 32 + 23] = _mm256_extract_epi8(ix[6], 29);
        qlut_i8[k * 256 + 6 * 32 + 24] = _mm256_extract_epi8(ix[8], 21);
        qlut_i8[k * 256 + 6 * 32 + 25] = _mm256_extract_epi8(ix[8], 29);
        qlut_i8[k * 256 + 6 * 32 + 26] = _mm256_extract_epi8(ix[10], 21);
        qlut_i8[k * 256 + 6 * 32 + 27] = _mm256_extract_epi8(ix[10], 29);
        qlut_i8[k * 256 + 6 * 32 + 28] = _mm256_extract_epi8(ix[12], 21);
        qlut_i8[k * 256 + 6 * 32 + 29] = _mm256_extract_epi8(ix[12], 29);
        qlut_i8[k * 256 + 6 * 32 + 30] = _mm256_extract_epi8(ix[14], 21);
        qlut_i8[k * 256 + 6 * 32 + 31] = _mm256_extract_epi8(ix[14], 29);

        qlut_i8[k * 256 + 7 * 32 + 0] = _mm256_extract_epi8(ix[0], 22);
        qlut_i8[k * 256 + 7 * 32 + 1] = _mm256_extract_epi8(ix[0], 30);
        qlut_i8[k * 256 + 7 * 32 + 2] = _mm256_extract_epi8(ix[2], 22);
        qlut_i8[k * 256 + 7 * 32 + 3] = _mm256_extract_epi8(ix[2], 30);
        qlut_i8[k * 256 + 7 * 32 + 4] = _mm256_extract_epi8(ix[4], 22);
        qlut_i8[k * 256 + 7 * 32 + 5] = _mm256_extract_epi8(ix[4], 30);
        qlut_i8[k * 256 + 7 * 32 + 6] = _mm256_extract_epi8(ix[6], 22);
        qlut_i8[k * 256 + 7 * 32 + 7] = _mm256_extract_epi8(ix[6], 30);
        qlut_i8[k * 256 + 7 * 32 + 8] = _mm256_extract_epi8(ix[8], 22);
        qlut_i8[k * 256 + 7 * 32 + 9] = _mm256_extract_epi8(ix[8], 30);
        qlut_i8[k * 256 + 7 * 32 + 10] = _mm256_extract_epi8(ix[10], 22);
        qlut_i8[k * 256 + 7 * 32 + 11] = _mm256_extract_epi8(ix[10], 30);
        qlut_i8[k * 256 + 7 * 32 + 12] = _mm256_extract_epi8(ix[12], 22);
        qlut_i8[k * 256 + 7 * 32 + 13] = _mm256_extract_epi8(ix[12], 30);
        qlut_i8[k * 256 + 7 * 32 + 14] = _mm256_extract_epi8(ix[14], 22);
        qlut_i8[k * 256 + 7 * 32 + 15] = _mm256_extract_epi8(ix[14], 30);

        qlut_i8[k * 256 + 7 * 32 + 16] = _mm256_extract_epi8(ix[0], 23);
        qlut_i8[k * 256 + 7 * 32 + 17] = _mm256_extract_epi8(ix[0], 31);
        qlut_i8[k * 256 + 7 * 32 + 18] = _mm256_extract_epi8(ix[2], 23);
        qlut_i8[k * 256 + 7 * 32 + 19] = _mm256_extract_epi8(ix[2], 31);
        qlut_i8[k * 256 + 7 * 32 + 20] = _mm256_extract_epi8(ix[4], 23);
        qlut_i8[k * 256 + 7 * 32 + 21] = _mm256_extract_epi8(ix[4], 31);
        qlut_i8[k * 256 + 7 * 32 + 22] = _mm256_extract_epi8(ix[6], 23);
        qlut_i8[k * 256 + 7 * 32 + 23] = _mm256_extract_epi8(ix[6], 31);
        qlut_i8[k * 256 + 7 * 32 + 24] = _mm256_extract_epi8(ix[8], 23);
        qlut_i8[k * 256 + 7 * 32 + 25] = _mm256_extract_epi8(ix[8], 31);
        qlut_i8[k * 256 + 7 * 32 + 26] = _mm256_extract_epi8(ix[10], 23);
        qlut_i8[k * 256 + 7 * 32 + 27] = _mm256_extract_epi8(ix[10], 31);
        qlut_i8[k * 256 + 7 * 32 + 28] = _mm256_extract_epi8(ix[12], 23);
        qlut_i8[k * 256 + 7 * 32 + 29] = _mm256_extract_epi8(ix[12], 31);
        qlut_i8[k * 256 + 7 * 32 + 30] = _mm256_extract_epi8(ix[14], 23);
        qlut_i8[k * 256 + 7 * 32 + 31] = _mm256_extract_epi8(ix[14], 31);
    }

    *lut_scales = scales;
    *lut_biases = biases;
#endif
    return 0;
}

inline int32_t tbl_g4_int8_int32_update_impl(int32_t m, int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);

    const int KK = BK / 2;
    __m256i vec_lut[2 * KK];
    // __m256i vec_lut_sec[KK];
#pragma unroll
    // each K has 128i, so k * 16 * (int8)
    for (int k = 0; k < KK; k++) {
        // vec_lut[k] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 8 * k));
        __m256i vec_k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(lut + 32 * k));
        __m128i vec_k_fir = _mm256_castsi256_si128(vec_k);
        __m128i vec_k_sec = _mm256_extracti128_si256(vec_k, 1);
        vec_lut[2 * k] = _mm256_set_m128i(vec_k_fir, vec_k_fir);
        vec_lut[2 * k + 1] = _mm256_set_m128i(vec_k_sec, vec_k_sec);
    }

#pragma unroll
    for (int i = 0; i < m / 2; i += 16) {
        // each 4 num / 4 * 8 = 32 num
        // printf("i:%d\n", i);
        __m256i vec_c0 = _mm256_set1_epi16(0);
        __m256i vec_c1 = _mm256_set1_epi16(0);
#pragma unroll
        // KK / 4 for 32 row each row 8index
        for (int k = 0; k < KK / 8; k++) {
            // 256i in a (int8) -> 32 int8  -> 64 int4 -> 64 index -> 192num -> 32row each row 6num
            // 256i * 2 -> 32 row each row 4index(12num) for signindex
            // 256i in signindex -> 128index -> 32row each row 4index
            // 256i in k (int8) -> 8 * 16 int8  -> (1 index) * 16 possibilities
            // 64 int4
            // printf("%d\n", i * KK / 4 + k * 32);
            // 16 * 16
            #pragma unorll
            for (int j = 0; j < 4; j++) {
                // printf("%d\n", i * KK + k * 32 * 4 + j * 32);

                __m256i vec_a = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK + k * 32 * 4 + j * 32));

                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                // printf("%d\n", 2 * k * 8 + j * 4 + 0);
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(vec_lut[2 * k * 8 + j * 4 + 0], vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(vec_lut[2 * k * 8 + j * 4 + 1], vec_v_top);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(vec_lut[2 * k * 8 + j * 4 + 2], vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(vec_lut[2 * k * 8 + j * 4 + 3], vec_v_bot);

                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);
                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo); 
            }
        }

        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 8));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 16));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 24));

        // 8 * int32
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 8), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 16), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 24), vec_gc3);
    }
#endif
    return 0;
}

 int32_t qgemm_lut_t1_int8_m256_k8640_n1_b2(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C) {
  alignas(32) uint32_t CBits[BM];
  memset(&(CBits[0]), 0, BM * sizeof(int32_t));
#pragma unroll
  // compute 96 nums in one loop
  // 96 = 8640 / 96
  // 16 * BM = 96 * BM / 3 / 2
  // 512 = 96 / 3 * 32
  // 8 * BM = 96 / 3 * BM / 8
  for (int32_t k_outer = 0; k_outer < K / BK; ++k_outer) {
    tbl_g4_int8_int32_update_impl(BM, (&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK / 2 / 2 * BM)])));
  }

#pragma unroll
  for (int i = 0; i < BM; i++) {
    ((float*)C)[i] = (float)(((int32_t*)CBits)[i]) * ((float*)LUT_Scales)[0] * ((float*)Scales)[0];
  }

  if (0 != 0) {
    return -1;
  }
  return 0;
}

extern "C"
 int32_t preprocessor_t1_int8_m6400_k8640_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT) {
  
  partial_max_reset((&(((float*)LUT_Scales)[0])));
  // 8640 / 24 == 200
//   for (int32_t k_outer = 0; k_outer < K / 24; ++k_outer) {
    partial_max(K, (&(((float*)LUT_Scales)[0])), (&(((float*)B)[0])));
//   }
  lut_ctor_g4_int8_k0_b2(K, (&(((int8_t*)QLUT)[0])), (&(((float*)B)[0])), (&(((float*)LUT_Scales)[0])), (&(((float*)LUT_Biases)[0])));
  return 0;
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void quantize_row_i8_s(const float * x, void * y, int64_t n, float* act_scales, int32_t* act_sums) {
// void quantize_row_i8_s(const float * x, void * y, int64_t n, float* act_scales) {
    int8_t* dst = (int8_t*)y;
    double min = 0.00001;
    double max = min;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, (double)fabs((double)x[i]));
    }
    float s = 127 / max;
    act_scales[0] = s;
    float temp;
    int32_t sum = 0;
    for (int i = 0; i < n; ++i) {
        temp = round((double)(x[i] * s));
        if (temp >  127) temp = 127;
        if (temp < -128) temp = -128;
        sum += temp;
        dst[i] = (int8_t)(temp);
    }
    act_sums[0] = sum;
}

#define QK_I2_S 128

#if defined(__AVX2__)
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#endif

void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t*)vx;
    const int8_t  *    y = (int8_t*)vy;

#if defined(__AVX2__)

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i accu = _mm256_setzero_si256();

    for (int i=0; i < group32_num; i++){
        __m256i accu32 = _mm256_setzero_si256();
        for (int j=0; j < 32; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + i * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
    }

    for (int i = 0; i < groupla_num; i++){
        __m256i accula = _mm256_setzero_si256();
        for (int j = 0; j < la_num; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + group32_num * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
    }
    int sumi = hsum_i32_8(accu);
    *s = (float)sumi;
#else

    int sumi = 0;

    for (int i = 0; i < n / 4; i++) {
        const int8_t* weight = (const int8_t *)(i2s_i8s + x[i]);
        sumi += (int)y[i*4+0] * weight[0];
        sumi += (int)y[i*4+1] * weight[1];
        sumi += (int)y[i*4+2] * weight[2];
        sumi += (int)y[i*4+3] * weight[3];
    }
    *s = (float)sumi;
#endif
}

#define QK_I2 128

void quantize_i2_s(const float * src, void * dst) {
    // 2 bits per weight
    int n = K;

    // f32 -> q8
    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n * sizeof(uint8_t));

    // q8 -> 0, 1, 2
    //       |  |  |
    //      -1, 0, 1

    // uint8_t* i2_weight = (uint8_t*)dst;
    // for (int i=0; i<n; i++) {
    //     int group_idx = i / 4;
    //     int group_pos = i % 4;
    //     uint8_t temp = (q8[i] << (6 - 2 * group_pos));
    //     i2_weight[group_idx] |= temp;
    // }

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / QK_I2; i++) {
        for (int j = 0; j < QK_I2; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;            
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    // 32B for alignment
}


void matrixMultiply(int N, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0;
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

int main() {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    const int N = 1;
    const int g = 3;

    float* B = (float *)malloc(N * K * sizeof(float));
    for (int i = 0; i < N * K; i++) {
        // B[i] = 1;
        B[i] = rand() % (1000 + 1) / (float)(1000);
    }

    float* oA = (float *)malloc(M * K * sizeof(float));
    std::ifstream ori_in("origin_3200_8640_lut3.weight", std::ios::binary);
    ori_in.seekg(0, std::ios::end);
    std::streampos ori_fileSize = ori_in.tellg();
    ori_in.seekg(0, std::ios::beg);
    ori_in.read(reinterpret_cast<char*>(oA), ori_fileSize);
    ori_in.close();

    // 5bit -> 3value
    // one index -> 3num
    // one index -> 5bit
    // 3^2 possibilities stored in 8 index, 0 for 0
    // 2^4 store 3^2 possibitiles, kind of waste space but helpful for speed
    // uint8_t* A = (uint8_t *)malloc(M * K / 4 * sizeof(uint8_t));

    // // uint8_t* sign = (uint8_t *)malloc(M * K / 12 * sizeof(uint8_t));
    // std::ifstream tra_in("trans_3200_8640_lut3.weight", std::ios::binary);
    // tra_in.seekg(0, std::ios::end);
    // std::streampos tra_fileSize = tra_in.tellg();
    // tra_in.seekg(0, std::ios::beg);
    // tra_in.read(reinterpret_cast<char*>(A), tra_fileSize);
    // tra_in.close();
    // for (int i = 0; i < M * K; i++) {
    //     oA[i] = -0.22;
    // }
    // uint8_t* sign = ((uint8_t *)(A)) + M * K / 3 / 2;
    // for (int i = 0; i < 10; i++) {
    //     printf("At:%d\n", A[i]);
    // }
    // for (int i = M * K / 6 - 10; i < M * K / 6; i++) {
    //     printf("Ab:%d\n", A[i]);
    // }
    // for (int i = 0; i < 10; i++) {
    //     printf("signt:%d\n", sign[i]);
    // }
    // for (int i = M * K / 12 - 10; i < M * K / 12; i++) {
    //     printf("signb:%d\n", sign[i]);
    // }
    // for (int i = 0; i < M * K / 4; i++) {
    //     A[i] = 0x55;
    // }
    // // 1 -> + 0 -> -
    // for (int i = 0; i < M * K / 24; i++) {
    //     sign[i] = 0xff;
    // }

    float* ori_C = (float *)malloc(N * M * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        ori_C[i] = 0;
    }

    float* C = (float *)malloc(N * M * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        C[i] = 0;
    }

    matrixMultiply(1, oA, B, ori_C);

    // for (int i=0; i<10; i++) {
    //     printf("%f ", ori_C[i]);
    // }
    // printf("\n");

    float Scales[1] = {0.22f};
    // float LUT_Scales[1];
    // float LUT_Biases[1];

    int32_t* act_sums = (int32_t*)malloc(sizeof(int32_t) * N);
    float* act_scales = (float*)malloc(sizeof(float) * N);

    int8_t* qy = (int8_t*)malloc(sizeof(int8_t) * N * K);

    for (int i=0; i<N; i++) {
        quantize_row_i8_s(B, qy, K, act_scales, act_sums);
    }

    uint8_t* qx = (uint8_t*)malloc(sizeof(uint8_t) * M * K / 4);
    for (int i=0; i<M; i++) {
        quantize_i2_s(oA + i * K, qx + i * K / 4);
    }

    for (int i=0; i<M; i++) {
        ggml_vec_dot_i2_i8_s(K, C + i, 0, qx + i * K / 4, 0, qy, 0, 0);
    }

    for (int i=0; i<M; i++) {
        C[i] = (C[i] - act_sums[0]) / act_scales[0] * Scales[0];
    }

    // // int16
    // int8_t* QLUT = (int8_t *)malloc(N * 16 * (K / 2) * sizeof(int8_t) * 2);

    // double preprocess_time = 0;
    // double gemm_time = 0;

    // // for (int i = 0; i < 1000; i++) {
    // auto pre_start = std::chrono::high_resolution_clock::now();
    // preprocessor_t1_int8_m6400_k8640_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);
    // auto pre_end   = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed = pre_end - pre_start;
    // preprocess_time += double(elapsed.count());

    // const int n_tile_num = M / BM;
    // const int nth = 1;
    // const int ith = 0;
    // const int w_size           = M * K / (2 * 2); // int8 
    // const int sign_size        = 0; //int8
    // const int lut_size         = N * 16 * (K / 2) * 2; // int8
    // const int c_size           = N * M; // float
    // const int w_tile_size      = w_size / n_tile_num;
    // const int lut_tile_size    = lut_size / n_tile_num;
    // const int sign_tile_size   = sign_size / n_tile_num;
    // const int c_tile_size      = c_size / n_tile_num;

    // const int th_tile_num = (n_tile_num + nth - 1) / nth;
    // const int th_tile_beg = ith * th_tile_num;
    // const int th_tile_end = n_tile_num;

    // auto gemm_start = std::chrono::high_resolution_clock::now();
    // for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
    //     const int w_offset          = i_tile * w_tile_size;
    //     const int sign_offset       = i_tile * sign_tile_size;
    //     const int scales_offset     = 0;

    //     const int qlut_offset       = i_tile * lut_tile_size;
    //     const int lut_scales_offset = 0;
    //     const int dst_offset        = i_tile * c_tile_size;

    //     qgemm_lut_t1_int8_m256_k8640_n1_b2(A + w_offset, nullptr, QLUT, Scales, LUT_Scales, LUT_Biases, C + dst_offset);
    // }
    // auto gemm_end   = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> gemm_elapsed = gemm_end - gemm_start;       
    // gemm_time += double(gemm_elapsed.count());
    // // }
    // std::cout << "preprocess:" << preprocess_time << "ms" << std::endl;
    // std::cout << "gemm:" << gemm_time << "ms" << std::endl; 

    for (int i=0; i<M; i++) {
        // printf("%f ", C[i]);
        if (fabs(ori_C[i] - C[i]) > 0.1){
            printf("index:%d\n", i);
            printf("ori:%f\n", ori_C[i]);
            printf("tra:%f\n", C[i]);
        }
    }
    printf("\n");
    printf("done\n");
}