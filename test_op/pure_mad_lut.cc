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

#define M 4096
#define K 8192
#define N 1
#define BM 128
#define BK3 96
#define BK2 32

static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

#if __AVX__ || __AVX2__ || __AVX512F__
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#endif

#if defined __AVX2__

inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi32(va, vb);
    *vh = _mm256_unpackhi_epi32(va, vb);
}

inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi64(va, vb);
    *vh = _mm256_unpackhi_epi64(va, vb);
}

inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));
}


inline void Transpose_8_8(
    __m256i *v0,
    __m256i *v1,
    __m256i *v2,
    __m256i *v3,
    __m256i *v4,
    __m256i *v5,
    __m256i *v6,
    __m256i *v7)
{
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;

    _mm256_merge_epi32(*v0, *v1, &w0, &w1);
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);

    _mm256_merge_epi64(w0, w2, &x0, &x1);
    _mm256_merge_epi64(w1, w3, &x2, &x3);
    _mm256_merge_epi64(w4, w6, &x4, &x5);
    _mm256_merge_epi64(w5, w7, &x6, &x7);

    _mm256_merge_si128(x0, x4, v0, v1);
    _mm256_merge_si128(x1, x5, v2, v3);
    _mm256_merge_si128(x2, x6, v4, v5);
    _mm256_merge_si128(x3, x7, v6, v7);
}

#endif

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
    float scales = 127 / _mm_cvtss_f32(max1);
    *lut_scales = scales;
#endif

    return 0;
}

int32_t partial_max_reset(void* lut_scales_) {
    float_type* lut_scales = (float_type*)lut_scales_;
    *lut_scales = 0.0;
    return 0;
}

inline int32_t three_lut_preprocess(int32_t act_k, int8_t* qlut, float_type* b, float_type* lut_scales, float_type* lut_biases) {
#if defined __AVX2__
    __m256 vec_lut[16];
    float biases = 0.0;
    const __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);
    float scales = *lut_scales;
    __m256i shuffle_mask = _mm256_set_epi8(
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
                                            );
#pragma unroll
    for (int k = 0; k < act_k / 24; ++k) {
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 24 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 24 + 1, vec_bi, 1);
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 24 + 2, vec_bi, 1);

        __m256i vec_b0i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b2i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b2, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        vec_lut[15] = _mm256_setzero_si256();

        vec_lut[14] = _mm256_setzero_si256();

        // 1 1 1
        vec_lut[13] = vec_b0i;
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b1i);
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b2i);

        // 1 1 0
        vec_lut[12] = vec_b0i;
        vec_lut[12] = _mm256_add_epi32(vec_lut[12], vec_b1i);

        // 1 1 -1
        vec_lut[11] = vec_b0i;
        vec_lut[11] = _mm256_add_epi32(vec_lut[11], vec_b1i);
        vec_lut[11] = _mm256_sub_epi32(vec_lut[11], vec_b2i);

        // 1 0 1
        vec_lut[10] = vec_b0i;
        vec_lut[10] = _mm256_add_epi32(vec_lut[10], vec_b2i);

        // 1 0 0
        vec_lut[9] = vec_b0i;

        // 1 0 -1
        vec_lut[8] = vec_b0i;
        vec_lut[8] = _mm256_sub_epi32(vec_lut[8], vec_b2i);

        // 1 -1 1
        vec_lut[7] = vec_b0i;
        vec_lut[7] = _mm256_sub_epi32(vec_lut[7], vec_b1i);
        vec_lut[7] = _mm256_add_epi32(vec_lut[7], vec_b2i);

        // 1 -1 0
        vec_lut[6] = vec_b0i;
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1i);

        // 1 -1 -1
        vec_lut[5] = vec_b0i;
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b1i);
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b2i);

        // 0 1 1
        vec_lut[4] = vec_b1i;
        vec_lut[4] = _mm256_add_epi32(vec_lut[4], vec_b2i);

        // 0 1 0
        vec_lut[3] = vec_b1i;

        // 0 1 -1
        vec_lut[2] = vec_b1i;
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b2i);

        // 0 0 1
        vec_lut[1] = vec_b2i;

        // 0 0 0
        vec_lut[0] = _mm256_setzero_si256();

// #pragma unroll
//         for (int g = 0; g < 14; ++g) {
//             vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(t_scales));
//         }

        __m256i ix[16];

#pragma unroll
        for (int g = 0; g < 16; ++g) {
            ix[g] = vec_lut[g];
        }

        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
        }

        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);

    }

    *lut_scales = scales;
    *lut_biases = biases;
#endif
    return 0;
}

inline int32_t two_lut_preprocess(int32_t act_k, int8_t* qlut, float_type* b, float_type* lut_scales, float_type* lut_biases) {
#if defined __AVX2__
    __m256 vec_lut[16];
    float biases = 0.0;
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);
    float scales = *lut_scales;
    __m256i shuffle_mask = _mm256_set_epi8(
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
                                            );
#pragma unroll
    for (int k = 0; k < act_k / 16; ++k) {
        __m256 vec_b0f = _mm256_i32gather_ps(b + k * 16 + 0, vec_bi, 1);
        __m256 vec_b1f = _mm256_i32gather_ps(b + k * 16 + 1, vec_bi, 1);

        __m256i vec_b0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        vec_lut[15] = _mm256_setzero_si256();

        vec_lut[14] = _mm256_setzero_si256();

        vec_lut[13] = _mm256_setzero_si256();

        vec_lut[12] = _mm256_setzero_si256();

        vec_lut[11] = _mm256_setzero_si256();

        vec_lut[10] = _mm256_setzero_si256();

        vec_lut[9] = _mm256_setzero_si256();

        // 1 1
        vec_lut[8] = vec_b0;
        vec_lut[8] = _mm256_add_epi32(vec_lut[8], vec_b1);

        // 1 0
        vec_lut[7] = vec_b0;

        // 1 -1
        vec_lut[6] = vec_b0;
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1);

        // 0 1
        vec_lut[5] = vec_b1;

        // 0 0
        vec_lut[4] = _mm256_setzero_si256();

        // 0 -1
        vec_lut[3] = _mm256_setzero_si256();
        vec_lut[3] = _mm256_sub_epi32(vec_lut[3], vec_b1);

        // -1 1
        vec_lut[2] = _mm256_setzero_si256();
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b0);
        vec_lut[2] = _mm256_add_epi32(vec_lut[2], vec_b1);

        // -1 0
        vec_lut[1] = _mm256_setzero_si256();
        vec_lut[1] = _mm256_sub_epi32(vec_lut[1], vec_b0);

        // -1 -1
        vec_lut[0] = _mm256_setzero_si256();
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b0);
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b1);

        __m256i ix[16];
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            ix[g] = vec_lut[g];
        }

        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
        }

        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);

    }

    *lut_scales = scales;
    *lut_biases = biases;
#endif
    return 0;
}

inline int32_t two_gemm_impl(int32_t m, int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);

    const int KK = BK2 / 2;
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

inline int32_t three_gemm_impl(int32_t m, int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    // compute 96 num
    // one K for 3 num / 32 K
    const int KK = BK3 / 3;
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
            __m256i vec_sign = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 4 + k * 32));

            #pragma unorll
            for (int j = 0; j < 4; j++) {

                __m256i vec_a = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK + k * 32 * 4 + j * 32));

                __m256i vec_sign_left_hi = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * j)), 15);
                __m256i vec_sign_left_lo = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * j + 1)), 15);

                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);

                __m256i vec_v_top_fir = _mm256_shuffle_epi8(vec_lut[2 * k * 8 + j * 4 + 0], vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(vec_lut[2 * k * 8 + j * 4 + 1], vec_v_top);

                __m256i vec_sign_right_hi = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * j + 2)), 15);
                __m256i vec_sign_right_lo = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * j + 3)), 15);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(vec_lut[2 * k * 8 + j * 4 + 2], vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(vec_lut[2 * k * 8 + j * 4 + 3], vec_v_bot);

                __m256i vec_v_top_lo = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec), vec_sign_left_lo), vec_sign_left_lo);
                __m256i vec_v_top_hi = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec), vec_sign_left_hi), vec_sign_left_hi);
                __m256i vec_v_bot_lo = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec), vec_sign_right_lo), vec_sign_right_lo);
                __m256i vec_v_bot_hi = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec), vec_sign_right_hi), vec_sign_right_hi);
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

// template<int three_K>
 int32_t three_qgemm_lut(int k, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C) {
  alignas(32) uint32_t CBits[BM];
  memset(&(CBits[0]), 0, BM * sizeof(int32_t));
#pragma unroll
  // compute 96 nums in one loop
  // 96 = 8640 / 96
  // 16 * BM = 96 * BM / 3 / 2
  // 512 = 96 / 3 * 32
  // 8 * BM = 96 / 3 * BM / 8
  for (int32_t k_outer = 0; k_outer < k / BK3; ++k_outer) {
    three_gemm_impl(BM, (&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK3 / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BK3 / 3 / 2 * BM)])), (&(((uint8_t*)sign)[(k_outer * BK3 / 3 / 8 * BM)])));
  }

#pragma unroll
  for (int i = 0; i < BM; i++) {
    ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i]);
    // ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i]);
    // ((float*)C)[i] = (float)(((int32_t*)C)[i]) * ((float*)(LUT_Scales))[0] * ((float*)(Scales))[0];
  }

  if (0 != 0) {
    return -1;
  }
  return 0;
}

// template<int two_K>
 int32_t two_qgemm_lut(int k, void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C) {
  alignas(32) uint32_t CBits[BM];
  memset(&(CBits[0]), 0, BM * sizeof(int32_t));
#pragma unroll
  // compute 96 nums in one loop
  // 96 = 8640 / 96
  // 16 * BM = 96 * BM / 3 / 2
  // 512 = 96 / 3 * 32
  // 8 * BM = 96 / 3 * BM / 8
  for (int32_t k_outer = 0; k_outer < k / BK2; ++k_outer) {
    two_gemm_impl(BM, (&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM)])));
  }

#pragma unroll
  for (int i = 0; i < BM; i++) {
    // ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i]);
    ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i]);
    ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)(LUT_Scales))[0] * ((float*)(Scales))[0];
  }

  if (0 != 0) {
    return -1;
  }
  return 0;
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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
        int v = nearest_int(x[i] * s);
        if (v >  127) v = 127;
        if (v < -128) v = -128;
        sum += v;
        dst[i] = (int8_t)(v);
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

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

#if defined(__AVX2__)

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

#elif defined(__ARM_NEON)

    int32x4_t accu = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3);

    for (int i=0; i < group32_num; i++) {

#if defined(__ARM_FEATURE_DOTPROD)

#else
        int16x8_t accu32_0 = vdupq_n_s16(0);
        int16x8_t accu32_1 = vdupq_n_s16(0);
#endif

        for (int j=0; j < 32; j++) {
            uint8x16_t xq8_6 = vld1q_u8(x + i * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(x + i * 32 * 32 + j * 32 + 16);
            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            const int8x16_t yq8_0 = vld1q_s8(y + i * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + i * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + i * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + i * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(y + i * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(y + i * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(y + i * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(y + i * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            accu = vdotq_s32(accu, q8_0, yq8_0);
            accu = vdotq_s32(accu, q8_1, yq8_1);
            accu = vdotq_s32(accu, q8_2, yq8_2);
            accu = vdotq_s32(accu, q8_3, yq8_3);
            accu = vdotq_s32(accu, q8_4, yq8_4);
            accu = vdotq_s32(accu, q8_5, yq8_5);
            accu = vdotq_s32(accu, q8_6, yq8_6);
            accu = vdotq_s32(accu, q8_7, yq8_7);
#else
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_3), vget_low_s8(yq8_3));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_5), vget_low_s8(yq8_5));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_5), vget_high_s8(yq8_5));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_7), vget_low_s8(yq8_7));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
        }

#if defined(__ARM_FEATURE_DOTPROD)

#else
        accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu32_0)));
        accu = vaddq_s32(accu, vmovl_high_s16(accu32_0));
        accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu32_1)));
        accu = vaddq_s32(accu, vmovl_high_s16(accu32_1));
#endif
    }

    for (int i = 0; i < groupla_num; i++){
#if defined(__ARM_FEATURE_DOTPROD)

#else
        int16x8_t accu32_0 = vdupq_n_s16(0);
        int16x8_t accu32_1 = vdupq_n_s16(0);
#endif
        for (int j = 0; j < la_num; j++) {
            uint8x16_t xq8_6 = vld1q_u8(x + i * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(x + i * 32 * 32 + j * 32 + 16);
            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            const int8x16_t yq8_0 = vld1q_s8(y + i * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + i * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + i * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + i * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(y + i * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(y + i * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(y + i * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(y + i * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            accu = vdotq_s32(accu, q8_0, yq8_0);
            accu = vdotq_s32(accu, q8_1, yq8_1);
            accu = vdotq_s32(accu, q8_2, yq8_2);
            accu = vdotq_s32(accu, q8_3, yq8_3);
            accu = vdotq_s32(accu, q8_4, yq8_4);
            accu = vdotq_s32(accu, q8_5, yq8_5);
            accu = vdotq_s32(accu, q8_6, yq8_6);
            accu = vdotq_s32(accu, q8_7, yq8_7);
#else
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_3), vget_low_s8(yq8_3));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_5), vget_low_s8(yq8_5));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_5), vget_high_s8(yq8_5));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_7), vget_low_s8(yq8_7));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
        }
#if defined(__ARM_FEATURE_DOTPROD)

#else
        accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accula_0)));
        accu = vaddq_s32(accu, vmovl_high_s16(accula_0));
        accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accula_1)));
        accu = vaddq_s32(accu, vmovl_high_s16(accula_1));
#endif
    }

    int sumi = vaddlvq_s32(accu);
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

    memset(dst, 0, n * sizeof(uint8_t) / 4);

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

void float_act_quant(float* B, float* dst) {
    double min = 0.00001;
    double max = min;
    for (int i = 0; i < K; ++i) {
        max = MAX(max, (double)fabs((double)B[i]));
    }
    float s = 127 / max;
    float temp;
    for (int i = 0; i < K; ++i) {
        temp = round((double)(B[i] * s));
        if (temp >  127) temp = 127;
        if (temp < -128) temp = -128;
        dst[i] = temp / s;
    }
}

#define QK_K 256
void quantize_row_q8_K_ref(const float *x, int8_t * qy, float* d, int16_t* bsums) {
    const int64_t nb = K / QK_K;

    for (int i = 0; i < nb; i++) {
        int8_t* t_qy = qy + i * QK_K;
        float* t_d = d + i;
        int16_t* t_bsums = bsums + i * QK_K / 16;
        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            t_qy[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += t_qy[j*16 + ii];
            }
            t_bsums[j] = sum;
        }
        t_d[0] = 1/iscale;
        x += QK_K;
    }
}

void quantize_row_tq2_0_ref(const float * x, uint8_t * qx, float* d_) {
    const int64_t nb = K / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        uint8_t* t_qx = qx + i * QK_K / 4;
        float* t_d_ = d_ + i;
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        t_d_[0] = d;

        for (size_t j = 0; j < QK_K / 4; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    // -1, 0, 1 -> 0, 1, 2
                    int xi = lroundf(x[m + n*32] * id) + 1;
                    q += (xi & 3) << (2*n);
                }
                t_qx[j + m] = q;
            }
            x += 4*32;
        }
    }
}

void ggml_vec_dot_tq2_0_q8_K(int n, float * s, const uint8_t * x, float* dx, const int8_t * y, float* dy, int16_t* bsums) {

    const int nb = n / QK_K;

#if defined(__AVX2__)
    __m256 sumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const int8_t* t_y = y + i * QK_K;
        float* t_dy = dy + i;
        float* t_dx = dx + i;
        int16_t* t_bsums = bsums + i * QK_K / 16;
        const uint8_t* t_x = x + i * QK_K / 4;
        // 16-bit sums, because 256*127 still fits
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

        for (size_t j = 0; j < QK_K / 4; j += 32) {
            __m256i qx0 = _mm256_loadu_si256((const __m256i *) (t_x + j));
            __m256i qx1 = _mm256_srli_epi16(qx0, 2);
            __m256i qx2 = _mm256_srli_epi16(qx0, 4);
            __m256i qx3 = _mm256_srli_epi16(qx0, 6);

            // 0, 1, 2 (should not be 3)
            qx0 = _mm256_and_si256(qx0, _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(qx1, _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(qx2, _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(qx3, _mm256_set1_epi8(3));

            const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (t_y + j*4 +  0));
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (t_y + j*4 + 32));
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (t_y + j*4 + 64));
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (t_y + j*4 + 96));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);

            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(qx0, qx1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(qx2, qx3));
        }

        const __m256i ysum = _mm256_loadu_si256((const __m256i *) t_bsums);
        const __m256 d = _mm256_set1_ps(t_dy[0] * t_dx[0]);

        sumi0 = _mm256_add_epi16(sumi0, sumi1);
        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }

    *s = hsum_float_8(sumf);
#endif
}

void matrixMultiply(const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[j * M + i] = 0.0;
            for (int k = 0; k < K; ++k) {
                C[j * M + i] += A[i * K + k] * B[j * K + k];
            }
        }
    }
}

void float_act_quant(float* B, int32_t* dst, float* act_scale) {
    double min = 0.00001;
    double max = min;
    for (int i = 0; i < K; ++i) {
        max = MAX(max, (double)fabs((double)B[i]));
    }
    float s = 127 / max;
    act_scale[0] = s;
    for (int i = 0; i < K; ++i) {
        int v = nearest_int(B[i] * s);
        if (v >  127) v = 127;
        if (v < -128) v = -128;
        dst[i] = (int32_t)v;
    }
}

void weight_quant(float* A, int32_t* dst, float i2_scale) {
    for (int i = 0; i < M * K; ++i) {
        if (fabs((double)(A[i])) < 1e-6) {
            dst[i] = 0;
            continue;
        } else {
            dst[i] = (double)A[i] * i2_scale > 0 ? 1 : -1;
        }
    }
}

void matrixMultiply_int(const int32_t* A, const int32_t* B, int32_t* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[j * M + i] = 0.0;
            for (int k = 0; k < K; ++k) {
                C[j * M + i] += A[i * K + k] * B[j * K + k];
            }
        }
    }
}

void float_compute(float* A, float* B, float* C, float i2_scale) {
    for (int i = 0; i < M * 1; i++) {
        C[i] = 0;
    }
    int32_t* int_C = (int32_t*)malloc(1 * M * sizeof(int32_t));
    for (int i = 0; i < M * 1; i++) {
        int_C[i] = 0;
    }
    int32_t* int_B = (int32_t*)malloc(1 * K * sizeof(int32_t));
    int32_t* int_A = (int32_t*)malloc(M * K * sizeof(int32_t));
    float* act_scale = (float*)malloc(sizeof(float));
    float_act_quant(B, int_B, act_scale);
    weight_quant(A, int_A, i2_scale);
    matrixMultiply_int(int_A, int_B, int_C);
    for (int i=0; i < M * 1; i++) {
        C[i] = int_C[i] / act_scale[0] * i2_scale;
    }
}


void i2_s_compute(float* A, float* B, float* C) {
    for (int i = 0; i < M * 1; i++) {
        C[i] = 0;
    }
    int32_t* act_sums = (int32_t*)malloc(sizeof(int32_t) * N);
    float* act_scales = (float*)malloc(sizeof(float) * N);

    int8_t* qy = (int8_t*)malloc(sizeof(int8_t) * N * K);

    for (int i=0; i<N; i++) {
        quantize_row_i8_s(B, qy, K, act_scales, act_sums);
    }

    uint8_t* qx = (uint8_t*)malloc(sizeof(uint8_t) * M * K / 4 + sizeof(float));
    for (int i=0; i<M; i++) {
        quantize_i2_s(A + i * K, qx + i * K / 4);
    }

    for (int i=0; i<M; i++) {
        ggml_vec_dot_i2_i8_s(K, C + i, 0, qx + i * K / 4, 0, qy, 0, 0);
    }

    for (int i=0; i<M; i++) {
        C[i] = (C[i] - act_sums[0]) / act_scales[0] * ((float*)(qx + M * K / 4))[0];
    }
}

void tq20_compute(float* A, float* B, float* C) {
    for (int i = 0; i < M * 1; i++) {
        C[i] = 0;
    }
    int16_t* bsums = (int16_t*)malloc(sizeof(int16_t) * K / 16);
    float* dy = (float*)malloc(sizeof(float) * K / 256);
    int8_t* tq20_qy = (int8_t*)malloc(sizeof(int8_t) * K);
    float* dx = (float*)malloc(sizeof(float) * M * K / 256);
    uint8_t* tq20_qx = (uint8_t*)malloc(sizeof(uint8_t) * M * K / 4);

    for (int i=0; i<N; i++) {
        quantize_row_q8_K_ref(B, tq20_qy, dy, bsums);
    }

    for (int i=0; i<M; i++) {
        quantize_row_tq2_0_ref(A + i * K, tq20_qx + i * K / 4, dx + i * K / 256);
    }

    for (int i=0; i<M; i++) {
        ggml_vec_dot_tq2_0_q8_K(K, C + i, tq20_qx + i * K / 4, dx + i * K / 256, tq20_qy, dy, bsums);
    }
}

// void tl_compute(uint8_t* A, float* B, float* C) {
//     for (int i = 0; i < M * 1; i++) {
//         C[i] = 0;
//     }
//     const int three_k = (int)(K / BK3) * BK3;
//     const int two_k = K - three_k;

//     uint8_t* sign = ((uint8_t *)(A)) + M * three_k / 3 / 2;
//     uint8_t* two_A = ((uint8_t *)(A)) + M * three_k / 4;

//     float Scales[1] = {0.22f};
//     float LUT_Scales[1];
//     float LUT_Biases[1];

//     int8_t* three_QLUT = (int8_t *)malloc(1 * 16 * (three_k / 3) * sizeof(int8_t) * 2);
//     int8_t* two_QLUT = (int8_t *)malloc(1 * 16 * (two_k / 2) * sizeof(int8_t) * 2);

//     partial_max_reset((&(((float*)LUT_Scales)[0])));
//   // 8640 / 24 == 200
//     partial_max(K, (&(((float*)LUT_Scales)[0])), (&(((float*)B)[0])));
//     three_lut_preprocess(three_k, (&(((int8_t*)three_QLUT)[0])), (&(((float*)B)[0])), (&(((float*)LUT_Scales)[0])), (&(((float*)LUT_Biases)[0])));
//     two_lut_preprocess(two_k, (&(((int8_t*)two_QLUT)[0])), (&(((float*)B)[three_k])), (&(((float*)LUT_Scales)[0])), (&(((float*)LUT_Biases)[0])));

//     const int n_tile_num = M / BM;
//     const int nth = 1;
//     const int ith = 0;
//     const int w_size           = M * three_k / (2 * 3); // int8 
//     const int sign_size        = M * three_k / 24; //int8
//     const int lut_size         = 1 * 16 * (three_k / 3) * 2; // int8
//     const int c_size           = 1 * M; // float
//     const int w_tile_size      = w_size / n_tile_num;
//     const int lut_tile_size    = lut_size / n_tile_num;
//     const int sign_tile_size   = sign_size / n_tile_num;
//     const int c_tile_size      = c_size / n_tile_num;

//     const int th_tile_num = (n_tile_num + nth - 1) / nth;
//     const int th_tile_beg = ith * th_tile_num;
//     const int th_tile_end = n_tile_num;

//     // auto gemm_start = std::chrono::high_resolution_clock::now();
//     for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
//         const int w_offset          = i_tile * w_tile_size;
//         const int sign_offset       = i_tile * sign_tile_size;
//         const int scales_offset     = 0;

//         const int qlut_offset       = i_tile * lut_tile_size;
//         const int lut_scales_offset = 0;
//         const int dst_offset        = i_tile * c_tile_size;

//         three_qgemm_lut(three_k, A + w_offset, sign + sign_offset, three_QLUT, Scales, LUT_Scales, LUT_Biases, C + dst_offset);
//     }

//     const int two_w_size           = M * two_k / (2 * 2); // int8 
//     const int two_lut_size         = 1 * 16 * (two_k / 2) * 2; // int8
//     const int two_w_tile_size      = two_w_size / n_tile_num;
//     const int two_lut_tile_size    = two_lut_size / n_tile_num;

//     // auto gemm_start = std::chrono::high_resolution_clock::now();
//     for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
//         const int two_w_offset          = i_tile * two_w_tile_size;

//         const int two_qlut_offset       = i_tile * two_lut_tile_size;
//         const int two_dst_offset        = i_tile * c_tile_size;

//         two_qgemm_lut(two_k, two_A + two_w_offset, two_QLUT, Scales, LUT_Scales, LUT_Biases, C + two_dst_offset);
//     }

//     // for (int i=0; i<M; i++) {
//     //     C[i] = C[i] * LUT_Scales[0] * Scales[0];
//     // }
// }

int main() {
    float* B = (float *)malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) {
        // B[i] = 1;
        B[i] = rand() % (1000 + 1) / (float)(1000);
    }

    float* oA = (float *)malloc(M * K * sizeof(float));
    std::ifstream ori_in("origin_3200_3200_lut3.weight", std::ios::binary);
    ori_in.seekg(0, std::ios::end);
    std::streampos ori_fileSize = ori_in.tellg();
    ori_in.seekg(0, std::ios::beg);
    ori_in.read(reinterpret_cast<char*>(oA), ori_fileSize);
    ori_in.close();

    uint8_t* A = (uint8_t *)malloc(M * K / 4 * sizeof(uint8_t));

    std::ifstream tra_in("trans_3200_3200_lut3.weight", std::ios::binary);
    tra_in.seekg(0, std::ios::end);
    std::streampos tra_fileSize = tra_in.tellg();
    tra_in.seekg(0, std::ios::beg);
    tra_in.read(reinterpret_cast<char*>(A), tra_fileSize);
    tra_in.close();

    float* ori_C = (float *)malloc(1 * M * sizeof(float));
    float_compute(oA, B, ori_C, 0.22f);

    // float* lut_C = (float *)malloc(1 * M * sizeof(float));
    // tl_compute(A, B, lut_C);

    float* i2_s_C = (float *)malloc(1 * M * sizeof(float));
    i2_s_compute(oA, B, i2_s_C);

    // float* tq20_C = (float *)malloc(1 * M * sizeof(float));
    // tq20_compute(oA, B, tq20_C);

    for (int i=0; i<M; i++) {
        // printf("%f ", C[i]);
        // if (fabs(ori_C[i] - lut_C[i]) > 1e-20){
            printf("index:%d\n", i);
            printf("float:%.10f\n", ori_C[i]);
            // printf("tl2:%.10f\n", lut_C[i]);
            printf("i2_s:%.10f\n", i2_s_C[i]);
            // printf("tq20:%.10f\n", tq20_C[i]);
        // }
    }
    printf("\n");
    printf("done\n");
}