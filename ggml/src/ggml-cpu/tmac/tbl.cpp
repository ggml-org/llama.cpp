#include "tbl.h"
#include "lut_ctor.h"
#include "../../common/log.h"

#include "string.h"
#include <cmath>
#include <type_traits>
#include <chrono>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <stdexcept>


#ifdef __ARM_NEON
template <int N>
struct SignedHalvingAdder {
    SignedHalvingAdder<N / 2> adder;
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k < N / 2) {
            adder.push(v, k);
            if (k == N / 2 - 1) {
                lhs = adder.get();
            }
        } else {
            adder.push(v, k - N / 2);
            if (k == N - 1) {
                lhs = vrhaddq_s8(lhs, adder.get());
            }
        }
    }

    inline int8x16_t get() {
        return lhs;
    }

    inline int16x8_t get_low() {
        return vmovl_s8(vget_low_s8(lhs));
    }

    inline int16x8_t get_high() {
        return vmovl_high_s8(lhs);
    }
};

template <>
struct SignedHalvingAdder<2> {
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs = vrhaddq_s8(lhs, v);
        }
    }

    inline int8x16_t get() {
        return lhs;
    }

    inline int16x8_t get_low() {
        return vmovl_s8(vget_low_s8(lhs));
    }

    inline int16x8_t get_high() {
        return vmovl_high_s8(lhs);
    }
};

struct SignedLongAdder {
    int16x8_t lhs_low;
    int16x8_t lhs_high;
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs_low = vaddl_s8(vget_low_s8(lhs), vget_low_s8(v));
            lhs_high = vaddl_high_s8(lhs, v);
        }
    }

    inline int16x8_t get_low() {
        return lhs_low;
    }

    inline int16x8_t get_high() {
        return lhs_high;
    }
};

template <int N>
struct SignedWideningAdder {
    SignedLongAdder adder;
    int16x8_t lhs_low;
    int16x8_t lhs_high;

    inline void push(int8x16_t v, int k) {
        if (k % 2 == 0) {
            adder.push(v, 0);
        } else {
            adder.push(v, 1);
            if (k == 1) {
                lhs_low = adder.get_low();
                lhs_high = adder.get_high();
            } else {
                lhs_low += adder.get_low();
                lhs_high += adder.get_high();
            }
        }
    }

    inline int16x8_t get_low() {
        return lhs_low;
    }

    inline int16x8_t get_high() {
        return lhs_high;
    }
};
#elif defined __AVX2__
#define extract_low_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
#define extract_high_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))
#define extract_low_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
#define extract_high_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))

template <int N>
struct SignedHalvingAdder {
    SignedHalvingAdder<N / 2> adder;
    __m256i lhs;

    inline void push(__m256i v, int k) {
        if (k < N / 2) {
            adder.push(v, k);
            if (k == N / 2 - 1) {
                lhs = adder.get();
            }
        } else {
            adder.push(v, k - N / 2);
            if (k == N - 1) {
                lhs = _mm256_avg_epu8(lhs, adder.get());
            }
        }
    }

    inline __m256i get() {
        return lhs;
    }

    inline __m256i get_low() {
        return extract_low_epi8_epi16(lhs);
    }

    inline __m256i get_high() {
        return extract_high_epi8_epi16(lhs);
    }
};

template <>
struct SignedHalvingAdder<2> {
    __m256i lhs;

    inline void push(__m256i v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs = _mm256_avg_epu8(lhs, v);
        }
    }

    inline __m256i get() {
        return lhs;
    }

    inline __m256i get_low() {
        return extract_low_epi8_epi16(lhs);
    }

    inline __m256i get_high() {
        return extract_high_epi8_epi16(lhs);
    }
};

template <int N>
struct SignedWideningAdder {
    __m256i lhs_low;
    __m256i lhs_high;

    inline void push(__m256i v, int k) {
        if (k == 0) {
            lhs_low = extract_low_epi8_epi16(v);
            lhs_high = extract_high_epi8_epi16(v);
        } else {
            lhs_low = _mm256_add_epi16(lhs_low, extract_low_epi8_epi16(v));
            lhs_high = _mm256_add_epi16(lhs_high, extract_high_epi8_epi16(v));
        }
    }

    inline __m256i get_low() {
        return lhs_low;
    }

    inline __m256i get_high() {
        return lhs_high;
    }
};

#endif

template <bool FastAggregation, int ActK>
using SignedAdder = typename std::conditional<FastAggregation, SignedHalvingAdder<ActK>, SignedWideningAdder<ActK>>::type;


template <int K>
struct mylog2 {
    enum {
        value = 1 + mylog2<K / 2>::value
    };
};

template <>
struct mylog2<0> {
    enum {
        value = -1
    };
};



template <bool has_scale, int K, int Bits>
inline int32_t tbl_g4_float_float_update_impl(int32_t m, tmac_float_type* c, tmac_float_type* lut, uint8_t* a, tmac_float_type* scales) {
#ifdef __ARM_NEON
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    uint8x16x2_t vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld2q_u8(reinterpret_cast<uint8_t*>(lut + k * 16));
    }

    for (int i = 0; i < m / 2; i += 16) {
        float16x8_t vec_c0 = vld1q_f16(c + i * 2);
        float16x8_t vec_c1 = vld1q_f16(c + i * 2 + 8);
        float16x8_t vec_c2 = vld1q_f16(c + i * 2 + 16);
        float16x8_t vec_c3 = vld1q_f16(c + i * 2 + 24);
        // Currently assume K * 4 weights share the same group of scale
        float16x8_t vec_s0 = vld1q_f16(scales + i * 2);
        float16x8_t vec_s1 = vld1q_f16(scales + i * 2 + 8);
        float16x8_t vec_s2 = vld1q_f16(scales + i * 2 + 16);
        float16x8_t vec_s3 = vld1q_f16(scales + i * 2 + 24);

#pragma unroll
        for (int k = 0; k < K; k++) {
            // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
            uint8x16_t vec_as = vld1q_u8(a + i * K + k * 16);
            uint8x16_t vec_a_bot = vandq_u8(vec_as, vec_mask);
            uint8x16_t vec_a_top = vshrq_n_u8(vec_as, 4);

            uint8x16_t vec_v_bot_low = vqtbl1q_u8(vec_lut[k].val[0], vec_a_bot);
            uint8x16_t vec_v_bot_high = vqtbl1q_u8(vec_lut[k].val[1], vec_a_bot);
            uint8x16x2_t vec_v_bot = vzipq_u8(vec_v_bot_low, vec_v_bot_high);

            uint8x16_t vec_v_top_low = vqtbl1q_u8(vec_lut[k].val[0], vec_a_top);
            uint8x16_t vec_v_top_high = vqtbl1q_u8(vec_lut[k].val[1], vec_a_top);
            uint8x16x2_t vec_v_top = vzipq_u8(vec_v_top_low, vec_v_top_high);

            if (has_scale) {
                // TODO: optimize scales
                vec_c0 += vreinterpretq_f16_u8(vec_v_bot.val[0]) * vec_s0;
                vec_c1 += vreinterpretq_f16_u8(vec_v_bot.val[1]) * vec_s1;
                vec_c2 += vreinterpretq_f16_u8(vec_v_top.val[0]) * vec_s2;
                vec_c3 += vreinterpretq_f16_u8(vec_v_top.val[1]) * vec_s3;
            } else {
                vec_c0 += vreinterpretq_f16_u8(vec_v_bot.val[0]);
                vec_c1 += vreinterpretq_f16_u8(vec_v_bot.val[1]);
                vec_c2 += vreinterpretq_f16_u8(vec_v_top.val[0]);
                vec_c3 += vreinterpretq_f16_u8(vec_v_top.val[1]);
            }
        }

        vst1q_f16(c + i * 2, vec_c0);
        vst1q_f16(c + i * 2 + 8, vec_c1);
        vst1q_f16(c + i * 2 + 16, vec_c2);
        vst1q_f16(c + i * 2 + 24, vec_c3);
    }
#endif

    return 0;
}

template<int bits>
constexpr int get_bias_scale() {
    // The bias scale will be added to the first bit
    // 15 = (1/2 + 1 + 2 + 4) / (1/2)
    // 7 = (1/2 + 1 + 2) / (1/2)
    // 3 = (1/2 + 1) / (1/2)
    // 1 = (1/2) / (1/2)
    if constexpr (bits == 4) {
        return 15;
    } else if constexpr (bits == 3) {
        return 7;
    } else if constexpr (bits == 2) {
        return 3;
    } else if constexpr (bits == 1) {
        return 1;
    } else {
        return 0;
    }
}


// When FastAggregation is enabled, FastAggregationK = ActK
// zero_points is merged into scales to maintain API
template <bool has_scale, int K, int Bits, int ActK, bool FastAggregation, bool ZeroPoint, bool OneScale>
inline int32_t tbl_g4_int8_float_update_impl(int32_t m, tmac_float_type* c, int8_t* lut, uint8_t* a, tmac_float_type* scales, tmac_float_type* lut_scales, tmac_float_type* lut_biases) {
#ifdef __ARM_NEON
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    int8x16_t vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

    SignedAdder<FastAggregation, ActK> adder_bot, adder_top;
    for (int i = 0; i < m / 2; i += 16) {
        float16x8_t vec_c0, vec_c1, vec_c2, vec_c3;

        tmac_float_type partial_sum = (tmac_float_type) -0.0f;
#pragma unroll
        for (int kk = 0; kk < K; kk += ActK) {
#pragma unroll
            for (int k = 0; k < ActK; k++) {
                // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
                uint8x16_t vec_as = vld1q_u8(a + i * K + (kk + k) * 16);
                uint8x16_t vec_a_top = vshrq_n_u8(vec_as, 4);
                uint8x16_t vec_a_bot = vandq_u8(vec_as, vec_mask);

                int8x16_t vec_v_bot_tmp = vqtbl1q_s8(vec_lut[kk + k], vec_a_bot);
                int8x16_t vec_v_top_tmp = vqtbl1q_s8(vec_lut[kk + k], vec_a_top);
                adder_bot.push(vec_v_bot_tmp, k);
                adder_top.push(vec_v_top_tmp, k);
            }

            float16x8_t vec_v_bot_low  = vcvtq_f16_s16(adder_bot.get_low());
            float16x8_t vec_v_bot_high = vcvtq_f16_s16(adder_bot.get_high());
            float16x8_t vec_v_top_low  = vcvtq_f16_s16(adder_top.get_low());
            float16x8_t vec_v_top_high = vcvtq_f16_s16(adder_top.get_high());

            tmac_float_type lut_s = lut_scales[kk / ActK];
            tmac_float_type lut_b = lut_biases[kk / ActK];

            // lut_b = -sum(xi for i in range(ActK * 4))
            if (ZeroPoint) {
                partial_sum += lut_b;
            }

            // https://arxiv.org/pdf/2106.10860.pdf
            // Fast aggregation bias: -FastAggregationK * log2(FastAggregationK) / 4 * (act_k / FastAggregationK)
            if (FastAggregation) {
                lut_s = lut_s * ActK;
                lut_b -= lut_s * (mylog2<ActK>::value / 4 * get_bias_scale<Bits>());
            }

#define lut_fma(vs, ib) \
    ((ib) % Bits) ? ((vs) * lut_s) \
                  : ((vs) * lut_s + lut_b)
            if (kk == 0) {
                vec_c0  = lut_fma(vec_v_bot_low,  (i / 4    ));
                vec_c1  = lut_fma(vec_v_bot_high, (i / 4 + 1));
                vec_c2  = lut_fma(vec_v_top_low,  (i / 4 + 2));
                vec_c3  = lut_fma(vec_v_top_high, (i / 4 + 3));
            } else {
                vec_c0 += lut_fma(vec_v_bot_low,  (i / 4    ));
                vec_c1 += lut_fma(vec_v_bot_high, (i / 4 + 1));
                vec_c2 += lut_fma(vec_v_top_low,  (i / 4 + 2));
                vec_c3 += lut_fma(vec_v_top_high, (i / 4 + 3));
            }
#undef lut_fma
        }

        if (ZeroPoint) {
            // OneScale mode is disabled for ZeroPoint = True
            float16x8_t vec_s0 = vld1q_f16(scales + ((i / 4    ) / Bits) * 16);
            float16x8_t vec_s1 = vld1q_f16(scales + ((i / 4 + 1) / Bits) * 16);
            float16x8_t vec_s2 = vld1q_f16(scales + ((i / 4 + 2) / Bits) * 16);
            float16x8_t vec_s3 = vld1q_f16(scales + ((i / 4 + 3) / Bits) * 16);
            // default_zero = 2 ** (bits - 1)
            // w = (w - default_zero - (zeros - default_zero)) * scales
            vec_c0 = vld1q_f16(c + i * 2)      + vec_c0 * vec_s0;
            vec_c1 = vld1q_f16(c + i * 2 + 8)  + vec_c1 * vec_s1;
            vec_c2 = vld1q_f16(c + i * 2 + 16) + vec_c2 * vec_s2;
            vec_c3 = vld1q_f16(c + i * 2 + 24) + vec_c3 * vec_s3;
            float16x8_t vec_z0 = vld1q_f16(scales + ((i / 4    ) / Bits) * 16 + 8);
            float16x8_t vec_z1 = vld1q_f16(scales + ((i / 4 + 1) / Bits) * 16 + 8);
            float16x8_t vec_z2 = vld1q_f16(scales + ((i / 4 + 2) / Bits) * 16 + 8);
            float16x8_t vec_z3 = vld1q_f16(scales + ((i / 4 + 3) / Bits) * 16 + 8);
            partial_sum *= 2;
#define add_zero(cs, zs, ib) \
    ((ib) % Bits) ? ((cs)) \
                  : ((cs) + zs * partial_sum)
            vst1q_f16(c + i * 2,      add_zero(vec_c0, vec_z0, (i / 4    )));
            vst1q_f16(c + i * 2 + 8,  add_zero(vec_c1, vec_z1, (i / 4 + 1)));
            vst1q_f16(c + i * 2 + 16, add_zero(vec_c2, vec_z2, (i / 4 + 2)));
            vst1q_f16(c + i * 2 + 24, add_zero(vec_c3, vec_z3, (i / 4 + 3)));
#undef add_zero
        } else {
            if (OneScale) {
                tmac_float_type vec_s = scales[0];
                vst1q_f16(c + i * 2,      vld1q_f16(c + i * 2     ) + vec_c0 * vec_s);
                vst1q_f16(c + i * 2 + 8,  vld1q_f16(c + i * 2 + 8 ) + vec_c1 * vec_s);
                vst1q_f16(c + i * 2 + 16, vld1q_f16(c + i * 2 + 16) + vec_c2 * vec_s);
                vst1q_f16(c + i * 2 + 24, vld1q_f16(c + i * 2 + 24) + vec_c3 * vec_s);
            } else {
                float16x8_t vec_s0 = vld1q_f16(scales + ((i / 4    ) / Bits) * 8);
                float16x8_t vec_s1 = vld1q_f16(scales + ((i / 4 + 1) / Bits) * 8);
                float16x8_t vec_s2 = vld1q_f16(scales + ((i / 4 + 2) / Bits) * 8);
                float16x8_t vec_s3 = vld1q_f16(scales + ((i / 4 + 3) / Bits) * 8);
                vst1q_f16(c + i * 2,      vld1q_f16(c + i * 2     ) + vec_c0 * vec_s0);
                vst1q_f16(c + i * 2 + 8,  vld1q_f16(c + i * 2 + 8 ) + vec_c1 * vec_s1);
                vst1q_f16(c + i * 2 + 16, vld1q_f16(c + i * 2 + 16) + vec_c2 * vec_s2);
                vst1q_f16(c + i * 2 + 24, vld1q_f16(c + i * 2 + 24) + vec_c3 * vec_s3);
            }
        }
    }
#elif defined __AVX2__
    const __m128i vec_mask = _mm_set1_epi8(0x0f);
    __m128i vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 16));
    }

    SignedAdder<FastAggregation, ActK> adder;
    for (int i = 0; i < m / 2; i += 16) {
        __m256 vec_c0, vec_c1, vec_c2, vec_c3;

        tmac_float_type partial_sum = (tmac_float_type)-0.0f;
#pragma unroll
        for (int kk = 0; kk < K; kk += ActK) {
#pragma unroll
            for (int k = 0; k < ActK; k++) {
                // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
                __m128i vec_as = _mm_loadu_si128(reinterpret_cast<__m128i*>(a + i * K + (kk + k) * 16));
                __m128i vec_a_bot = _mm_and_si128(vec_as, vec_mask);
                __m128i vec_a_top = _mm_and_si128(_mm_srli_epi16(vec_as, 4), vec_mask);

                __m256i vec_lut_ = _mm256_set_m128i(vec_lut[kk + k], vec_lut[kk + k]);
                __m256i vec_a = _mm256_set_m128i(vec_a_top, vec_a_bot);
                __m256i vec_v = _mm256_shuffle_epi8(vec_lut_, vec_a);
                adder.push(vec_v, k);
            }

            __m256 vec_v_low_low = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_low()));
            __m256 vec_v_low_high = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_low()));
            __m256 vec_v_high_low = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_high()));
            __m256 vec_v_high_high = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_high()));
        
            tmac_float_type lut_s = lut_scales[kk / ActK];
            tmac_float_type lut_b = lut_biases[kk / ActK];

            partial_sum += lut_b;

            if (FastAggregation) {
                lut_s = lut_s * ActK;
                lut_b -= lut_s * (mylog2<ActK>::value / 4 * get_bias_scale<Bits>());
            }

#define lut_fma(vs, ib) \
    ((ib) % Bits) ? (_mm256_mul_ps((vs),   _mm256_set1_ps(lut_s))) \
                  : (_mm256_fmadd_ps((vs), _mm256_set1_ps(lut_s), _mm256_set1_ps(lut_b)))
            if (kk == 0) {
                vec_c0 = lut_fma(vec_v_low_low,   (i / 4    ));
                vec_c1 = lut_fma(vec_v_low_high,  (i / 4 + 1));
                vec_c2 = lut_fma(vec_v_high_low,  (i / 4 + 2));
                vec_c3 = lut_fma(vec_v_high_high, (i / 4 + 3));
            } else {
                vec_c0 = _mm256_add_ps(vec_c0, lut_fma(vec_v_low_low,   (i / 4    )));
                vec_c1 = _mm256_add_ps(vec_c1, lut_fma(vec_v_low_high,  (i / 4 + 1)));
                vec_c2 = _mm256_add_ps(vec_c2, lut_fma(vec_v_high_low,  (i / 4 + 2)));
                vec_c3 = _mm256_add_ps(vec_c3, lut_fma(vec_v_high_high, (i / 4 + 3)));
            }
#undef lut_fma
        }

        if (ZeroPoint) {
            __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 16);
            __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 16);
            __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 16);
            __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 16);
            vec_c0 = _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2));
            vec_c1 = _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8));
            vec_c2 = _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16));
            vec_c3 = _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24));
            __m256 vec_z0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 16 + 8);
            __m256 vec_z1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 16 + 8);
            __m256 vec_z2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 16 + 8);
            __m256 vec_z3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 16 + 8);
            partial_sum *= 2;
#define add_zero(cs, zs, ib) \
    ((ib) % Bits) ? ((cs)) \
                  : (_mm256_fmadd_ps((zs), _mm256_set1_ps(partial_sum), (cs)))
            _mm256_storeu_ps(c + i * 2,      add_zero(vec_c0, vec_z0, (i / 4    )));
            _mm256_storeu_ps(c + i * 2 + 8,  add_zero(vec_c1, vec_z1, (i / 4 + 1)));
            _mm256_storeu_ps(c + i * 2 + 16, add_zero(vec_c2, vec_z2, (i / 4 + 2)));
            _mm256_storeu_ps(c + i * 2 + 24, add_zero(vec_c3, vec_z3, (i / 4 + 3)));
#undef add_zero
        } else if (OneScale) {
            tmac_float_type single_scale = scales[0];
            __m256 vec_s = _mm256_set1_ps(single_scale);
            _mm256_storeu_ps(c + i * 2,      _mm256_fmadd_ps(vec_c0, vec_s, _mm256_loadu_ps(c + i * 2)));
            _mm256_storeu_ps(c + i * 2 + 8,  _mm256_fmadd_ps(vec_c1, vec_s, _mm256_loadu_ps(c + i * 2 + 8)));
            _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(vec_c2, vec_s, _mm256_loadu_ps(c + i * 2 + 16)));
            _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(vec_c3, vec_s, _mm256_loadu_ps(c + i * 2 + 24)));
        } else {
            __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 8);
            __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 8);
            __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 8);
            __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 8);
            _mm256_storeu_ps(c + i * 2,      _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2)));
            _mm256_storeu_ps(c + i * 2 + 8,  _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8)));
            _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16)));
            _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24)));
        }
    }
#endif

    return 0;
}

// Unified scale
// TODO: implement fast aggregation for unified scale
template <int K, int Bits>
inline int32_t tbl_g4_int8_int32_update_impl(int32_t m, int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __ARM_NEON
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    int8x16_t vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

    SignedAdder<false, K> adder_bot, adder_top;
    for (int i = 0; i < m / 2; i += 16) {
#pragma unroll
        for (int k = 0; k < K; k++) {
            // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
            uint8x16_t vec_as = vld1q_u8(a + i * K + k * 16);
            uint8x16_t vec_a_top = vshrq_n_u8(vec_as, 4);
            uint8x16_t vec_a_bot = vandq_u8(vec_as, vec_mask);

            int8x16_t vec_v_bot_tmp = vqtbl1q_s8(vec_lut[k], vec_a_bot);
            int8x16_t vec_v_top_tmp = vqtbl1q_s8(vec_lut[k], vec_a_top);
            adder_bot.push(vec_v_bot_tmp, k);
            adder_top.push(vec_v_top_tmp, k);
        }

        int16x8_t vec_v_bot_low  = adder_bot.get_low();
        int16x8_t vec_v_bot_high = adder_bot.get_high();
        int16x8_t vec_v_top_low  = adder_top.get_low();
        int16x8_t vec_v_top_high = adder_top.get_high();

        int32x4_t vec_v_bot_low_low = vmovl_s16(vget_low_s16(vec_v_bot_low));
        int32x4_t vec_v_bot_low_high = vmovl_high_s16(vec_v_bot_low);
        int32x4_t vec_v_bot_high_low = vmovl_s16(vget_low_s16(vec_v_bot_high));
        int32x4_t vec_v_bot_high_high = vmovl_high_s16(vec_v_bot_high);
        int32x4_t vec_v_top_low_low = vmovl_s16(vget_low_s16(vec_v_top_low));
        int32x4_t vec_v_top_low_high = vmovl_high_s16(vec_v_top_low);
        int32x4_t vec_v_top_high_low = vmovl_s16(vget_low_s16(vec_v_top_high));
        int32x4_t vec_v_top_high_high = vmovl_high_s16(vec_v_top_high);

        vst1q_s32(c + i * 2,      vld1q_s32(c + i * 2     ) + vec_v_bot_low_low  );
        vst1q_s32(c + i * 2 + 4,  vld1q_s32(c + i * 2 + 4 ) + vec_v_bot_low_high );
        vst1q_s32(c + i * 2 + 8,  vld1q_s32(c + i * 2 + 8 ) + vec_v_bot_high_low );
        vst1q_s32(c + i * 2 + 12, vld1q_s32(c + i * 2 + 12) + vec_v_bot_high_high);
        vst1q_s32(c + i * 2 + 16, vld1q_s32(c + i * 2 + 16) + vec_v_top_low_low  );
        vst1q_s32(c + i * 2 + 20, vld1q_s32(c + i * 2 + 20) + vec_v_top_low_high );
        vst1q_s32(c + i * 2 + 24, vld1q_s32(c + i * 2 + 24) + vec_v_top_high_low );
        vst1q_s32(c + i * 2 + 28, vld1q_s32(c + i * 2 + 28) + vec_v_top_high_high);
    }

#elif defined __AVX2__
    const __m128i vec_mask = _mm_set1_epi8(0x0f);
    __m128i vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 16));
    }

    SignedAdder<false, K> adder;
    for (int i = 0; i < m / 2; i += 16) {
#pragma unroll
        for (int k = 0; k < K; k++) {
            // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
            __m128i vec_as = _mm_loadu_si128(reinterpret_cast<__m128i*>(a + i * K + k * 16));
            __m128i vec_a_bot = _mm_and_si128(vec_as, vec_mask);
            __m128i vec_a_top = _mm_and_si128(_mm_srli_epi16(vec_as, 4), vec_mask);

            __m256i vec_lut_ = _mm256_set_m128i(vec_lut[k], vec_lut[k]);
            __m256i vec_a = _mm256_set_m128i(vec_a_top, vec_a_bot);
            __m256i vec_v = _mm256_shuffle_epi8(vec_lut_, vec_a);
            adder.push(vec_v, k);
        }

        __m256i vec_v_low_low   = extract_low_epi16_epi32(adder.get_low());
        __m256i vec_v_low_high  = extract_high_epi16_epi32(adder.get_low());
        __m256i vec_v_high_low  = extract_low_epi16_epi32(adder.get_high());
        __m256i vec_v_high_high = extract_high_epi16_epi32(adder.get_high());
        __m256i vec_c0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2));
        __m256i vec_c1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 8));
        __m256i vec_c2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 16));
        __m256i vec_c3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 24));
        vec_c0 = _mm256_add_epi32(vec_c0, vec_v_low_low);
        vec_c1 = _mm256_add_epi32(vec_c1, vec_v_low_high);
        vec_c2 = _mm256_add_epi32(vec_c2, vec_v_high_low);
        vec_c3 = _mm256_add_epi32(vec_c3, vec_v_high_high);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2     ), vec_c0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 8 ), vec_c1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 16), vec_c2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i * 2 + 24), vec_c3);
    }

#endif
    return 0;
}

template <int K, int Bits>
inline int32_t tbl_g4_int8_int16_update_impl(int32_t m, int16_t* c, int8_t* lut, uint8_t* a) {
#ifdef __ARM_NEON
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    int8x16_t vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

    SignedAdder<false, K> adder_bot, adder_top;
    for (int i = 0; i < m / 2; i += 16) {
#pragma unroll
        for (int k = 0; k < K; k++) {
            // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
            uint8x16_t vec_as = vld1q_u8(a + i * K + k * 16);
            uint8x16_t vec_a_top = vshrq_n_u8(vec_as, 4);
            uint8x16_t vec_a_bot = vandq_u8(vec_as, vec_mask);

            int8x16_t vec_v_bot_tmp = vqtbl1q_s8(vec_lut[k], vec_a_bot);
            int8x16_t vec_v_top_tmp = vqtbl1q_s8(vec_lut[k], vec_a_top);
            adder_bot.push(vec_v_bot_tmp, k);
            adder_top.push(vec_v_top_tmp, k);
        }

        int16x8_t vec_v_bot_low  = adder_bot.get_low();
        int16x8_t vec_v_bot_high = adder_bot.get_high();
        int16x8_t vec_v_top_low  = adder_top.get_low();
        int16x8_t vec_v_top_high = adder_top.get_high();
        vst1q_s16(c + i * 2,      vld1q_s16(c + i * 2     ) + vec_v_bot_low);
        vst1q_s16(c + i * 2 + 8,  vld1q_s16(c + i * 2 + 8 ) + vec_v_bot_high);
        vst1q_s16(c + i * 2 + 16, vld1q_s16(c + i * 2 + 16) + vec_v_top_low);
        vst1q_s16(c + i * 2 + 24, vld1q_s16(c + i * 2 + 24) + vec_v_top_high);
    }
#elif defined __AVX2__
    // TODO: implement this
#endif
}


inline void tbl_g4_int8_float_gather_bit1_impl(int32_t m, tmac_float_type* C_global, tmac_float_type* CBits, tmac_float_type* C) {
    constexpr int32_t bits = 1;

    int32_t m_c_outer_max = m / 32;
    for (int32_t m_c_outer = 0; m_c_outer < m_c_outer_max; ++m_c_outer) {
        int32_t cse_var_2 = (m_c_outer * 32 * bits);
        int32_t cse_var_1 = (m_c_outer * 32);
        #pragma unroll
        for (int32_t m_c_inner = 0; m_c_inner < 32; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            C_global[cse_var_1 + m_c_inner] = (CBits[cse_var_2 + bit_offset_0] * (tmac_float_type)5.000000e-01f);

        }
    }

    for (int32_t m_inner_outer = 0; m_inner_outer < m_c_outer_max; ++m_inner_outer) {
        #pragma unroll
        for (int32_t m_inner = 0; m_inner < 32; ++m_inner) {
            int offset = m_inner_outer * 32 + m_inner;
            C[offset] = C_global[offset];
        }
    }
}

inline void tbl_g4_int8_float_gather_bit2_impl(int32_t m, tmac_float_type* C_global, tmac_float_type* CBits, tmac_float_type* C) {
    constexpr int32_t bits = 2;

    int32_t m_c_outer_max = m / 32;
    for (int32_t m_c_outer = 0; m_c_outer < m_c_outer_max; ++m_c_outer) {
        int32_t cse_var_2 = (m_c_outer * 32 * bits);
        int32_t cse_var_1 = (m_c_outer * 32);
        #pragma unroll
        for (int32_t m_c_inner = 0; m_c_inner < 32; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            int32_t bit_offset_1 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 8;
            C_global[cse_var_1 + m_c_inner] = (CBits[cse_var_2 + bit_offset_0] * (tmac_float_type)5.000000e-01f)
                                            + (CBits[cse_var_2 + bit_offset_1]);
        }
    }

    for (int32_t m_inner_outer = 0; m_inner_outer < m_c_outer_max; ++m_inner_outer) {
        #pragma unroll
        for (int32_t m_inner = 0; m_inner < 32; ++m_inner) {
            int offset = m_inner_outer * 32 + m_inner;
            C[offset] = C_global[offset];
        }
    }
}

inline void tbl_g4_int8_float_gather_bit3_impl(int32_t m, tmac_float_type* C_global, tmac_float_type* CBits, tmac_float_type* C) {
    constexpr int32_t bits = 3;

    int32_t m_c_outer_max = m / 32;
    for (int32_t m_c_outer = 0; m_c_outer < m_c_outer_max; ++m_c_outer) {
        int32_t cse_var_2 = (m_c_outer * 32 * bits);
        int32_t cse_var_1 = (m_c_outer * 32);
        #pragma unroll
        for (int32_t m_c_inner = 0; m_c_inner < 32; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            int32_t bit_offset_1 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 8;
            int32_t bit_offset_2 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 16;
            C_global[cse_var_1 + m_c_inner] = (CBits[cse_var_2 + bit_offset_0] * (tmac_float_type)5.000000e-01f)
                                            + (CBits[cse_var_2 + bit_offset_1])
                                            + (CBits[cse_var_2 + bit_offset_2] * (tmac_float_type)2.000000e+00f);
        }
    }

    for (int32_t m_inner_outer = 0; m_inner_outer < m_c_outer_max; ++m_inner_outer) {
        #pragma unroll
        for (int32_t m_inner = 0; m_inner < 32; ++m_inner) {
            int offset = m_inner_outer * 32 + m_inner;
            C[offset] = C_global[offset];
        }
    }
}

inline void tbl_g4_int8_float_gather_bit4_impl(int32_t m, tmac_float_type* C_global, tmac_float_type* CBits, tmac_float_type* C) {
    constexpr int32_t bits = 4;

    int32_t m_c_outer_max = m / 32;
    for (int32_t m_c_outer = 0; m_c_outer < m_c_outer_max; ++m_c_outer) {
        int32_t cse_var_2 = (m_c_outer * 32 * bits);
        int32_t cse_var_1 = (m_c_outer * 32);
        #pragma unroll
        for (int32_t m_c_inner = 0; m_c_inner < 32; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            int32_t bit_offset_1 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 8;
            int32_t bit_offset_2 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 16;
            int32_t bit_offset_3 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 24;
            C_global[cse_var_1 + m_c_inner] = (CBits[cse_var_2 + bit_offset_0] * (tmac_float_type)5.000000e-01f)
                                            + (CBits[cse_var_2 + bit_offset_1])
                                            + (CBits[cse_var_2 + bit_offset_2] * (tmac_float_type)2.000000e+00f)
                                            + (CBits[cse_var_2 + bit_offset_3] * (tmac_float_type)4.000000e+00f);
        }
    }

    for (int32_t m_inner_outer = 0; m_inner_outer < m_c_outer_max; ++m_inner_outer) {
        #pragma unroll
        for (int32_t m_inner = 0; m_inner < 32; ++m_inner) {
            int offset = m_inner_outer * 32 + m_inner;
            C[offset] = C_global[offset];
        }
    }
}




#ifdef __cplusplus
extern "C" {
#endif

int32_t tbl_int8_reset(int32_t m, int8_t* c) {
    memset(c, 0, m);
    return 0;
}

int32_t tbl_float_reset(int32_t m, void* c) {
    memset(c, 0, m * sizeof(tmac_float_type));
    return 0;
}

int32_t tbl_int32_reset(int32_t m, int32_t* c) {
    memset(c, 0, m * sizeof(int32_t));
    return 0;
}

int32_t tbl_int16_reset(int32_t m, int16_t* c) {
    memset(c, 0, m * sizeof(int16_t));
    return 0;
}

#ifdef __cplusplus
}
#endif


void qgemm_lut_int8_g4(
            void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C,
            int bm, int K, int N, const struct tmac_kernel_config * const kernel_config) {
    // TODO: support N > 1
    if (N != 1) {
        throw std::runtime_error("N > 1 is not supported yet");
    }

    const int g = kernel_config->g;
    const int ngroups_per_elem = 8 / g;
    int q_group_size = kernel_config->q_group_size;
    int act_group_size = kernel_config->act_group_size;
    bool has_scale = kernel_config->has_scale;
    int kfactor = kernel_config->kfactor;
    int bits = kernel_config->bits;
    int actk = kernel_config->actk;
    bool has_zero_point = kernel_config->has_zero_point;
    bool one_scale = kernel_config->one_scale;
    int m = bm / bits;

    tmac_float_type *CBits = new tmac_float_type[bm];
    tmac_float_type *C_global = new tmac_float_type[m];
    tbl_int32_reset(bm * sizeof(tmac_float_type) / sizeof(int32_t), (&(((int32_t*)CBits)[0])));
    
    int32_t k_outer_max = K / (kfactor * g);
    int32_t scale_gs = q_group_size / (kfactor * g);
    int32_t scale_idx_shfr = 0;
    if (scale_gs == 1) {
        scale_idx_shfr = 0;
    } else if (scale_gs == 2) {
        scale_idx_shfr = 1;
    } else if (scale_gs == 4) {
        scale_idx_shfr = 2;
    } else if (scale_gs == 8) {
        scale_idx_shfr = 3;
    } else {
        fprintf(stderr, "q_group_size=%d, kfactor=%d, g=%d\n", q_group_size, kfactor, g);
        fprintf(stderr, "Unsupported scale group size over kfactor. Expected {1,2,4,8}, got %d.\n", scale_gs);
        throw std::runtime_error("");
    }

    for (int32_t k_outer = 0; k_outer < k_outer_max; k_outer++) {
        uint8_t * a = ((uint8_t *)A) + k_outer * bm * kfactor / ngroups_per_elem;
        tmac_float_type * scales = one_scale ? (tmac_float_type *)Scales :
                              has_zero_point ? ((tmac_float_type *)Scales) + (k_outer >> scale_idx_shfr) * m * 2:
                                               ((tmac_float_type *)Scales) + (k_outer >> scale_idx_shfr) * m;
        int8_t * lut = ((int8_t *)LUT) + k_outer * kfactor * int(pow(2, g));
        tmac_float_type * lut_scales = ((tmac_float_type *)LUT_Scales) + (k_outer * kfactor * g / act_group_size);
        tmac_float_type * lut_biases = ((tmac_float_type *)LUT_Biases) + (k_outer * kfactor * g / act_group_size);

        if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, true, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, true, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 16, false, true, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, false, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, false, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 16, false, false, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, false, true>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, false, true>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 16, false, false, true>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        }
        
        else if (has_scale && kfactor == 8 && bits == 4 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 4, 8, false, true, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 4 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 4, 8, false, true, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 4 && actk == 16 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 4, 16, false, true, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 8 && bits == 4 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 4, 8, false, false, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 4 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 4, 8, false, false, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 4 && actk == 16 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 4, 16, false, false, false>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 8 && bits == 4 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 4, 8, false, false, true>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 4 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 4, 8, false, false, true>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 4 && actk == 16 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 4, 16, false, false, true>(
                (int32_t)bm, CBits, lut, a, scales, lut_scales, lut_biases);
        }
    }
    // if (!(((uint8_t *)A)[0] == 0 && ((uint8_t *)A)[1] == 0 && ((uint8_t *)A)[2] == 0 && ((uint8_t *)A)[3] == 0
    //     && ((uint8_t *)A)[4] == 0 && ((uint8_t *)A)[5] == 0 && ((uint8_t *)A)[6] == 0 && ((uint8_t *)A)[7] == 0)) {
    //     printf("\n\n\n\nCBits:\n\n\n");
    //     for (int i = 0; i < bm; i++) {
    //         printf("%f ", CBits[i]);
    //     }
    //     printf("\n");
    // }

    if (bits == 1) {
        tbl_g4_int8_float_gather_bit1_impl(m, C_global, CBits, (tmac_float_type *)C);
    } else if (bits == 2) {
        tbl_g4_int8_float_gather_bit2_impl(m, C_global, CBits, (tmac_float_type *)C);
    } else if (bits == 3) {
        tbl_g4_int8_float_gather_bit3_impl(m, C_global, CBits, (tmac_float_type *)C);
    } else if (bits == 4) {
        tbl_g4_int8_float_gather_bit4_impl(m, C_global, CBits, (tmac_float_type *)C);
    } else {
        throw std::runtime_error("Unsupported bits");
    }

    delete[] C_global;
    delete[] CBits;
}

