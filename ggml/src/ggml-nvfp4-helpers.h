#pragma once

#include <stdint.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>  // For NAN and ldexpf
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(__CUDACC__)
#   include <cuda_fp16.h>
#   include <cuda_fp8.h>
    // cuda_fp4.h requires CUDA 13
#   if __CUDACC_VER_MAJOR__ >= 13
#       include <cuda_fp4.h>
#       define GGML_CUDA_FP4_NATIVE 1
#   else
#       define GGML_CUDA_FP4_NATIVE 0
#   endif
#   define GGML_HD __host__ __device__ __forceinline__
#   define GGML_DEVICE __device__ __forceinline__
#else
#   define GGML_HD static inline
#   define GGML_DEVICE static inline
#   define GGML_CUDA_FP4_NATIVE 0
#endif


#define GGML_FP8_UE4M3_TO_FP32(x)  ggml_fp8_ue4m3_to_fp32(x)
#define GGML_FP8_UE4M3_TO_FP32_HALF(x)  ggml_fp8_ue4m3_to_fp32_half(x)
#define GGML_FP32_TO_UE4M3(v) ggml_fp8_ue4m3_from_fp32(v)

// Four-over-Six (MIT) constants
#define NVFP4_E2M1_MAX_VALUE 6.0f

// Forward declarations for functions used in macros
GGML_HD uint8_t ggml_fp8_ue4m3_from_fp32(float v);
GGML_HD float ggml_fp8_ue4m3_to_fp32(uint8_t b);
#define NVFP4_E4M3_MAX_VALUE 448.0f

#ifndef GGML_NVFP4_MAX_FP8_TENSOR_SCALE
// Keep 448 as the default tensor-scale cap to match feb5 quantizer behavior.
// Per-run overrides are still available via MAX_FP8_TENSOR_SCALE / GGML_NVFP4_MAX_FP8_TENSOR_SCALE.
#define GGML_NVFP4_MAX_FP8_TENSOR_SCALE 448.0f
#endif

// can use on both host and device to avoid double definitions across files
#define KVALUES_NVFP4_FLOAT_LIST { \
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f, \
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  \
}
#define KVALUES_NVFP4_VAL2_LIST { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 }

GGML_HD float kvalues_nvfp4_float(int i) {
#if defined(__cplusplus)
    constexpr float t[16] = KVALUES_NVFP4_FLOAT_LIST;
#else
    static const float t[16] = KVALUES_NVFP4_FLOAT_LIST;
#endif
    return t[i & 15];
}

static inline float ggml_fp8_ue4m3_to_fp32_half(uint8_t u) {
    uint8_t b = u & 0x7Fu;          // unsigned UE4M3
    if (b == 0) return 0.0f;
    if (b == 0x7F) b = 0x7E;        // clamp NaN code -> max finite

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const __nv_fp8_storage_t fp8 = (__nv_fp8_storage_t) b;
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
    __half h;
    memcpy(&h, &hr, sizeof(h));
    return __half2float(h);
#else
    const uint32_t exp8  = b >> 3;
    const uint32_t mant8 = b & 7u;

    uint32_t bits;
    if (exp8) {
        bits = ((exp8 + 120u) << 23) | (mant8 << 20);
    } else {
        // subnormal: mant * 2^-9, prebuilt exact fp32 patterns
        static const uint32_t sub_bits[8] = {
            0x00000000u, // 0
            0x3B000000u, // 1 * 2^-9
            0x3B800000u, // 2 * 2^-9
            0x3BC00000u, // 3 * 2^-9
            0x3C000000u, // 4 * 2^-9
            0x3C200000u, // 5 * 2^-9
            0x3C400000u, // 6 * 2^-9
            0x3C600000u, // 7 * 2^-9
        };
        bits = sub_bits[mant8];
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
#endif
}

GGML_HD uint8_t ggml_fp8_ue4m3_from_fp32(float v);

// ============================================================================
// FP4 (E2M1) Conversion Functions - Native intrinsics when available
// ============================================================================


// Convert FP4 (E2M1) nibble to float
GGML_DEVICE float ggml_fp4_to_fp32(uint8_t fp4_nibble) {
#if GGML_CUDA_FP4_NATIVE
    __half_raw hr = __nv_cvt_fp4_to_halfraw(fp4_nibble, __NV_E2M1);
    __half h;
    memcpy(&h, &hr, sizeof(__half));
    return __half2float(h);
#else
    // Use lookup table on cpu or non-blackwell 
    return kvalues_nvfp4_float(fp4_nibble & 0xF);
#endif
}

// Convert float to FP4 (E2M1) nibble
GGML_DEVICE uint8_t ggml_fp32_to_fp4(float v) {
#if GGML_CUDA_FP4_NATIVE
    return __nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest);
#else
    // Software fallback: find best matching FP4 value
    float best_err = FLT_MAX;
    uint8_t best = 0;
    for (int i = 0; i < 16; ++i) {
        const float qi = kvalues_nvfp4_float(i);
        const float err = (v - qi) * (v - qi);
        if (err < best_err) {
            best_err = err;
            best = (uint8_t)i;
        }
    }
    return best;
#endif
}

// device-native and fallback is host bit-identical .
GGML_HD float ggml_nvfp4_fp16_to_fp32(const ggml_half h) {
    uint16_t hu;
    memcpy(&hu, &h, sizeof(hu));

#    if defined(__CUDA_ARCH__)
    // Use CUDA's half conversion helpers directly for bit-stable decode on device.
    __half_raw hr;
    hr.x = hu;
    __half hv;
    memcpy(&hv, &hr, sizeof(hv));
    return __half2float(hv);
#    else
    unsigned int sign     = (hu >> 15) & 1u;
    unsigned int exponent = (hu >> 10) & 0x1fu;
    unsigned int mantissa = (unsigned int) (hu & 0x3ffu) << 13;

    if (exponent == 0x1fu) {  // NaN/Inf
        sign     = mantissa ? (sign >> 1) : sign;
        mantissa = mantissa ? 0x7fffffu : 0u;
        exponent = 0xffu;
    } else if (exponent == 0u) {  // denorm/zero
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71u;
            do {
                msb = mantissa & 0x400000u;
                mantissa <<= 1;
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffffu;
        }
    } else {
        exponent += 0x70u;
    }

    const unsigned int u = (sign << 31) | (exponent << 23) | mantissa;
    float              f;
    memcpy(&f, &u, sizeof(u));
    return f;
#    endif
}

// NVFP4 code layout (current packed nibble order within each 32-value group):
// byte j stores nibble pair (code[j], code[j + 16]), j in [0, 15].
// This is repeated for 8 groups to cover 256 values.
GGML_HD int ggml_nvfp4_qbyte_from_elem(const int i) {
    return i / 2;
}

GGML_HD int ggml_nvfp4_qshift_from_elem(const int i) {
    return (i % 2) * 4;
}

GGML_HD uint8_t ggml_nvfp4_get_q4(const uint8_t qs[QK_K/2], const int i) {
    const int qb = ggml_nvfp4_qbyte_from_elem(i);
    const int sh = ggml_nvfp4_qshift_from_elem(i);
    return (uint8_t) ((qs[qb] >> sh) & 0x0F);
}

GGML_HD int ggml_nvfp4_q4_to_val2(const uint8_t q) {
    static const int8_t v2[16] = KVALUES_NVFP4_VAL2_LIST;
    return (int) v2[q & 0x0F];
}

GGML_HD void ggml_nvfp4_pack_codes_256(const uint8_t codes[QK_K], uint8_t qs[QK_K/2]) {
    for (int j = 0; j < QK_K/2; ++j) {
        qs[j] = (uint8_t) ((codes[2*j] & 0x0F) | ((codes[2*j + 1] & 0x0F) << 4));
    }
}




// Pack 8 consecutive NVFP4 codes into 4 bytes (q0|q1<<4, q2|q3<<4, ...), returned as a little-endian u32.
// Current NVFP4 layout is contiguous nibble-pair packing:
//   qs[j] = codes[2*j] | (codes[2*j + 1] << 4)
// so chunk8 maps directly to 4 consecutive bytes.
GGML_HD uint32_t ggml_nvfp4_pack8(const uint8_t qs[QK_K/2], const int chunk8) {
    const int base = chunk8 * 4;
    return ((uint32_t) qs[base + 0])
        | (((uint32_t) qs[base + 1]) << 8)
        | (((uint32_t) qs[base + 2]) << 16)
        | (((uint32_t) qs[base + 3]) << 24);
}


GGML_HD uint8_t best_index_nvfp4(float v, float s) {
#if GGML_CUDA_FP4_NATIVE
    // UE4M3 scale => must be positive
    if (!(s > 0.0f) || !isfinite(s) || !isfinite(v)) return 0;
    float scaled = v / s;
    if (!isfinite(scaled)) return 0;
    // CUDA docs: out-of-range saturates; NaN becomes +MAXNORM :contentReference[oaicite:1]{index=1}
    const __nv_fp4_storage_t fp4 = __nv_cvt_float_to_fp4(scaled, __NV_E2M1, cudaRoundNearest);
    return ((uint8_t) fp4) & 0x0F;
#else
    if (!(s > 0.0f) || !isfinite(s) || !isfinite(v)) return 0;
    float best_err = FLT_MAX;
    uint8_t best = 0;
    for (int i = 0; i < 16; ++i) {
        const float qi  = s * kvalues_nvfp4_float(i);
        const float err = (v - qi) * (v - qi);
        if (err < best_err) { best_err = err; best = (uint8_t) i; }
    }
    return best;
#endif
}


GGML_HD float ggml_fp8_ue4m3_to_fp32(uint8_t b) {
    // Native conversion: interpret as +E4M3 (sign bit is 0 after masking).
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint8_t u = b & 0x7Fu;      // force unsigned
    if (u == 0x7Fu)
       {
          u = 0x7Eu;  // never allow NaN code
       }
    const __nv_fp8_storage_t fp8 = (__nv_fp8_storage_t) u;
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);

    __half h;
    memcpy(&h, &hr, sizeof(__half));
    return __half2float(h);
#else
    const uint32_t u = b & 0x7Fu;          // unsigned UE4M3 (force sign=0)
    if (u == 0) return 0.0f;

    const uint32_t exp  = u >> 3;          // 0..15
    const uint32_t mant = u & 0x7u;        // 0..7
    // Clamp NaN code 0x7F to max finite UE4M3 value.
    // Important: this is format decode (always max=448), not tensor-scale policy.
    if (exp == 0xFu && mant == 0x7u) return NVFP4_E4M3_MAX_VALUE;

    uint32_t bits;

    if (exp != 0) {
        // normal: (1 + mant/8) * 2^(exp-7)
        // fp32 exponent = (exp-7)+127 = exp+120
        bits = ((exp + 120u) << 23) | (mant << 20);  // mant: 3 -> 23 bits (shift 20)
    } else {
        // subnormal: mant * 2^-9
        // normalize mant (1..7) without a loop
        // p = floor(log2(mant)) in {0,1,2}
        const uint32_t p = 31u - (uint32_t)__builtin_clz(mant);
        const uint32_t r = mant - (1u << p);
        // value = (1 + r/2^p) * 2^(p-9)
        const uint32_t exp32  = (p + 118u);                 // (p-9)+127 = p+118
        const uint32_t mant32 = r << (23u - p);             // fraction bits
        bits = (exp32 << 23) | mant32;
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
#endif
}

GGML_HD uint8_t ggml_fp8_e4m3_from_fp32(float v) {
    uint32_t i;
    memcpy(&i, &v, sizeof(uint32_t));

    uint32_t sign = (i >> 31) & 1;
    uint32_t exp  = (i >> 23) & 0xFF;
    uint32_t mant = i & 0x7FFFFF;

    if (exp == 0xFF) {
        return (uint8_t)((sign << 7) | 0x7E);
    }

    if (exp == 0 && mant == 0) {
        return (uint8_t)(sign << 7);
    }

    int e = (int)exp - 127;
    int biased_exp = e + 7;

    if (biased_exp > 15) {
        return (uint8_t)((sign << 7) | 0x7E);
    }

    uint32_t m = mant | 0x800000;

    if (biased_exp < 1) {
        int shift = 1 - biased_exp;
        if (shift > 24) {
            return (uint8_t)(sign << 7);
        }

        int total_shift = 20 + shift;

        if (total_shift >= 24) {
             if (total_shift > 25) return (uint8_t)(sign << 7);
        }

        uint32_t round_bit_mask = 1 << (total_shift - 1);
        uint32_t sticky_mask = round_bit_mask - 1;
        uint32_t lsb_mask = 1 << total_shift;

        uint32_t round_bit = m & round_bit_mask;
        uint32_t sticky_bit = (m & sticky_mask) != 0;
        uint32_t lsb = m & lsb_mask;

        m >>= total_shift;

        if (round_bit && (sticky_bit || lsb)) {
            m++;
        }

        return (uint8_t)((sign << 7) | m);
    }

    uint32_t round_bit_mask = 1 << 19;
    uint32_t sticky_mask = round_bit_mask - 1;
    uint32_t lsb_mask = 1 << 20;

    uint32_t round_bit = m & round_bit_mask;
    uint32_t sticky_bit = (m & sticky_mask) != 0;
    uint32_t lsb = m & lsb_mask;

    m >>= 20;
    m &= 0x7;

    if (round_bit && (sticky_bit || lsb)) {
        m++;
        if (m > 7) {
            m = 0;
            biased_exp++;
        }
    }

    if (biased_exp > 15) {
        return (uint8_t)((sign << 7) | 0x7E);
    }

    if (biased_exp == 15 && m == 7) {
        return (uint8_t)((sign << 7) | 0x7E);
    }

    return (uint8_t)((sign << 7) | (biased_exp << 3) | m);
}
static const int8_t ggml_nvfp4_lut_i8_dbl_bytes[16] = {
     0,   1,   2,   3,
     4,   6,   8,  12,
     0,  -1,  -2,  -3,
    -4,  -6,  -8, -12,
};

#if defined(__CUDACC__)
__device__ __constant__ int8_t ksum2_nvfp4_byte[256];

static __device__ __forceinline__ int sum2_nvfp4_byte(uint8_t b) {
    return (int) ksum2_nvfp4_byte[b];
}

static inline void ggml_cuda_init_nvfp4_sum2_lut(void) {
    int8_t v2[16] = KVALUES_NVFP4_VAL2_LIST;
    int8_t h[256];

    for (int b = 0; b < 256; ++b) {
        h[b] = (int8_t) (v2[b & 0x0F] + v2[b >> 4]);
    }

    cudaMemcpyToSymbol(ksum2_nvfp4_byte, h, sizeof(h));
}
#else
static inline void ggml_cuda_init_nvfp4_sum2_lut(void) {}
#endif

static inline int32_t ggml_dot16_nvfp4_nvfp4(const uint8_t * qx, const uint8_t * qy) {
#if defined(__SSE4_1__)
    const __m128i lut   = _mm_loadu_si128((const __m128i *) ggml_nvfp4_lut_i8_dbl_bytes);
    const __m128i mask0 = _mm_set1_epi8(0x0f);

    const __m128i bx = _mm_loadl_epi64((const __m128i *) qx);
    const __m128i by = _mm_loadl_epi64((const __m128i *) qy);

    const __m128i x_lo  = _mm_and_si128(bx, mask0);
    const __m128i x_hi  = _mm_and_si128(_mm_srli_epi16(bx, 4), mask0);
    const __m128i y_lo  = _mm_and_si128(by, mask0);
    const __m128i y_hi  = _mm_and_si128(_mm_srli_epi16(by, 4), mask0);

    const __m128i x_idx = _mm_unpacklo_epi8(x_lo, x_hi);
    const __m128i y_idx = _mm_unpacklo_epi8(y_lo, y_hi);

    const __m128i x_i8  = _mm_shuffle_epi8(lut, x_idx);
    const __m128i y_i8  = _mm_shuffle_epi8(lut, y_idx);

    const __m128i x16_0 = _mm_cvtepi8_epi16(x_i8);
    const __m128i y16_0 = _mm_cvtepi8_epi16(y_i8);
    const __m128i x16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(x_i8, 8));
    const __m128i y16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(y_i8, 8));

    __m128i s = _mm_add_epi32(_mm_madd_epi16(x16_0, y16_0),
                             _mm_madd_epi16(x16_1, y16_1));

    // horizontal sum 4x int32 -> int32
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return _mm_cvtsi128_si32(s);
#else
    int32_t sum = 0;
    for (int j = 0; j < 8; ++j) {
        const uint8_t ax = qx[j];
        const uint8_t ay = qy[j];

        const int8_t x0 = ggml_nvfp4_lut_i8_dbl_bytes[ax & 0x0f];
        const int8_t x1 = ggml_nvfp4_lut_i8_dbl_bytes[ax >> 4];
        const int8_t y0 = ggml_nvfp4_lut_i8_dbl_bytes[ay & 0x0f];
        const int8_t y1 = ggml_nvfp4_lut_i8_dbl_bytes[ay >> 4];

        sum += (int32_t) x0 * (int32_t) y0;
        sum += (int32_t) x1 * (int32_t) y1;
    }
    return sum;
#endif
}


#if !defined(__CUDACC__)
// Dot product of 16 int8 values.
static inline int32_t ggml_dot16_i8(const int8_t * x, const int8_t * y) {
#if defined(__SSE4_1__)
    const __m128i vx = _mm_loadu_si128((const __m128i *) x);
    const __m128i vy = _mm_loadu_si128((const __m128i *) y);

    const __m128i x16_0 = _mm_cvtepi8_epi16(vx);
    const __m128i y16_0 = _mm_cvtepi8_epi16(vy);
    const __m128i x16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(vx, 8));
    const __m128i y16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(vy, 8));

    __m128i s = _mm_add_epi32(_mm_madd_epi16(x16_0, y16_0),
                             _mm_madd_epi16(x16_1, y16_1));

    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return _mm_cvtsi128_si32(s);
#else
    int32_t sum = 0;
    for (int i = 0; i < 16; ++i) sum += (int32_t) x[i] * (int32_t) y[i];
    return sum;
#endif
}

#endif


// ============================================================================
// NVFP4 adaptive 4/6 sub-block scaling chooser (HOST-ONLY)
//
// This is used by the CPU quantizers (compiled by the host compiler).
// CUDA quantization has an in-kernel implementation and does not need this.
//
// Chooses per-16-value sub-block scale: M=6 (normal) vs M=4 ("zoom").
// Scores using REALIZED scales (after FP8 UE4M3 encode+decode) and a mild
// gap-bias term around the FP4 E2M1 gap between 4.0 and 6.0.
//
// Returns chosen FP8 scale byte (UE4M3).
// ============================================================================
// NOTE: This logic is used by both CPU and CUDA paths. Keep it GGML_HD.
GGML_HD float nvfp4_sse_16_for_scale(
    const float * GGML_RESTRICT x16,
    const float * GGML_RESTRICT qw16,
    const float scale
) {
    if (!(scale > 0.0f) || !isfinite(scale)) return INFINITY;

    float sse = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float w = 1.0f;
        if (qw16) {
            const float qi = qw16[i];
            w = (isfinite(qi) && qi > 0.0f) ? qi : 0.0f;
        }
        if (w == 0.0f) continue;

        const float v = x16[i];
        const uint8_t code = best_index_nvfp4(v, scale);
        const float q = kvalues_nvfp4_float(code & 0xF) * scale;
        const float e = v - q;
        sse += w * e * e;
    }
    return sse;
}

// MIT Four-over-Six reference quantizer for E2M1 rounding.
GGML_HD float nvfp4_fake_quantize_positive_e2m1(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }

    float step1 = roundf(2.0f * x) * 0.5f;
    float step2 = roundf(x);
    float step3 = 2.0f * roundf(x * 0.5f);

    if (step3 > NVFP4_E2M1_MAX_VALUE) {
        step3 = NVFP4_E2M1_MAX_VALUE;
    }

    if (x < 2.0f) {
        return step1;
    }
    if (x < 4.0f) {
        return step2;
    }
    return step3;
}

GGML_HD float nvfp4_fake_quantize_signed_e2m1(float x) {
    const float ax = fabsf(x);
    const float q = nvfp4_fake_quantize_positive_e2m1(ax);
    return copysignf(q, x);
}

// MIT-style local objective with optional imatrix weighting.
// This mirrors Q4_K's weighted local-fit spirit: better preserve important weights.
GGML_HD float nvfp4_mse_16_for_scale_mit(
    const float * GGML_RESTRICT x16,
    const float scale
) {
    if (!(scale > 0.0f) || !isfinite(scale)) return INFINITY;

    float sse = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const float v = x16[i];
        const float q = nvfp4_fake_quantize_signed_e2m1(v / scale);
        const float deq = q * scale;
        const float e = deq - v;
        sse += e * e;
    }
    return sse;
}

GGML_HD float nvfp4_mse_16_for_scale_mit_w(
    const float * GGML_RESTRICT x16,
    const float * GGML_RESTRICT qw16,
    const float scale
) {
    if (!(scale > 0.0f) || !isfinite(scale)) return INFINITY;

    float sse = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float w = 1.0f;
        if (qw16) {
            const float qi = qw16[i];
            w = (isfinite(qi) && qi > 0.0f) ? qi : 0.0f;
        }
        if (w == 0.0f) continue;
        const float v = x16[i];
        const float q = nvfp4_fake_quantize_signed_e2m1(v / scale);
        const float deq = q * scale;
        const float e = deq - v;
        sse += w * e * e;
    }
    return sse;
}

// Adaptive chooser logic for  NVFP4 scale chooser "Four over Six" with WMSE scoring.
// Credit goes to authors of the paper:
// Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling
// Jack Cook and Junxian Guo and Guangxuan Xiao and Yujun Lin and Song Han
// (2025): See https://arxiv.org/abs/2512.02010} and https://github.com/mit-han-lab/fouroversix 
/* MIT License
Copyright (c) 2025 Jack Cook
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/ 
GGML_HD uint8_t adaptive_block_scale_4_or_6(const float * GGML_RESTRICT x16, const float * GGML_RESTRICT qw16, const float d_fp32) {
    const float eps        = 1e-20f;
    const float m6_anchor  = 6.00f;
    const float m4_anchor  = 4.00f;
    const float u_guard    = 7.00;
    const float tail_thr   = 0.609f;
    const int   tail_min   = 2;
    const float gap_lo     = 4.409f;
    const float gap_hi     = 5.474f;
    const float gap_lambda = 0.02f;
    const float top2_thr   = 0.957f;

    if (!(d_fp32 > 0.0f) || !isfinite(d_fp32)) return 0;

    float max_abs = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const float ax = fabsf(x16[i]);
        if (ax > max_abs) max_abs = ax;
    }
    if (!(max_abs > 0.0f)) return 0;

    // ideal sb in "units of d"
    const float sb6_ideal = (max_abs * (1.0f / m6_anchor)) / d_fp32;
    const float sb4_ideal = (max_abs * (1.0f / m4_anchor)) / d_fp32;

    const uint8_t b6 = GGML_FP32_TO_UE4M3(sb6_ideal);
    const uint8_t b4 = GGML_FP32_TO_UE4M3(sb4_ideal);

    if (b6 == b4) return b4;

    const float s6u = GGML_FP8_UE4M3_TO_FP32(b6);
    const float s4u = GGML_FP8_UE4M3_TO_FP32(b4);

    if (s6u == 0.0f) return b4;
    if (s4u == 0.0f) return b6;

    const float scale6 = d_fp32 * s6u;
    const float scale4 = d_fp32 * s4u;
    const float u_max6 = max_abs / (scale6 + eps);

    float max1 = 0.0f;
    float max2 = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float ax = fabsf(x16[i]);
        if (ax > max1) { max2 = max1; max1 = ax; }
        else if (ax > max2) { max2 = ax; }
    }

    int tail_cnt = 0;
    const float tail_cut = tail_thr * max_abs;
    for (int i = 0; i < 16; ++i) {
        const float ax = fabsf(x16[i]);
        if (ax >= tail_cut) {
            ++tail_cnt;
        }
    }

    // Q4_K-style weighted local objective: use imatrix weights when available.
    float L6 = nvfp4_mse_16_for_scale_mit_w(x16, qw16, scale6);
    float L4 = nvfp4_mse_16_for_scale_mit_w(x16, qw16, scale4);

    // Heuristic penalties
    if (max2 >= top2_thr * max1) {
        const float u6_2 = max2 / (scale6 + eps);
        if (u6_2 >= gap_lo && u6_2 <= gap_hi) {
            L6 += gap_lambda * (scale6 * scale6);
        }
    }
    if (u_max6 > u_guard) {
        L6 += gap_lambda * (scale6 * scale6);
    }

    if (tail_cnt >= tail_min) {
        for (int i = 0; i < 16; ++i) {
            const float u = x16[i] / (scale6 + eps);
            const float ua = fabsf(u);
            if (ua >= gap_lo && ua <= gap_hi) {
                L6 += gap_lambda * (scale6 * scale6);
            }
        }
    }

    return (L4 < L6) ? b4 : b6;
}

// Shared host/device tensor-scale cap chooser for NVFP4 256-element blocks.
// Keeps cap decision logic identical across CPU and CUDA quantization paths.
GGML_HD float ggml_nvfp4_pick_max_fp8_tensor_scale(
    const float * GGML_RESTRICT x256,
    const float * GGML_RESTRICT qw256,
    const float gmax,
    const float max_fp8_tensor_4) {

    const float high = 448.0f;
    const float low = (isfinite(max_fp8_tensor_4) && max_fp8_tensor_4 > 0.0f) ? fminf(max_fp8_tensor_4, high) : high;

    const float d_min_guess = (gmax * 0.25f) / high;
    if (!(d_min_guess > 0.0f) || !isfinite(d_min_guess)) {
        return high;
    }

    int use_fp4_4 = 0;
    for (int sub = 0; sub < 16; ++sub) {
        const float * GGML_RESTRICT x16 = x256 + sub * 16;
        const float * GGML_RESTRICT qw16 = qw256 ? (qw256 + sub * 16) : NULL;

        float max_abs = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float ax = fabsf(x16[j]);
            if (ax > max_abs) {
                max_abs = ax;
            }
        }

        if (!(max_abs > 0.0f)) {
            continue;
        }

        const float sb6_ideal = (max_abs * (1.0f / 6.0f)) / d_min_guess;
        const float sb4_ideal = (max_abs * (1.0f / 4.0f)) / d_min_guess;

        const uint8_t b6 = ggml_fp8_ue4m3_from_fp32(sb6_ideal);
        const uint8_t b4 = ggml_fp8_ue4m3_from_fp32(sb4_ideal);
        const uint8_t b = adaptive_block_scale_4_or_6(x16, qw16, d_min_guess);

        if (b == b4 && b4 != b6) {
            use_fp4_4 = 1;
            break;
        }
    }

    return use_fp4_4 ? low : high;
}

// Per-16 sub-block cap chooser used by both CPU/CUDA quantization paths.
// If adaptive 4/6 chooses the M=4 branch (and it is distinguishable from M=6),
// enforce the lower cap (typically 256) for that sub-block; otherwise use high cap.
GGML_HD float ggml_nvfp4_pick_subblock_fp8_cap(
    const float * GGML_RESTRICT x16,
    const float * GGML_RESTRICT qw16,
    const float d_fp32,
    const float max_fp8_high,
    const float max_fp8_low) {

    const float high = (isfinite(max_fp8_high) && max_fp8_high > 0.0f) ? max_fp8_high : 448.0f;
    const float low = (isfinite(max_fp8_low) && max_fp8_low > 0.0f) ? fminf(max_fp8_low, high) : high;
    if (!(d_fp32 > 0.0f) || !isfinite(d_fp32)) {
        return high;
    }

    float max_abs = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const float ax = fabsf(x16[j]);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }
    if (!(max_abs > 0.0f)) {
        return high;
    }

    const float sb6_ideal = (max_abs * (1.0f / 6.0f)) / d_fp32;
    const float sb4_ideal = (max_abs * (1.0f / 4.0f)) / d_fp32;

    const uint8_t b6 = ggml_fp8_ue4m3_from_fp32(sb6_ideal);
    const uint8_t b4 = ggml_fp8_ue4m3_from_fp32(sb4_ideal);
    const uint8_t b = adaptive_block_scale_4_or_6(x16, qw16, d_fp32);

    return (b == b4 && b4 != b6) ? low : high;
}

GGML_HD uint8_t ggml_fp8_ue4m3_from_fp32(float v) {
    if (!(v > 0.0f)) return 0;
    if (!(v < NVFP4_E4M3_MAX_VALUE)) return 0x7E;
    if (!isfinite(v)) return 0x7E;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    uint8_t c = (uint8_t) s & 0x7Fu;
    return (c == 0x7Fu) ? 0x7Eu : c;
#else
 // this seems unnecessarily omplex, but it is an exact CPU replica of the CUDA native __nv_cvt_float_to_fp8
 // this avoids even the tiniest discrepancies so CPU=GPU at all times

    uint32_t u;
    memcpy(&u, &v, sizeof(u));

    if (u & 0x80000000u) {
        return 0;
    }

    const uint32_t exp  = (u >> 23) & 0xFFu;
    const uint32_t mant =  u        & 0x7FFFFFu;

    if (exp == 0) {
        return 0;
    }

    // NaN/Inf goes to SATFINITE which is return MAXNORM
    if (exp == 0xFFu) {
        return 0x7Eu;
    }

    const int e  = (int)exp - 127;   // unbiased 
    int e4 = e + 7;                  // FP8 E4 with bias=7

    // if any overflow goes to MAXNORM
    if (e4 > 15) {
        return 0x7Eu;
    }

    // 24-bit mantissa with hidden leading 1
    const uint32_t m24 = (1u << 23) | mant;

    // subnormal 
    if (e4 <= 0) {
        // mant3 = RNE( m24 * 2^(e-14) )
        // since e4<=0 => e<=-7 => (e-14)<=-21 => right shift by (14-e)
        const int s = 14 - e;  // >= 1
        uint32_t q;

        if (s >= 32) {
            q = 0;
        } else {
            const uint32_t r   = m24 >> s;
            const uint32_t rem = m24 & ((1u << s) - 1u);
            const uint32_t half = 1u << (s - 1);

            q = r + (rem > half || (rem == half && (r & 1u)));
        }

        // if round reaches 8, reduce to the smallest smallest normal
        if (q >= 8u) {
            return 0x08u;
        }
        return (uint8_t) q; //
    }

    // "Normal FP8" has exp4 in [1..14], mant3 is top 3 mantissa bits with RNE
    uint32_t mant3 = mant >> 20;              // 3 bits
    const uint32_t rem  = mant & ((1u << 20) - 1u);
    const uint32_t half = 1u << 19;

    // Round NE for  mant3
    if (rem > half || (rem == half && (mant3 & 1u))) {
        mant3 += 1u;
        if (mant3 == 8u) {
            mant3 = 0u;
            e4 += 1;
            if (e4 > 15) {
                return 0x7Eu;
            }
        }
    }

    uint8_t out = (uint8_t)(((uint32_t)e4 << 3) | mant3);

    // reduce to max finite if NaN code
    if (out == 0x7Fu) out = 0x7Eu;
    return out;
#endif
}


// ============================================================================
// Shared NVFP4 helpers (CPU/CUDA)
// ============================================================================

GGML_HD float ggml_nvfp4_qw(const float * qw, int idx) {
    if (!qw) return 1.0f;
    const float w = qw[idx];
    if (!isfinite(w) || w <= 0.0f) return 0.0f;
    return w > 64.0f ? 64.0f : w;
}

GGML_HD float ggml_nvfp4_qw16(const float * qw16, int j) {
    if (!qw16) return 1.0f;
    const float w = qw16[j];
    return (isfinite(w) && w > 0.0f) ? w : 0.0f;
}

GGML_HD float ggml_nvfp4_subblock_sse_w_best(
        const float * x16,
        const float * qw16,
        const float scale) {
    if (!(scale > 0.0f) || !isfinite(scale)) {
        float sse = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float w = ggml_nvfp4_qw16(qw16, j);
            if (w == 0.0f) {
                continue;
            }
            const float v = x16[j];
            sse += w * (v * v);
        }
        return sse;
    }

    float sse = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const float w = ggml_nvfp4_qw16(qw16, j);
        if (w == 0.0f) {
            continue;
        }
        const float v = x16[j];
        const uint8_t code = best_index_nvfp4(v, scale);
        const float q = scale * kvalues_nvfp4_float(code & 0xF);
        const float e = v - q;
        sse += w * (e * e);
    }
    return sse;
}

GGML_HD uint8_t ggml_nvfp4_refine_sbscale_fp8_local(
    const float * x16,
    const float * qw16,
    const float d,
    uint8_t b_init
) {
    if (!(d > 0.0f) || !isfinite(d)) {
        return 0;
    }

    uint8_t best_b = b_init;
    float best_sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d * GGML_FP8_UE4M3_TO_FP32(best_b));

    // neighborhood search to fix FP8 round-to-nearest ties.
    for (int delta = 1; delta <= 2; ++delta) {
        const int cand0 = (int) b_init - delta;
        const int cand1 = (int) b_init + delta;

        if (cand0 >= 0) {
            const uint8_t b = (uint8_t) cand0;
            const float sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d * GGML_FP8_UE4M3_TO_FP32(b));
            if (sse < best_sse) {
                best_sse = sse;
                best_b = b;
            }
        }

        if (cand1 <= 0x7E) {
            const uint8_t b = (uint8_t) cand1;
            const float sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d * GGML_FP8_UE4M3_TO_FP32(b));
            if (sse < best_sse) {
                best_sse = sse;
                best_b = b;
            }
        }
    }

    return best_b;
}

GGML_HD float ggml_nvfp4_obj_256_codes(
    const float * x256,
    const float * qw256,
    const uint8_t * codes256,
    const float d,
    const uint8_t * sb_scales
) {
    float sse = 0.0f;
    float sum_err = 0.0f;
    float sum_w = 0.0f;

    for (int sub = 0; sub < 16; ++sub) {
        const float scale = d * GGML_FP8_UE4M3_TO_FP32(sb_scales[sub]);
        if (!(scale > 0.0f) || !isfinite(scale)) continue;

        const float * x16 = x256 + sub * 16;
        const float * qw16 = qw256 ? (qw256 + sub * 16) : NULL;

        for (int j = 0; j < 16; ++j) {
            const float w = ggml_nvfp4_qw(qw16, j);
            const uint8_t qi = codes256[sub * 16 + j] & 0xF;
            const float q = scale * kvalues_nvfp4_float(qi & 0xF);
            const float e = x16[j] - q;
            sse     += w * e * e;
            sum_err += w * e;
            sum_w   += w;
        }
    }

    float obj = sse;
    if (sum_w > 0.0f && isfinite(sum_w)) {
        const float mean_err = sum_err / sum_w;
        obj = sse + 32.0f * sum_w * mean_err * mean_err;
    }
    return obj;
}

GGML_HD float ggml_nvfp4_sse_256_codes(
    const float * x256,
    const float * qw256,
    const uint8_t * codes256,
    const float d,
    const uint8_t * sb_scales
) {
    float sse = 0.0f;

    for (int sub = 0; sub < 16; ++sub) {
        const float scale = d * GGML_FP8_UE4M3_TO_FP32(sb_scales[sub]);
        if (!(scale > 0.0f) || !isfinite(scale)) continue;

        const float * x16 = x256 + sub * 16;
        const float * qw16 = qw256 ? (qw256 + sub * 16) : NULL;

        for (int j = 0; j < 16; ++j) {
            const float w = ggml_nvfp4_qw(qw16, j);
            const uint8_t qi = codes256[sub * 16 + j] & 0xF;
            const float q = scale * kvalues_nvfp4_float(qi & 0xF);
            const float e = x16[j] - q;
            sse += w * e * e;
        }
    }

    return sse;
}
