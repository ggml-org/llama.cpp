#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#ifndef GGML_ROCMFP4_USE_SCALE_LUT
#define GGML_ROCMFP4_USE_SCALE_LUT 0
#endif

#if defined(GGML_USE_HIP) && GGML_ROCMFP4_USE_SCALE_LUT
#define ROCMFP4_SCALE_SUB(M) ((M) * 0x1p-10f)
#define ROCMFP4_SCALE_E1(M)  ((8 + (M)) * 0x1p-10f)
#define ROCMFP4_SCALE_E2(M)  ((8 + (M)) * 0x1p-9f)
#define ROCMFP4_SCALE_E3(M)  ((8 + (M)) * 0x1p-8f)
#define ROCMFP4_SCALE_E4(M)  ((8 + (M)) * 0x1p-7f)
#define ROCMFP4_SCALE_E5(M)  ((8 + (M)) * 0x1p-6f)
#define ROCMFP4_SCALE_E6(M)  ((8 + (M)) * 0x1p-5f)
#define ROCMFP4_SCALE_E7(M)  ((8 + (M)) * 0x1p-4f)
#define ROCMFP4_SCALE_E8(M)  ((8 + (M)) * 0x1p-3f)
#define ROCMFP4_SCALE_E9(M)  ((8 + (M)) * 0x1p-2f)
#define ROCMFP4_SCALE_E10(M) ((8 + (M)) * 0x1p-1f)
#define ROCMFP4_SCALE_E11(M) ((8 + (M)) * 0x1p0f)
#define ROCMFP4_SCALE_E12(M) ((8 + (M)) * 0x1p1f)
#define ROCMFP4_SCALE_E13(M) ((8 + (M)) * 0x1p2f)
#define ROCMFP4_SCALE_E14(M) ((8 + (M)) * 0x1p3f)
#define ROCMFP4_SCALE_E15(M) ((8 + (M)) * 0x1p4f)

static __device__ __constant__ const float rocmfp4_scale_ue4m3_half_lut[127] = {
    ROCMFP4_SCALE_SUB(0), ROCMFP4_SCALE_SUB(1), ROCMFP4_SCALE_SUB(2), ROCMFP4_SCALE_SUB(3),
    ROCMFP4_SCALE_SUB(4), ROCMFP4_SCALE_SUB(5), ROCMFP4_SCALE_SUB(6), ROCMFP4_SCALE_SUB(7),
    ROCMFP4_SCALE_E1(0),  ROCMFP4_SCALE_E1(1),  ROCMFP4_SCALE_E1(2),  ROCMFP4_SCALE_E1(3),
    ROCMFP4_SCALE_E1(4),  ROCMFP4_SCALE_E1(5),  ROCMFP4_SCALE_E1(6),  ROCMFP4_SCALE_E1(7),
    ROCMFP4_SCALE_E2(0),  ROCMFP4_SCALE_E2(1),  ROCMFP4_SCALE_E2(2),  ROCMFP4_SCALE_E2(3),
    ROCMFP4_SCALE_E2(4),  ROCMFP4_SCALE_E2(5),  ROCMFP4_SCALE_E2(6),  ROCMFP4_SCALE_E2(7),
    ROCMFP4_SCALE_E3(0),  ROCMFP4_SCALE_E3(1),  ROCMFP4_SCALE_E3(2),  ROCMFP4_SCALE_E3(3),
    ROCMFP4_SCALE_E3(4),  ROCMFP4_SCALE_E3(5),  ROCMFP4_SCALE_E3(6),  ROCMFP4_SCALE_E3(7),
    ROCMFP4_SCALE_E4(0),  ROCMFP4_SCALE_E4(1),  ROCMFP4_SCALE_E4(2),  ROCMFP4_SCALE_E4(3),
    ROCMFP4_SCALE_E4(4),  ROCMFP4_SCALE_E4(5),  ROCMFP4_SCALE_E4(6),  ROCMFP4_SCALE_E4(7),
    ROCMFP4_SCALE_E5(0),  ROCMFP4_SCALE_E5(1),  ROCMFP4_SCALE_E5(2),  ROCMFP4_SCALE_E5(3),
    ROCMFP4_SCALE_E5(4),  ROCMFP4_SCALE_E5(5),  ROCMFP4_SCALE_E5(6),  ROCMFP4_SCALE_E5(7),
    ROCMFP4_SCALE_E6(0),  ROCMFP4_SCALE_E6(1),  ROCMFP4_SCALE_E6(2),  ROCMFP4_SCALE_E6(3),
    ROCMFP4_SCALE_E6(4),  ROCMFP4_SCALE_E6(5),  ROCMFP4_SCALE_E6(6),  ROCMFP4_SCALE_E6(7),
    ROCMFP4_SCALE_E7(0),  ROCMFP4_SCALE_E7(1),  ROCMFP4_SCALE_E7(2),  ROCMFP4_SCALE_E7(3),
    ROCMFP4_SCALE_E7(4),  ROCMFP4_SCALE_E7(5),  ROCMFP4_SCALE_E7(6),  ROCMFP4_SCALE_E7(7),
    ROCMFP4_SCALE_E8(0),  ROCMFP4_SCALE_E8(1),  ROCMFP4_SCALE_E8(2),  ROCMFP4_SCALE_E8(3),
    ROCMFP4_SCALE_E8(4),  ROCMFP4_SCALE_E8(5),  ROCMFP4_SCALE_E8(6),  ROCMFP4_SCALE_E8(7),
    ROCMFP4_SCALE_E9(0),  ROCMFP4_SCALE_E9(1),  ROCMFP4_SCALE_E9(2),  ROCMFP4_SCALE_E9(3),
    ROCMFP4_SCALE_E9(4),  ROCMFP4_SCALE_E9(5),  ROCMFP4_SCALE_E9(6),  ROCMFP4_SCALE_E9(7),
    ROCMFP4_SCALE_E10(0), ROCMFP4_SCALE_E10(1), ROCMFP4_SCALE_E10(2), ROCMFP4_SCALE_E10(3),
    ROCMFP4_SCALE_E10(4), ROCMFP4_SCALE_E10(5), ROCMFP4_SCALE_E10(6), ROCMFP4_SCALE_E10(7),
    ROCMFP4_SCALE_E11(0), ROCMFP4_SCALE_E11(1), ROCMFP4_SCALE_E11(2), ROCMFP4_SCALE_E11(3),
    ROCMFP4_SCALE_E11(4), ROCMFP4_SCALE_E11(5), ROCMFP4_SCALE_E11(6), ROCMFP4_SCALE_E11(7),
    ROCMFP4_SCALE_E12(0), ROCMFP4_SCALE_E12(1), ROCMFP4_SCALE_E12(2), ROCMFP4_SCALE_E12(3),
    ROCMFP4_SCALE_E12(4), ROCMFP4_SCALE_E12(5), ROCMFP4_SCALE_E12(6), ROCMFP4_SCALE_E12(7),
    ROCMFP4_SCALE_E13(0), ROCMFP4_SCALE_E13(1), ROCMFP4_SCALE_E13(2), ROCMFP4_SCALE_E13(3),
    ROCMFP4_SCALE_E13(4), ROCMFP4_SCALE_E13(5), ROCMFP4_SCALE_E13(6), ROCMFP4_SCALE_E13(7),
    ROCMFP4_SCALE_E14(0), ROCMFP4_SCALE_E14(1), ROCMFP4_SCALE_E14(2), ROCMFP4_SCALE_E14(3),
    ROCMFP4_SCALE_E14(4), ROCMFP4_SCALE_E14(5), ROCMFP4_SCALE_E14(6), ROCMFP4_SCALE_E14(7),
    ROCMFP4_SCALE_E15(0), ROCMFP4_SCALE_E15(1), ROCMFP4_SCALE_E15(2), ROCMFP4_SCALE_E15(3),
    ROCMFP4_SCALE_E15(4), ROCMFP4_SCALE_E15(5), ROCMFP4_SCALE_E15(6),
};

#undef ROCMFP4_SCALE_SUB
#undef ROCMFP4_SCALE_E1
#undef ROCMFP4_SCALE_E2
#undef ROCMFP4_SCALE_E3
#undef ROCMFP4_SCALE_E4
#undef ROCMFP4_SCALE_E5
#undef ROCMFP4_SCALE_E6
#undef ROCMFP4_SCALE_E7
#undef ROCMFP4_SCALE_E8
#undef ROCMFP4_SCALE_E9
#undef ROCMFP4_SCALE_E10
#undef ROCMFP4_SCALE_E11
#undef ROCMFP4_SCALE_E12
#undef ROCMFP4_SCALE_E13
#undef ROCMFP4_SCALE_E14
#undef ROCMFP4_SCALE_E15
#endif

static __device__ __forceinline__ float rocmfp4_u32_as_f32(uint32_t bits) {
#if defined(GGML_USE_HIP)
    return __uint_as_float(bits);
#else
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif
}

// ROCmFP4 validates scale bytes before backend execution, so HIP/ROCm hot
// paths can decode finite unsigned E4M3 half-scales directly without the
// generic FP8 NaN handling used by other formats.
static __device__ __forceinline__ float rocmfp4_ue4m3_to_fp32_half_finite(uint8_t x) {
#if defined(GGML_USE_HIP) && GGML_ROCMFP4_USE_SCALE_LUT
    return x <= 0x7e ? rocmfp4_scale_ue4m3_half_lut[x] : 0.0f;
#else
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;

    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }

    const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    return rocmfp4_u32_as_f32(bits);
#endif
}

static __device__ __forceinline__ float rocmfpx_ue4m3_to_fp32_finite(uint8_t x) {
    if (x > 0x7e) {
        return 0.0f;
    }

    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;

    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }

    const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    return rocmfp4_u32_as_f32(bits);
}

static __device__ __forceinline__ uint8_t rocmfpx_nearest_scale_ue4m3_cuda(float target_scale) {
    if (!(target_scale > 0.0f) || !isfinite(target_scale)) {
        return 0;
    }

    uint8_t lo = 1;
    uint8_t hi = 0x7e;
    while (lo < hi) {
        const uint8_t mid = lo + (hi - lo) / 2;
        if (rocmfpx_ue4m3_to_fp32_finite(mid) < target_scale) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == 1) {
        return 1;
    }

    const float hi_scale = rocmfpx_ue4m3_to_fp32_finite(lo);
    const float lo_scale = rocmfpx_ue4m3_to_fp32_finite((uint8_t) (lo - 1));
    return (target_scale - lo_scale <= hi_scale - target_scale) ? (uint8_t) (lo - 1) : lo;
}

static __device__ __forceinline__ int8_t rocmfp4_decode_i8(uint8_t q) {
    q &= 0x0f;
    const int mag3 = q & 0x07;
    const int mag = mag3 <= 4 ? mag3 : 2*mag3 - 4;
    return (q & 0x08) ? -mag : mag;
}
