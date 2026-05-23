#pragma once

#include <cstdint>
#include <cstring>

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
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;

    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }

    const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    return rocmfp4_u32_as_f32(bits);
}

static __device__ __forceinline__ int8_t rocmfp4_decode_i8(uint8_t q) {
    q &= 0x0f;
    const int mag3 = q & 0x07;
    const int mag = mag3 <= 4 ? mag3 : 2*mag3 - 4;
    return (q & 0x08) ? -mag : mag;
}
