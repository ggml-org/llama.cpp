#pragma once
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#define GGML_SSM_INLINE static __device__ __forceinline__
#else
#define GGML_SSM_INLINE static inline
#endif

GGML_SSM_INLINE uint32_t ggml_ssm_low_mask(int bits) {
    return bits >= 32 ? UINT32_MAX : (1U << bits) - 1;
}

// Return IEEE FP16 bits. The probability of choosing the larger magnitude is rem / 2^shift.
// Philox4x32-10 is counter based: the same logical state write has the same draw on CPU and CUDA.
GGML_SSM_INLINE uint16_t ggml_ssm_round_f16(uint32_t bits, uint32_t seed, uint32_t layer, uint32_t pos, uint64_t elem) {
    const uint32_t magnitude = bits & 0x7fffffff;
    const uint32_t sign = (bits >> 16) & 0x8000;
    if (magnitude >= 0x477fe000) {
        const uint32_t value = magnitude > 0x7f800000 ? 0x7e00 : magnitude >= 0x477ff000 ? 0x7c00 : 0x7bff;
        return (uint16_t) (sign | value);
    }

    const uint32_t exp = magnitude >> 23;
    uint32_t lower, rem;
    int shift;
    if (exp >= 113) {
        lower = (magnitude - 0x38000000) >> 13;
        rem = magnitude & 0x1fff;
        shift = 13;
    } else {
        const uint32_t mant = (magnitude & 0x7fffff) | (exp ? 0x800000 : 0);
        shift = exp ? 126 - (int) exp : 125;
        lower = shift >= 24 ? 0 : mant >> shift;
        rem = mant & ggml_ssm_low_mask(shift);
    }
    if (rem == 0) {
        return (uint16_t) (sign | lower);
    }

    uint32_t a = (uint32_t)elem, b = (uint32_t)(elem >> 32), c = pos, d = layer;
    uint32_t k0 = seed, k1 = 0;
    for (int round = 0; round < 10; ++round) {
        const uint64_t a_prod = (uint64_t)a * 0xd2511f53U;
        const uint64_t c_prod = (uint64_t)c * 0xcd9e8d57U;
        a = (uint32_t)(c_prod >> 32) ^ b ^ k0;
        c = (uint32_t)(a_prod >> 32) ^ d ^ k1;
        b = (uint32_t)c_prod;
        d = (uint32_t)a_prod;
        k0 += 0x9e3779b9U;
        k1 += 0xbb67ae85U;
    }
    // FP32 subnormals can be as little as 2^-125 of a half-subnormal step.
    const bool up = (a & ggml_ssm_low_mask(shift)) < rem &&
        (shift <= 32 || !(b & ggml_ssm_low_mask(shift - 32))) &&
        (shift <= 64 || !(c & ggml_ssm_low_mask(shift - 64))) &&
        (shift <= 96 || !(d & ggml_ssm_low_mask(shift - 96)));
    return (uint16_t)(sign | (lower + up));
}

#undef GGML_SSM_INLINE
