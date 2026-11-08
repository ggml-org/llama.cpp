#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <limits>

inline uint8_t float_to_e4m3(float f)
{
    if (sycl::isnan(f)) {
        return 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t bits = sycl::bit_cast<uint32_t>(f);
    uint32_t sign = (bits >> 31) & 0x1u;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;

    // Zero
    if (exp == 0 && mant == 0) {
        return static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    int e = static_cast<int>(exp) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    if (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        return static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    if (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        int shift = -6 - e;
        m = (m | 0x800000u) >> (shift + 1);        // +1 because we lose the implicit 1 position
        if (shift > 23) m = 0;
    } else {
        // Normal number: adjust exponent bias from 127 to 7
        int new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t round_bit = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    if (round_bit) {
        m += 1;
        // Carry into exponent if mantissa overflows
        if ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            if ((m >> 3) > 14) {
                return static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t result = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    return result;
}

inline float e4m3_to_float(uint8_t x)
{
    if (x == 0) return 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t exp  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    if (exp == 0xF && mant != 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (exp == 0xF) {                     // 0x7F or 0xFF treated as NaN
        return std::numeric_limits<float>::quiet_NaN();
    }

    float val;

    if (exp == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } else {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::pow(2.0f, static_cast<float>(exp) - 7.0f);
    }

    return sign ? -val : val;
}

// The actual type definition
struct __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = default;

    explicit __nv_fp8_e4m3(float f) : raw(float_to_e4m3(f)) {}
    explicit __nv_fp8_e4m3(sycl::half h) : raw(float_to_e4m3(static_cast<float>(h))) {}

    operator float() const { return e4m3_to_float(raw); }
    operator sycl::half() const { return static_cast<sycl::half>(static_cast<float>(*this)); }

    // Allow direct access for vector loads/stores
    operator uint8_t&() { return raw; }
    operator uint8_t() const { return raw; }
};

using __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
using __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

