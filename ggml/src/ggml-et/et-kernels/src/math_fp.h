//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#ifndef MATH_FP_H
#define MATH_FP_H

#include <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
static inline float et_fdiv(float a, float b) {
    float d;
    unsigned long temp;

    __asm__ volatile (
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [temp] "=&r"(temp), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b)
    );

    return d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
static inline float et_powf(float base, float exp) {
    // Handle special cases
    if (base <= 0.0f) {
        if (base == 0.0f) {
            if (exp > 0.0f) return 0.0f;
            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            union { float f; uint32_t i; } inf = { .i = 0x7F800000 };
            return inf.f;
        }
        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        union { float f; uint32_t i; } nan = { .i = 0x7FC00000 };
        return nan.f;
    }
    if (base == 1.0f) return 1.0f;
    if (exp == 0.0f) return 1.0f;
    if (exp == 1.0f) return base;

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    float result;
    unsigned long temp;

    __asm__ volatile (
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"  // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"  // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [base] "f"(base), [exp] "f"(exp)
    );

    return result;
}

// Natural logarithm.
static inline float et_logf(float x) {
    // Handle special cases
    if (x < 0.0f) {
        // Return NaN for negative input
        union { float f; uint32_t i; } nan = { .i = 0x7FC00000 };
        return nan.f;
    }
    if (x == 0.0f) {
        // Return -infinity for log(0)
        union { float f; uint32_t i; } inf = { .i = 0xFF800000 };
        return inf.f;
    }
    if (x == 1.0f) return 0.0f;

    float log2_result;
    unsigned long temp;

    __asm__ volatile (
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(log2_result)
        : [x] "f"(x)
    );

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    const float ln2 = 0.69314718055994530942f;
    return log2_result * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
static inline float et_sqrtf(float x) {
    // Handle special cases
    if (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        union { float f; uint32_t i; } nan = { .i = 0x7FC00000 };
        return nan.f;
    }
    if (x == 0.0f) return 0.0f;

    return et_powf(x, 0.5f);
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
static inline float et_expf(float x) {
    // Handle special cases
    if (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        union { float f; uint32_t i; } inf = { .i = 0x7F800000 };
        return inf.f;
    }
    if (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        return 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    const float log2e = 1.4426950408889634f;  // log2(e)
    float x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    float result;
    unsigned long temp;

    __asm__ volatile (
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [x_log2e] "f"(x_log2e)
    );

    return result;
}

//******************************************************************************
// Trigonometric Functions (Taylor Series)
// Simple implementations using only basic arithmetic operations
//******************************************************************************

// Sine function using Taylor series approximation
static inline float et_sinf(float x) {
    // Normalize to [-pi, pi] range first
    const float pi = 3.14159265f;
    const float two_pi = 6.28318531f;

    // Simple range reduction (not perfect but good enough for ROPE)
    while (x > pi) x -= two_pi;
    while (x < -pi) x += two_pi;

    // Taylor series: sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float x5 = x3 * x2;
    const float x7 = x5 * x2;

    return x - (x3 * et_fdiv(1.0f, 6.0f)) + (x5 * et_fdiv(1.0f, 120.0f)) - (x7 * et_fdiv(1.0f, 5040.0f));
}

// Cosine function using Taylor series approximation
static inline float et_cosf(float x) {
    // Normalize to [-pi, pi] range first
    const float pi = 3.14159265f;
    const float two_pi = 6.28318531f;

    // Simple range reduction (not perfect but good enough for ROPE)
    while (x > pi) x -= two_pi;
    while (x < -pi) x += two_pi;

    // Taylor series: cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
    const float x2 = x * x;
    const float x4 = x2 * x2;
    const float x6 = x4 * x2;

    return 1.0f - (x2 * et_fdiv(1.0f, 2.0f)) + (x4 * et_fdiv(1.0f, 24.0f)) - (x6 * et_fdiv(1.0f, 720.0f));
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
static inline float fp16_to_fp32(uint16_t h) {
    // Extract sign, exponent, mantissa
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h & 0x7C00);
    uint32_t mantissa = (h & 0x03FF);

    if (exp == 0) {
        // Subnormal or zero
        if (mantissa == 0) {
            return *(float*)&sign; // Return signed zero
        }
        // Convert subnormal
        exp = 0x38800000; // 2^-14 in fp32
        mantissa <<= 13;
        // Normalize
        while ((mantissa & 0x00800000) == 0) {
            mantissa <<= 1;
            exp -= 0x00800000;
        }
        mantissa &= 0x007FFFFF;
        uint32_t result = sign | exp | mantissa;
        return *(float*)&result;
    } else if (exp == 0x7C00) {
        // Infinity or NaN
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *(float*)&result;
    } else {
        // Normal number
        exp = ((exp >> 10) - 15 + 127) << 23; // Convert exponent
        mantissa <<= 13;
        uint32_t result = sign | exp | mantissa;
        return *(float*)&result;
    }
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
static inline uint16_t fp32_to_fp16(float f) {
    float result_f;
    unsigned long temp;

    __asm__ volatile (
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result_f)
        : [f] "f"(f)
    );

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t result_bits = *(uint32_t*)&result_f;
    return (uint16_t)result_bits;
}

#endif // MATH_FP_H
