#pragma once

#include "ggml-metal-impl.h"

#include <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#if __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#if defined(GGML_METAL_HAS_BF16)
typedef matrix<bfloat, 4, 4> bfloat4x4;
typedef matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static float kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

constexpr constant static float kvalues_jq2k_f[4] = {
    -19.f, -4.f, 8.f, 26.f
};

constexpr constant static float kvalues_jq3k_f[8] = {
    -51.f, -30.f, -16.f, -4.f, 7.f, 20.f, 36.f, 58.f
};

constexpr constant static float kvalues_jq4k_f[16] = {
    -127.f, -97.f, -78.f, -62.f, -48.f, -35.f, -23.f, -12.f, -1.f, 11.f, 23.f, 36.f, 51.f, 67.f, 87.f, 105.f
};

constexpr constant static float kvalues_jq5k_f[32] = {
    -115.f, -105.f, -94.f, -83.f, -73.f, -65.f, -58.f, -51.f, -45.f, -39.f, -33.f, -27.f, -22.f, -17.f, -11.f,  -6.f,
      -1.f,    4.f,   9.f,  15.f,  20.f,  26.f,  32.f,  39.f,  46.f,  53.f,  61.f,  69.f,  77.f,  86.f, 103.f, 111.f
};

constexpr constant static float kvalues_jq6k_f[64] = {
    -120.f, -115.f, -108.f, -103.f, -99.f, -93.f, -88.f, -83.f, -79.f, -75.f, -70.f, -66.f, -62.f, -59.f, -55.f, -51.f,
     -48.f,  -45.f,  -41.f,  -38.f, -35.f, -32.f, -29.f, -26.f, -24.f, -21.f, -18.f, -15.f, -13.f, -10.f,  -7.f,  -5.f,
      -2.f,    0.f,    3.f,    6.f,   8.f,  11.f,  14.f,  16.f,  19.f,  22.f,  25.f,  28.f,  31.f,  34.f,  38.f,  41.f,
      44.f,   48.f,   52.f,   56.f,  60.f,  64.f,  68.f,  73.f,  77.f,  82.f,  86.f,  93.f,  97.f, 104.f, 109.f, 114.f
};

constexpr constant static float JQ2K_PHASE_F = 5.f;
constexpr constant static float JQ3K_PHASE_F = 5.f;
constexpr constant static float JQ4K_PHASE_F = 18.f;
constexpr constant static float JQ5K_PHASE_F = 15.f;
constexpr constant static float JQ6K_PHASE_F = 13.f;

static inline int best_index_int8(int n, constant float * val, float x) {
    // val is sorted, so x picks level i once it reaches the midpoint of val[i-1] and val[i].
    // n must be a power of 2
    const float x2 = x + x;
    int idx = 0;
    for (int step = n >> 1; step > 0; step >>= 1) {
        const int j = idx + step;
        idx = (x2 >= val[j-1] + val[j]) ? j : idx;
    }
    return idx;
}

static inline float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    return as_type<float>(bits);
}

static inline float dot(float x, float y) {
    return x*y;
}

static inline float sum(float x) {
    return x;
}

static inline float sum(float4 x) {
    return x[0] + x[1] + x[2] + x[3];
}

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant float p_erf  = 0.3275911f;
constant float a1_erf = 0.254829592f;
constant float a2_erf = -0.284496736f;
constant float a3_erf = 1.421413741f;
constant float a4_erf = -1.453152027f;
constant float a5_erf = 1.061405429f;

template<typename T>
inline T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    return sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> inline float elu_approx<float>(float x) {
    return (x > 0.f) ? x : (exp(x) - 1);
}

template<> inline float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    return res;
}
