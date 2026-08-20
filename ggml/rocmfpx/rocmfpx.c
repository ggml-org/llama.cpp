#include "rocmfpx.h"

#include <assert.h>
#include <math.h>
#include <string.h>

// Finite unsigned E4M3 scale bytes decoded to FP32. Precomputed from the same
// exp/mant formula rocmfpx_ue4m3_to_fp32() used to evaluate with ldexpf():
//   exp == 0 -> mant * 2^-10 ; otherwise (8 + mant) * 2^(exp - 11).
// The scale search re-decodes candidate bytes for every block, and dequant
// decodes a scale for every element, so keeping this as a table (identical to
// the former per-call ldexpf result) removes the transcendental from both hot
// paths without changing any produced value.
#define ROCMFPX_SCALE_SUB(M) ((M) * 0x1p-10f)
#define ROCMFPX_SCALE_E(B, M) ((8 + (M)) * (B))

static const float rocmfpx_scale_ue4m3[127] = {
    ROCMFPX_SCALE_SUB(0),      ROCMFPX_SCALE_SUB(1),      ROCMFPX_SCALE_SUB(2),      ROCMFPX_SCALE_SUB(3),
    ROCMFPX_SCALE_SUB(4),      ROCMFPX_SCALE_SUB(5),      ROCMFPX_SCALE_SUB(6),      ROCMFPX_SCALE_SUB(7),
    ROCMFPX_SCALE_E(0x1p-10f,0), ROCMFPX_SCALE_E(0x1p-10f,1), ROCMFPX_SCALE_E(0x1p-10f,2), ROCMFPX_SCALE_E(0x1p-10f,3),
    ROCMFPX_SCALE_E(0x1p-10f,4), ROCMFPX_SCALE_E(0x1p-10f,5), ROCMFPX_SCALE_E(0x1p-10f,6), ROCMFPX_SCALE_E(0x1p-10f,7),
    ROCMFPX_SCALE_E(0x1p-9f,0),  ROCMFPX_SCALE_E(0x1p-9f,1),  ROCMFPX_SCALE_E(0x1p-9f,2),  ROCMFPX_SCALE_E(0x1p-9f,3),
    ROCMFPX_SCALE_E(0x1p-9f,4),  ROCMFPX_SCALE_E(0x1p-9f,5),  ROCMFPX_SCALE_E(0x1p-9f,6),  ROCMFPX_SCALE_E(0x1p-9f,7),
    ROCMFPX_SCALE_E(0x1p-8f,0),  ROCMFPX_SCALE_E(0x1p-8f,1),  ROCMFPX_SCALE_E(0x1p-8f,2),  ROCMFPX_SCALE_E(0x1p-8f,3),
    ROCMFPX_SCALE_E(0x1p-8f,4),  ROCMFPX_SCALE_E(0x1p-8f,5),  ROCMFPX_SCALE_E(0x1p-8f,6),  ROCMFPX_SCALE_E(0x1p-8f,7),
    ROCMFPX_SCALE_E(0x1p-7f,0),  ROCMFPX_SCALE_E(0x1p-7f,1),  ROCMFPX_SCALE_E(0x1p-7f,2),  ROCMFPX_SCALE_E(0x1p-7f,3),
    ROCMFPX_SCALE_E(0x1p-7f,4),  ROCMFPX_SCALE_E(0x1p-7f,5),  ROCMFPX_SCALE_E(0x1p-7f,6),  ROCMFPX_SCALE_E(0x1p-7f,7),
    ROCMFPX_SCALE_E(0x1p-6f,0),  ROCMFPX_SCALE_E(0x1p-6f,1),  ROCMFPX_SCALE_E(0x1p-6f,2),  ROCMFPX_SCALE_E(0x1p-6f,3),
    ROCMFPX_SCALE_E(0x1p-6f,4),  ROCMFPX_SCALE_E(0x1p-6f,5),  ROCMFPX_SCALE_E(0x1p-6f,6),  ROCMFPX_SCALE_E(0x1p-6f,7),
    ROCMFPX_SCALE_E(0x1p-5f,0),  ROCMFPX_SCALE_E(0x1p-5f,1),  ROCMFPX_SCALE_E(0x1p-5f,2),  ROCMFPX_SCALE_E(0x1p-5f,3),
    ROCMFPX_SCALE_E(0x1p-5f,4),  ROCMFPX_SCALE_E(0x1p-5f,5),  ROCMFPX_SCALE_E(0x1p-5f,6),  ROCMFPX_SCALE_E(0x1p-5f,7),
    ROCMFPX_SCALE_E(0x1p-4f,0),  ROCMFPX_SCALE_E(0x1p-4f,1),  ROCMFPX_SCALE_E(0x1p-4f,2),  ROCMFPX_SCALE_E(0x1p-4f,3),
    ROCMFPX_SCALE_E(0x1p-4f,4),  ROCMFPX_SCALE_E(0x1p-4f,5),  ROCMFPX_SCALE_E(0x1p-4f,6),  ROCMFPX_SCALE_E(0x1p-4f,7),
    ROCMFPX_SCALE_E(0x1p-3f,0),  ROCMFPX_SCALE_E(0x1p-3f,1),  ROCMFPX_SCALE_E(0x1p-3f,2),  ROCMFPX_SCALE_E(0x1p-3f,3),
    ROCMFPX_SCALE_E(0x1p-3f,4),  ROCMFPX_SCALE_E(0x1p-3f,5),  ROCMFPX_SCALE_E(0x1p-3f,6),  ROCMFPX_SCALE_E(0x1p-3f,7),
    ROCMFPX_SCALE_E(0x1p-2f,0),  ROCMFPX_SCALE_E(0x1p-2f,1),  ROCMFPX_SCALE_E(0x1p-2f,2),  ROCMFPX_SCALE_E(0x1p-2f,3),
    ROCMFPX_SCALE_E(0x1p-2f,4),  ROCMFPX_SCALE_E(0x1p-2f,5),  ROCMFPX_SCALE_E(0x1p-2f,6),  ROCMFPX_SCALE_E(0x1p-2f,7),
    ROCMFPX_SCALE_E(0x1p-1f,0),  ROCMFPX_SCALE_E(0x1p-1f,1),  ROCMFPX_SCALE_E(0x1p-1f,2),  ROCMFPX_SCALE_E(0x1p-1f,3),
    ROCMFPX_SCALE_E(0x1p-1f,4),  ROCMFPX_SCALE_E(0x1p-1f,5),  ROCMFPX_SCALE_E(0x1p-1f,6),  ROCMFPX_SCALE_E(0x1p-1f,7),
    ROCMFPX_SCALE_E(0x1p0f,0),   ROCMFPX_SCALE_E(0x1p0f,1),   ROCMFPX_SCALE_E(0x1p0f,2),   ROCMFPX_SCALE_E(0x1p0f,3),
    ROCMFPX_SCALE_E(0x1p0f,4),   ROCMFPX_SCALE_E(0x1p0f,5),   ROCMFPX_SCALE_E(0x1p0f,6),   ROCMFPX_SCALE_E(0x1p0f,7),
    ROCMFPX_SCALE_E(0x1p1f,0),   ROCMFPX_SCALE_E(0x1p1f,1),   ROCMFPX_SCALE_E(0x1p1f,2),   ROCMFPX_SCALE_E(0x1p1f,3),
    ROCMFPX_SCALE_E(0x1p1f,4),   ROCMFPX_SCALE_E(0x1p1f,5),   ROCMFPX_SCALE_E(0x1p1f,6),   ROCMFPX_SCALE_E(0x1p1f,7),
    ROCMFPX_SCALE_E(0x1p2f,0),   ROCMFPX_SCALE_E(0x1p2f,1),   ROCMFPX_SCALE_E(0x1p2f,2),   ROCMFPX_SCALE_E(0x1p2f,3),
    ROCMFPX_SCALE_E(0x1p2f,4),   ROCMFPX_SCALE_E(0x1p2f,5),   ROCMFPX_SCALE_E(0x1p2f,6),   ROCMFPX_SCALE_E(0x1p2f,7),
    ROCMFPX_SCALE_E(0x1p3f,0),   ROCMFPX_SCALE_E(0x1p3f,1),   ROCMFPX_SCALE_E(0x1p3f,2),   ROCMFPX_SCALE_E(0x1p3f,3),
    ROCMFPX_SCALE_E(0x1p3f,4),   ROCMFPX_SCALE_E(0x1p3f,5),   ROCMFPX_SCALE_E(0x1p3f,6),   ROCMFPX_SCALE_E(0x1p3f,7),
    ROCMFPX_SCALE_E(0x1p4f,0),   ROCMFPX_SCALE_E(0x1p4f,1),   ROCMFPX_SCALE_E(0x1p4f,2),   ROCMFPX_SCALE_E(0x1p4f,3),
    ROCMFPX_SCALE_E(0x1p4f,4),   ROCMFPX_SCALE_E(0x1p4f,5),   ROCMFPX_SCALE_E(0x1p4f,6),
};

#undef ROCMFPX_SCALE_SUB
#undef ROCMFPX_SCALE_E

float rocmfpx_ue4m3_to_fp32(uint8_t e) {
    return rocmfpx_scale_is_valid(e) ? rocmfpx_scale_ue4m3[e] : 0.0f;
}

bool rocmfpx_scale_is_valid(uint8_t e) {
    return e <= 0x7e;
}

// Precomputed table of all 127 valid UE4M3 scale values (e = 0x00..0x7e).
// Avoids repeated ldexpf calls in the hot MSE inner loops and in nearest-scale
// binary search. Matches the style of rocmfp4_scale_ue4m3_half in rocmfp4.c.
#define ROCMFPX_SCALE_SUB(M) ((M) * 0x1p-10f)
#define ROCMFPX_SCALE_E1(M)  ((8 + (M)) * 0x1p-10f)
#define ROCMFPX_SCALE_E2(M)  ((8 + (M)) * 0x1p-9f)
#define ROCMFPX_SCALE_E3(M)  ((8 + (M)) * 0x1p-8f)
#define ROCMFPX_SCALE_E4(M)  ((8 + (M)) * 0x1p-7f)
#define ROCMFPX_SCALE_E5(M)  ((8 + (M)) * 0x1p-6f)
#define ROCMFPX_SCALE_E6(M)  ((8 + (M)) * 0x1p-5f)
#define ROCMFPX_SCALE_E7(M)  ((8 + (M)) * 0x1p-4f)
#define ROCMFPX_SCALE_E8(M)  ((8 + (M)) * 0x1p-3f)
#define ROCMFPX_SCALE_E9(M)  ((8 + (M)) * 0x1p-2f)
#define ROCMFPX_SCALE_E10(M) ((8 + (M)) * 0x1p-1f)
#define ROCMFPX_SCALE_E11(M) ((8 + (M)) * 0x1p0f)
#define ROCMFPX_SCALE_E12(M) ((8 + (M)) * 0x1p1f)
#define ROCMFPX_SCALE_E13(M) ((8 + (M)) * 0x1p2f)
#define ROCMFPX_SCALE_E14(M) ((8 + (M)) * 0x1p3f)
#define ROCMFPX_SCALE_E15(M) ((8 + (M)) * 0x1p4f)

static const float rocmfpx_scale_table[127] = {
    ROCMFPX_SCALE_SUB(0), ROCMFPX_SCALE_SUB(1), ROCMFPX_SCALE_SUB(2), ROCMFPX_SCALE_SUB(3),
    ROCMFPX_SCALE_SUB(4), ROCMFPX_SCALE_SUB(5), ROCMFPX_SCALE_SUB(6), ROCMFPX_SCALE_SUB(7),
    ROCMFPX_SCALE_E1(0),  ROCMFPX_SCALE_E1(1),  ROCMFPX_SCALE_E1(2),  ROCMFPX_SCALE_E1(3),
    ROCMFPX_SCALE_E1(4),  ROCMFPX_SCALE_E1(5),  ROCMFPX_SCALE_E1(6),  ROCMFPX_SCALE_E1(7),
    ROCMFPX_SCALE_E2(0),  ROCMFPX_SCALE_E2(1),  ROCMFPX_SCALE_E2(2),  ROCMFPX_SCALE_E2(3),
    ROCMFPX_SCALE_E2(4),  ROCMFPX_SCALE_E2(5),  ROCMFPX_SCALE_E2(6),  ROCMFPX_SCALE_E2(7),
    ROCMFPX_SCALE_E3(0),  ROCMFPX_SCALE_E3(1),  ROCMFPX_SCALE_E3(2),  ROCMFPX_SCALE_E3(3),
    ROCMFPX_SCALE_E3(4),  ROCMFPX_SCALE_E3(5),  ROCMFPX_SCALE_E3(6),  ROCMFPX_SCALE_E3(7),
    ROCMFPX_SCALE_E4(0),  ROCMFPX_SCALE_E4(1),  ROCMFPX_SCALE_E4(2),  ROCMFPX_SCALE_E4(3),
    ROCMFPX_SCALE_E4(4),  ROCMFPX_SCALE_E4(5),  ROCMFPX_SCALE_E4(6),  ROCMFPX_SCALE_E4(7),
    ROCMFPX_SCALE_E5(0),  ROCMFPX_SCALE_E5(1),  ROCMFPX_SCALE_E5(2),  ROCMFPX_SCALE_E5(3),
    ROCMFPX_SCALE_E5(4),  ROCMFPX_SCALE_E5(5),  ROCMFPX_SCALE_E5(6),  ROCMFPX_SCALE_E5(7),
    ROCMFPX_SCALE_E6(0),  ROCMFPX_SCALE_E6(1),  ROCMFPX_SCALE_E6(2),  ROCMFPX_SCALE_E6(3),
    ROCMFPX_SCALE_E6(4),  ROCMFPX_SCALE_E6(5),  ROCMFPX_SCALE_E6(6),  ROCMFPX_SCALE_E6(7),
    ROCMFPX_SCALE_E7(0),  ROCMFPX_SCALE_E7(1),  ROCMFPX_SCALE_E7(2),  ROCMFPX_SCALE_E7(3),
    ROCMFPX_SCALE_E7(4),  ROCMFPX_SCALE_E7(5),  ROCMFPX_SCALE_E7(6),  ROCMFPX_SCALE_E7(7),
    ROCMFPX_SCALE_E8(0),  ROCMFPX_SCALE_E8(1),  ROCMFPX_SCALE_E8(2),  ROCMFPX_SCALE_E8(3),
    ROCMFPX_SCALE_E8(4),  ROCMFPX_SCALE_E8(5),  ROCMFPX_SCALE_E8(6),  ROCMFPX_SCALE_E8(7),
    ROCMFPX_SCALE_E9(0),  ROCMFPX_SCALE_E9(1),  ROCMFPX_SCALE_E9(2),  ROCMFPX_SCALE_E9(3),
    ROCMFPX_SCALE_E9(4),  ROCMFPX_SCALE_E9(5),  ROCMFPX_SCALE_E9(6),  ROCMFPX_SCALE_E9(7),
    ROCMFPX_SCALE_E10(0), ROCMFPX_SCALE_E10(1), ROCMFPX_SCALE_E10(2), ROCMFPX_SCALE_E10(3),
    ROCMFPX_SCALE_E10(4), ROCMFPX_SCALE_E10(5), ROCMFPX_SCALE_E10(6), ROCMFPX_SCALE_E10(7),
    ROCMFPX_SCALE_E11(0), ROCMFPX_SCALE_E11(1), ROCMFPX_SCALE_E11(2), ROCMFPX_SCALE_E11(3),
    ROCMFPX_SCALE_E11(4), ROCMFPX_SCALE_E11(5), ROCMFPX_SCALE_E11(6), ROCMFPX_SCALE_E11(7),
    ROCMFPX_SCALE_E12(0), ROCMFPX_SCALE_E12(1), ROCMFPX_SCALE_E12(2), ROCMFPX_SCALE_E12(3),
    ROCMFPX_SCALE_E12(4), ROCMFPX_SCALE_E12(5), ROCMFPX_SCALE_E12(6), ROCMFPX_SCALE_E12(7),
    ROCMFPX_SCALE_E13(0), ROCMFPX_SCALE_E13(1), ROCMFPX_SCALE_E13(2), ROCMFPX_SCALE_E13(3),
    ROCMFPX_SCALE_E13(4), ROCMFPX_SCALE_E13(5), ROCMFPX_SCALE_E13(6), ROCMFPX_SCALE_E13(7),
    ROCMFPX_SCALE_E14(0), ROCMFPX_SCALE_E14(1), ROCMFPX_SCALE_E14(2), ROCMFPX_SCALE_E14(3),
    ROCMFPX_SCALE_E14(4), ROCMFPX_SCALE_E14(5), ROCMFPX_SCALE_E14(6), ROCMFPX_SCALE_E14(7),
    ROCMFPX_SCALE_E15(0), ROCMFPX_SCALE_E15(1), ROCMFPX_SCALE_E15(2), ROCMFPX_SCALE_E15(3),
    ROCMFPX_SCALE_E15(4), ROCMFPX_SCALE_E15(5), ROCMFPX_SCALE_E15(6),
};

#undef ROCMFPX_SCALE_SUB
#undef ROCMFPX_SCALE_E1
#undef ROCMFPX_SCALE_E2
#undef ROCMFPX_SCALE_E3
#undef ROCMFPX_SCALE_E4
#undef ROCMFPX_SCALE_E5
#undef ROCMFPX_SCALE_E6
#undef ROCMFPX_SCALE_E7
#undef ROCMFPX_SCALE_E8
#undef ROCMFPX_SCALE_E9
#undef ROCMFPX_SCALE_E10
#undef ROCMFPX_SCALE_E11
#undef ROCMFPX_SCALE_E12
#undef ROCMFPX_SCALE_E13
#undef ROCMFPX_SCALE_E14
#undef ROCMFPX_SCALE_E15

// O(1) table lookup — use this everywhere inside this file instead of calling
// the public rocmfpx_ue4m3_to_fp32() which goes through the ldexpf path.
static inline float rocmfpx_scale_lookup(uint8_t e) {
    return e <= 0x7e ? rocmfpx_scale_table[e] : 0.0f;
}

// Binary search for the UE4M3 entry nearest to `target`.
// rocmfpx_scale_table is monotonically increasing in [0..0x7e], so we can
// binary-search to a two-element window then pick the closer neighbor.
// This replaces the original O(126) linear scan. Matches rocmfp4_nearest_scale_ue4m3.
static uint8_t rocmfpx_nearest_scale_ue4m3(float target) {
    if (!(target > 0.0f) || !isfinite(target)) {
        return 0;
    }

    int lo = 1;
    int hi = 126;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (rocmfpx_scale_table[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == 1) {
        return 1;
    }

    const float hi_scale = rocmfpx_scale_table[lo];
    const float lo_scale = rocmfpx_scale_table[lo - 1];

    // Ties keep the lower (smaller) scale byte, matching the old ascending scan.
    return (target - lo_scale <= hi_scale - target) ? (uint8_t)(lo - 1) : (uint8_t) lo;
}


size_t rocmfpx_row_size_fp3(int64_t k) {
    assert(k % QK_ROCMFP3 == 0);
    return (size_t) (k / QK_ROCMFP3) * sizeof(block_rocmfp3);
}

size_t rocmfpx_row_size_fp2(int64_t k) {
    assert(k % QK_ROCMFP2 == 0);
    return (size_t) (k / QK_ROCMFP2) * sizeof(block_rocmfp2);
}

size_t rocmfpx_row_size_fp6(int64_t k) {
    assert(k % QK_ROCMFP6 == 0);
    return (size_t) (k / QK_ROCMFP6) * sizeof(block_rocmfp6);
}

size_t rocmfpx_row_size_fp8(int64_t k) {
    assert(k % QK_ROCMFP8 == 0);
    return (size_t) (k / QK_ROCMFP8) * sizeof(block_rocmfp8);
}



static float rocmfpx_max_abs(const float * x, int n) {
    float max_abs = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float ax = fabsf(x[i]);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    return max_abs;
}

static void rocmfpx_prepare_mse_weights(
        float * dst, const float * x, int n, const float * quant_weights, float sigma2,
        float * max_abs, float * max_abs_weight, bool * all_finite) {
    *max_abs = 0.0f;
    *max_abs_weight = 0.0f;
    *all_finite = true;

    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        const float qw = quant_weights[i];
        const float weight = isfinite(qw) && qw > 0.0f && isfinite(x[i]) ? qw * sqrtf(sigma2 + x[i]*x[i]) : 0.0f;
        *all_finite = *all_finite && isfinite(x[i]);

        if (isfinite(x[i])) {
            if (ax > *max_abs) {
                *max_abs = ax;
                *max_abs_weight = weight;
            } else if (ax == *max_abs && weight > *max_abs_weight) {
                *max_abs_weight = weight;
            }
        }

        // Match llama.cpp imatrix weighting style: calibration importance is
        // scaled by row energy so large activations stay protected.
        dst[i] = weight;
    }
}

// ROCmFP2 S40 uses the frozen MORD code order {-4, -1, +1, +4}.
static inline int rocmfpx_decode_fp2_code(uint8_t code) {
    static const int8_t values[4] = { -4, -1, 1, 4 };
    return values[code & 3u];
}

static inline uint8_t rocmfpx_quantize_fp2_code(float x, float inv_scale) {
    if (!isfinite(x) || !(inv_scale > 0.0f)) {
        return 0;
    }

    const float magnitude = fabsf(x * inv_scale);
    const bool outer = magnitude > 2.5f;
    if (signbit(x)) {
        return outer ? 0u : 1u;
    }
    return outer ? 3u : 2u;
}

static float rocmfpx_fp2_group_mse_for_scale(
        const float * x, const float * mse_weights, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float reconstructed = (float) rocmfpx_decode_fp2_code(rocmfpx_quantize_fp2_code(x[i], inv_scale)) * scale;
        const float delta = x[i] - reconstructed;
        err += (mse_weights ? mse_weights[i] : 1.0f) * delta * delta;
        if (err > best_err) {
            return err;
        }
    }
    return err;
}

static uint8_t rocmfpx_choose_scale_fp2_mse(
        const float * x, int n, const float * quant_weights, float sigma2) {
    float mse_weights[QK_ROCMFP2/2];
    float max_abs = 0.0f;
    float max_abs_weight = 0.0f;
    bool all_finite = true;

    if (quant_weights) {
        rocmfpx_prepare_mse_weights(
                mse_weights, x, n, quant_weights, sigma2,
                &max_abs, &max_abs_weight, &all_finite);
    } else {
        max_abs = rocmfpx_max_abs(x, n);
        max_abs_weight = 1.0f;
    }
    GGML_UNUSED(all_finite);

    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const float * weights = quant_weights ? mse_weights : NULL;
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 4.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            const float clip_delta = max_abs - 4.0f * scale;
            const float clip_err = max_abs_weight * clip_delta * clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = rocmfpx_fp2_group_mse_for_scale(x, weights, n, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfpx_fp2_group_mse_for_scale(x, weights, n, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }
    return best_e;
}

static void rocmfpx_quantize_row_fp2_impl(
        const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y,
        int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP2 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i] * x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP2;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib * QK_ROCMFP2;
        const float * qw = quant_weights ? quant_weights + ib * QK_ROCMFP2 : NULL;
        block_rocmfp2 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP2 / 2);
            const float * xh = xb + half_off;
            const float * qh = qw ? qw + half_off : NULL;
            yb->e[half] = rocmfpx_choose_scale_fp2_mse(xh, QK_ROCMFP2 / 2, qh, sigma2);

            const float scale = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (int packed = 0; packed < 4; ++packed) {
                uint8_t byte = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    const int j = 4 * packed + lane;
                    byte |= (uint8_t) (rocmfpx_quantize_fp2_code(xh[j], inv_scale) << (2 * lane));
                }
                yb->qs[half * 4 + packed] = byte;
            }
        }
    }
}

void rocmfpx_quantize_row_fp2_ref(
        const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp2_impl(x, y, k, NULL);
}

void rocmfpx_dequantize_row_fp2(
        const block_rocmfp2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP2 == 0);
    const int64_t nb = k / QK_ROCMFP2;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp2 * xb = x + ib;
        float * yb = y + ib * QK_ROCMFP2;
        for (int half = 0; half < 2; ++half) {
            const float scale = rocmfpx_scale_lookup(xb->e[half]);
            for (int j = 0; j < QK_ROCMFP2 / 2; ++j) {
                const uint8_t code = (xb->qs[half * 4 + j / 4] >> (2 * (j % 4))) & 3u;
                yb[half * (QK_ROCMFP2 / 2) + j] = (float) rocmfpx_decode_fp2_code(code) * scale;
            }
        }
    }
}

void rocmfpx_quantize_row_fp2(
        const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp2_ref(x, (block_rocmfp2 *) y, k);
}

size_t rocmfpx_quantize_fp2(
        const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp2(n_per_row);
    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrows; ++row) {
        rocmfpx_quantize_row_fp2_impl(
                src + row * n_per_row, (block_rocmfp2 *) qrow, n_per_row, imatrix);
        qrow += row_size;
    }
    return (size_t) nrows * row_size;
}

// ---------------------------------------------------------------------------
// FP3 group pack/unpack: 8 × 3-bit codes → 3 bytes (24 bits)
//
// Bit layout across 3 bytes (LSB first within each byte):
//   byte 0: [v0 2:0][v1 2:0][v2 1:0]
//   byte 1: [v2.2  ][v3 2:0][v4 2:0][v5.0]
//   byte 2: [v5 2:1][v6 2:0][v7 2:0]
//
// A 16-element half-block is two consecutive pack8 groups.
// Half 0 → qs[0..5], Half 1 → qs[6..11].
// ---------------------------------------------------------------------------
static inline void rocmfpx_fp3_pack8(uint8_t * dst, const uint8_t * c) {
    dst[0] = (uint8_t)( (c[0] & 7u)        | ((c[1] & 7u) << 3) | ((c[2] & 3u) << 6) );
    dst[1] = (uint8_t)( ((c[2] >> 2) & 1u) | ((c[3] & 7u) << 1) | ((c[4] & 7u) << 4) | ((c[5] & 1u) << 7) );
    dst[2] = (uint8_t)( ((c[5] >> 1) & 3u) | ((c[6] & 7u) << 2) | ((c[7] & 7u) << 5) );
}

static inline void rocmfpx_fp3_unpack8(const uint8_t * src, uint8_t * c) {
    c[0] =  src[0]        & 7u;
    c[1] = (src[0] >> 3)  & 7u;
    c[2] = ((src[0] >> 6) & 3u) | ((src[1] & 1u) << 2);
    c[3] = (src[1] >> 1)  & 7u;
    c[4] = (src[1] >> 4)  & 7u;
    c[5] = ((src[1] >> 7) & 1u) | ((src[2] & 3u) << 1);
    c[6] = (src[2] >> 2)  & 7u;
    c[7] = (src[2] >> 5)  & 7u;
}

// ---------------------------------------------------------------------------
// FP6 group pack/unpack: 4 × 6-bit codes → 3 bytes (24 bits)
//
// Bit layout:
//   byte 0: [v0 5:0][v1 1:0]
//   byte 1: [v1 5:2][v2 3:0]
//   byte 2: [v2 5:4][v3 5:0]
//
// A 16-element half-block is four consecutive pack4 groups.
// Half 0 → qs[0..11], Half 1 → qs[12..23].
// ---------------------------------------------------------------------------
static inline void rocmfpx_fp6_pack4(uint8_t * dst, const uint8_t * c) {
    dst[0] = (uint8_t)( (c[0] & 0x3fu)        | ((c[1] & 0x03u) << 6) );
    dst[1] = (uint8_t)( ((c[1] >> 2) & 0x0fu) | ((c[2] & 0x0fu) << 4) );
    dst[2] = (uint8_t)( ((c[2] >> 4) & 0x03u) | ((c[3] & 0x3fu) << 2) );
}

static inline void rocmfpx_fp6_unpack4(const uint8_t * src, uint8_t * c) {
    c[0] =  src[0]         & 0x3fu;
    c[1] = ((src[0] >> 6)  & 0x03u) | ((src[1] & 0x0fu) << 2);
    c[2] = ((src[1] >> 4)  & 0x0fu) | ((src[2] & 0x03u) << 4);
    c[3] =  (src[2] >> 2)  & 0x3fu;
}

static int rocmfpx_decode_fp3_code(uint8_t code) {
    static const int mag[4] = { 0, 1, 2, 4 };
    const int value = mag[code & 3u];
    return (code & 4u) ? -value : value;
}

static uint8_t rocmfpx_quantize_fp3_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    const float ax = fabsf(x * inv_scale);
    uint8_t mag;

    if (ax <= 0.5f) {
        mag = 0;
    } else if (ax <= 1.5f) {
        mag = 1;
    } else if (ax <= 3.0f) {
        mag = 2;
    } else {
        mag = 3;
    }

    return mag == 0 ? 0 : (uint8_t) ((x < 0.0f ? 4u : 0u) | mag);
}

// Fused threshold + decode for finite values in the exhaustive scale search.
// The result exactly matches quantize_fp3_code followed by decode_fp3_code.
static inline float rocmfpx_fp3_decoded_mag(float x, float inv_scale) {
    const float a = fabsf(x * inv_scale);
    float mag;
    if (a <= 0.5f) {
        return 0.0f;
    } else if (a <= 1.5f) {
        mag = 1.0f;
    } else if (a <= 3.0f) {
        mag = 2.0f;
    } else {
        mag = 4.0f;
    }
    return x < 0.0f ? -mag : mag;
}

static float rocmfpx_fp3_block_mse_for_scale(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite — skips isfinite() per element.
static float rocmfpx_fp3_block_mse_for_scale_finite(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfpx_fp3_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite.
static float rocmfpx_fp3_block_weighted_mse_for_scale_finite(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp3_mse_impl(
        const float * x, int n, const float * mse_weights,
        float max_abs, float max_abs_weight, bool all_finite) {
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 4.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            const float clip_delta = max_abs - 4.0f*scale;
            const float clip_err = mse_weights ? max_abs_weight*clip_delta*clip_delta : clip_delta*clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = mse_weights ?
                    (all_finite ?
                        rocmfpx_fp3_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e0, best_err) :
                        rocmfpx_fp3_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err)) :
                    (all_finite ?
                        rocmfpx_fp3_block_mse_for_scale_finite(x, n, (uint8_t) e0, best_err) :
                        rocmfpx_fp3_block_mse_for_scale(x, n, (uint8_t) e0, best_err));
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = mse_weights ?
                (all_finite ?
                    rocmfpx_fp3_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e1, best_err) :
                    rocmfpx_fp3_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err)) :
                (all_finite ?
                    rocmfpx_fp3_block_mse_for_scale_finite(x, n, (uint8_t) e1, best_err) :
                    rocmfpx_fp3_block_mse_for_scale(x, n, (uint8_t) e1, best_err));
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

static uint8_t rocmfpx_choose_scale_fp3_mse(const float * x, int n) {
    const float max_abs = rocmfpx_max_abs(x, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }
    // rocmfpx_max_abs skips non-finite, but we need to know if all are finite
    // to select the fast path. Use a simple scan here since max_abs already ran.
    bool all_finite = true;
    for (int i = 0; i < n; ++i) {
        all_finite = all_finite && isfinite(x[i]);
    }

    return rocmfpx_choose_scale_fp3_mse_impl(x, n, NULL, max_abs, 0.0f, all_finite);
}

static uint8_t rocmfpx_choose_scale_fp3_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP3);
    float mse_weights[QK_ROCMFP3];
    float max_abs;
    float max_abs_weight;
    bool all_finite;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight, &all_finite);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp3_mse_impl(x, n, mse_weights, max_abs, max_abs_weight, all_finite);
}

static int rocmfpx_decode_fp6_code(uint8_t code) {
    const int mag = code & 31u;
    return (code & 32u) ? -(mag == 0 ? 32 : mag) : mag;
}

static uint8_t rocmfpx_quantize_fp6_code(float x, float inv_scale) {
    if (!isfinite(x) || !isfinite(inv_scale) || inv_scale <= 0.0f) {
        return 0;
    }

    int q = (int) lroundf(x * inv_scale);
    if (q > 31) {
        q = 31;
    } else if (q < -32) {
        q = -32;
    }

    return q == 0 ? 0 : (uint8_t) (q < 0 ? (32u | ((uint8_t) -q & 31u)) : (uint8_t) q);
}

// Fused round, clamp, and decode for finite values in the scale search. Keep
// current main's asymmetric signed range [-32, 31], including the encoded -32
// endpoint, rather than the older experimental branch's [-31, 31] behavior.
static inline float rocmfpx_fp6_decoded_value(float x, float inv_scale) {
    int q = (int) lroundf(x * inv_scale);
    if (q > 31) {
        q = 31;
    } else if (q < -32) {
        q = -32;
    }

    return (float) q;
}

static float rocmfpx_fp6_block_mse_for_scale(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float y = rocmfpx_fp6_decoded_value(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite.
static float rocmfpx_fp6_block_mse_for_scale_finite(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfpx_fp6_decoded_value(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfpx_fp6_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float y = rocmfpx_fp6_decoded_value(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite.
static float rocmfpx_fp6_block_weighted_mse_for_scale_finite(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfpx_fp6_decoded_value(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp6_mse_impl(
        const float * x, int n, const float * mse_weights,
        float max_abs, float max_abs_weight, bool all_finite) {
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 31.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            const float clip_delta = max_abs - 31.0f*scale;
            const float clip_err = mse_weights ? max_abs_weight*clip_delta*clip_delta : clip_delta*clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = mse_weights ?
                    (all_finite ?
                        rocmfpx_fp6_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e0, best_err) :
                        rocmfpx_fp6_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err)) :
                    (all_finite ?
                        rocmfpx_fp6_block_mse_for_scale_finite(x, n, (uint8_t) e0, best_err) :
                        rocmfpx_fp6_block_mse_for_scale(x, n, (uint8_t) e0, best_err));
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = mse_weights ?
                (all_finite ?
                    rocmfpx_fp6_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e1, best_err) :
                    rocmfpx_fp6_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err)) :
                (all_finite ?
                    rocmfpx_fp6_block_mse_for_scale_finite(x, n, (uint8_t) e1, best_err) :
                    rocmfpx_fp6_block_mse_for_scale(x, n, (uint8_t) e1, best_err));
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

static uint8_t rocmfpx_choose_scale_fp6_mse(const float * x, int n) {
    const float max_abs = rocmfpx_max_abs(x, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }
    bool all_finite = true;
    for (int i = 0; i < n; ++i) {
        all_finite = all_finite && isfinite(x[i]);
    }

    return rocmfpx_choose_scale_fp6_mse_impl(x, n, NULL, max_abs, 0.0f, all_finite);
}

static uint8_t rocmfpx_choose_scale_fp6_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP6);
    float mse_weights[QK_ROCMFP6];
    float max_abs;
    float max_abs_weight;
    bool all_finite;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight, &all_finite);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp6_mse_impl(x, n, mse_weights, max_abs, max_abs_weight, all_finite);
}

static int8_t rocmfpx_quantize_fp8_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    int q = (int) lroundf(x * inv_scale);
    if (q > 127) {
        q = 127;
    } else if (q < -127) {
        q = -127;
    }

    return (int8_t) q;
}

static float rocmfpx_fp8_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const int8_t code = rocmfpx_quantize_fp8_code(x[i], inv_scale);
        const float y = (float) code * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite.
static float rocmfpx_fp8_block_weighted_mse_for_scale_finite(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const int8_t code = rocmfpx_quantize_fp8_code(x[i], inv_scale);
        const float y = (float) code * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp8_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP8);
    float mse_weights[QK_ROCMFP8];
    float max_abs;
    float max_abs_weight;
    bool all_finite;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight, &all_finite);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 127.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            const float clip_delta = max_abs - 127.0f*scale;
            if (clip_delta > 0.0f && max_abs_weight*clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = all_finite ?
                    rocmfpx_fp8_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e0, best_err) :
                    rocmfpx_fp8_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = all_finite ?
                rocmfpx_fp8_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e1, best_err) :
                rocmfpx_fp8_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

void rocmfpx_quantize_row_fp3_ref(const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP3 == 0);

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP3;
        block_rocmfp3 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP3/2);
            const float * xh   = xb + half_off;
            yb->e[half] = rocmfpx_choose_scale_fp3_mse(xh, QK_ROCMFP3/2);

            const float scale     = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            // Compute 16 codes then pack into 6 bytes (2 groups of 8).
            uint8_t codes[QK_ROCMFP3/2];
            for (int j = 0; j < QK_ROCMFP3/2; ++j) {
                codes[j] = rocmfpx_quantize_fp3_code(xh[j], inv_scale);
            }
            rocmfpx_fp3_pack8(yb->qs + half_off*3/8,     codes);
            rocmfpx_fp3_pack8(yb->qs + half_off*3/8 + 3, codes + 8);
        }
    }
}

static void rocmfpx_quantize_row_fp3_weighted(
        const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP3 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i]*x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP3;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP3 : NULL;
        block_rocmfp3 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP3/2);
            const float * xh   = xb + half_off;
            const float * qh   = qw ? qw + half_off : NULL;
            yb->e[half] = qh ?
                rocmfpx_choose_scale_fp3_weighted_mse(xh, QK_ROCMFP3/2, qh, sigma2) :
                rocmfpx_choose_scale_fp3_mse(xh, QK_ROCMFP3/2);

            const float scale     = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            uint8_t codes[QK_ROCMFP3/2];
            for (int j = 0; j < QK_ROCMFP3/2; ++j) {
                codes[j] = rocmfpx_quantize_fp3_code(xh[j], inv_scale);
            }
            rocmfpx_fp3_pack8(yb->qs + half_off*3/8,     codes);
            rocmfpx_fp3_pack8(yb->qs + half_off*3/8 + 3, codes + 8);
        }
    }
}

void rocmfpx_dequantize_row_fp3(const block_rocmfp3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP3 == 0);

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp3 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP3;

        // Unpack all 32 codes in 4 groups of 8 (4 × 3 bytes).
        uint8_t codes[QK_ROCMFP3];
        rocmfpx_fp3_unpack8(xb->qs,      codes);
        rocmfpx_fp3_unpack8(xb->qs + 3,  codes + 8);
        rocmfpx_fp3_unpack8(xb->qs + 6,  codes + 16);
        rocmfpx_fp3_unpack8(xb->qs + 9,  codes + 24);

        for (int half = 0; half < 2; ++half) {
            const float scale = rocmfpx_scale_lookup(xb->e[half]);
            for (int j = 0; j < QK_ROCMFP3/2; ++j) {
                const int i = half*(QK_ROCMFP3/2) + j;
                yb[i] = (float) rocmfpx_decode_fp3_code(codes[i]) * scale;
            }
        }
    }
}

void rocmfpx_quantize_row_fp3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp3_ref(x, (block_rocmfp3 *) y, k);
}

size_t rocmfpx_quantize_fp3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp3(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp3_weighted(src + row*n_per_row, (block_rocmfp3 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp3_ref(src + row*n_per_row, (block_rocmfp3 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

static void rocmfpx_quantize_row_fp6_weighted(
        const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP6 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i]*x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP6;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP6 : NULL;
        block_rocmfp6 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP6/2);
            const float * xh   = xb + half_off;
            const float * qh   = qw ? qw + half_off : NULL;
            yb->e[half] = qh ?
                rocmfpx_choose_scale_fp6_weighted_mse(xh, QK_ROCMFP6/2, qh, sigma2) :
                rocmfpx_choose_scale_fp6_mse(xh, QK_ROCMFP6/2);

            const float scale     = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            // 16 codes packed into 4 groups of 4 (4 × 3 bytes = 12 bytes per half).
            uint8_t codes[QK_ROCMFP6/2];
            for (int j = 0; j < QK_ROCMFP6/2; ++j) {
                codes[j] = rocmfpx_quantize_fp6_code(xh[j], inv_scale);
            }
            uint8_t * qsdst = yb->qs + half_off * 6 / 8;
            rocmfpx_fp6_pack4(qsdst,      codes);
            rocmfpx_fp6_pack4(qsdst +  3, codes +  4);
            rocmfpx_fp6_pack4(qsdst +  6, codes +  8);
            rocmfpx_fp6_pack4(qsdst +  9, codes + 12);
        }
    }
}

void rocmfpx_quantize_row_fp6_ref(const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP6 == 0);

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP6;
        block_rocmfp6 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP6/2);
            const float * xh   = xb + half_off;
            yb->e[half] = rocmfpx_choose_scale_fp6_mse(xh, QK_ROCMFP6/2);

            const float scale     = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            uint8_t codes[QK_ROCMFP6/2];
            for (int j = 0; j < QK_ROCMFP6/2; ++j) {
                codes[j] = rocmfpx_quantize_fp6_code(xh[j], inv_scale);
            }
            uint8_t * qsdst = yb->qs + half_off * 6 / 8;
            rocmfpx_fp6_pack4(qsdst,      codes);
            rocmfpx_fp6_pack4(qsdst +  3, codes +  4);
            rocmfpx_fp6_pack4(qsdst +  6, codes +  8);
            rocmfpx_fp6_pack4(qsdst +  9, codes + 12);
        }
    }
}

void rocmfpx_dequantize_row_fp6(const block_rocmfp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP6 == 0);

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp6 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP6;

        // Unpack all 32 codes in 8 groups of 4 (8 × 3 bytes = 24 bytes).
        uint8_t codes[QK_ROCMFP6];
        rocmfpx_fp6_unpack4(xb->qs,      codes);
        rocmfpx_fp6_unpack4(xb->qs +  3, codes +  4);
        rocmfpx_fp6_unpack4(xb->qs +  6, codes +  8);
        rocmfpx_fp6_unpack4(xb->qs +  9, codes + 12);
        rocmfpx_fp6_unpack4(xb->qs + 12, codes + 16);
        rocmfpx_fp6_unpack4(xb->qs + 15, codes + 20);
        rocmfpx_fp6_unpack4(xb->qs + 18, codes + 24);
        rocmfpx_fp6_unpack4(xb->qs + 21, codes + 28);

        for (int half = 0; half < 2; ++half) {
            const float scale = rocmfpx_scale_lookup(xb->e[half]);
            for (int j = 0; j < QK_ROCMFP6/2; ++j) {
                const int i = half*(QK_ROCMFP6/2) + j;
                yb[i] = (float) rocmfpx_decode_fp6_code(codes[i]) * scale;
            }
        }
    }
}

void rocmfpx_quantize_row_fp6(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp6_ref(x, (block_rocmfp6 *) y, k);
}

size_t rocmfpx_quantize_fp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp6(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp6_weighted(src + row*n_per_row, (block_rocmfp6 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp6_ref(src + row*n_per_row, (block_rocmfp6 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

static void rocmfpx_quantize_row_fp8_weighted(
        const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP8 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i]*x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP8;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP8 : NULL;
        block_rocmfp8 * yb = y + ib;

        yb->e = qw ? rocmfpx_choose_scale_fp8_weighted_mse(xb, QK_ROCMFP8, qw, sigma2) :
                     rocmfpx_nearest_scale_ue4m3(rocmfpx_max_abs(xb, QK_ROCMFP8) / 127.0f);

        const float scale = rocmfpx_scale_lookup(yb->e);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb->qs[i] = rocmfpx_quantize_fp8_code(xb[i], inv_scale);
        }
    }
}

void rocmfpx_quantize_row_fp8_ref(const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP8 == 0);

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP8;
        block_rocmfp8 * yb = y + ib;

        const float max_abs = rocmfpx_max_abs(xb, QK_ROCMFP8);
        yb->e = rocmfpx_nearest_scale_ue4m3(max_abs / 127.0f);

        const float scale = rocmfpx_scale_lookup(yb->e);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb->qs[i] = rocmfpx_quantize_fp8_code(xb[i], inv_scale);
        }
    }
}

void rocmfpx_dequantize_row_fp8(const block_rocmfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP8 == 0);

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp8 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP8;

        const float scale = rocmfpx_scale_lookup(xb->e);
        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb[i] = (float) xb->qs[i] * scale;
        }
    }
}

void rocmfpx_quantize_row_fp8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp8_ref(x, (block_rocmfp8 *) y, k);
}

size_t rocmfpx_quantize_fp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp8(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp8_weighted(src + row*n_per_row, (block_rocmfp8 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp8_ref(src + row*n_per_row, (block_rocmfp8 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

bool rocmfpx_validate_row_data_fp2(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp2) != 0) {
        return false;
    }

    const block_rocmfp2 * blocks = (const block_rocmfp2 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp2);
    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }
    return true;
}

bool rocmfpx_validate_row_data_fp3(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp3) != 0) {
        return false;
    }

    const block_rocmfp3 * blocks = (const block_rocmfp3 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp3);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }

    return true;
}

bool rocmfpx_validate_row_data_fp6(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp6) != 0) {
        return false;
    }

    const block_rocmfp6 * blocks = (const block_rocmfp6 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp6);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }

    return true;
}

bool rocmfpx_validate_row_data_fp8(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp8) != 0) {
        return false;
    }

    const block_rocmfp8 * blocks = (const block_rocmfp8 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp8);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e)) {
            return false;
        }
    }

    return true;
}
