#define GGML_COMMON_DECL_C
#include "../src/ggml-common.h"

#include "rocmfp4.h"

#include <assert.h>
#include <float.h>
#include <math.h>

// ggml-base is compiled architecture-neutral (no -mavx2), so SIMD for the hot
// CPU dot product is enabled per-function via a target attribute plus a runtime
// CPU check. This keeps the AVX2 path in one translation unit without exporting
// internals to ggml-cpu. Non-GNU/non-x86 builds fall back to the scalar loop.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)) && !defined(ROCMFP4_NO_AVX2)
#include <immintrin.h>
#define ROCMFP4_X86_AVX2_DISPATCH 1
#endif

// ROCmFP4 stores a signed integer FP4-like codebook at half-scale. It is
// E2M1-derived, but the largest magnitude is retuned from 12 to 10 after
// sampling Qwen3 dense tensors; this reduces outlier pull without changing the
// packed 4-bit layout or integer dot-product path.
static const int8_t rocmfp4_codebook[16] = {
     0,  1,  2,  3,  4,  6,  8, 10,
     0, -1, -2, -3, -4, -6, -8,-10,
};

static inline int8_t rocmfp4_decode(uint8_t q) {
    q &= 0x0f;
    const int mag3 = q & 0x07;
    const int mag = mag3 <= 4 ? mag3 : 2*mag3 - 4;
    return (q & 0x08) ? -mag : mag;
}

static inline int8_t rocmfp4_decode_table(uint8_t q) {
    return rocmfp4_codebook[q & 0x0f];
}

// Finite unsigned E4M3 scale bytes decoded to the half-scale values used by
// ROCmFP4. Keeping this as a table avoids rebuilding identical FP32 values for
// every candidate during exhaustive scale search.
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

static const float rocmfp4_scale_ue4m3_half[127] = {
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

static inline float rocmfp4_ue4m3_to_fp32_half(uint8_t e) {
    return e <= 0x7e ? rocmfp4_scale_ue4m3_half[e] : 0.0f;
}

static inline uint8_t rocmfp4_best_index_scaled_finite(float x, float inv_scale_half) {
    // Exact nearest-neighbor thresholds for Codebook10:
    //   0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10
    // Ties intentionally choose the lower-magnitude code, matching the former
    // linear scan because the positive codes and zero appear first.
    const float a = fabsf(x * inv_scale_half);
    if (a <= 0.5f) {
        return 0;
    }

    const bool neg = x < 0.0f;
    if (a <= 1.5f) {
        return neg ?  9 : 1;
    }
    if (a <= 2.5f) {
        return neg ? 10 : 2;
    }
    if (a <= 3.5f) {
        return neg ? 11 : 3;
    }
    if (a <= 5.0f) {
        return neg ? 12 : 4;
    }
    if (a <= 7.0f) {
        return neg ? 13 : 5;
    }
    if (a <= 9.0f) {
        return neg ? 14 : 6;
    }

    return neg ? 15 : 7;
}

static inline uint8_t rocmfp4_best_index_scaled(float x, float inv_scale_half) {
    if (!isfinite(x)) {
        return 0;
    }

    return rocmfp4_best_index_scaled_finite(x, inv_scale_half);
}

// Fused best-index + decode used only inside the exhaustive scale search. The
// scale search re-scans every block element for every candidate scale byte, so
// avoiding the code -> decode round-trip on the hottest quantize path matters.
// Returns the same signed Codebook10 magnitude that
// rocmfp4_decode(rocmfp4_best_index_scaled_finite(x, inv_scale_half)) produces,
// so quantized output is bit-identical to the previous path.
static inline float rocmfp4_decoded_mag_scaled_finite(float x, float inv_scale_half) {
    const float a = fabsf(x * inv_scale_half);

    float mag;
    if (a <= 0.5f) {
        mag = 0.0f;
    } else if (a <= 1.5f) {
        mag = 1.0f;
    } else if (a <= 2.5f) {
        mag = 2.0f;
    } else if (a <= 3.5f) {
        mag = 3.0f;
    } else if (a <= 5.0f) {
        mag = 4.0f;
    } else if (a <= 7.0f) {
        mag = 6.0f;
    } else if (a <= 9.0f) {
        mag = 8.0f;
    } else {
        mag = 10.0f;
    }

    return x < 0.0f ? -mag : mag;
}

static inline float rocmfp4_decoded_mag_scaled(float x, float inv_scale_half) {
    if (!isfinite(x)) {
        return 0.0f;
    }

    return rocmfp4_decoded_mag_scaled_finite(x, inv_scale_half);
}

static inline bool rocmfp4_scale_is_valid(uint8_t e) {
    // ROCmFP4 scale bytes are unsigned finite E4M3 values. 0x7f is NaN in the
    // unsigned encoding and values with the sign bit set are not valid scales.
    return e <= 0x7e;
}

static float rocmfp4_block_mse_for_scale_unweighted(
        const float * x, int n, int e, float best_err) {
    const float scale_half = rocmfp4_ue4m3_to_fp32_half((uint8_t) e);
    const float inv_scale_half = 1.0f / scale_half;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfp4_decoded_mag_scaled(x[i], inv_scale_half) * scale_half;
        const float d = x[i] - y;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfp4_block_mse_for_scale_unweighted_finite(
        const float * x, int n, int e, float best_err) {
    const float scale_half = rocmfp4_ue4m3_to_fp32_half((uint8_t) e);
    const float inv_scale_half = 1.0f / scale_half;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfp4_decoded_mag_scaled_finite(x[i], inv_scale_half) * scale_half;
        const float d = x[i] - y;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfp4_block_mse_for_scale_weighted(
        const float * x, int n, const float * mse_weights, int e, float best_err) {
    const float scale_half = rocmfp4_ue4m3_to_fp32_half((uint8_t) e);
    const float inv_scale_half = 1.0f / scale_half;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfp4_decoded_mag_scaled(x[i], inv_scale_half) * scale_half;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfp4_block_mse_for_scale_weighted_finite(
        const float * x, int n, const float * mse_weights, int e, float best_err) {
    const float scale_half = rocmfp4_ue4m3_to_fp32_half((uint8_t) e);
    const float inv_scale_half = 1.0f / scale_half;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfp4_decoded_mag_scaled_finite(x[i], inv_scale_half) * scale_half;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static void rocmfp4_prepare_mse_weights(
        float * dst, const float * x, int n, const float * quant_weights, float sigma2,
        float * max_abs, float * max_abs_weight, bool * all_finite) {
    *max_abs = 0.0f;
    *max_abs_weight = 0.0f;
    *all_finite = true;

    for (int i = 0; i < n; ++i) {
        const float qw = quant_weights[i];
        const float ax = fabsf(x[i]);
        const float weight = isfinite(qw) && qw > 0.0f ? qw * sqrtf(sigma2 + x[i]*x[i]) : 0.0f;
        *all_finite = *all_finite && isfinite(x[i]);

        if (ax > *max_abs) {
            *max_abs = ax;
            *max_abs_weight = weight;
        } else if (ax == *max_abs && weight > *max_abs_weight) {
            *max_abs_weight = weight;
        }

        // Match llama.cpp's imatrix weighting style for Q4_0: calibration
        // importance is scaled by row energy so large activations remain protected.
        dst[i] = weight;
    }
}

static int rocmfp4_nearest_scale_ue4m3(float target_scale_half) {
    if (!(target_scale_half > 0.0f) || !isfinite(target_scale_half)) {
        return 1;
    }

    int lo = 1;
    int hi = 126;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (rocmfp4_ue4m3_to_fp32_half((uint8_t) mid) < target_scale_half) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == 1) {
        return 1;
    }

    const float hi_scale = rocmfp4_ue4m3_to_fp32_half((uint8_t) lo);
    const float lo_scale = rocmfp4_ue4m3_to_fp32_half((uint8_t) (lo - 1));

    // Match the former ascending nearest scan: exact midpoint ties keep the
    // lower scale byte.
    return (target_scale_half - lo_scale <= hi_scale - target_scale_half) ? lo - 1 : lo;
}

static uint8_t rocmfp4_choose_scale_ue4m3_exhaustive_unweighted(
        const float * x, int n, float max_abs, bool all_finite) {
    const int start_e = rocmfp4_nearest_scale_ue4m3(max_abs / 10.0f);

    int best_e = 0;
    float best_err = FLT_MAX;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale_half = rocmfp4_ue4m3_to_fp32_half((uint8_t) e0);
            const float clip_delta = max_abs - 10.0f*scale_half;
            if (clip_delta > 0.0f && clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = all_finite ?
                    rocmfp4_block_mse_for_scale_unweighted_finite(x, n, e0, best_err) :
                    rocmfp4_block_mse_for_scale_unweighted(x, n, e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = e0;
                }
            }
        }

        const int e1 = start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = all_finite ?
                rocmfp4_block_mse_for_scale_unweighted_finite(x, n, e1, best_err) :
                rocmfp4_block_mse_for_scale_unweighted(x, n, e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return (uint8_t) best_e;
}

static uint8_t rocmfp4_choose_scale_ue4m3_exhaustive_weighted(
        const float * x, int n, const float * mse_weights, float max_abs, float max_abs_weight, bool all_finite) {
    const int start_e = rocmfp4_nearest_scale_ue4m3(max_abs / 10.0f);

    int best_e = 0;
    float best_err = FLT_MAX;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale_half = rocmfp4_ue4m3_to_fp32_half((uint8_t) e0);
            const float clip_delta = max_abs - 10.0f*scale_half;
            if (max_abs_weight > 0.0f && clip_delta > 0.0f && max_abs_weight*clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = all_finite ?
                    rocmfp4_block_mse_for_scale_weighted_finite(x, n, mse_weights, e0, best_err) :
                    rocmfp4_block_mse_for_scale_weighted(x, n, mse_weights, e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = e0;
                }
            }
        }

        const int e1 = start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = all_finite ?
                rocmfp4_block_mse_for_scale_weighted_finite(x, n, mse_weights, e1, best_err) :
                rocmfp4_block_mse_for_scale_weighted(x, n, mse_weights, e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return (uint8_t) best_e;
}

static uint8_t rocmfp4_choose_scale_ue4m3(const float * x, int n, const float * quant_weights, float sigma2) {
    if (quant_weights) {
        assert(n <= QK_ROCMFP4);
        float mse_weights_buf[QK_ROCMFP4];
        float weighted_max_abs;
        float max_abs_weight;
        bool all_finite;
        rocmfp4_prepare_mse_weights(mse_weights_buf, x, n, quant_weights, sigma2, &weighted_max_abs, &max_abs_weight, &all_finite);
        if (!(weighted_max_abs > 0.0f) || !isfinite(weighted_max_abs)) {
            return 0;
        }
        return rocmfp4_choose_scale_ue4m3_exhaustive_weighted(x, n, mse_weights_buf, weighted_max_abs, max_abs_weight, all_finite);
    }

    float max_abs = 0.0f;
    bool all_finite = true;
    for (int i = 0; i < n; ++i) {
        all_finite = all_finite && isfinite(x[i]);
        const float ax = fabsf(x[i]);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfp4_choose_scale_ue4m3_exhaustive_unweighted(x, n, max_abs, all_finite);
}

static void rocmfp4_quantize_row_q4_0_weighted(
        const float * GGML_RESTRICT x, block_rocmfp4 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP4 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += x[i]*x[i];
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP4;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP4 : NULL;
        const uint8_t e0 = rocmfp4_choose_scale_ue4m3(xb,                 QK_ROCMFP4/2, qw,                              sigma2);
        const uint8_t e1 = rocmfp4_choose_scale_ue4m3(xb + QK_ROCMFP4/2, QK_ROCMFP4/2, qw ? qw + QK_ROCMFP4/2 : NULL, sigma2);
        const float scale_half0 = rocmfp4_ue4m3_to_fp32_half(e0);
        const float scale_half1 = rocmfp4_ue4m3_to_fp32_half(e1);
        const float inv_scale_half0 = scale_half0 > 0.0f ? 1.0f / scale_half0 : 0.0f;
        const float inv_scale_half1 = scale_half1 > 0.0f ? 1.0f / scale_half1 : 0.0f;

        y[ib].e[0] = e0;
        y[ib].e[1] = e1;

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            const uint8_t q0 = rocmfp4_best_index_scaled(xb[j],                 inv_scale_half0);
            const uint8_t q1 = rocmfp4_best_index_scaled(xb[j + QK_ROCMFP4/2], inv_scale_half1);
            y[ib].qs[j] = q0 | (q1 << 4);
        }
    }
}

static void rocmfp4_quantize_row_q4_0_fast_weighted(
        const float * GGML_RESTRICT x, block_rocmfp4_fast * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP4 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += x[i]*x[i];
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP4;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP4 : NULL;
        const uint8_t e = rocmfp4_choose_scale_ue4m3(xb, QK_ROCMFP4, qw, sigma2);
        const float scale_half = rocmfp4_ue4m3_to_fp32_half(e);
        const float inv_scale_half = scale_half > 0.0f ? 1.0f / scale_half : 0.0f;

        y[ib].e = e;

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            const uint8_t q0 = rocmfp4_best_index_scaled(xb[j],                 inv_scale_half);
            const uint8_t q1 = rocmfp4_best_index_scaled(xb[j + QK_ROCMFP4/2], inv_scale_half);
            y[ib].qs[j] = q0 | (q1 << 4);
        }
    }
}

void rocmfp4_quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_rocmfp4 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP4 == 0);

    const int64_t nb = k / QK_ROCMFP4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP4;
        const uint8_t e0 = rocmfp4_choose_scale_ue4m3(xb,                 QK_ROCMFP4/2, NULL, 0.0f);
        const uint8_t e1 = rocmfp4_choose_scale_ue4m3(xb + QK_ROCMFP4/2, QK_ROCMFP4/2, NULL, 0.0f);
        const float scale_half0 = rocmfp4_ue4m3_to_fp32_half(e0);
        const float scale_half1 = rocmfp4_ue4m3_to_fp32_half(e1);
        const float inv_scale_half0 = scale_half0 > 0.0f ? 1.0f / scale_half0 : 0.0f;
        const float inv_scale_half1 = scale_half1 > 0.0f ? 1.0f / scale_half1 : 0.0f;

        y[ib].e[0] = e0;
        y[ib].e[1] = e1;

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            const uint8_t q0 = rocmfp4_best_index_scaled(xb[j],                 inv_scale_half0);
            const uint8_t q1 = rocmfp4_best_index_scaled(xb[j + QK_ROCMFP4/2], inv_scale_half1);
            y[ib].qs[j] = q0 | (q1 << 4);
        }
    }
}

void rocmfp4_quantize_row_q4_0_fast_ref(const float * GGML_RESTRICT x, block_rocmfp4_fast * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP4 == 0);

    const int64_t nb = k / QK_ROCMFP4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP4;
        const uint8_t e = rocmfp4_choose_scale_ue4m3(xb, QK_ROCMFP4, NULL, 0.0f);
        const float scale_half = rocmfp4_ue4m3_to_fp32_half(e);
        const float inv_scale_half = scale_half > 0.0f ? 1.0f / scale_half : 0.0f;

        y[ib].e = e;

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            const uint8_t q0 = rocmfp4_best_index_scaled(xb[j],                 inv_scale_half);
            const uint8_t q1 = rocmfp4_best_index_scaled(xb[j + QK_ROCMFP4/2], inv_scale_half);
            y[ib].qs[j] = q0 | (q1 << 4);
        }
    }
}

void rocmfp4_dequantize_row_q4_0(const block_rocmfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP4 == 0);

    const int64_t nb = k / QK_ROCMFP4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float d0 = rocmfp4_ue4m3_to_fp32_half(x[ib].e[0]);
        const float d1 = rocmfp4_ue4m3_to_fp32_half(x[ib].e[1]);

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            y[ib*QK_ROCMFP4 + j]                 = (float) rocmfp4_decode(x[ib].qs[j] & 0x0f) * d0;
            y[ib*QK_ROCMFP4 + j + QK_ROCMFP4/2] = (float) rocmfp4_decode(x[ib].qs[j] >> 4)   * d1;
        }
    }
}

void rocmfp4_dequantize_row_q4_0_fast(const block_rocmfp4_fast * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP4 == 0);

    const int64_t nb = k / QK_ROCMFP4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float d = rocmfp4_ue4m3_to_fp32_half(x[ib].e);

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            y[ib*QK_ROCMFP4 + j]                 = (float) rocmfp4_decode(x[ib].qs[j] & 0x0f) * d;
            y[ib*QK_ROCMFP4 + j + QK_ROCMFP4/2] = (float) rocmfp4_decode(x[ib].qs[j] >> 4)   * d;
        }
    }
}

void rocmfp4_quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfp4_quantize_row_q4_0_ref(x, (block_rocmfp4 *) y, k);
}

void rocmfp4_quantize_row_q4_0_fast(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfp4_quantize_row_q4_0_fast_ref(x, (block_rocmfp4_fast *) y, k);
}

size_t rocmfp4_quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_0_ROCMFP4, n_per_row);

    if (!imatrix) {
        rocmfp4_quantize_row_q4_0_ref(src, (block_rocmfp4 *) dst, nrows*n_per_row);
        return nrows * row_size;
    }

    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrows; ++row) {
        rocmfp4_quantize_row_q4_0_weighted(src, (block_rocmfp4 *) qrow, n_per_row, imatrix);
        src += n_per_row;
        qrow += row_size;
    }

    return nrows * row_size;
}

size_t rocmfp4_quantize_q4_0_fast(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_0_ROCMFP4_FAST, n_per_row);

    if (!imatrix) {
        rocmfp4_quantize_row_q4_0_fast_ref(src, (block_rocmfp4_fast *) dst, nrows*n_per_row);
        return nrows * row_size;
    }

    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrows; ++row) {
        rocmfp4_quantize_row_q4_0_fast_weighted(src, (block_rocmfp4_fast *) qrow, n_per_row, imatrix);
        src += n_per_row;
        qrow += row_size;
    }

    return nrows * row_size;
}

bool rocmfp4_validate_row_data(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp4) != 0) {
        return false;
    }

    const block_rocmfp4 * blocks = (const block_rocmfp4 *) data;
    const size_t nblocks = nbytes / sizeof(block_rocmfp4);
    for (size_t i = 0; i < nblocks; ++i) {
        if (!rocmfp4_scale_is_valid(blocks[i].e[0]) || !rocmfp4_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }

    return true;
}

bool rocmfp4_validate_row_data_fast(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp4_fast) != 0) {
        return false;
    }

    const block_rocmfp4_fast * blocks = (const block_rocmfp4_fast *) data;
    const size_t nblocks = nbytes / sizeof(block_rocmfp4_fast);
    for (size_t i = 0; i < nblocks; ++i) {
        if (!rocmfp4_scale_is_valid(blocks[i].e)) {
            return false;
        }
    }

    return true;
}

#ifdef ROCMFP4_X86_AVX2_DISPATCH
__attribute__((target("avx2")))
static inline int rocmfp4_hsum_i32_8_avx2(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// Decode one 32-weight block's low and high nibble streams through the
// Codebook10 table with a single PSHUFB, then integer-dot each against its half
// of the q8_0 block. Integer sums are order-independent, so sumi0/sumi1 match
// the scalar reference exactly and the float result is bit-identical.
__attribute__((target("avx2")))
static inline void rocmfp4_block_isums_avx2(
        const uint8_t * qs, const int8_t * q8, int * sumi0, int * sumi1) {
    const __m128i tbl = _mm_loadu_si128((const __m128i *) rocmfp4_codebook);
    const __m128i q   = _mm_loadu_si128((const __m128i *) qs);
    const __m128i lo  = _mm_and_si128(q, _mm_set1_epi8(0x0F));
    const __m128i hi  = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0F));
    const __m128i dlo = _mm_shuffle_epi8(tbl, lo);
    const __m128i dhi = _mm_shuffle_epi8(tbl, hi);
    const __m128i ylo = _mm_loadu_si128((const __m128i *) q8);
    const __m128i yhi = _mm_loadu_si128((const __m128i *) (q8 + QK_ROCMFP4/2));
    const __m256i pl  = _mm256_madd_epi16(_mm256_cvtepi8_epi16(dlo), _mm256_cvtepi8_epi16(ylo));
    const __m256i ph  = _mm256_madd_epi16(_mm256_cvtepi8_epi16(dhi), _mm256_cvtepi8_epi16(yhi));
    *sumi0 = rocmfp4_hsum_i32_8_avx2(pl);
    *sumi1 = rocmfp4_hsum_i32_8_avx2(ph);
}

__attribute__((target("avx2")))
static void rocmfp4_vec_dot_q4_0_q8_0_avx2(
        int nb, float * GGML_RESTRICT s, const block_rocmfp4 * GGML_RESTRICT x, const block_q8_0 * GGML_RESTRICT y) {
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = rocmfp4_ue4m3_to_fp32_half(x[ib].e[0]) * ggml_fp16_to_fp32(y[ib].d);
        const float d1 = rocmfp4_ue4m3_to_fp32_half(x[ib].e[1]) * ggml_fp16_to_fp32(y[ib].d);
        int sumi0, sumi1;
        rocmfp4_block_isums_avx2(x[ib].qs, y[ib].qs, &sumi0, &sumi1);
        sumf += d0 * (float) sumi0 + d1 * (float) sumi1;
    }
    *s = sumf;
}

__attribute__((target("avx2")))
static void rocmfp4_vec_dot_q4_0_fast_q8_0_avx2(
        int nb, float * GGML_RESTRICT s, const block_rocmfp4_fast * GGML_RESTRICT x, const block_q8_0 * GGML_RESTRICT y) {
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const float d = rocmfp4_ue4m3_to_fp32_half(x[ib].e) * ggml_fp16_to_fp32(y[ib].d);
        int sumi0, sumi1;
        rocmfp4_block_isums_avx2(x[ib].qs, y[ib].qs, &sumi0, &sumi1);
        sumf += d * (float) (sumi0 + sumi1);
    }
    *s = sumf;
}
#endif // ROCMFP4_X86_AVX2_DISPATCH

void rocmfp4_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    assert(n % QK_ROCMFP4 == 0);
    assert(QK_ROCMFP4 == QK8_0);

    const block_rocmfp4 * GGML_RESTRICT x = (const block_rocmfp4 *) vx;
    const block_q8_0    * GGML_RESTRICT y = (const block_q8_0 *) vy;

    const int nb = n / QK_ROCMFP4;

#ifdef ROCMFP4_X86_AVX2_DISPATCH
    if (__builtin_cpu_supports("avx2")) {
        rocmfp4_vec_dot_q4_0_q8_0_avx2(nb, s, x, y);
        return;
    }
#endif

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = rocmfp4_ue4m3_to_fp32_half(x[ib].e[0]) * ggml_fp16_to_fp32(y[ib].d);
        const float d1 = rocmfp4_ue4m3_to_fp32_half(x[ib].e[1]) * ggml_fp16_to_fp32(y[ib].d);
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            const uint8_t q = x[ib].qs[j];
            sumi0 += rocmfp4_decode_table(q)      * y[ib].qs[j];
            sumi1 += rocmfp4_decode_table(q >> 4) * y[ib].qs[j + QK_ROCMFP4/2];
        }

        sumf += d0 * (float) sumi0 + d1 * (float) sumi1;
    }

    *s = sumf;
}

void rocmfp4_vec_dot_q4_0_fast_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    assert(n % QK_ROCMFP4 == 0);
    assert(QK_ROCMFP4 == QK8_0);

    const block_rocmfp4_fast * GGML_RESTRICT x = (const block_rocmfp4_fast *) vx;
    const block_q8_0         * GGML_RESTRICT y = (const block_q8_0 *) vy;

    const int nb = n / QK_ROCMFP4;

#ifdef ROCMFP4_X86_AVX2_DISPATCH
    if (__builtin_cpu_supports("avx2")) {
        rocmfp4_vec_dot_q4_0_fast_q8_0_avx2(nb, s, x, y);
        return;
    }
#endif

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const float d = rocmfp4_ue4m3_to_fp32_half(x[ib].e) * ggml_fp16_to_fp32(y[ib].d);
        int sumi = 0;

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            const uint8_t q = x[ib].qs[j];
            sumi += rocmfp4_decode_table(q)      * y[ib].qs[j];
            sumi += rocmfp4_decode_table(q >> 4) * y[ib].qs[j + QK_ROCMFP4/2];
        }

        sumf += d * (float) sumi;
    }

    *s = sumf;
}
