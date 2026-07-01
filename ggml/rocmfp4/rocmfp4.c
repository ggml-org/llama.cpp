#define GGML_COMMON_DECL_C
#include "../src/ggml-common.h"

#include "rocmfp4.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

// ROCmFP4 stores a signed integer FP4-like codebook at half-scale. It is
// E2M1-derived, but the largest magnitude is retuned from 12 to 10 after
// sampling Qwen3 dense tensors; this reduces outlier pull without changing the
// packed 4-bit layout or integer dot-product path.
static const int8_t rocmfp4_codebook[16] = {
     0,  1,  2,  3,  4,  6,  8, 10,
     0, -1, -2, -3, -4, -6, -8,-10,
};

static inline int8_t rocmfp4_decode(uint8_t q) {
    return rocmfp4_codebook[q & 0x0f];
}

static inline float rocmfp4_ue4m3_to_fp32_half(uint8_t e) {
    // Unsigned E4M3 scale. Return half the raw value because the codebook
    // stores half-scale integer levels (e.g. 10 represents 5.0 raw-scale units).
    if (e == 0 || e == 0x7f || e == 0xff) {
        return 0.0f;
    }

    const int exp = (e >> 3) & 0x0f;
    const int man = e & 0x07;
    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }

    const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static inline uint8_t rocmfp4_best_index_scaled(float x, float inv_scale_half) {
    if (!isfinite(x)) {
        return 0;
    }

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

static uint8_t rocmfp4_best_index(float x, float scale_half) {
    if (!(scale_half > 0.0f) || !isfinite(scale_half)) {
        return 0;
    }

    return rocmfp4_best_index_scaled(x, 1.0f / scale_half);
}

static inline bool rocmfp4_scale_is_valid(uint8_t e) {
    // ROCmFP4 scale bytes are unsigned finite E4M3 values. 0x7f is NaN in the
    // unsigned encoding and values with the sign bit set are not valid scales.
    return e <= 0x7e;
}

static float rocmfp4_block_mse_for_scale(
        const float * x, int n, const float * quant_weights, float sigma2, int e, float best_err) {
    const float scale_half = rocmfp4_ue4m3_to_fp32_half((uint8_t) e);
    const float inv_scale_half = 1.0f / scale_half;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const uint8_t q = rocmfp4_best_index_scaled(x[i], inv_scale_half);
        const float y = (float) rocmfp4_decode(q) * scale_half;
        const float d = x[i] - y;

        float w = 1.0f;
        if (quant_weights) {
            // Match llama.cpp's imatrix weighting style for Q4_0: calibration
            // importance is scaled by row energy so large activations remain protected.
            const float qw = quant_weights[i];
            w = isfinite(qw) && qw > 0.0f ? qw * sqrtf(sigma2 + x[i]*x[i]) : 0.0f;
        }

        err += w*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
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

static uint8_t rocmfp4_choose_scale_ue4m3_exhaustive(
        const float * x, int n, const float * quant_weights, float sigma2, float max_abs) {
    const int start_e = rocmfp4_nearest_scale_ue4m3(max_abs / 10.0f);

    int best_e = 0;
    float best_err = FLT_MAX;

    for (int delta = 0; delta <= 125; ++delta) {
        const int candidates[2] = { start_e - delta, start_e + delta };
        for (int ci = 0; ci < 2; ++ci) {
            const int e = candidates[ci];
            if (e < 1 || e > 126 || (delta == 0 && ci == 1)) {
                continue;
            }

            const float err = rocmfp4_block_mse_for_scale(x, n, quant_weights, sigma2, e, best_err);
            if (err < best_err || (err == best_err && e < best_e)) {
                best_err = err;
                best_e = e;
            }
        }
    }

    return (uint8_t) best_e;
}

static uint8_t rocmfp4_choose_scale_ue4m3(const float * x, int n, const float * quant_weights, float sigma2) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float xi = x[i];
        max_abs = fmaxf(max_abs, fabsf(xi));
    }

    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfp4_choose_scale_ue4m3_exhaustive(x, n, quant_weights, sigma2, max_abs);
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
        memset(y[ib].qs, 0, sizeof(y[ib].qs));

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
        memset(y[ib].qs, 0, sizeof(y[ib].qs));

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
        memset(y[ib].qs, 0, sizeof(y[ib].qs));

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
        memset(y[ib].qs, 0, sizeof(y[ib].qs));

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
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = rocmfp4_ue4m3_to_fp32_half(x[ib].e[0]) * ggml_fp16_to_fp32(y[ib].d);
        const float d1 = rocmfp4_ue4m3_to_fp32_half(x[ib].e[1]) * ggml_fp16_to_fp32(y[ib].d);
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            sumi0 += rocmfp4_decode(x[ib].qs[j] & 0x0f) * y[ib].qs[j];
            sumi1 += rocmfp4_decode(x[ib].qs[j] >> 4)   * y[ib].qs[j + QK_ROCMFP4/2];
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
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const float d = rocmfp4_ue4m3_to_fp32_half(x[ib].e) * ggml_fp16_to_fp32(y[ib].d);
        int sumi = 0;

        for (int j = 0; j < QK_ROCMFP4/2; ++j) {
            sumi += rocmfp4_decode(x[ib].qs[j] & 0x0f) * y[ib].qs[j];
            sumi += rocmfp4_decode(x[ib].qs[j] >> 4)   * y[ib].qs[j + QK_ROCMFP4/2];
        }

        sumf += d * (float) sumi;
    }

    *s = sumf;
}
