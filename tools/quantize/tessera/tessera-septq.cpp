//
// tessera-septq.cpp
//
// SEPTQ (banded-Cholesky Hessian) weight quantizer. Faithful C++ port of
// tools/tile640/quantize_v3.py: quantize_2d_septq and its helpers
// (_septq_banded_cholesky / _septq_gptq_M / _septq_build_hessian).
//
// See tessera-septq.h for the dtype policy and algorithm overview.
//

#include "tessera-septq.h"

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define TS_HAS_CBLAS 1
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#define TS_HAS_CBLAS 1
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// --- Tile640 layout constants (match tools/tile640/quantize_v3.py) ---
static const int64_t TS_TILE640_PAGE_SIZE      = 640;
static const int64_t TS_TILE640_LANE_SIZE      = 20;
static const int64_t TS_TILE640_LANES_PER_PAGE = 32;  // 32 * 20 = 640

// ===========================================================================
// Hessian
// ===========================================================================

// Accumulate X^T X / n in float64 (long reduction) and store as float32.
// This is the most accurate scalar choice and lands within 1 ULP of
// numpy's float32 BLAS Hessian. Matches _septq_build_hessian in Python.
void ts_septq_build_hessian(int64_t in_dim,
                            const float * act_scales,
                            const float * activations, int64_t n_tokens,
                            float ridge_fraction,
                            float * H) {
    // Initialize to zero.
    std::memset(H, 0, sizeof(float) * (size_t)(in_dim * in_dim));

    if (activations != nullptr && n_tokens > 0) {
#if defined(TS_HAS_CBLAS)
        // activations is (n_tokens x in_dim) row-major. Compute X^T @ X via
        // dgemm: H = (1/n) * A^T @ A (row-major, Upper). Then copy upper
        // to lower for full matrix.
        std::vector<double> Hacc((size_t)(in_dim * in_dim), 0.0);
        // Promote activations to double for parity with the scalar path.
        std::vector<double> Ad((size_t)(n_tokens * in_dim));
        for (int64_t i = 0; i < n_tokens * in_dim; i++)
            Ad[i] = (double)activations[i];
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    (int)in_dim, (int)in_dim, (int)n_tokens,
                    1.0, Ad.data(), (int)in_dim, Ad.data(), (int)in_dim,
                    0.0, Hacc.data(), (int)in_dim);
        double inv_n = 1.0 / (double)std::max<int64_t>(n_tokens, 1);
        for (int64_t i = 0; i < in_dim; i++)
            for (int64_t j = 0; j < in_dim; j++)
                H[i * in_dim + j] = (float)(Hacc[i * in_dim + j] * inv_n);
#else
        // Accumulate X^T X in double, then cast per element to float32.
        std::vector<double> Hacc((size_t)(in_dim * in_dim), 0.0);
        for (int64_t t = 0; t < n_tokens; t++) {
            const float * xrow = activations + (size_t)(t * in_dim);
            for (int64_t i = 0; i < in_dim; i++) {
                double xi = (double)xrow[i];
                if (xi == 0.0) continue;
                double * row = Hacc.data() + (size_t)(i * in_dim);
                for (int64_t j = 0; j < in_dim; j++) {
                    row[j] += xi * (double)xrow[j];
                }
            }
        }
        double inv_n = 1.0 / (double)std::max<int64_t>(n_tokens, 1);
        for (int64_t i = 0; i < in_dim * in_dim; i++) {
            H[i] = (float)(Hacc[(size_t)i] * inv_n);
        }
#endif
        // diag_mean + ridge, computed in float32 to match Python's np.float32
        // arithmetic on the float32 Hessian.
        double diag_sum = 0.0;
        for (int64_t i = 0; i < in_dim; i++) {
            diag_sum += (double)H[i * in_dim + i];
        }
        float diag_mean = (float)(diag_sum / (double)in_dim);
        float ridge_f32 = (float)ridge_fraction;
        float ridge = std::max(ridge_f32 * diag_mean, 1e-2f * diag_mean);
        if (ridge > 0.0f) {
            for (int64_t i = 0; i < in_dim; i++) {
                H[i * in_dim + i] += ridge;
            }
        }
    } else if (act_scales != nullptr) {
        for (int64_t i = 0; i < in_dim; i++) {
            float s = std::max(act_scales[i], 1e-8f);
            H[i * in_dim + i] = s * s;
        }
    } else {
        for (int64_t i = 0; i < in_dim; i++) {
            H[i * in_dim + i] = 1.0f;
        }
    }
}

// ===========================================================================
// Banded Cholesky
// ===========================================================================

// Scalar float32 banded outer-product Cholesky, matching Python's
// _septq_banded_cholesky loop exactly (same f32 ops in the same order).
// On a non-positive pivot it falls back to a full Cholesky.
static void ts_septq_full_cholesky(const float * H, int64_t n, float * L) {
    // Standard LL^T Cholesky, f32, row-major. Matches numpy's
    // np.linalg.cholesky to within f32 rounding for the fallback path
    // (the fallback path is only used when the banded factorization fails;
    // its outputs flow through _septq_gptq_M unchanged).
    std::memset(L, 0, sizeof(float) * (size_t)(n * n));
    for (int64_t j = 0; j < n; j++) {
        double s = 0.0;
        for (int64_t k = 0; k < j; k++) {
            double lkj = L[j * n + k];
            s += lkj * lkj;
        }
        float diag = (float)((double)H[j * n + j] - s);
        if (diag <= 0.0f) {
            // Numerically rank-deficient; emulate numpy's behavior by
            // clamping to a tiny positive so sqrt is finite. numpy would
            // raise; the SEPTQ fallback path is only entered when the
            // banded factor failed, so we keep the contract finite.
            diag = std::max(diag, 0.0f);
        }
        L[j * n + j] = std::sqrt(diag);
        float inv_diag = L[j * n + j] > 0.0f ? 1.0f / L[j * n + j] : 0.0f;
        for (int64_t i = j + 1; i < n; i++) {
            double s2 = 0.0;
            for (int64_t k = 0; k < j; k++) {
                s2 += (double)L[i * n + k] * (double)L[j * n + k];
            }
            L[i * n + j] = (float)(((double)H[i * n + j] - s2) * (double)inv_diag);
        }
    }
}

void ts_septq_banded_cholesky(const float * H, int64_t n, int64_t bandwidth,
                              float * L_out) {
    std::memset(L_out, 0, sizeof(float) * (size_t)(n * n));
    bool fallback = false;
    for (int64_t j = 0; j < n; j++) {
        int64_t k_min = std::max<int64_t>(0, j - bandwidth);
        float s;
        if (k_min < j) {
            // s = H[j,j] - dot(L[j, k_min:j], L[j, k_min:j])
            double acc = 0.0;
            for (int64_t k = k_min; k < j; k++) {
                acc += (double)L_out[j * n + k] * (double)L_out[j * n + k];
            }
            s = (float)((double)H[j * n + j] - acc);
        } else {
            s = H[j * n + j];
        }
        if (s <= 0.0f) {
            fallback = true;
            break;
        }
        L_out[j * n + j] = std::sqrt(s);
        int64_t i_max = std::min(n, j + bandwidth + 1);
        if (i_max > j + 1) {
            float inv_ljj = 1.0f / L_out[j * n + j];
            if (k_min < j) {
                // L[i, j] = (H[i, j] - L[i, k_min:j] . L[j, k_min:j]) / L[j, j]
                for (int64_t i = j + 1; i < i_max; i++) {
                    double acc = 0.0;
                    for (int64_t k = k_min; k < j; k++) {
                        acc += (double)L_out[i * n + k] * (double)L_out[j * n + k];
                    }
                    L_out[i * n + j] = (float)(((double)H[i * n + j] - acc) * (double)inv_ljj);
                }
            } else {
                for (int64_t i = j + 1; i < i_max; i++) {
                    L_out[i * n + j] = H[i * n + j] * inv_ljj;
                }
            }
        }
    }
    if (!fallback) {
        return;
    }
    // Fallback: full Cholesky.
    ts_septq_full_cholesky(H, n, L_out);
}

// ===========================================================================
// GPTQ-M
// ===========================================================================

void ts_septq_gptq_M(const float * L, int64_t n, int64_t bandwidth,
                     float * M_out) {
    std::memset(M_out, 0, sizeof(float) * (size_t)(n * n));
    // Per-column banded forward substitution: solve L x = e_j, keep x[k] for
    // k in (j, j + bandwidth + 1). M[j, k] = x[k] * L[j, j].
    // We avoid a full scratch array by walking j innermost.
    for (int64_t j = 0; j < n; j++) {
        float ljj = L[j * n + j];
        if (ljj == 0.0f) continue;
        float x_j = 1.0f / ljj;
        int64_t k_max = std::min(n, j + bandwidth + 1);
        // x[k] for k in (j, k_max). Store locally.
        std::vector<float> x((size_t)(k_max - (j + 1)), 0.0f);
        for (int64_t k = j + 1; k < k_max; k++) {
            int64_t row_min = std::max<int64_t>(0, k - bandwidth);
            double s = 0.0;
            // s = -dot(L[k, row_min:k], x[row_min:k]). In the Python
            // reference x is a full length-n zero vector with only x[j]
            // and x[j+1:k] set, so x[m] = 0 for m < j (no contribution).
            for (int64_t m = row_min; m < k; m++) {
                float xm;
                if (m < j) {
                    xm = 0.0f;             // never set in the reference
                } else if (m == j) {
                    xm = x_j;
                } else {
                    xm = x[(size_t)(m - (j + 1))];
                }
                s -= (double)L[k * n + m] * (double)xm;
            }
            float xk = (float)(s / (double)L[k * n + k]);
            x[(size_t)(k - (j + 1))] = xk;
        }
        if (k_max > j + 1) {
            for (int64_t k = j + 1; k < k_max; k++) {
                M_out[j * n + k] = x[(size_t)(k - (j + 1))] * ljj;
            }
        }
    }
}

// ===========================================================================
// Tile640 pack + scales (ports of pack_tile640 / compute_scales)
// ===========================================================================

// {-1, 0, +1} -> {2, 0, 1} base-3 digit.
static int64_t ts_septq_trit_to_digit(int8_t t) {
    if (t > 0) return 1;
    if (t < 0) return 2;
    return 0;
}

static void ts_septq_pack_tile640(const int8_t * ternary_flat,
                                  int64_t out_dim, int64_t in_dim,
                                  std::vector<uint32_t> & packed,
                                  int64_t & pages_per_row) {
    pages_per_row = (in_dim + TS_TILE640_PAGE_SIZE - 1) / TS_TILE640_PAGE_SIZE;
    int64_t padded = pages_per_row * TS_TILE640_PAGE_SIZE;
    // pow3[i] = 3^i for i in [0, LANE_SIZE). 3^19 = 1162261467 < 2^31, so the
    // per-word sum fits in uint32 (max word = sum of pow3 = (3^20-1)/2 ~ 1.7e9).
    static const uint32_t pow3[TS_TILE640_LANE_SIZE] = {
        1u, 3u, 9u, 27u, 81u, 243u, 729u, 2187u, 6561u, 19683u,
        59049u, 177147u, 531441u, 1594323u, 4782969u, 14348907u,
        43046721u, 129140163u, 387420489u, 1162261467u
    };
    packed.assign((size_t)(out_dim * pages_per_row * TS_TILE640_LANES_PER_PAGE), 0);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t p = 0; p < pages_per_row; p++) {
            for (int64_t lane = 0; lane < TS_TILE640_LANES_PER_PAGE; lane++) {
                uint32_t word = 0;
                for (int64_t s = 0; s < TS_TILE640_LANE_SIZE; s++) {
                    int64_t col = p * TS_TILE640_PAGE_SIZE
                                + lane * TS_TILE640_LANE_SIZE + s;
                    uint32_t digit = 0;
                    if (col < in_dim) {
                        digit = (uint32_t)ts_septq_trit_to_digit(
                            ternary_flat[r * in_dim + col]);
                    }
                    word += digit * pow3[s];
                }
                size_t idx = (size_t)(((r * pages_per_row) + p)
                                      * TS_TILE640_LANES_PER_PAGE + lane);
                packed[idx] = word;
            }
        }
    }
    (void)padded;
}

// f32 -> f16 bit cast (IEEE 754). Returns the raw uint16 bits as an int32.
static uint16_t ts_septq_f32_to_f16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 31) & 0x1u;
    uint32_t exp  = (u >> 23) & 0xFFu;
    uint32_t mant = u & 0x7FFFFFu;
    uint16_t h;
    if (exp == 0xFF) {
        // inf / nan
        h = (uint16_t)((sign << 15) | 0x7C00u | (mant ? 0x200u : 0u));
    } else {
        int32_t e = (int32_t)exp - 127 + 15;
        if (e >= 0x1F) {
            // overflow -> inf
            h = (uint16_t)((sign << 15) | 0x7C00u);
        } else if (e <= 0) {
            // subnormal or zero
            if (e < -10) {
                h = (uint16_t)(sign << 15);  // underflow to zero
            } else {
                uint32_t m = mant | 0x800000u;
                uint32_t shift = (uint32_t)(14 - e);
                uint32_t mm = m >> shift;
                // round-to-nearest-even
                uint32_t rem = m & ((1u << shift) - 1u);
                uint32_t half = 1u << (shift - 1);
                if (rem > half || (rem == half && (mm & 1u))) mm++;
                h = (uint16_t)((sign << 15) | mm);
            }
        } else {
            // normal: shift f32 mantissa (23 bits) down to f16 (10 bits),
            // rounding the dropped 13 bits to nearest-even.
            uint32_t mant13 = mant >> 13;
            uint32_t dropped = mant & 0x1FFFu;
            uint32_t half = 1u << 12;
            uint32_t mm = mant13;
            if (dropped > half || (dropped == half && (mm & 1u))) {
                mm++;
                if (mm == 0x400u) { mm = 0; e++; }
            }
            if (e >= 0x1F) {
                h = (uint16_t)((sign << 15) | 0x7C00u);
            } else {
                h = (uint16_t)((sign << 15)
                               | ((uint32_t)e << 10) | (mm & 0x3FFu));
            }
        }
    }
    return h;
}

static void ts_septq_compute_scales(const float * W,
                                    const int8_t * ternary_flat,
                                    int64_t out_dim, int64_t in_dim,
                                    std::vector<uint16_t> & page_scales,
                                    std::vector<int8_t> & lane_scales) {
    int64_t pages_per_row = (in_dim + TS_TILE640_PAGE_SIZE - 1) / TS_TILE640_PAGE_SIZE;
    int64_t padded = pages_per_row * TS_TILE640_PAGE_SIZE;
    page_scales.assign((size_t)(out_dim * pages_per_row), 0);
    lane_scales.assign((size_t)(out_dim * pages_per_row * TS_TILE640_LANES_PER_PAGE), 0);

    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t p = 0; p < pages_per_row; p++) {
            // lane_target[lane] = mean(|W|) over nonzero ternary entries in lane
            float lane_target[TS_TILE640_LANES_PER_PAGE];
            for (int64_t lane = 0; lane < TS_TILE640_LANES_PER_PAGE; lane++) {
                double sum = 0.0;
                int64_t count = 0;
                for (int64_t s = 0; s < TS_TILE640_LANE_SIZE; s++) {
                    int64_t col = p * TS_TILE640_PAGE_SIZE
                                + lane * TS_TILE640_LANE_SIZE + s;
                    if (col >= in_dim) continue;
                    int8_t t = ternary_flat[r * in_dim + col];
                    if (t != 0) {
                        sum += std::fabs((double)W[r * in_dim + col]);
                        count++;
                    }
                }
                lane_target[lane] = count > 0 ? (float)(sum / (double)count) : 0.0f;
            }
            float page_max = 0.0f;
            for (int64_t lane = 0; lane < TS_TILE640_LANES_PER_PAGE; lane++) {
                page_max = std::max(page_max, lane_target[lane]);
            }
            if (page_max < 1e-30f) page_max = 1.0f;
            // page_scales stored as raw f16 bits (canonical Tessera convention).
            page_scales[(size_t)(r * pages_per_row + p)] =
                ts_septq_f32_to_f16_bits(page_max);
            for (int64_t lane = 0; lane < TS_TILE640_LANES_PER_PAGE; lane++) {
                float raw = (lane_target[lane] / page_max) * 127.0f;
                float rounded = std::round(raw);
                int32_t v = (int32_t)rounded;
                if (v < 1) v = 1;
                if (v > 127) v = 127;
                lane_scales[(size_t)((r * pages_per_row + p)
                                     * TS_TILE640_LANES_PER_PAGE + lane)] =
                    (int8_t)v;
            }
        }
    }
    (void)padded;
}

// ===========================================================================
// Importance scoring
// ===========================================================================

// Per-row stable rank -> [0, 1] CDF weight, matching the inv_cdf mode
// (1 - CDF(|W|)) computed via stable argsort.
static void ts_septq_inv_cdf_weight(const float * abs_W,
                                    int64_t out_dim, int64_t in_dim,
                                    std::vector<float> & out) {
    out.assign((size_t)(out_dim * in_dim), 0.0f);
    std::vector<int64_t> order(in_dim);
    for (int64_t i = 0; i < in_dim; i++) order[i] = i;
    for (int64_t r = 0; r < out_dim; r++) {
        const float * row = abs_W + r * in_dim;
        // stable sort of indices by ascending |W|
        std::stable_sort(order.begin(), order.end(),
                         [&](int64_t a, int64_t b) { return row[a] < row[b]; });
        float denom = (float)std::max<int64_t>(in_dim - 1, 1);
        // rank[k] = position of element order[k] in the sorted order, in [0,1]
        std::vector<float> ranks((size_t)in_dim);
        for (int64_t k = 0; k < in_dim; k++) {
            ranks[order[k]] = (float)k / denom;
        }
        for (int64_t j = 0; j < in_dim; j++) {
            out[r * in_dim + j] = 1.0f - ranks[j];
        }
    }
}

static double ts_septq_median(const float * v, int64_t n) {
    if (n <= 0) return 0.0;
    std::vector<float> tmp(v, v + n);
    std::nth_element(tmp.begin(), tmp.begin() + n / 2, tmp.end());
    if (n % 2 == 1) return (double)tmp[n / 2];
    auto lo = std::max_element(tmp.begin(), tmp.begin() + n / 2);
    return ((double)*lo + (double)tmp[n / 2]) / 2.0;
}

// ===========================================================================
// Top-level quantizer
// ===========================================================================

int ts_septq_quantize_2d(const float * weights, int64_t out_dim, int64_t in_dim,
                         const float * activations, int64_t n_tokens,
                         const float * act_scales,
                         const ts_septq_params * params,
                         ts_septq_result * result) {
    if (!weights || !params || !result) return 1;
    if (out_dim <= 0 || in_dim <= 0) return 1;
    if (!(params->septq_ratio > 0.0f && params->septq_ratio <= 1.0f)) return 2;
    if (params->septq_iterations < 1) return 3;
    if (!(params->ternary_threshold >= 0.3f && params->ternary_threshold <= 3.0f)) return 4;
    if (params->hessian_bandwidth < 0) return 5;
    if (params->importance_lambda < 0.0f) return 6;
    if (activations != nullptr && n_tokens <= 0) return 7;

    // Effective Hessian mode: banded requires activations.
    bool banded = (params->hessian_mode == TS_SEPTQ_HESSIAN_BANDED);
    if (banded && activations == nullptr) {
        banded = false;
    }
    int64_t bandwidth = std::min(params->hessian_bandwidth, in_dim - 1);

    int64_t n = out_dim * in_dim;

    // h_diag: column-importance proxy.
    std::vector<float> h_diag((size_t)in_dim);
    if (act_scales != nullptr) {
        for (int64_t j = 0; j < in_dim; j++) {
            h_diag[j] = std::max(act_scales[j], 1e-8f);
        }
    } else {
        for (int64_t j = 0; j < in_dim; j++) h_diag[j] = 1.0f;
    }

    // abs_W, row_mean_abs, threshold, keep_2d, sign_W.
    std::vector<float> abs_W((size_t)n);
    std::vector<int8_t> sign_W((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        float w = weights[i];
        abs_W[i] = std::fabs(w);
        sign_W[i] = (w > 0.0f) ? 1 : (w < 0.0f ? -1 : 0);
    }
    std::vector<float> threshold_1d((size_t)out_dim);
    for (int64_t r = 0; r < out_dim; r++) {
        double sum = 0.0;
        for (int64_t j = 0; j < in_dim; j++) {
            sum += (double)abs_W[r * in_dim + j];
        }
        float mean = (float)(sum / (double)in_dim);
        threshold_1d[r] = mean * params->ternary_threshold;
    }

    // keep_2d: |W| >= threshold (per row).
    std::vector<uint8_t> keep_2d((size_t)n, 0);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t j = 0; j < in_dim; j++) {
            if (abs_W[r * in_dim + j] >= threshold_1d[r]) {
                keep_2d[r * in_dim + j] = 1;
            }
        }
    }

    // ternary_init = where(keep_2d, sign_W, 0); quant_error_init = (W - ternary_init)^2
    std::vector<float> quant_error_init((size_t)n, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        float t = keep_2d[i] ? (float)sign_W[i] : 0.0f;
        float diff = weights[i] - t;
        quant_error_init[i] = diff * diff;
    }

    // importance_2d.
    std::vector<float> importance_flat((size_t)n, 0.0f);
    if (params->importance_weight == TS_SEPTQ_IMP_QUANT_ERROR_H) {
        for (int64_t i = 0; i < n; i++) {
            int64_t j = i % in_dim;
            importance_flat[i] = quant_error_init[i] * h_diag[j];
        }
    } else if (params->importance_weight == TS_SEPTQ_IMP_INV_ABS_W) {
        for (int64_t i = 0; i < n; i++) {
            int64_t j = i % in_dim;
            float w = 1.0f / (abs_W[i] + 1e-8f);
            importance_flat[i] = quant_error_init[i] * h_diag[j] * w;
        }
    } else if (params->importance_weight == TS_SEPTQ_IMP_INV_CDF) {
        std::vector<float> cdf_w;
        ts_septq_inv_cdf_weight(abs_W.data(), out_dim, in_dim, cdf_w);
        for (int64_t i = 0; i < n; i++) {
            int64_t j = i % in_dim;
            importance_flat[i] = quant_error_init[i] * h_diag[j] * cdf_w[i];
        }
    } else if (params->importance_weight == TS_SEPTQ_IMP_HYBRID) {
        // base_importance + normalized_lambda * inv_abs * h_diag
        std::vector<float> inv_abs((size_t)n);
        std::vector<float> base((size_t)n);
        for (int64_t i = 0; i < n; i++) {
            int64_t j = i % in_dim;
            inv_abs[i] = 1.0f / (abs_W[i] + 1e-8f);
            base[i] = quant_error_init[i] * h_diag[j];
        }
        // normalized_lambda via medians (matches the Python hybrid path)
        std::vector<float> inv_abs_h((size_t)n);
        for (int64_t i = 0; i < n; i++) {
            int64_t j = i % in_dim;
            inv_abs_h[i] = inv_abs[i] * h_diag[j];
        }
        double base_scale = ts_septq_median(base.data(), n);
        double inv_scale  = ts_septq_median(inv_abs_h.data(), n);
        float normalized_lambda = 0.0f;
        if (inv_scale > 0.0 && base_scale > 0.0) {
            normalized_lambda = params->importance_lambda
                              * (float)(base_scale / inv_scale);
        }
        for (int64_t i = 0; i < n; i++) {
            int64_t j = i % in_dim;
            importance_flat[i] = base[i] + normalized_lambda * inv_abs[i] * h_diag[j];
        }
    } else {
        return 8;
    }

    // Step 3: static global mask via quickselect threshold.
    int64_t k = std::max<int64_t>(1, std::min<int64_t>(n,
                 (int64_t)std::ceil((double)n * (double)params->septq_ratio)));
    std::vector<uint8_t> mask_flat((size_t)n, 0);
    if (k >= n) {
        std::fill(mask_flat.begin(), mask_flat.end(), 1);
    } else {
        // kth largest value (0-indexed: index k-1 of descending partition).
        // nth_element on the negative -> the element at position k-1 of the
        // descending order. We use a comparator that picks the k-th largest.
        std::vector<float> tmp(importance_flat.begin(), importance_flat.end());
        // nth_element with greater<> puts the (k-1)-th largest at position k-1.
        std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(),
                         std::greater<float>());
        float threshold = tmp[(size_t)(k - 1)];
        for (int64_t i = 0; i < n; i++) {
            if (importance_flat[i] >= threshold) {
                mask_flat[i] = 1;
            }
        }
        // Cap overshoots (rare; only when ties at the boundary).
        int64_t count = 0;
        for (int64_t i = 0; i < n; i++) if (mask_flat[i]) count++;
        int64_t excess = count - k;
        if (excess > 0) {
            // Among the masked entries, clear the lowest-importance `excess`.
            std::vector<int64_t> true_idx;
            true_idx.reserve((size_t)count);
            for (int64_t i = 0; i < n; i++) if (mask_flat[i]) true_idx.push_back(i);
            std::vector<float> true_imp((size_t)count);
            for (int64_t t = 0; t < count; t++) {
                true_imp[(size_t)t] = importance_flat[true_idx[(size_t)t]];
            }
            // smallest `excess` by nth_element
            std::vector<int64_t> order((size_t)count);
            for (int64_t t = 0; t < count; t++) order[(size_t)t] = t;
            std::nth_element(order.begin(), order.begin() + (excess - 1),
                             order.end(), [&](int64_t a, int64_t b) {
                                 return true_imp[(size_t)a] < true_imp[(size_t)b];
                             });
            for (int64_t t = 0; t < excess; t++) {
                mask_flat[true_idx[(size_t)order[(size_t)t]]] = 0;
            }
        }
    }

    // Step 4: ternary_2d and error_2d.
    // quantized_mask_2d = mask_2d & keep_2d
    std::vector<int8_t> ternary_2d((size_t)n, 0);
    std::vector<float>  error_2d((size_t)n, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        uint8_t qm = mask_flat[i] & keep_2d[i];
        ternary_2d[i] = qm ? sign_W[i] : 0;
        float err = ((float)ternary_2d[i] - weights[i]) * (float)qm;
        error_2d[i] = err;
    }

    // W_compensated = W - E @ M  (banded) or W (diagonal).
    std::vector<float> W_comp((size_t)n);
    if (banded) {
        std::vector<float> H((size_t)(in_dim * in_dim));
        ts_septq_build_hessian(in_dim, act_scales, activations, n_tokens,
                               params->ridge_fraction, H.data());
        std::vector<float> L((size_t)(in_dim * in_dim));
        ts_septq_banded_cholesky(H.data(), in_dim, bandwidth, L.data());
        std::vector<float> M((size_t)(in_dim * in_dim));
        ts_septq_gptq_M(L.data(), in_dim, bandwidth, M.data());
        // W_comp = W - error_2d @ M;  error_2d is (out_dim x in_dim),
        // M is (in_dim x in_dim).
#if defined(TS_HAS_CBLAS)
        // Copy W into W_comp first, then W_comp -= error_2d @ M.
        std::copy(weights, weights + n, W_comp.begin());
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)out_dim, (int)in_dim, (int)in_dim,
                    -1.0f, error_2d.data(), (int)in_dim, M.data(), (int)in_dim,
                    1.0f, W_comp.data(), (int)in_dim);
#else
        for (int64_t r = 0; r < out_dim; r++) {
            const float * erow = error_2d.data() + r * in_dim;
            for (int64_t j = 0; j < in_dim; j++) {
                double acc = 0.0;
                for (int64_t p = 0; p < in_dim; p++) {
                    acc += (double)erow[p] * (double)M[p * in_dim + j];
                }
                W_comp[r * in_dim + j] = weights[r * in_dim + j] - (float)acc;
            }
        }
#endif
    } else {
        std::copy(weights, weights + n, W_comp.begin());
    }

    // Step 5: pack into Tessera format. The "outliers" are the UNIMPORTANT
    // elements (where mask is False), stored at full precision after the
    // cross-column update. CSR by row.
    // Build the outlier index list (stable by row, then column).
    std::vector<int64_t> outlier_idx;
    outlier_idx.reserve((size_t)(n / 2));
    for (int64_t i = 0; i < n; i++) {
        if (!mask_flat[i]) outlier_idx.push_back(i);
    }
    // outlier_rows = idx // in_dim, already ascending since idx is row-major.
    // Stable sort by row (the natural index order is already row-major, so
    // the iteration order is row-then-col, matching argsort(stable)).
    result->outlier_cols.clear();
    result->outlier_vals.clear();
    result->outlier_cols.reserve(outlier_idx.size());
    result->outlier_vals.reserve(outlier_idx.size());
    std::vector<int64_t> row_counts((size_t)out_dim, 0);
    for (int64_t idx : outlier_idx) {
        int64_t r = idx / in_dim;
        int64_t c = idx % in_dim;
        result->outlier_cols.push_back((int32_t)c);
        // outlier_resid = W_comp[idx] as f16 bits (canonical Tessera convention).
        result->outlier_vals.push_back(ts_septq_f32_to_f16_bits(W_comp[idx]));
        row_counts[(size_t)r]++;
    }
    result->outlier_row_offsets.assign((size_t)(out_dim + 1), 0);
    for (int64_t r = 0; r < out_dim; r++) {
        result->outlier_row_offsets[(size_t)(r + 1)] =
            result->outlier_row_offsets[(size_t)r] + (int32_t)row_counts[(size_t)r];
    }

    // ternary_final + mask_2d (diagnostics).
    result->ternary.assign(ternary_2d.begin(), ternary_2d.end());
    result->mask_2d.assign(mask_flat.begin(), mask_flat.end());

    // pack + scales (computed against the ORIGINAL weights, not W_comp).
    int64_t pages_per_row = 0;
    ts_septq_pack_tile640(ternary_2d.data(), out_dim, in_dim,
                          result->packed, pages_per_row);
    ts_septq_compute_scales(weights, ternary_2d.data(), out_dim, in_dim,
                            result->page_scales, result->lane_scales);
    return 0;
}
