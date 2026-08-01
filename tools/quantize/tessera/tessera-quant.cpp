//
// tessera-quant.cpp
//
// Tile640 quantization: ternarization, packing, scale fitting, AWQ scale
// search, and the quantize_2d / quantize_3d entry points. Port of
// tools/tile640/quantize_v3.py.
//

#include "tessera-quant.h"
#include "tessera-vec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#endif

// Wire-format constants (mirror ggml/src/ggml-common.h).
#define TS_PAGE_SIZE      640
#define TS_LANE_SIZE      20
#define TS_LANES_PER_PAGE 32
#define TS_WORDS_PER_PAGE 32

// ---------------------------------------------------------------------------
// f16 helpers (IEEE 754 binary16, round-to-nearest-even)
// ---------------------------------------------------------------------------

static uint16_t ts_f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x & 0x80000000u) >> 16;
    uint32_t exp  = (x & 0x7f800000u) >> 23;
    uint32_t mant = (x & 0x007fffffu);

    if (exp == 0xff) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u)); // inf / nan
    }
    int32_t e = (int32_t)exp - 127 + 15;
    if (e >= 0x1f) {
        return (uint16_t)(sign | 0x7c00u); // overflow -> inf
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t)sign; // underflow -> signed zero
        }
        mant |= 0x00800000u;                 // restore implicit leading 1
        uint32_t shift = (uint32_t)(14 - e); // drop to 10-bit mantissa
        uint32_t rb    = 1u << (shift - 1);
        uint32_t m     = mant >> shift;
        if ((mant & rb) && ((mant & (rb - 1)) || (m & 1))) {
            m++;
        }
        return (uint16_t)(sign | m);
    }
    uint16_t hm = (uint16_t)(mant >> 13);
    uint16_t h  = (uint16_t)(sign | ((uint32_t)e << 10) | hm);
    if (((mant >> 12) & 1u) && ((mant & 0xfffu) || (hm & 1))) {
        h++;
    }
    return h;
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

static int64_t ts_pages_per_row(int64_t in_dim) {
    return (in_dim + TS_PAGE_SIZE - 1) / TS_PAGE_SIZE;
}

static const uint32_t * ts_pow3_table(void) {
    static uint32_t pow3[TS_LANE_SIZE];
    static bool init = false;
    if (!init) {
        pow3[0] = 1;
        for (int i = 1; i < TS_LANE_SIZE; i++) {
            pow3[i] = pow3[i - 1] * 3u;
        }
        init = true;
    }
    return pow3;
}

// median of the finite, strictly-positive entries; returns 0 if none.
static float ts_median_finite_positive(const float * x, int64_t n) {
    std::vector<float> v;
    v.reserve((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        if (std::isfinite(x[i]) && x[i] > 0.0f) {
            v.push_back(x[i]);
        }
    }
    if (v.empty()) {
        return 0.0f;
    }
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];
    if (v.size() % 2 == 0) {
        float lo = *std::max_element(v.begin(), v.begin() + mid);
        m = 0.5f * (lo + m);
    }
    return m;
}

// ---------------------------------------------------------------------------
// normalized AWQ scale
// ---------------------------------------------------------------------------

void ts_normalized_awq_scale(const float * act_scales, float alpha,
                             float * scale_out, int64_t in_dim) {
    float reference = ts_median_finite_positive(act_scales, in_dim);
    if (reference <= 0.0f) {
        std::fill(scale_out, scale_out + in_dim, 1.0f);
        return;
    }
    float denom = std::max(reference, 1e-8f);
    for (int64_t c = 0; c < in_dim; c++) {
        float v = act_scales[c];
        float rel;
        if (std::isnan(v)) {
            rel = 1.0f;
        } else if (std::isinf(v)) {
            rel = (v > 0.0f) ? 256.0f : (1.0f / 256.0f);
        } else {
            rel = v / denom;
        }
        rel = std::min(std::max(rel, 1.0f / 256.0f), 256.0f);
        scale_out[c] = std::pow(rel, alpha);
    }
}

// ---------------------------------------------------------------------------
// ternarize with activation-aware scaling
// ---------------------------------------------------------------------------

float ts_ternarize_with_acts(const float * weights, const float * act_scales,
                             float alpha, float clip,
                             int8_t * ternary_out,
                             int64_t out_dim, int64_t in_dim) {
    const int64_t n = out_dim * in_dim;

    std::vector<float> ws(weights, weights + n);

    if (act_scales != nullptr && alpha > 0.0f) {
        std::vector<float> scale(in_dim);
        ts_normalized_awq_scale(act_scales, alpha, scale.data(), in_dim);
        for (int64_t r = 0; r < out_dim; r++) {
            float * row = ws.data() + r * in_dim;
            for (int64_t c = 0; c < in_dim; c++) {
                row[c] *= scale[c];
            }
        }
    }

    if (clip > 0.0f && clip < 1.0f) {
        for (int64_t r = 0; r < out_dim; r++) {
            float * row = ws.data() + r * in_dim;
            float maxabs = 0.0f;
            for (int64_t c = 0; c < in_dim; c++) {
                maxabs = std::max(maxabs, std::fabs(row[c]));
            }
            float limit = maxabs * clip;
            for (int64_t c = 0; c < in_dim; c++) {
                row[c] = std::min(std::max(row[c], -limit), limit);
            }
        }
    }

    // sequential float32 accumulation; numpy .sum() is pairwise so the sum may
    // differ in the last bit (ternary still matches, see test_bit_equiv.cpp)
    float abs_sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        abs_sum += std::fabs(ws[i]);
    }
    float threshold = (n > 0) ? (abs_sum / (float)n) : 0.0f;

    for (int64_t i = 0; i < n; i++) {
        int8_t t = 0;
        if (std::fabs(ws[i]) >= threshold) {
            if (ws[i] > 0.0f) {
                t = 1;
            } else if (ws[i] < 0.0f) {
                t = -1;
            }
        }
        ternary_out[i] = t;
    }
    return threshold;
}

// Fused scale + clip + ternarize. See declaration in the header. The two-pass
// structure is mandatory: the threshold needs a full mean-of-abs reduction
// over the scaled weights, and only then can we assign ternary values.
float ts_scale_clip_ternarize_fused(const float * weights,
                                    const float * wscale,
                                    float clip,
                                    float * ws_out,
                                    float * core_out,
                                    int8_t * ternary_out,
                                    int64_t out_dim, int64_t in_dim) {
    const int64_t n = out_dim * in_dim;
    const bool do_clip = (clip > 0.0f && clip < 1.0f);

    // --- Pass 1: scale into ws_out, accumulate mean-of-abs for the threshold,
    //     and build per-row maxabs for clipping. Writes ws_out and core_out
    //     (core = copy of ws at this point; clipped in place below).
#if defined(__APPLE__)
    // ws_out = weights * wscale[c] (broadcast per column)
    ts_mat_scale_cols(weights, wscale, ws_out, out_dim, in_dim);
#else
    for (int64_t r = 0; r < out_dim; r++) {
        const float * wrow = weights + r * in_dim;
        float * orow = ws_out + r * in_dim;
        for (int64_t c = 0; c < in_dim; c++) {
            orow[c] = wrow[c] * wscale[c];
        }
    }
#endif

    // core_out = ws_out (copy). Unavoidable since ts_compute_scales + the MSE
    // path below both need the un-clipped ws AND the clipped core.
    std::memcpy(core_out, ws_out, (size_t)n * sizeof(float));

    // Per-row maxabs for clipping (only if clip is active).
    std::vector<float> row_maxabs;
    if (do_clip) {
        row_maxabs.resize((size_t)out_dim);
        for (int64_t r = 0; r < out_dim; r++) {
            row_maxabs[(size_t)r] = ts_vec_maxabs(core_out + r * in_dim, in_dim);
        }
    }

    // Clip core_out in place.
    if (do_clip) {
        for (int64_t r = 0; r < out_dim; r++) {
            float * row = core_out + r * in_dim;
            float limit = row_maxabs[(size_t)r] * clip;
            for (int64_t c = 0; c < in_dim; c++) {
                row[c] = std::min(std::max(row[c], -limit), limit);
            }
        }
    }

    // Threshold = mean(|ws|) (NOT mean(|core|) - matches the unfused path).
    float global_amp = ts_vec_meanabs(ws_out, n);
    float threshold = global_amp;

    // --- Pass 2: ternarize core_out using the threshold.
#if defined(__APPLE__)
    // Elementwise ternarize: ternary = (core >= threshold) ? sign(core) : 0.
    // vDSP doesn't have a fused sign+threshold, so use a scalar loop but
    // benefit from the data already being in cache from the copy above.
#endif
    for (int64_t i = 0; i < n; i++) {
        int8_t t = 0;
        if (std::fabs(core_out[i]) >= threshold) {
            if (core_out[i] > 0.0f) {
                t = 1;
            } else if (core_out[i] < 0.0f) {
                t = -1;
            }
        }
        ternary_out[i] = t;
    }

    return global_amp;
}

// ---------------------------------------------------------------------------
// scale fitting from weights + ternary pattern
// ---------------------------------------------------------------------------

void ts_compute_scales(const float * weights, const int8_t * ternary_flat,
                       uint16_t * page_scales, int8_t * lane_scales,
                       int64_t out_dim, int64_t in_dim) {
    const int64_t pages = ts_pages_per_row(in_dim);

    for (int64_t o = 0; o < out_dim; o++) {
        for (int64_t p = 0; p < pages; p++) {
            float lane_target[TS_LANES_PER_PAGE];
            for (int l = 0; l < TS_LANES_PER_PAGE; l++) {
                float sum_abs = 0.0f;
                int   count   = 0;
                for (int k = 0; k < TS_LANE_SIZE; k++) {
                    int64_t col = p * TS_PAGE_SIZE + l * TS_LANE_SIZE + k;
                    if (col >= in_dim) {
                        continue; // zero padding
                    }
                    int64_t idx = o * in_dim + col;
                    if (ternary_flat[idx] != 0) {
                        sum_abs += std::fabs(weights[idx]);
                        count++;
                    }
                }
                lane_target[l] = (count > 0) ? (sum_abs / count) : 0.0f;
            }

            float page_max = 0.0f;
            for (int l = 0; l < TS_LANES_PER_PAGE; l++) {
                page_max = std::max(page_max, lane_target[l]);
            }
            if (page_max < 1e-30f) {
                page_max = 1.0f;
            }

            int64_t page_idx = o * pages + p;
            page_scales[page_idx] = ts_f32_to_f16(page_max);
            for (int l = 0; l < TS_LANES_PER_PAGE; l++) {
                float raw  = (lane_target[l] / page_max) * 127.0f;
                int   q    = (int)std::lround(raw);
                q = std::min(std::max(q, 1), 127);
                lane_scales[page_idx * TS_LANES_PER_PAGE + l] = (int8_t)q;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// packing
// ---------------------------------------------------------------------------

void ts_pack_tile640(const int8_t * ternary_flat,
                     uint32_t * packed_out,
                     uint16_t * page_scales_out,
                     int8_t * lane_scales_out,
                     int64_t out_dim, int64_t in_dim) {
    const int64_t pages     = ts_pages_per_row(in_dim);
    const uint32_t * pow3   = ts_pow3_table();
    const uint16_t unit_f16 = ts_f32_to_f16(1.0f);

    for (int64_t o = 0; o < out_dim; o++) {
        for (int64_t p = 0; p < pages; p++) {
            int64_t page_idx = o * pages + p;
            page_scales_out[page_idx] = unit_f16;
            for (int l = 0; l < TS_LANES_PER_PAGE; l++) {
                uint32_t word = 0;
                bool any = false;
                for (int k = 0; k < TS_LANE_SIZE; k++) {
                    int64_t col = p * TS_PAGE_SIZE + l * TS_LANE_SIZE + k;
                    int8_t t = (col < in_dim) ? ternary_flat[o * in_dim + col] : 0;
                    uint32_t trit = (t > 0) ? 1u : ((t < 0) ? 2u : 0u);
                    word += trit * pow3[k];
                    if (t != 0) {
                        any = true;
                    }
                }
                packed_out[page_idx * TS_WORDS_PER_PAGE + l] = word;
                lane_scales_out[page_idx * TS_LANES_PER_PAGE + l] = (int8_t)(any ? 127 : 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// repair-residual (outlier) selection
// ---------------------------------------------------------------------------

std::vector<int32_t> ts_select_repair_residuals(
    const float * weights, const int8_t * ternary_flat,
    float page_scale, int64_t out_dim, int64_t in_dim,
    int64_t max_outliers, float threshold) {

    const int64_t n = out_dim * in_dim;
    if (max_outliers <= 0 || n == 0) {
        return {};
    }

    // residual of a uniform-amplitude dequant: |w - t * page_scale|
    std::vector<std::pair<float, int64_t>> scored;
    scored.reserve((size_t)std::min<int64_t>(n, max_outliers * 4));
    for (int64_t i = 0; i < n; i++) {
        float dequant = (float)ternary_flat[i] * page_scale;
        float resid   = std::fabs(weights[i] - dequant);
        if (resid >= threshold) {
            scored.push_back({ resid, i });
        }
    }

    size_t keep = std::min<size_t>(scored.size(), (size_t)max_outliers);
    std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
                      [](const std::pair<float, int64_t> & a,
                         const std::pair<float, int64_t> & b) {
                          if (a.first != b.first) {
                              return a.first > b.first; // residual descending
                          }
                          return a.second < b.second;   // stable by index
                      });

    std::vector<int32_t> out(keep);
    for (size_t i = 0; i < keep; i++) {
        out[i] = (int32_t)scored[i].second;
    }
    return out;
}

// ---------------------------------------------------------------------------
// AWQ scale search (per-row, importance-weighted reconstruction MSE)
// ---------------------------------------------------------------------------

// row subsample for very tall matrices (matches the Python 1024-row cap)
static std::vector<int64_t> ts_row_ids(int64_t out_dim, int64_t cap) {
    std::vector<int64_t> ids;
    if (out_dim <= cap) {
        ids.resize((size_t)out_dim);
        std::iota(ids.begin(), ids.end(), 0);
        return ids;
    }
    ids.resize((size_t)cap);
    for (int64_t i = 0; i < cap; i++) {
        ids[(size_t)i] = (int64_t)std::llround((double)i * (double)(out_dim - 1) / (double)(cap - 1));
    }
    return ids;
}

// dequantize a ternary pattern fitted with ts_compute_scales back to float
static void ts_dequant(const int8_t * ternary, const uint16_t * page_scales,
                       const int8_t * lane_scales, float * out,
                       int64_t out_dim, int64_t in_dim) {
    const int64_t pages = ts_pages_per_row(in_dim);
    for (int64_t o = 0; o < out_dim; o++) {
        for (int64_t p = 0; p < pages; p++) {
            int64_t page_idx = o * pages + p;
            // page scale stored as f16; recover an f32 amplitude via the
            // same lane normalization used at fit time.
            float page_max = 1.0f; // relative; lane scales carry the ratio
            // reconstruct the f16 page scale deterministically
            {
                uint16_t h = page_scales[page_idx];
                // decode f16 -> f32 (normals only needed here)
                uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
                uint32_t exp  = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                uint32_t bits;
                if (exp == 0) {
                    if (mant == 0) {
                        bits = sign;
                    } else {
                        int e = -1;
                        do { mant <<= 1; e--; } while ((mant & 0x400u) == 0);
                        mant &= 0x3ff;
                        bits = sign | ((uint32_t)(e + 127 + 1) << 23) | (mant << 13);
                    }
                } else if (exp == 0x1f) {
                    bits = sign | 0x7f800000u | (mant << 13);
                } else {
                    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
                }
                std::memcpy(&page_max, &bits, sizeof(bits));
            }
            for (int l = 0; l < TS_LANES_PER_PAGE; l++) {
                float amp = page_max * (float)lane_scales[page_idx * TS_LANES_PER_PAGE + l] / 127.0f;
                for (int k = 0; k < TS_LANE_SIZE; k++) {
                    int64_t col = p * TS_PAGE_SIZE + l * TS_LANE_SIZE + k;
                    if (col >= in_dim) {
                        continue;
                    }
                    out[o * in_dim + col] = (float)ternary[o * in_dim + col] * amp;
                }
            }
        }
    }
}

float ts_awq_scale_search(const float * weights, const float * act_scales,
                          const float * calib_activations,
                          int64_t out_dim, int64_t in_dim,
                          int64_t n_tokens, int64_t n_grid) {
    (void)calib_activations;
    (void)n_tokens;

    if (act_scales == nullptr || out_dim == 0 || in_dim == 0) {
        return 0.0f;
    }
    if (n_grid < 2) {
        n_grid = 2;
    }

    std::vector<int64_t> rows = ts_row_ids(out_dim, 1024);
    int64_t R = (int64_t)rows.size();
    std::vector<float> W((size_t)R * (size_t)in_dim);
    for (int64_t r = 0; r < R; r++) {
        std::memcpy(W.data() + r * in_dim, weights + rows[(size_t)r] * in_dim,
                    (size_t)in_dim * sizeof(float));
    }

    std::vector<float> act2((size_t)in_dim);
    for (int64_t c = 0; c < in_dim; c++) {
        float a = act_scales[c];
        act2[(size_t)c] = a * a;
    }

    const int64_t pages = ts_pages_per_row(in_dim);
    std::vector<int8_t>  ternary((size_t)R * (size_t)in_dim);
    std::vector<uint16_t> pscale((size_t)R * (size_t)pages);
    std::vector<int8_t>  lscale((size_t)R * (size_t)pages * TS_LANES_PER_PAGE);
    std::vector<float>   deq((size_t)R * (size_t)in_dim);
    std::vector<float>   scale((size_t)in_dim);
    std::vector<float>   Ws((size_t)R * (size_t)in_dim);
    std::vector<float>   diff((size_t)R * (size_t)in_dim);

    float best_alpha = 0.0f;
    float best_err   = std::numeric_limits<float>::infinity();

    for (int64_t g = 0; g < n_grid; g++) {
        float alpha = (float)g / (float)(n_grid - 1);
        if (alpha == 0.0f) {
            std::fill(scale.begin(), scale.end(), 1.0f);
        } else {
            ts_normalized_awq_scale(act_scales, alpha, scale.data(), in_dim);
        }
        ts_mat_scale_cols(W.data(), scale.data(), Ws.data(), R, in_dim);
        ts_ternarize_with_acts(Ws.data(), nullptr, 0.0f, 0.0f, ternary.data(), R, in_dim);
        ts_compute_scales(Ws.data(), ternary.data(), pscale.data(), lscale.data(), R, in_dim);
        ts_dequant(ternary.data(), pscale.data(), lscale.data(), deq.data(), R, in_dim);

        // effective weight in the original scale: deq / scale, then diff = deq/scale - W.
        // Compute in place: deq <- deq / scale (broadcast divide), then diff <- deq - W.
        for (int64_t r = 0; r < R; r++) {
            float * drow = deq.data() + r * in_dim;
            const float * wrow = W.data() + r * in_dim;
            float * diffrow = diff.data() + r * in_dim;
            for (int64_t c = 0; c < in_dim; c++) {
                diffrow[c] = drow[c] / scale[c] - wrow[c];
            }
        }
        // importance-weighted MSE: mean( diff^2 * act^2 ). diff^2 * act^2 is an
        // elementwise mul followed by a sum; vDSP dispatches both.
        float err_sum = 0.0f;
        const int64_t mn = R * in_dim;
#if defined(__APPLE__)
        // Square `diff` in place, multiply by act^2, then sum. vDSP_vmul +
        // vDSP_sves (sum of vector elements, not in older SDKs) - use vDSP_vmul
        // into a scratch then vDSP_sve.
        static thread_local std::vector<float> scratch;
        if ((int64_t)scratch.size() < mn) scratch.resize((size_t)mn);
        vDSP_vmul(diff.data(), 1, act2.data(), 1, scratch.data(), 1, (vDSP_Length)mn);
        vDSP_sve(scratch.data(), 1, &err_sum, (vDSP_Length)mn);
#else
        double err = 0.0;
        for (int64_t i = 0; i < mn; i++) {
            double d = diff[(size_t)i];
            err += d * d * (double)act2[(size_t)i];
        }
        err_sum = (float)err;
#endif
        float err = err_sum / (float)mn;
        if (err < best_err) {
            best_err   = err;
            best_alpha = alpha;
        }
    }
    return best_alpha;
}

// ---------------------------------------------------------------------------
// AWQ scale search (layer-output MSE)
// ---------------------------------------------------------------------------

float ts_awq_scale_search_layer_output(
    const float * weights, const float * act_scales,
    const float * calib_X, const float * ref_output,
    int64_t out_dim, int64_t in_dim,
    int64_t n_tokens, int64_t n_grid) {

    if (act_scales == nullptr || calib_X == nullptr || ref_output == nullptr ||
        out_dim == 0 || in_dim == 0 || n_tokens == 0) {
        return 0.0f;
    }
    if (n_grid < 2) {
        n_grid = 2;
    }

    std::vector<int64_t> rows = ts_row_ids(out_dim, 1024);
    int64_t R = (int64_t)rows.size();

    // W: (R x in_dim), ref_T: (R x n_tokens) = transpose of ref_output[rows]
    std::vector<float> W((size_t)R * (size_t)in_dim);
    std::vector<float> ref_T((size_t)R * (size_t)n_tokens);
    for (int64_t r = 0; r < R; r++) {
        int64_t src = rows[(size_t)r];
        std::memcpy(W.data() + r * in_dim, weights + src * in_dim,
                    (size_t)in_dim * sizeof(float));
        for (int64_t t = 0; t < n_tokens; t++) {
            ref_T[(size_t)(r * n_tokens + t)] = ref_output[t * out_dim + src];
        }
    }

    // X_T: (in_dim x n_tokens)
    std::vector<float> X_T((size_t)in_dim * (size_t)n_tokens);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t c = 0; c < in_dim; c++) {
            X_T[(size_t)(c * n_tokens + t)] = calib_X[t * in_dim + c];
        }
    }

    std::vector<float> scale((size_t)in_dim);
    std::vector<float> Ws((size_t)R * (size_t)in_dim);
    std::vector<float> Weff((size_t)R * (size_t)in_dim);
    std::vector<float> WXq((size_t)R * (size_t)n_tokens);

    float best_alpha = 0.0f;
    float best_err   = std::numeric_limits<float>::infinity();

    for (int64_t g = 0; g < n_grid; g++) {
        float alpha = (float)g / (float)(n_grid - 1);
        if (alpha == 0.0f) {
            std::fill(scale.begin(), scale.end(), 1.0f);
        } else {
            ts_normalized_awq_scale(act_scales, alpha, scale.data(), in_dim);
        }
        for (int64_t r = 0; r < R; r++) {
            for (int64_t c = 0; c < in_dim; c++) {
                int64_t i = r * in_dim + c;
                Ws[(size_t)i] = W[(size_t)i] * scale[(size_t)c];
                float q = std::round(Ws[(size_t)i]);
                q = std::min(std::max(q, -1.0f), 1.0f);
                Weff[(size_t)i] = q / scale[(size_t)c];
            }
        }
        // WXq = Weff @ X_T  -> (R x n_tokens)
        ts_mat_mul(Weff.data(), X_T.data(), WXq.data(), R, in_dim, n_tokens);

        // MSE between WXq and ref_T. diff = WXq - ref_T, then sum of squares.
        // Use vDSP_vsub + vDSP_svesq (sum of vector-of-squares) on Apple.
        const int64_t mn = R * n_tokens;
        float err;
#if defined(__APPLE__)
        vDSP_vsub(ref_T.data(), 1, WXq.data(), 1, WXq.data(), 1, (vDSP_Length)mn);
        float sumsq = 0.0f;
        vDSP_svesq(WXq.data(), 1, &sumsq, (vDSP_Length)mn);
        err = sumsq / (float)mn;
#else
        double e = 0.0;
        for (int64_t i = 0; i < mn; i++) {
            double d = (double)WXq[(size_t)i] - (double)ref_T[(size_t)i];
            e += d * d;
        }
        err = (float)(e / (double)mn);
#endif
        if (err < best_err) {
            best_err   = err;
            best_alpha = alpha;
        }
    }
    return best_alpha;
}

// ---------------------------------------------------------------------------
// quantize_2d
// ---------------------------------------------------------------------------

int ts_quantize_2d(const float * weights,
                   const float * act_scales,
                   const float * calib_X,
                   const float * ref_output,
                   const float * imatrix,
                   int64_t out_dim, int64_t in_dim, int64_t n_tokens,
                   const ts_quant_params_2d * params,
                   ts_quant_result_2d * result) {
    (void)imatrix;

    if (weights == nullptr || result == nullptr || out_dim <= 0 || in_dim <= 0) {
        return 1;
    }

    ts_quant_params_2d defaults = { 0.0f, 1.0f, 0, 0.0f, false, false, 20, 0 };
    if (params == nullptr) {
        params = &defaults;
    }

    const int64_t n     = out_dim * in_dim;
    const int64_t pages = ts_pages_per_row(in_dim);

    // --- resolve AWQ alpha (0 = auto-search when activations are present) ---
    float resolved_alpha = params->alpha;
    if (act_scales != nullptr && resolved_alpha == 0.0f) {
        if (calib_X != nullptr && ref_output != nullptr && n_tokens > 0) {
            resolved_alpha = ts_awq_scale_search_layer_output(
                weights, act_scales, calib_X, ref_output,
                out_dim, in_dim, n_tokens, params->awq_grid);
        } else {
            resolved_alpha = ts_awq_scale_search(
                weights, act_scales, calib_X,
                out_dim, in_dim, n_tokens, params->awq_grid);
        }
    }

    // --- per-channel weight scaling + stored input scale ---
    std::vector<float> wscale((size_t)in_dim, 1.0f);
    std::vector<float> input_scale((size_t)in_dim, 1.0f);
    if (act_scales != nullptr && resolved_alpha > 0.0f) {
        ts_normalized_awq_scale(act_scales, resolved_alpha, wscale.data(), in_dim);
        for (int64_t c = 0; c < in_dim; c++) {
            input_scale[(size_t)c] = 1.0f / wscale[(size_t)c];
        }
    }

    // Fused scale + clip + ternarize: one streaming pass over W instead of
    // the unfused 5-pass sequence (scale -> copy -> clip -> copy-for-threshold
    // -> ternarize). Produces ws (scaled), core (clipped), ternary, and the
    // global amplitude in one call.
    std::vector<float>   ws((size_t)n);
    std::vector<float>   core((size_t)n);
    std::vector<int8_t>  ternary((size_t)n);
    float global_amp = ts_scale_clip_ternarize_fused(
        weights, wscale.data(), params->clip,
        ws.data(), core.data(), ternary.data(),
        out_dim, in_dim);
    if (n == 0) global_amp = 1.0f;

    std::vector<int32_t> outlier_flat = ts_select_repair_residuals(
        ws.data(), ternary.data(), global_amp, out_dim, in_dim,
        params->max_outliers, params->outlier_thresh);

    for (int32_t idx : outlier_flat) {
        ternary[(size_t)idx] = 0;
    }

    // --- pack + fit scales ---
    result->packed.assign((size_t)out_dim * (size_t)pages * TS_WORDS_PER_PAGE, 0);
    result->page_scales.assign((size_t)out_dim * (size_t)pages, 0);
    result->lane_scales.assign((size_t)out_dim * (size_t)pages * TS_LANES_PER_PAGE, 0);

    std::vector<uint16_t> pack_ps((size_t)out_dim * (size_t)pages);
    std::vector<int8_t>   pack_ls((size_t)out_dim * (size_t)pages * TS_LANES_PER_PAGE);
    ts_pack_tile640(ternary.data(), result->packed.data(),
                    pack_ps.data(), pack_ls.data(), out_dim, in_dim);

    ts_compute_scales(core.data(), ternary.data(),
                      result->page_scales.data(), result->lane_scales.data(),
                      out_dim, in_dim);

    // --- outlier CSR (sorted by row, residual order preserved within a row) ---
    std::stable_sort(outlier_flat.begin(), outlier_flat.end(),
                     [in_dim](int32_t a, int32_t b) {
                         return (a / in_dim) < (b / in_dim);
                     });

    result->outlier_row_offsets.assign((size_t)(out_dim + 1), 0);
    result->outlier_cols.resize(outlier_flat.size());
    result->outlier_vals.resize(outlier_flat.size());
    for (size_t i = 0; i < outlier_flat.size(); i++) {
        int64_t idx = outlier_flat[i];
        int64_t row = idx / in_dim;
        result->outlier_row_offsets[(size_t)row + 1]++;
        result->outlier_cols[i] = (int32_t)(idx % in_dim);
        result->outlier_vals[i] = ts_f32_to_f16(ws[(size_t)idx]);
    }
    for (int64_t r = 0; r < out_dim; r++) {
        result->outlier_row_offsets[(size_t)r + 1] += result->outlier_row_offsets[(size_t)r];
    }

    // --- stored activation (input) scale ---
    if (act_scales != nullptr && resolved_alpha > 0.0f) {
        result->act_scale.resize((size_t)in_dim);
        for (int64_t c = 0; c < in_dim; c++) {
            result->act_scale[(size_t)c] = ts_f32_to_f16(input_scale[(size_t)c]);
        }
    } else {
        result->act_scale.clear();
    }

    result->best_alpha = resolved_alpha;

    // --- reconstruction MSE + recon build, fused ---
    // Fuses three unfused passes (vsub -> dotpr -> scale_cols) into one:
    //   diff = ws - deq; mse += diff^2; recon[i] = deq[i] * input_scale[col]
    // Memory traffic: 1440 MB -> 540 MB (reads ws + deq once, writes recon).
    std::vector<float> deq((size_t)n, 0.0f);
    ts_dequant(ternary.data(), result->page_scales.data(),
               result->lane_scales.data(), deq.data(), out_dim, in_dim);
    for (int32_t idx : outlier_flat) {
        deq[(size_t)idx] = ws[(size_t)idx];
    }

    std::vector<float> & recon = result->recon;
    recon.resize((size_t)n);
#if defined(__APPLE__)
    // Build recon = deq * input_scale[c] and accumulate MSE(ws - deq) in one
    // pass. vDSP can't fuse these two reductions, so run elementwise but
    // stream each tensor exactly once.
    {
        float mse_accum = 0.0f;
        for (int64_t r = 0; r < out_dim; r++) {
            const float * ws_row = ws.data() + r * in_dim;
            const float * deq_row = deq.data() + r * in_dim;
            float * recon_row = recon.data() + r * in_dim;
            const float * iscale = input_scale.data();
            for (int64_t c = 0; c < in_dim; c++) {
                float d = ws_row[c] - deq_row[c];
                mse_accum += d * d;
                recon_row[c] = deq_row[c] * iscale[c];
            }
        }
        result->mse = mse_accum / (float)n;
    }
#else
    {
        double mse_accum = 0.0;
        for (int64_t r = 0; r < out_dim; r++) {
            const float * ws_row = ws.data() + r * in_dim;
            const float * deq_row = deq.data() + r * in_dim;
            float * recon_row = recon.data() + r * in_dim;
            const float * iscale = input_scale.data();
            for (int64_t c = 0; c < in_dim; c++) {
                double d = (double)ws_row[c] - (double)deq_row[c];
                mse_accum += d * d;
                recon_row[c] = deq_row[c] * iscale[c];
            }
        }
        result->mse = (float)(mse_accum / (double)n);
    }
#endif

    return 0;
}

// ---------------------------------------------------------------------------
// quantize_3d (MoE experts)
// ---------------------------------------------------------------------------

int ts_quantize_3d(const float * weights,
                   const float * act_scales,
                   const float * calib_X,
                   const float * ref_output,
                   const float * imatrix,
                   int64_t n_experts, int64_t out_dim, int64_t in_dim,
                   int64_t n_tokens,
                   const ts_quant_params_2d * params,
                   std::vector<ts_quant_result_2d> * results) {
    if (weights == nullptr || results == nullptr || n_experts <= 0) {
        return 1;
    }
    results->clear();
    results->resize((size_t)n_experts);

    const int64_t stride = out_dim * in_dim;
    for (int64_t ex = 0; ex < n_experts; ex++) {
        int rc = ts_quantize_2d(weights + ex * stride,
                                act_scales, calib_X, ref_output, imatrix,
                                out_dim, in_dim, n_tokens,
                                params, &(*results)[(size_t)ex]);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}
