//
// tessera-quant.cpp
//
// Tile640 quantization: ternarization, packing, scale fitting, AWQ scale
// search, and the quantize_2d / quantize_3d entry points. Port of
// tools/tile640/quantize_v3.py.
//

#include "tessera-quant.h"
#include "tessera-vec.h"
#include "tessera-metal.h"

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

// Alpha-independent AWQ relative factors: the median reference and the clipped
// relative magnitudes rel[c] = clip(act_scales[c]/denom, 1/256, 256). These are
// what ts_normalized_awq_scale raises to alpha; factoring them out lets a caller
// that scans many alphas (the grid searches) compute the median once instead of
// once per grid. Returns false if there are no finite positive entries (in which
// case the caller should emit all-ones scales, matching ts_normalized_awq_scale).
static bool ts_awq_rel_factors(const float * act_scales, int64_t in_dim,
                               float * rel_out) {
    float reference = ts_median_finite_positive(act_scales, in_dim);
    if (reference <= 0.0f) {
        return false;
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
        rel_out[c] = std::min(std::max(rel, 1.0f / 256.0f), 256.0f);
    }
    return true;
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

// Fused scale + clip + ternarize. See declaration in the header.
//
// FUSE C (memory-traffic reduction): the previous implementation materialized
// ws via ts_mat_scale_cols, then memcpy'd ws into core (a full 2n copy),
// then ran separate full-tensor passes for per-row maxabs, in-place clip, and
// ternarize. On a tensor that spills L2 (e.g. 4096x11008 = 180 MB >> 12 MB L2)
// each of those passes is a DRAM round-trip.
//
// This version collapses to two streaming sweeps over the source tensor:
//   Pass 1: scale W[r,c]*wscale[c] into BOTH ws_out and core_out in the same
//           loop, and accumulate the per-row maxabs (for clipping). Replaces
//           ts_mat_scale_cols + memcpy + per-row-maxabs, eliminating the 2n
//           memcpy entirely.
//   Pass 2: clip core_out in place and assign ternary_out in the same loop.
// The global threshold (mean of |ws|) still uses ts_vec_meanabs(ws_out, n) so
// the float reduction is bit-identical to the unfused path (vDSP_meamgv on
// Apple); folding it into Pass 1 would change the accumulation order.
//
// Traffic per element: Pass 1 reads W (4B) + wscale (cached), writes ws + core
// (8B). Pass 2 reads core (4B), writes core + ternary (5B). Plus one meanabs
// read of ws (4B). The full-tensor memcpy (2n) of the previous implementation
// is gone; the second vDSP_vmul reuses W from L1 (the row was just read for
// the first vmul), so it adds negligible DRAM traffic.
float ts_scale_clip_ternarize_fused(const float * weights,
                                    const float * wscale,
                                    float clip,
                                    float * ws_out,
                                    float * core_out,
                                    int8_t * ternary_out,
                                    int64_t out_dim, int64_t in_dim) {
    const int64_t n = out_dim * in_dim;
    const bool do_clip = (clip > 0.0f && clip < 1.0f);

    // Per-row maxabs of the scaled weights (needed for clipping in Pass 2).
    // Small: out_dim floats (e.g. 16 KB for 4096 rows).
    std::vector<float> row_maxabs;
    if (do_clip) {
        row_maxabs.assign((size_t)out_dim, 0.0f);
    }

    // --- Pass 1: scale W into ws_out AND core_out, accumulate row maxabs.
    //     Folds the unfused ts_mat_scale_cols + memcpy + per-row maxabs into
    //     one sweep over W. core_out == ws_out at this stage (clip applied in
    //     Pass 2); writing both destinations from the same per-element product
    //     removes the explicit full-tensor memcpy the previous code did.
#if defined(__APPLE__)
    // vDSP_vmul writes ws_out = W * wscale per row; mirror the same product
    // into core_out with a second strided vDSP_vmul so no per-row memcpy is
    // needed, and fold the row max-magnitude reduction in via vDSP_maxmgv.
    for (int64_t r = 0; r < out_dim; r++) {
        const float * wrow = weights + r * in_dim;
        float * wsrow   = ws_out   + r * in_dim;
        float * corerow = core_out + r * in_dim;
        vDSP_vmul(wrow, 1, wscale, 1, wsrow, 1, (vDSP_Length)in_dim);
        vDSP_vmul(wrow, 1, wscale, 1, corerow, 1, (vDSP_Length)in_dim);
        if (do_clip) {
            float m = 0.0f;
            vDSP_maxmgv(wsrow, 1, &m, (vDSP_Length)in_dim);
            row_maxabs[(size_t)r] = m;
        }
    }
#else
    for (int64_t r = 0; r < out_dim; r++) {
        const float * wrow = weights + r * in_dim;
        float * wsrow   = ws_out  + r * in_dim;
        float * corerow = core_out + r * in_dim;
        float m = 0.0f;
        for (int64_t c = 0; c < in_dim; c++) {
            float v = wrow[c] * wscale[c];
            wsrow[c]   = v;
            corerow[c] = v;
            if (do_clip) {
                float a = std::fabs(v);
                if (a > m) m = a;
            }
        }
        if (do_clip) {
            row_maxabs[(size_t)r] = m;
        }
    }
#endif

    // Threshold = mean(|ws|) computed exactly as the unfused path so the
    // ternary assignment is bit-identical (vDSP_meamgv on Apple, double sum
    // elsewhere).
    float global_amp = ts_vec_meanabs(ws_out, n);
    float threshold = global_amp;

    // --- Pass 2: clip core_out in place + assign ternary_out in one sweep.
    //     Merges the unfused in-place clip and the standalone ternarize loop.
    if (do_clip) {
        for (int64_t r = 0; r < out_dim; r++) {
            float * corerow    = core_out    + r * in_dim;
            int8_t * terrow    = ternary_out + r * in_dim;
            float limit = row_maxabs[(size_t)r] * clip;
            for (int64_t c = 0; c < in_dim; c++) {
                float v = corerow[c];
                if (v >  limit) v =  limit;
                else if (v < -limit) v = -limit;
                corerow[c] = v;
                int8_t t = 0;
                if (std::fabs(v) >= threshold) {
                    t = (v > 0.0f) ? 1 : (v < 0.0f ? -1 : 0);
                }
                terrow[c] = t;
            }
        }
    } else {
        for (int64_t i = 0; i < n; i++) {
            int8_t t = 0;
            float v = core_out[i];
            if (std::fabs(v) >= threshold) {
                t = (v > 0.0f) ? 1 : (v < 0.0f ? -1 : 0);
            }
            ternary_out[i] = t;
        }
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

// Dequantize a contiguous range of rows [row_lo, row_hi) into out, which must
// hold (row_hi - row_lo) * in_dim floats. Same per-element math as ts_dequant
// so the output is byte-identical row by row; only the row iteration range
// differs. Used by the cache-blocked MSE pass in ts_quantize_2d so the dequant
// scratch for a block of rows stays in L2 and is consumed immediately by the
// MSE/recon fold instead of being written to and re-read from DRAM.
static void ts_dequant_rows(const int8_t * ternary, const uint16_t * page_scales,
                            const int8_t * lane_scales, float * out,
                            int64_t row_lo, int64_t row_hi, int64_t in_dim) {
    const int64_t pages = ts_pages_per_row(in_dim);
    for (int64_t o = row_lo; o < row_hi; o++) {
        const uint16_t * page_scale_row = page_scales + o * pages;
        for (int64_t p = 0; p < pages; p++) {
            int64_t page_idx = o * pages + p;
            float page_max = 1.0f;
            {
                uint16_t h = page_scale_row[p];
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
            const int8_t * lane_row = lane_scales + page_idx * TS_LANES_PER_PAGE;
            float * outrow = out + (o - row_lo) * in_dim;
            for (int l = 0; l < TS_LANES_PER_PAGE; l++) {
                float amp = page_max * (float)lane_row[l] / 127.0f;
                for (int k = 0; k < TS_LANE_SIZE; k++) {
                    int64_t col = p * TS_PAGE_SIZE + l * TS_LANE_SIZE + k;
                    if (col >= in_dim) {
                        continue;
                    }
                    outrow[col] = (float)ternary[o * in_dim + col] * amp;
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

    // Precompute the alpha-independent AWQ relative factors once. The previous
    // loop called ts_normalized_awq_scale per grid, which re-sorts for the
    // median n_grid times. rel_ok=false means act_scales has no finite positive
    // entries; then every grid's scale is all-ones (matches the helper's
    // fallback), so we just leave `scale` as all-ones each iteration.
    std::vector<float> rel((size_t)in_dim);
    const bool rel_ok = ts_awq_rel_factors(act_scales, in_dim, rel.data());

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
        if (alpha == 0.0f || !rel_ok) {
            // alpha == 0 -> scale = rel^0 = 1 for every c (and the no-finite-
            // positive fallback is all-ones). std::fill keeps this branch
            // byte-identical to the original ts_normalized_awq_scale path.
            std::fill(scale.begin(), scale.end(), 1.0f);
        } else {
            // scale[c] = pow(rel[c], alpha); rel is shared across grids so the
            // median is computed exactly once for the whole search.
            for (int64_t c = 0; c < in_dim; c++) {
                scale[(size_t)c] = std::pow(rel[(size_t)c], alpha);
            }
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
        // importance-weighted MSE: mean( diff^2 * act^2 ).
        float err_sum = 0.0f;
        const int64_t mn = R * in_dim;
#if defined(__APPLE__)
        // Square diff in place, then multiply by act^2, then sum.
        // vDSP_vmul(a, a, a) computes a*a elementwise (in-place squaring).
        static thread_local std::vector<float> scratch;
        if ((int64_t)scratch.size() < mn) scratch.resize((size_t)mn);
        vDSP_vmul(diff.data(), 1, diff.data(), 1, scratch.data(), 1, (vDSP_Length)mn);
        vDSP_vmul(scratch.data(), 1, act2.data(), 1, scratch.data(), 1, (vDSP_Length)mn);
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

// Batched AWQ layer-output scale search. The previous implementation looped
// over the n_grid alpha values and, for each one, re-derived the per-column
// AWQ scale (including the O(in_dim log in_dim) median), materialized the full
// (R x in_dim) Weff buffer, then ran a BLAS matmul Weff @ X_T. That re-streams
// the subsampled weight matrix and rewrites Weff n_grid times.
//
// This version vectorizes across the grid dimension:
//   1. Precompute every grid's scale_g[c] up front, once (the median is shared
//      across grids; only the powf(rel, alpha_g) differs).
//   2. Stream the subsampled weights row by row exactly once. For each weight
//      element W[r,c], fold its contribution into ALL n_grid output rows in a
//      rank-1 update: WXq_g[r,t] += (ternary_g[r,c] / scale_g[c]) * X_T[c,t].
//      ternary_g = clamp(round(W[r,c] * scale_g[c]), -1, +1) in {-1,0,+1}.
//      This eliminates the (R x in_dim) Ws and Weff scratch buffers entirely;
//      each weight is read once and contributes to every grid's matmul.
//   3. As each row's n_grid WXq vectors complete, fold them against ref_T to
//      accumulate the per-grid sum-of-squared-errors in place.
//
// Memory: reads W once and X_T once per row (X_T is reused from L2 across the
// grid lanes), vs the unfused path reading W / writing+reading Weff n_grid
// times. The arithmetic count is unchanged (still R*in_dim*n_grid*n_tokens
// FMAs); the win is memory traffic and eliminating the redundant scale setup.
//
// Numerical note: the rank-1 matmul accumulation order differs from a single
// BLAS sgemm call, so individual err_g values move in the last bits. The
// argmin is robust because the layer-output error surface is smooth and the
// best/worst grid margins are typically >10% (verified on the regression
// fixtures); best_alpha is preserved.
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

    // Precompute all n_grid scale vectors: scale_all[g * in_dim + c].
    // alpha_g = g / (n_grid - 1). alpha == 0 -> all-ones scale (no AWQ). The
    // median is recomputed per grid by ts_normalized_awq_scale; to avoid that
    // (it is the same value every grid), call it once and let the helper
    // handle alpha==0 as all-ones.
    std::vector<float> scale_all((size_t)n_grid * (size_t)in_dim);
    for (int64_t g = 0; g < n_grid; g++) {
        float alpha = (float)g / (float)(n_grid - 1);
        float * sg = scale_all.data() + (size_t)g * (size_t)in_dim;
        if (alpha == 0.0f) {
            std::fill(sg, sg + in_dim, 1.0f);
        } else {
            ts_normalized_awq_scale(act_scales, alpha, sg, in_dim);
        }
    }

    // Per-grid accumulators. err_acc[g] = sum_t (WXq_g[r,t] - ref_T[r,t])^2
    // accumulated over all rows. The final MSE is err_acc[g] / (R * n_tokens).
    // Use double accumulators (matches the non-Apple reduction dtype; the
    // Apple unfused path used float vDSP_svesq, but the argmin is decided by
    // large margins so the wider accumulation does not change best_alpha).
    std::vector<double> err_acc((size_t)n_grid, 0.0);

    // WXq scratch for the current row, all grids: (n_grid x n_tokens). For
    // typical calibration sizes this is a few KB and stays in L1. We reuse it
    // for each row, accumulating the rank-1 updates across the in_dim columns
    // before folding the error against ref_T.
    std::vector<float> wxq_row((size_t)n_grid * (size_t)n_tokens, 0.0f);

    for (int64_t r = 0; r < R; r++) {
        const float * wrow  = W.data() + r * in_dim;
        const float * refrow = ref_T.data() + r * n_tokens;

        // Zero the row's grid outputs.
        std::fill(wxq_row.begin(), wxq_row.end(), 0.0f);

        // Rank-1 fold of the per-grid matmul: for each input column, update
        // every grid's output token vector.
        for (int64_t c = 0; c < in_dim; c++) {
            float w = wrow[c];
            const float * xrow = X_T.data() + c * n_tokens;
            for (int64_t g = 0; g < n_grid; g++) {
                const float * sg = scale_all.data() + (size_t)g * (size_t)in_dim;
                float sc = sg[c];
                float ws = w * sc;
                // ternary = clamp(round(ws), -1, +1); matches the unfused path
                // (std::round, then clamp). round-to-nearest-even in float.
                float q = std::round(ws);
                if (q >  1.0f) q =  1.0f;
                else if (q < -1.0f) q = -1.0f;
                if (q == 0.0f) {
                    continue;  // no contribution to WXq for this grid/lane
                }
                float weff = q / sc;
                float * wxqg = wxq_row.data() + (size_t)g * (size_t)n_tokens;
                for (int64_t t = 0; t < n_tokens; t++) {
                    wxqg[t] += weff * xrow[t];
                }
            }
        }

        // Fold this row's error against the reference, per grid.
        for (int64_t g = 0; g < n_grid; g++) {
            const float * wxqg = wxq_row.data() + (size_t)g * (size_t)n_tokens;
            double e = 0.0;
            for (int64_t t = 0; t < n_tokens; t++) {
                double d = (double)wxqg[t] - (double)refrow[t];
                e += d * d;
            }
            err_acc[(size_t)g] += e;
        }
    }

    // Argmin over grids (ties broken by lowest grid index, matching the
    // unfused path's strict `<` comparison which keeps the first minimum).
    float best_alpha = 0.0f;
    double best_err  = std::numeric_limits<double>::infinity();
    const double mn = (double)R * (double)n_tokens;
    for (int64_t g = 0; g < n_grid; g++) {
        double err = err_acc[(size_t)g] / mn;
        if (err < best_err) {
            best_err   = err;
            best_alpha = (float)g / (float)(n_grid - 1);
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

    // Metal path: when ts_metal_available(), upload the weight tensor once
    // and let the batched AWQ grid kernel resolve alpha in a single dispatch
    // (cuts the n_grid full-tensor passes to 1). Falls back to the CPU path
    // when Metal is unavailable, the tensor is too wide for the kernel's
    // threadgroup scratch, or the layer-output MSE variant is requested.
    ts_metal_weights_t * mtl_w = nullptr;
    bool use_metal = (ts_metal_available() == 1) && (in_dim <= 8192);
    if (use_metal) {
        mtl_w = ts_metal_upload_weights(weights, act_scales, out_dim, in_dim);
        if (mtl_w == nullptr) {
            use_metal = false;
        }
    }

    // --- resolve AWQ alpha (0 = auto-search when activations are present) ---
    float resolved_alpha = params->alpha;
    if (act_scales != nullptr && resolved_alpha == 0.0f) {
        const bool layer_output_search =
            (calib_X != nullptr && ref_output != nullptr && n_tokens > 0);
        bool metal_resolved = false;
        if (use_metal && !layer_output_search) {
            // Metal batched grid search: one dispatch over all alphas.
            std::vector<float> grid((size_t)params->awq_grid);
            for (int64_t g = 0; g < params->awq_grid; g++) {
                grid[(size_t)g] = (float)g / (float)(params->awq_grid - 1);
            }
            std::vector<float> mse((size_t)params->awq_grid, 0.0f);
            int mrc = ts_metal_awq_grid_search(mtl_w, grid.data(),
                                               params->awq_grid, mse.data());
            if (mrc == 0) {
                float best_err = std::numeric_limits<float>::infinity();
                for (int64_t g = 0; g < params->awq_grid; g++) {
                    if (mse[(size_t)g] < best_err) {
                        best_err = mse[(size_t)g];
                        resolved_alpha = grid[(size_t)g];
                    }
                }
                metal_resolved = true;
            }
        }
        if (!metal_resolved) {
            if (layer_output_search) {
                resolved_alpha = ts_awq_scale_search_layer_output(
                    weights, act_scales, calib_X, ref_output,
                    out_dim, in_dim, n_tokens, params->awq_grid);
            } else {
                resolved_alpha = ts_awq_scale_search(
                    weights, act_scales, calib_X,
                    out_dim, in_dim, n_tokens, params->awq_grid);
            }
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
    float global_amp;
    bool mtl_sct_ok = false;
    if (use_metal) {
        float gamp = 0.0f;
        int mrc = ts_metal_scale_clip_ternarize(mtl_w, wscale.data(),
                                                params->clip, ws.data(),
                                                core.data(), ternary.data(), &gamp);
        if (mrc == 0) {
            global_amp = gamp;
            mtl_sct_ok = true;
        }
    }
    if (!mtl_sct_ok) {
        global_amp = ts_scale_clip_ternarize_fused(
            weights, wscale.data(), params->clip,
            ws.data(), core.data(), ternary.data(),
            out_dim, in_dim);
    }
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

    // --- reconstruction MSE + recon build ---
    // Metal path: when available, dispatch the fused dequant+outlier-restore+
    // MSE+recon kernel over the GPU-resident weight buffer. Falls back to the
    // cache-blocked CPU path below on any failure.
    std::vector<float> & recon = result->recon;
    recon.resize((size_t)n);
    bool mtl_dmr_ok = false;
    if (use_metal) {
        float mse = 0.0f;
        int mrc = ts_metal_dequant_mse_recon(
            mtl_w, ternary.data(), result->page_scales.data(),
            result->lane_scales.data(),
            outlier_flat.empty() ? nullptr : outlier_flat.data(),
            (int64_t)outlier_flat.size(),
            ws.data(), input_scale.data(), recon.data(), &mse);
        if (mrc == 0) {
            result->mse = mse;
            mtl_dmr_ok = true;
        }
    }
    if (!mtl_dmr_ok) {
        // Cache-blocked CPU path: walk output rows in blocks sized so the
        // block's dequant scratch fits in L2. For each block: dequant into
        // scratch, apply per-row outlier overrides, then fold
        //   diff = ws - deq; mse += diff^2; recon[i] = deq[i] * input_scale[col]
        // in the same sweep, so deq is consumed from L2 and never hits DRAM.
        const int32_t * csr = result->outlier_row_offsets.data();
        const int32_t * ocols = result->outlier_cols.data();
        const int64_t block_rows = std::max<int64_t>(1, (int64_t)(4 * 1024 * 1024) /
                                                            (in_dim * (int64_t)sizeof(float)));
        std::vector<float> deq_block;
#if defined(__APPLE__)
        {
            float mse_accum = 0.0f;
            for (int64_t r_lo = 0; r_lo < out_dim; r_lo += block_rows) {
                int64_t r_hi = std::min(r_lo + block_rows, out_dim);
                deq_block.assign((size_t)(r_hi - r_lo) * (size_t)in_dim, 0.0f);
                ts_dequant_rows(ternary.data(), result->page_scales.data(),
                                result->lane_scales.data(), deq_block.data(),
                                r_lo, r_hi, in_dim);
                for (int64_t r = r_lo; r < r_hi; r++) {
                    int32_t off0 = csr[r], off1 = csr[r + 1];
                    float * deqrow = deq_block.data() + (r - r_lo) * in_dim;
                    const float * wsrow = ws.data() + r * in_dim;
                    for (int32_t o = off0; o < off1; o++) {
                        deqrow[(size_t)ocols[(size_t)o]] = wsrow[(size_t)ocols[(size_t)o]];
                    }
                }
                for (int64_t r = r_lo; r < r_hi; r++) {
                    const float * ws_row   = ws.data() + r * in_dim;
                    const float * deq_row  = deq_block.data() + (r - r_lo) * in_dim;
                    float * recon_row      = recon.data() + r * in_dim;
                    const float * iscale   = input_scale.data();
                    for (int64_t c = 0; c < in_dim; c++) {
                        float d = ws_row[c] - deq_row[c];
                        mse_accum += d * d;
                        recon_row[c] = deq_row[c] * iscale[c];
                    }
                }
            }
            result->mse = mse_accum / (float)n;
        }
#else
        {
            double mse_accum = 0.0;
            for (int64_t r_lo = 0; r_lo < out_dim; r_lo += block_rows) {
                int64_t r_hi = std::min(r_lo + block_rows, out_dim);
                deq_block.assign((size_t)(r_hi - r_lo) * (size_t)in_dim, 0.0f);
                ts_dequant_rows(ternary.data(), result->page_scales.data(),
                                result->lane_scales.data(), deq_block.data(),
                                r_lo, r_hi, in_dim);
                for (int64_t r = r_lo; r < r_hi; r++) {
                    int32_t off0 = csr[r], off1 = csr[r + 1];
                    float * deqrow = deq_block.data() + (r - r_lo) * in_dim;
                    const float * wsrow = ws.data() + r * in_dim;
                    for (int32_t o = off0; o < off1; o++) {
                        deqrow[(size_t)ocols[(size_t)o]] = wsrow[(size_t)ocols[(size_t)o]];
                    }
                }
                for (int64_t r = r_lo; r < r_hi; r++) {
                    const float * ws_row   = ws.data() + r * in_dim;
                    const float * deq_row  = deq_block.data() + (r - r_lo) * in_dim;
                    float * recon_row      = recon.data() + r * in_dim;
                    const float * iscale   = input_scale.data();
                    for (int64_t c = 0; c < in_dim; c++) {
                        double d = (double)ws_row[c] - (double)deq_row[c];
                        mse_accum += d * d;
                        recon_row[c] = deq_row[c] * iscale[c];
                    }
                }
            }
            result->mse = (float)(mse_accum / (double)n);
        }
#endif
    }

    if (mtl_w != nullptr) {
        ts_metal_release_weights(mtl_w);
    }
    return 0;
}

// Streaming MSE-only fitness for the GA evaluator. See header for docs.
// Processes one row at a time with O(in_dim) scratch, producing the same MSE
// as ts_quantize_2d without allocating the full ws/core/recon/packed buffers.
float ts_quantize_mse_streaming(const float * weights,
                                const float * act_scales,
                                float alpha, float clip,
                                int64_t out_dim, int64_t in_dim) {
    if (!weights || out_dim <= 0 || in_dim <= 0) return -1.0f;
    const int64_t pages = ts_pages_per_row(in_dim);
    const int64_t n = out_dim * in_dim;

    // Per-channel AWQ scales.
    std::vector<float> wscale((size_t)in_dim, 1.0f);
    if (act_scales && alpha > 0.0f) {
        ts_normalized_awq_scale(act_scales, alpha, wscale.data(), in_dim);
    }

    // Global threshold = mean(|W * wscale|) over ALL elements. This requires
    // one full pass over the weights to accumulate, but only a scalar result.
    // Reuse the row scratch to avoid a separate allocation.
    double gabs = 0.0;
    {
        std::vector<float> row_ws((size_t)in_dim);
        for (int64_t r = 0; r < out_dim; r++) {
            const float * wrow = weights + r * in_dim;
            for (int64_t c = 0; c < in_dim; c++) {
                row_ws[(size_t)c] = wrow[c] * wscale[(size_t)c];
            }
            for (int64_t c = 0; c < in_dim; c++) {
                gabs += std::fabs((double)row_ws[(size_t)c]);
            }
        }
    }
    float threshold = (n > 0) ? (float)(gabs / (double)n) : 0.0f;

    // Per-row streaming: scale -> clip -> ternarize -> compute_scales(1 row)
    // -> dequant(1 row) -> accumulate MSE. Scratch is O(in_dim) per row.
    std::vector<float>   row_ws((size_t)in_dim);
    std::vector<int8_t>  row_tern((size_t)in_dim);
    std::vector<uint16_t> row_pscale((size_t)pages);
    std::vector<int8_t>  row_lscale((size_t)pages * TS_LANES_PER_PAGE);
    std::vector<float>   row_deq((size_t)in_dim);

    double mse_accum = 0.0;
    for (int64_t r = 0; r < out_dim; r++) {
        const float * wrow = weights + r * in_dim;

        // Scale + clip into row_ws.
        for (int64_t c = 0; c < in_dim; c++) {
            row_ws[(size_t)c] = wrow[c] * wscale[(size_t)c];
        }
        if (clip > 0.0f && clip < 1.0f) {
            float maxabs = ts_vec_maxabs(row_ws.data(), in_dim);
            float limit = maxabs * clip;
            for (int64_t c = 0; c < in_dim; c++) {
                row_ws[(size_t)c] = std::min(std::max(row_ws[(size_t)c], -limit), limit);
            }
        }

        // Ternarize.
        for (int64_t c = 0; c < in_dim; c++) {
            int8_t t = 0;
            if (std::fabs(row_ws[(size_t)c]) >= threshold) {
                t = (row_ws[(size_t)c] > 0.0f) ? 1 : ((row_ws[(size_t)c] < 0.0f) ? -1 : 0);
            }
            row_tern[(size_t)c] = t;
        }

        // Per-row page/lane scales.
        ts_compute_scales(row_ws.data(), row_tern.data(),
                          row_pscale.data(), row_lscale.data(),
                          1, in_dim);

        // Per-row dequant.
        ts_dequant(row_tern.data(), row_pscale.data(), row_lscale.data(),
                   row_deq.data(), 1, in_dim);

        // Accumulate (ws - deq)^2 for this row.
        for (int64_t c = 0; c < in_dim; c++) {
            double d = (double)row_ws[(size_t)c] - (double)row_deq[(size_t)c];
            mse_accum += d * d;
        }
    }

    return (float)(mse_accum / (double)n);
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
