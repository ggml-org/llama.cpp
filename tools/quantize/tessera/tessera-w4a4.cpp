//
// tessera-w4a4.cpp
//
// W4A4 activation quantization: per-token / per-tensor activation scales,
// LLM.int8-style per-channel outlier decomposition, and an activation-aware
// wrapper around ts_quantize_2d. Calibration side only; the runtime dequant
// kernel is a later wave (see docs/w4a4-calibration-design.md section 7).
//

#include "tessera-w4a4.h"
#include "tessera-quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// f16 helpers (IEEE 754 binary16, round-to-nearest-even). Mirrors the
// static helper in tessera-quant.cpp so W4A4 outlier values round-trip
// deterministically against the weight-side convention.
// ---------------------------------------------------------------------------

static uint16_t ts_w4a4_f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x & 0x80000000u) >> 16;
    uint32_t exp  = (x & 0x7f800000u) >> 23;
    uint32_t mant = (x & 0x007fffffu);

    if (exp == 0xff) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    int32_t e = (int32_t)exp - 127 + 15;
    if (e >= 0x1f) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - e);
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

static float ts_w4a4_f16_to_f32(uint16_t h) {
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
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

struct ts_w4a4_config ts_w4a4_default_config(void) {
    struct ts_w4a4_config cfg;
    cfg.enable            = false;
    cfg.activation_bits   = 4;
    cfg.scale_mode        = TS_W4A4_PER_TOKEN;
    cfg.outlier_thresh    = 6.0f;
    cfg.outlier_frac      = 0.001f;
    cfg.static_percentile = 0.999f;
    return cfg;
}

int ts_w4a4_qmax(int activation_bits) {
    if (activation_bits < 2) {
        activation_bits = 2;
    }
    if (activation_bits > 8) {
        activation_bits = 8;
    }
    return (1 << (activation_bits - 1)) - 1;
}

// ---------------------------------------------------------------------------
// activation scales
// ---------------------------------------------------------------------------

void ts_w4a4_compute_act_scales(const float * calib_X,
                                int64_t n_tokens, int64_t in_dim,
                                const ts_w4a4_config * cfg,
                                ts_w4a4_act_scales * out) {
    struct ts_w4a4_config d = ts_w4a4_default_config();
    if (cfg == nullptr) {
        cfg = &d;
    }
    out->mode       = cfg->scale_mode;
    out->qmax       = ts_w4a4_qmax(cfg->activation_bits);
    out->per_token.clear();
    out->per_tensor = 0.0f;

    if (calib_X == nullptr || n_tokens <= 0 || in_dim <= 0 || out->qmax <= 0) {
        return;
    }
    const float inv = 1.0f / (float)out->qmax;

    if (cfg->scale_mode == TS_W4A4_PER_TENSOR) {
        // single scale from the static percentile of |X|
        std::vector<float> abs((size_t)(n_tokens * in_dim));
        for (int64_t i = 0; i < n_tokens * in_dim; i++) {
            abs[(size_t)i] = std::fabs(calib_X[i]);
        }
        float p = cfg->static_percentile;
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        size_t idx = (size_t)std::llround((double)p * (double)(abs.size() - 1));
        std::nth_element(abs.begin(), abs.begin() + idx, abs.end());
        out->per_tensor = abs[idx] * inv;
        return;
    }

    // per-token dynamic: scale[t] = max_c |X[t,c]| / qmax
    out->per_token.resize((size_t)n_tokens);
    for (int64_t t = 0; t < n_tokens; t++) {
        const float * row = calib_X + (size_t)t * in_dim;
        float maxabs = 0.0f;
        for (int64_t c = 0; c < in_dim; c++) {
            maxabs = std::max(maxabs, std::fabs(row[c]));
        }
        out->per_token[(size_t)t] = maxabs * inv;
    }
}

// ---------------------------------------------------------------------------
// outlier detection (LLM.int8)
// ---------------------------------------------------------------------------

void ts_w4a4_detect_outliers(const float * calib_X,
                             int64_t n_tokens, int64_t in_dim,
                             const ts_w4a4_config * cfg,
                             ts_w4a4_outliers * out) {
    struct ts_w4a4_config d = ts_w4a4_default_config();
    if (cfg == nullptr) {
        cfg = &d;
    }
    out->channels.clear();
    out->mask.assign((size_t)std::max<int64_t>(in_dim, 0), 0);
    out->frac = 0.0f;

    if (calib_X == nullptr || n_tokens <= 0 || in_dim <= 0) {
        return;
    }

    // per-channel max |X| across tokens
    std::vector<float> chan_max((size_t)in_dim, 0.0f);
    for (int64_t t = 0; t < n_tokens; t++) {
        const float * row = calib_X + (size_t)t * in_dim;
        for (int64_t c = 0; c < in_dim; c++) {
            chan_max[(size_t)c] = std::max(chan_max[(size_t)c], std::fabs(row[c]));
        }
    }

    // threshold rule
    std::vector<uint32_t> cand;
    cand.reserve((size_t)in_dim);
    for (int64_t c = 0; c < in_dim; c++) {
        if (chan_max[(size_t)c] > cfg->outlier_thresh) {
            cand.push_back((uint32_t)c);
        }
    }

    // cap at outlier_frac * in_dim (frac <= 0 disables the cap), keeping the
    // highest-magnitude channels
    if (cfg->outlier_frac > 0.0f) {
        int64_t cap = (int64_t)(cfg->outlier_frac * (float)in_dim);
        if ((int64_t)cand.size() > cap) {
            std::partial_sort(cand.begin(), cand.begin() + cap, cand.end(),
                              [&chan_max](uint32_t a, uint32_t b) {
                                  if (chan_max[a] != chan_max[b]) {
                                      return chan_max[a] > chan_max[b];
                                  }
                                  return a < b;
                              });
            cand.resize((size_t)cap);
        }
    }

    std::sort(cand.begin(), cand.end());
    out->channels = std::move(cand);
    for (uint32_t c : out->channels) {
        out->mask[c] = 1;
    }
    out->frac = (float)out->channels.size() / (float)in_dim;
}

// ---------------------------------------------------------------------------
// decompose / recompose
// ---------------------------------------------------------------------------

void ts_w4a4_decompose(const float * calib_X,
                       int64_t n_tokens, int64_t in_dim,
                       const ts_w4a4_config * cfg,
                       const ts_w4a4_outliers * outliers,
                       ts_w4a4_decomp * out) {
    struct ts_w4a4_config d = ts_w4a4_default_config();
    if (cfg == nullptr) {
        cfg = &d;
    }
    out->outlier_channels.clear();
    out->quant.clear();
    out->outlier_vals.clear();
    out->scales.mode       = cfg->scale_mode;
    out->scales.qmax       = ts_w4a4_qmax(cfg->activation_bits);
    out->scales.per_token.clear();
    out->scales.per_tensor = 0.0f;

    if (calib_X == nullptr || n_tokens <= 0 || in_dim <= 0 || out->scales.qmax <= 0) {
        return;
    }
    if (outliers != nullptr) {
        out->outlier_channels = outliers->channels;
    }

    const int64_t n_out  = (int64_t)out->outlier_channels.size();
    const int     qmax   = out->scales.qmax;
    const int     qmin   = -qmax - 1;   // INT4: [-8, 7] for qmax = 7
    const float   inv    = 1.0f / (float)qmax;

    // per-channel outlier position (-1 = not an outlier)
    std::vector<int64_t> pos((size_t)in_dim, -1);
    for (int64_t i = 0; i < n_out; i++) {
        pos[out->outlier_channels[(size_t)i]] = i;
    }

    // The INT4 range is set by the NON-outlier channels only: outliers are
    // stored at FP16 and must not inflate the scale (the whole point of the
    // LLM.int8 isolation). Per-tensor static uses the percentile of the
    // non-outlier magnitudes; per-token dynamic uses the per-row non-outlier
    // max.
    float tensor_scale = 0.0f;
    if (cfg->scale_mode == TS_W4A4_PER_TENSOR) {
        std::vector<float> abs;
        abs.reserve((size_t)(n_tokens * in_dim));
        for (int64_t t = 0; t < n_tokens; t++) {
            const float * row = calib_X + (size_t)t * in_dim;
            for (int64_t c = 0; c < in_dim; c++) {
                if (pos[(size_t)c] < 0) {
                    abs.push_back(std::fabs(row[c]));
                }
            }
        }
        if (!abs.empty()) {
            float p = std::min(std::max(cfg->static_percentile, 0.0f), 1.0f);
            size_t idx = (size_t)std::llround((double)p * (double)(abs.size() - 1));
            std::nth_element(abs.begin(), abs.begin() + idx, abs.end());
            tensor_scale = abs[idx] * inv;
        }
        out->scales.per_tensor = tensor_scale;
    } else {
        out->scales.per_token.resize((size_t)n_tokens);
        for (int64_t t = 0; t < n_tokens; t++) {
            const float * row = calib_X + (size_t)t * in_dim;
            float maxabs = 0.0f;
            for (int64_t c = 0; c < in_dim; c++) {
                if (pos[(size_t)c] < 0) {
                    maxabs = std::max(maxabs, std::fabs(row[c]));
                }
            }
            out->scales.per_token[(size_t)t] = maxabs * inv;
        }
    }

    out->quant.assign((size_t)(n_tokens * in_dim), 0);
    out->outlier_vals.assign((size_t)(n_tokens * n_out), 0);

    for (int64_t t = 0; t < n_tokens; t++) {
        const float * row = calib_X + (size_t)t * in_dim;
        float scale = (cfg->scale_mode == TS_W4A4_PER_TENSOR)
                          ? tensor_scale
                          : out->scales.per_token[(size_t)t];
        for (int64_t c = 0; c < in_dim; c++) {
            int64_t oi = pos[(size_t)c];
            if (oi >= 0) {
                out->outlier_vals[(size_t)(t * n_out + oi)] = ts_w4a4_f32_to_f16(row[c]);
            } else {
                int q = 0;
                if (scale > 0.0f) {
                    q = (int)std::lround(row[c] / scale);
                    q = std::min(std::max(q, qmin), qmax);
                }
                out->quant[(size_t)(t * in_dim + c)] = (int8_t)q;
            }
        }
    }
}

void ts_w4a4_recompose(const ts_w4a4_decomp * decomp,
                       int64_t n_tokens, int64_t in_dim,
                       float * out) {
    if (decomp == nullptr || out == nullptr || n_tokens <= 0 || in_dim <= 0) {
        return;
    }
    const int64_t n_out = (int64_t)decomp->outlier_channels.size();
    std::vector<int64_t> pos((size_t)in_dim, -1);
    for (int64_t i = 0; i < n_out; i++) {
        pos[decomp->outlier_channels[(size_t)i]] = i;
    }

    for (int64_t t = 0; t < n_tokens; t++) {
        float scale = (decomp->scales.mode == TS_W4A4_PER_TENSOR)
                          ? decomp->scales.per_tensor
                          : decomp->scales.per_token[(size_t)t];
        for (int64_t c = 0; c < in_dim; c++) {
            int64_t oi = pos[(size_t)c];
            if (oi >= 0) {
                out[t * in_dim + c] = ts_w4a4_f16_to_f32(
                    decomp->outlier_vals[(size_t)(t * n_out + oi)]);
            } else {
                out[t * in_dim + c] = (float)decomp->quant[(size_t)(t * in_dim + c)] * scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// activation-aware weight quantization
// ---------------------------------------------------------------------------

// effective bits/weight: the stored Tile640 weight components (ternary trits +
// page scales + lane scales + repair outliers) plus the amortized W4A4
// activation overhead (per-token scale + FP16 outlier channels).
static float ts_w4a4_effective_bits(const ts_quant_result_2d * r,
                                    int64_t out_dim, int64_t in_dim,
                                    float outlier_frac) {
    const int64_t n = out_dim * in_dim;
    if (n <= 0) {
        return 0.0f;
    }
    const double inv_n = 1.0 / (double)n;
    double bits = 0.0;
    bits += (double)r->packed.size()      * 32.0 * inv_n;  // ternary trits
    bits += (double)r->page_scales.size() * 16.0 * inv_n;  // f16 page scales
    bits += (double)r->lane_scales.size() *  8.0 * inv_n;  // i8 lane scales
    bits += (double)r->outlier_cols.size() * (32.0 + 16.0) * inv_n; // col + f16 val
    bits += (double)r->outlier_row_offsets.size() * 32.0 * inv_n;
    // activation overhead: FP16 outlier channels + one f32 per-token scale
    bits += (double)outlier_frac * 16.0;
    bits += 32.0 * inv_n * (double)out_dim;   // per-token scale amortized over a row
    return (float)bits;
}

int ts_w4a4_quantize_weights(const float * weights,
                             const float * calib_X,
                             int64_t out_dim, int64_t in_dim, int64_t n_tokens,
                             const ts_quant_params_2d * qparams,
                             const ts_w4a4_config * cfg,
                             ts_quant_result_2d * base_out,
                             ts_w4a4_weight_result * out) {
    if (weights == nullptr || base_out == nullptr || out == nullptr ||
        out_dim <= 0 || in_dim <= 0) {
        return 1;
    }
    struct ts_w4a4_config d = ts_w4a4_default_config();
    if (cfg == nullptr) {
        cfg = &d;
    }

    // per-channel activation magnitude (mean |X|) makes the weight quantizer
    // activation-aware via the existing AWQ scaling in ts_quantize_2d
    std::vector<float> act;
    const float * act_ptr = nullptr;
    if (calib_X != nullptr && n_tokens > 0) {
        act.assign((size_t)in_dim, 0.0f);
        for (int64_t t = 0; t < n_tokens; t++) {
            const float * row = calib_X + (size_t)t * in_dim;
            for (int64_t c = 0; c < in_dim; c++) {
                act[(size_t)c] += std::fabs(row[c]);
            }
        }
        for (int64_t c = 0; c < in_dim; c++) {
            act[(size_t)c] /= (float)n_tokens;
        }
        act_ptr = act.data();
    }

    int rc = ts_quantize_2d(weights, act_ptr, nullptr, nullptr, act_ptr,
                            out_dim, in_dim, 0, qparams, base_out);
    if (rc != 0) {
        return rc;
    }

    out->base = base_out;
    ts_w4a4_detect_outliers(calib_X, n_tokens, in_dim, cfg, &out->outliers);
    ts_w4a4_compute_act_scales(calib_X, n_tokens, in_dim, cfg, &out->scales);
    out->effective_bits = ts_w4a4_effective_bits(base_out, out_dim, in_dim,
                                                 out->outliers.frac);
    return 0;
}

// ---------------------------------------------------------------------------
// sidecar
// ---------------------------------------------------------------------------

std::string ts_w4a4_scale_mode_str(enum ts_w4a4_scale_mode mode) {
    return (mode == TS_W4A4_PER_TENSOR) ? "per_tensor" : "per_token";
}

std::string ts_w4a4_sidecar_json(const ts_w4a4_sidecar * sc) {
    std::string s = "\"w4a4\": {";
    s += "\"enabled\": ";
    s += (sc->enabled ? "true" : "false");
    s += ", \"activation_bits\": " + std::to_string(sc->activation_bits);
    s += ", \"scale_mode\": \"" + ts_w4a4_scale_mode_str(sc->scale_mode) + "\"";
    s += ", \"outlier_frac\": " + std::to_string(sc->outlier_frac);
    s += ", \"act_scale_static\": " + std::to_string(sc->act_scale_static);
    s += ", \"act_outlier_count\": " + std::to_string(sc->outlier_channels.size());
    s += ", \"act_outlier_indices\": [";
    for (size_t i = 0; i < sc->outlier_channels.size(); i++) {
        if (i) {
            s += ", ";
        }
        s += std::to_string(sc->outlier_channels[i]);
    }
    s += "]}";
    return s;
}
