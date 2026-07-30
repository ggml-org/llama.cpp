//
// tessera-mm-awq.cpp
//
// Per-modality AWQ scale computation. Runs the AWQ alpha search independently
// per modality and combines the per-modality MSE into a modality-weighted
// fitness (M1). Missing modalities either error (M2 default) or fall back to
// the text alpha (M8). Self-contained: the f16 cast, normalized scale, and
// grid search are inlined so this unit has no link dependency on
// tessera-quant; the quantization surrogate (round, clamp to [-1, 1]) and the
// two MSE objectives mirror tessera-quant.cpp.
//

#include "tessera-mm-awq.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// modality indices (match tessera_modality_t in the design doc)
enum {
    TS_MM_TEXT  = 0,
    TS_MM_IMAGE = 1,
    TS_MM_AUDIO = 2,
};

//
// f16 cast (IEEE 754 binary16, truncation)
//

static uint16_t ts_mm_f32_to_f16(float f) {
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
        uint32_t shift = (uint32_t)(14 - e);
        return (uint16_t)(sign | (mant >> shift));
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | (mant >> 13));
}

//
// normalized AWQ scale: s_c = (act_c / median)^alpha, clamped to [1/256, 256]
//

static float ts_mm_median_finite_positive(const float * x, int64_t n) {
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

static void ts_mm_normalized_awq_scale(const float * act, float alpha,
                                       float * scale_out, int64_t in_dim) {
    float reference = ts_mm_median_finite_positive(act, in_dim);
    if (reference <= 0.0f) {
        std::fill(scale_out, scale_out + in_dim, 1.0f);
        return;
    }
    float denom = std::max(reference, 1e-8f);
    for (int64_t c = 0; c < in_dim; c++) {
        float v = act[c];
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

//
// per-modality MSE at a fixed scale vector. With calibration activations and a
// reference output this is the layer-output reconstruction MSE; otherwise it is
// the importance-weighted weight MSE (weighted by act^2). The effective weight
// is Weff = clamp(round(W * s), -1, 1) / s.
//

static float ts_mm_awq_mse(const float * weights, const float * act,
                           const float * calib_X, const float * ref_output,
                           int64_t out_dim, int64_t in_dim, int64_t n_tokens,
                           const float * scale) {
    double err = 0.0;
    if (calib_X != nullptr && ref_output != nullptr && n_tokens > 0) {
        for (int64_t r = 0; r < out_dim; r++) {
            for (int64_t t = 0; t < n_tokens; t++) {
                float acc = 0.0f;
                for (int64_t c = 0; c < in_dim; c++) {
                    float q = std::round(weights[r * in_dim + c] * scale[c]);
                    q = std::min(std::max(q, -1.0f), 1.0f);
                    acc += (q / scale[c]) * calib_X[t * in_dim + c];
                }
                double d = (double)acc - (double)ref_output[t * out_dim + r];
                err += d * d;
            }
        }
        err /= (double)(out_dim * n_tokens);
    } else {
        for (int64_t r = 0; r < out_dim; r++) {
            for (int64_t c = 0; c < in_dim; c++) {
                float q = std::round(weights[r * in_dim + c] * scale[c]);
                q = std::min(std::max(q, -1.0f), 1.0f);
                double d = (double)(q / scale[c]) - (double)weights[r * in_dim + c];
                double a = act[c];
                err += d * d * a * a;
            }
        }
        err /= (double)(out_dim * in_dim);
    }
    return (float)err;
}

// Grid search over alpha in [0, 1] (or a single fixed alpha when fixed_alpha > 0).
static void ts_mm_awq_search_one(const float * weights, const float * act,
                                 const float * calib_X, const float * ref_output,
                                 int64_t out_dim, int64_t in_dim, int64_t n_tokens,
                                 int64_t n_grid, float fixed_alpha,
                                 float * best_alpha, float * best_mse) {
    std::vector<float> scale((size_t)in_dim);

    if (fixed_alpha > 0.0f) {
        ts_mm_normalized_awq_scale(act, fixed_alpha, scale.data(), in_dim);
        *best_alpha = fixed_alpha;
        *best_mse = ts_mm_awq_mse(weights, act, calib_X, ref_output,
                                  out_dim, in_dim, n_tokens, scale.data());
        return;
    }

    n_grid = std::max(n_grid, (int64_t)2);
    float ba = 0.0f;
    float bm = std::numeric_limits<float>::infinity();
    for (int64_t g = 0; g < n_grid; g++) {
        float alpha = (float)g / (float)(n_grid - 1);
        if (alpha == 0.0f) {
            std::fill(scale.begin(), scale.end(), 1.0f);
        } else {
            ts_mm_normalized_awq_scale(act, alpha, scale.data(), in_dim);
        }
        float m = ts_mm_awq_mse(weights, act, calib_X, ref_output,
                                out_dim, in_dim, n_tokens, scale.data());
        if (m < bm) {
            bm = m;
            ba = alpha;
        }
    }
    *best_alpha = ba;
    *best_mse = bm;
}

static std::vector<uint16_t> * ts_mm_scale_vec(ts_mm_awq_result * r, int i) {
    if (i == TS_MM_TEXT)  return &r->act_scale_text;
    if (i == TS_MM_IMAGE) return &r->act_scale_image;
    return &r->act_scale_audio;
}

//
// public API
//

ts_mm_awq_params ts_mm_awq_default_params() {
    ts_mm_awq_params p;
    for (int i = 0; i < 3; i++) {
        p.alpha[i] = 0.0f; // auto-search
        p.clip[i]  = 1.0f; // no clip
    }
    p.modality_weights[TS_MM_TEXT]  = 0.5f;
    p.modality_weights[TS_MM_IMAGE] = 0.3f;
    p.modality_weights[TS_MM_AUDIO] = 0.2f;
    p.error_on_missing = true;
    p.awq_grid = 20;
    return p;
}

int ts_mm_awq_compute(const float * weights,
                      const float * act_scales[3],
                      const float * calib_X[3],
                      const float * ref_output[3],
                      const int64_t n_tokens[3],
                      int64_t out_dim, int64_t in_dim,
                      const ts_mm_awq_params * params,
                      ts_mm_awq_result * result,
                      std::string * err_msg) {
    if (weights == nullptr || act_scales == nullptr || result == nullptr ||
        out_dim <= 0 || in_dim <= 0) {
        if (err_msg) {
            *err_msg = "ts_mm_awq_compute: invalid arguments";
        }
        return -1;
    }

    ts_mm_awq_params defaults = ts_mm_awq_default_params();
    if (params == nullptr) {
        params = &defaults;
    }
    const int64_t n_grid = params->awq_grid > 0 ? params->awq_grid : 20;

    bool present[3];
    int n_present = 0;
    for (int i = 0; i < 3; i++) {
        present[i] = (act_scales[i] != nullptr);
        if (present[i]) {
            n_present++;
        }
    }

    if (n_present == 0) {
        if (err_msg) {
            *err_msg = "ts_mm_awq_compute: no modality has activation scales";
        }
        return -1;
    }

    if (n_present < 3 && params->error_on_missing) {
        if (err_msg) {
            static const char * names[3] = { "text", "image", "audio" };
            for (int i = 0; i < 3; i++) {
                if (!present[i]) {
                    *err_msg = std::string("ts_mm_awq_compute: missing modality '") +
                               names[i] + "'; re-calibrate with the multi-modal imatrix " +
                               "or disable error_on_missing to use the text fallback";
                    break;
                }
            }
        }
        return -1;
    }

    const uint16_t unit_f16 = ts_mm_f32_to_f16(1.0f);
    for (int i = 0; i < 3; i++) {
        ts_mm_scale_vec(result, i)->assign((size_t)in_dim, unit_f16);
    }

    float mse[3] = { 0.0f, 0.0f, 0.0f };

    for (int i = 0; i < 3; i++) {
        if (!present[i]) {
            continue;
        }
        const float * cx  = calib_X    ? calib_X[i]    : nullptr;
        const float * ref = ref_output ? ref_output[i] : nullptr;
        const int64_t nt  = n_tokens   ? n_tokens[i]   : 0;

        float ba = 0.0f;
        float bm = 0.0f;
        ts_mm_awq_search_one(weights, act_scales[i], cx, ref,
                             out_dim, in_dim, nt, n_grid, params->alpha[i],
                             &ba, &bm);
        result->best_alpha[i] = ba;
        mse[i] = bm;

        // stored act_scale is the activation rescale 1/s (matches ts_quantize_2d)
        std::vector<float> s((size_t)in_dim, 1.0f);
        if (ba > 0.0f) {
            ts_mm_normalized_awq_scale(act_scales[i], ba, s.data(), in_dim);
        }
        std::vector<uint16_t> * dst = ts_mm_scale_vec(result, i);
        for (int64_t c = 0; c < in_dim; c++) {
            (*dst)[(size_t)c] = ts_mm_f32_to_f16(1.0f / s[(size_t)c]);
        }
    }

    // missing modalities fall back to the text alpha (M8); use the first present
    // modality as the source when text itself is absent
    if (n_present < 3) {
        int src = present[TS_MM_TEXT]  ? TS_MM_TEXT  :
                  present[TS_MM_IMAGE] ? TS_MM_IMAGE : TS_MM_AUDIO;
        for (int i = 0; i < 3; i++) {
            if (present[i]) {
                continue;
            }
            result->best_alpha[i] = result->best_alpha[src];
            mse[i] = mse[src];
            *ts_mm_scale_vec(result, i) = *ts_mm_scale_vec(result, src);
        }
    }

    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < 3; i++) {
        if (!present[i]) {
            continue;
        }
        num += (double)params->modality_weights[i] * (double)mse[i];
        den += (double)params->modality_weights[i];
    }
    result->weighted_mse = (den > 0.0) ? (float)(num / den) : 0.0f;
    for (int i = 0; i < 3; i++) {
        result->mse_per_modality[i] = mse[i];
    }

    return 0;
}
