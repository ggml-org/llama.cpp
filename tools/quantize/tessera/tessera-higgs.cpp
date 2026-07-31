//
// tessera-higgs.cpp
//
// HIGGS Algorithm 3: per-layer alpha_l estimation via Gaussian-noise
// perturbation sweep + through-origin least-squares fit.
//

#include "tessera-higgs.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

static const float TS_HIGGS_PI = 3.141592653589793f;

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

static uint32_t ts_higgs_xorshift32(uint32_t * state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float ts_higgs_gaussian(uint32_t * state) {
    float u1;
    do {
        u1 = (ts_higgs_xorshift32(state) + 1u) * (1.0f / 4294967296.0f);
    } while (u1 <= 0.0f);
    float u2 = ts_higgs_xorshift32(state) * (1.0f / 4294967296.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * TS_HIGGS_PI * u2);
}

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

int ts_higgs_estimate(const float ** weights,
                      const int64_t * out_dims,
                      const int64_t * in_dims,
                      int64_t n_layers,
                      ts_higgs_metric_fn metric_fn, void * metric_ctx,
                      const ts_higgs_params * params,
                      ts_higgs_result * result) {
    if (!weights || !out_dims || !in_dims || !metric_fn || !result) {
        return -1;
    }

    ts_higgs_params p;
    if (params) {
        p = *params;
    } else {
        p = {};
    }
    if (p.J < 2)          p.J = 15;
    if (p.t_min <= 0.0f)  p.t_min = 0.01f;
    if (p.t_max <= p.t_min) p.t_max = 0.10f;
    if (p.r2_threshold <= 0.0f) p.r2_threshold = 0.95f;
    if (p.alpha_floor <= 0.0f)  p.alpha_floor = 1e-6f;
    if (p.seed == 0)      p.seed = 42u;

    result->layers.clear();
    result->n_valid = 0;
    result->n_fallback_uniform = 0;
    result->mean_alpha = 0.0f;

    uint32_t rng = p.seed;

    for (int64_t l = 0; l < n_layers; l++) {
        const int64_t d_out = out_dims[l];
        const int64_t d_in  = in_dims[l];
        const int64_t n_elem = d_out * d_in;
        const float * W = weights[l];

        // ||W_l||_F
        float fro2 = 0.0f;
        for (int64_t i = 0; i < n_elem; i++) {
            fro2 += W[i] * W[i];
        }
        float fro = sqrtf(fro2);

        // noise scale: t * ||W||_F / sqrt(d_in * d_out)
        float dim_scale = 1.0f / sqrtf((float)(d_in * d_out));

        ts_higgs_layer_result lr;
        lr.name = "layer_" + std::to_string(l);
        lr.t_grid.resize(p.J);
        lr.deltas.resize(p.J);

        std::vector<float> G(n_elem);

        for (int64_t j = 0; j < p.J; j++) {
            float t_j;
            if (p.J == 1) {
                t_j = p.t_min;
            } else {
                t_j = p.t_min + (p.t_max - p.t_min) * (float)j / (float)(p.J - 1);
            }
            lr.t_grid[j] = t_j;

            // Eqn 9: G_l = W_l + t_j * ||W_l||_F / sqrt(d_in*d_out) * Sigma
            float scale = t_j * fro * dim_scale;
            for (int64_t i = 0; i < n_elem; i++) {
                G[i] = W[i] + scale * ts_higgs_gaussian(&rng);
            }

            lr.deltas[j] = metric_fn(G.data(), d_out, d_in, l, metric_ctx);
        }

        // through-origin least squares: alpha = Sum(delta_j * t_j^2) / Sum(t_j^4)
        float num = 0.0f, den = 0.0f;
        for (int64_t j = 0; j < p.J; j++) {
            float t2 = lr.t_grid[j] * lr.t_grid[j];
            num += lr.deltas[j] * t2;
            den += t2 * t2;
        }
        float alpha = (den > 0.0f) ? num / den : 0.0f;

        // R^2 (through-origin): 1 - SS_res / SS_tot, SS_tot = Sum(delta_j^2)
        float ss_res = 0.0f, ss_tot = 0.0f;
        for (int64_t j = 0; j < p.J; j++) {
            float t2 = lr.t_grid[j] * lr.t_grid[j];
            float resid = lr.deltas[j] - alpha * t2;
            ss_res += resid * resid;
            ss_tot += lr.deltas[j] * lr.deltas[j];
        }
        float r2 = (ss_tot > 0.0f) ? 1.0f - ss_res / ss_tot : 0.0f;

        lr.r_squared = r2;
        if (r2 >= p.r2_threshold && alpha > 0.0f) {
            lr.valid = true;
            lr.alpha_l = std::max(alpha, p.alpha_floor);
            result->n_valid++;
        } else {
            lr.valid = false;
            lr.alpha_l = 1.0f;
            result->n_fallback_uniform++;
        }

        if (p.verbose) {
            printf("  higgs: %-12s alpha=%.6e r2=%.4f %s\n",
                   lr.name.c_str(), lr.alpha_l, lr.r_squared,
                   lr.valid ? "" : "(fallback)");
        }

        result->layers.push_back(std::move(lr));
    }

    float sum = 0.0f;
    for (int64_t l = 0; l < n_layers; l++) {
        sum += result->layers[l].alpha_l;
    }
    result->mean_alpha = (n_layers > 0) ? sum / (float)n_layers : 0.0f;

    return 0;
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

std::string ts_higgs_to_json(const ts_higgs_result * result) {
    if (!result) return "{}";

    std::string s;
    s += "{\n";
    s += "  \"n_valid\": " + std::to_string(result->n_valid) + ",\n";
    s += "  \"n_fallback_uniform\": " + std::to_string(result->n_fallback_uniform) + ",\n";
    s += "  \"mean_alpha\": " + std::to_string(result->mean_alpha) + ",\n";
    s += "  \"layers\": [\n";
    for (size_t i = 0; i < result->layers.size(); i++) {
        const auto & lr = result->layers[i];
        s += "    {\"name\": \"" + lr.name + "\", ";
        s += "\"alpha\": " + std::to_string(lr.alpha_l) + ", ";
        s += "\"r2\": " + std::to_string(lr.r_squared) + ", ";
        s += "\"valid\": ";
        s += lr.valid ? "true" : "false";
        s += "}";
        if (i + 1 < result->layers.size()) s += ",";
        s += "\n";
    }
    s += "  ]\n";
    s += "}\n";
    return s;
}

// ---------------------------------------------------------------------------
// JSON deserialization (minimal: extract alpha values from "layers" array)
// ---------------------------------------------------------------------------

int ts_higgs_from_json(const char * json_str, float * alphas_out, int64_t max_layers) {
    if (!json_str || !alphas_out || max_layers < 1) return -1;

    const char * p = strstr(json_str, "\"layers\"");
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    p++;

    int64_t count = 0;
    while (count < max_layers) {
        const char * alpha_key = strstr(p, "\"alpha\"");
        if (!alpha_key) break;
        const char * colon = strchr(alpha_key, ':');
        if (!colon) break;
        colon++;
        while (*colon == ' ' || *colon == '\t') colon++;
        char * end = nullptr;
        float val = strtof(colon, &end);
        if (end == colon) break;
        alphas_out[count++] = val;
        p = end;
    }

    return (count > 0) ? (int)count : -1;
}

// ---------------------------------------------------------------------------
// Alpha extraction
// ---------------------------------------------------------------------------

std::vector<float> ts_higgs_extract_alphas(const ts_higgs_result * result) {
    std::vector<float> alphas;
    if (!result) return alphas;
    alphas.reserve(result->layers.size());
    for (const auto & lr : result->layers) {
        alphas.push_back(lr.alpha_l);
    }
    return alphas;
}
