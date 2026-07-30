//
// test_higgs.cpp
//
// Tests for tessera-higgs.cpp. Verifies alpha_l recovery from a
// synthetic linear-response metric, R^2, and JSON round-trip.
//

#include "tessera-higgs.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct mock_ctx {
    const float * alpha_true;
    int64_t J;
    float t_min;
    float t_max;
    float noise_amp;
    uint32_t rng;
    int64_t call_idx;   // tracks which (layer, j) call we're on
    bool flat;          // if true, return constant (no t^2 dependence)
};

static uint32_t mock_xorshift(uint32_t * s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

// simulates a stable metric: delta = alpha_true * t_j^2 + noise
// uses the nominal t_j grid (what an averaged KL/PPL probe would see)
static float mock_metric(const float * perturbed, int64_t out_dim, int64_t in_dim,
                         int64_t layer_idx, void * ctx) {
    (void)perturbed; (void)out_dim; (void)in_dim;
    mock_ctx * mc = (mock_ctx *)ctx;

    int64_t j = mc->call_idx % mc->J;
    mc->call_idx++;

    float t_j = mc->t_min + (mc->t_max - mc->t_min) * (float)j / (float)(mc->J - 1);
    float t2 = t_j * t_j;

    float signal = mc->flat ? 1.0f : mc->alpha_true[layer_idx] * t2;

    float noise = 0.0f;
    if (mc->noise_amp > 0.0f) {
        float u1;
        do {
            u1 = (mock_xorshift(&mc->rng) + 1u) * (1.0f / 4294967296.0f);
        } while (u1 <= 0.0f);
        float u2 = mock_xorshift(&mc->rng) * (1.0f / 4294967296.0f);
        noise = mc->noise_amp * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }

    return signal + noise;
}

static bool test_alpha_recovery() {
    // 3 synthetic layers: 4x8, 8x16, 16x4
    const int64_t outs[] = { 4, 8, 16 };
    const int64_t ins[]  = { 8, 16, 4 };
    const float alpha_true[] = { 2.5f, 0.8f, 5.0f };
    const int64_t n_layers = 3;

    std::vector<std::vector<float>> wbuf(n_layers);
    std::vector<const float *> wptrs(n_layers);
    for (int64_t l = 0; l < n_layers; l++) {
        int64_t n = outs[l] * ins[l];
        wbuf[l].resize(n);
        for (int64_t i = 0; i < n; i++) {
            wbuf[l][i] = 0.1f * (float)(i + 1) - 0.05f * (float)(i % 3);
        }
        wptrs[l] = wbuf[l].data();
    }

    mock_ctx mc;
    mc.alpha_true = alpha_true;
    mc.J = 15;
    mc.t_min = 0.01f;
    mc.t_max = 0.10f;
    mc.noise_amp = 0.0f;  // noiseless for tight recovery
    mc.rng = 999u;
    mc.call_idx = 0;
    mc.flat = false;

    ts_higgs_params params = {};
    params.J = 15;
    params.t_min = 0.01f;
    params.t_max = 0.10f;
    params.r2_threshold = 0.95f;
    params.alpha_floor = 1e-6f;
    params.seed = 12345u;
    params.verbose = false;

    ts_higgs_result result;
    int rc = ts_higgs_estimate(wptrs.data(), outs, ins, n_layers,
                               mock_metric, &mc, &params, &result);
    if (rc != 0) {
        printf("  alpha_recovery: ts_higgs_estimate returned %d\n", rc);
        return false;
    }

    bool ok = true;
    for (int64_t l = 0; l < n_layers; l++) {
        const auto & lr = result.layers[l];
        float err = fabsf(lr.alpha_l - alpha_true[l]) / alpha_true[l];
        printf("  layer %ld: alpha=%.4f true=%.4f err=%.1f%% r2=%.6f valid=%d\n",
               (long)l, lr.alpha_l, alpha_true[l], err * 100.0f, lr.r_squared, lr.valid);
        if (err > 0.10f) ok = false;
        if (lr.r_squared < 0.95f) ok = false;
        if (!lr.valid) ok = false;
    }
    if (result.n_valid != n_layers) ok = false;
    if (result.n_fallback_uniform != 0) ok = false;

    return ok;
}

static bool test_fallback() {
    // flat metric response (no t^2 dependence) -> R^2 near 0 -> fallback
    const int64_t outs[] = { 4 };
    const int64_t ins[]  = { 4 };
    const float alpha_true[] = { 1.0f };
    std::vector<float> wbuf(16, 1.0f);
    const float * wptrs[] = { wbuf.data() };

    mock_ctx mc;
    mc.alpha_true = alpha_true;
    mc.J = 15;
    mc.t_min = 0.01f;
    mc.t_max = 0.10f;
    mc.noise_amp = 0.0f;
    mc.rng = 77u;
    mc.call_idx = 0;
    mc.flat = true;

    ts_higgs_params params = {};
    params.J = 15;
    params.seed = 1u;

    ts_higgs_result result;
    int rc = ts_higgs_estimate(wptrs, outs, ins, 1, mock_metric, &mc, &params, &result);
    if (rc != 0) return false;

    bool ok = true;
    if (result.layers[0].valid) ok = false;
    if (result.layers[0].alpha_l != 1.0f) ok = false;
    if (result.n_fallback_uniform != 1) ok = false;
    printf("  fallback: alpha=%.4f valid=%d n_fallback=%ld\n",
           result.layers[0].alpha_l, result.layers[0].valid,
           (long)result.n_fallback_uniform);
    return ok;
}

static bool test_json_roundtrip() {
    // build a small result manually
    ts_higgs_result result;
    result.n_valid = 2;
    result.n_fallback_uniform = 1;
    result.mean_alpha = 2.0f;

    ts_higgs_layer_result l0;
    l0.name = "attn_v";
    l0.alpha_l = 3.7e-3f;
    l0.r_squared = 0.984f;
    l0.valid = true;
    result.layers.push_back(l0);

    ts_higgs_layer_result l1;
    l1.name = "ffn_up";
    l1.alpha_l = 1.0f;
    l1.r_squared = 0.5f;
    l1.valid = false;
    result.layers.push_back(l1);

    ts_higgs_layer_result l2;
    l2.name = "attn_k";
    l2.alpha_l = 0.012f;
    l2.r_squared = 0.99f;
    l2.valid = true;
    result.layers.push_back(l2);

    std::string json = ts_higgs_to_json(&result);
    printf("  json:\n%s", json.c_str());

    float alphas[8] = {};
    int n = ts_higgs_from_json(json.c_str(), alphas, 8);
    printf("  parsed %d layers: [%.6f, %.6f, %.6f]\n", n, alphas[0], alphas[1], alphas[2]);

    bool ok = true;
    if (n != 3) ok = false;
    if (fabsf(alphas[0] - 3.7e-3f) > 1e-4f) ok = false;
    if (fabsf(alphas[1] - 1.0f) > 1e-4f) ok = false;
    if (fabsf(alphas[2] - 0.012f) > 1e-4f) ok = false;

    // bad input
    if (ts_higgs_from_json("{}", alphas, 8) != -1) ok = false;
    if (ts_higgs_from_json(nullptr, alphas, 8) != -1) ok = false;

    return ok;
}

int main() {
    struct { const char * name; bool (*fn)(); } tests[] = {
        { "alpha_recovery",  test_alpha_recovery },
        { "fallback",        test_fallback },
        { "json_roundtrip",  test_json_roundtrip },
    };

    bool all = true;
    for (auto & t : tests) {
        bool ok = t.fn();
        printf("[%s] %s\n", ok ? "PASS" : "FAIL", t.name);
        all = all && ok;
    }
    return all ? 0 : 1;
}
