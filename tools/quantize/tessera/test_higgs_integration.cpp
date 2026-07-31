//
// test_higgs_integration.cpp
//
// Integration test: HIGGS alpha estimation -> cache round-trip ->
// alpha-weighted GA fitness. Verifies the full pipeline composes.
//

#include "tessera-higgs.h"
#include "tessera-higgs-cache.h"
#include "tessera-search.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char * msg) {
    if (cond) {
        g_pass++;
    } else {
        g_fail++;
        printf("  FAIL: %s\n", msg);
    }
}

// ---------------------------------------------------------------------------
// Mock metric: delta = alpha_true[layer] * t_j^2 (noiseless linear response)
// ---------------------------------------------------------------------------

struct mock_ctx {
    const float * alpha_true;
    int64_t J;
    float t_min;
    float t_max;
    int64_t call_idx;
};

static float mock_metric(const float * perturbed, int64_t out_dim, int64_t in_dim,
                         int64_t layer_idx, void * ctx) {
    (void)perturbed; (void)out_dim; (void)in_dim;
    mock_ctx * mc = (mock_ctx *)ctx;

    int64_t j = mc->call_idx % mc->J;
    mc->call_idx++;

    float t_j = mc->t_min + (mc->t_max - mc->t_min) * (float)j / (float)(mc->J - 1);
    float t2 = t_j * t_j;

    return mc->alpha_true[layer_idx] * t2;
}

// ---------------------------------------------------------------------------
// Test 1: alpha estimation on 3-layer synthetic model
// ---------------------------------------------------------------------------

static bool test_estimation() {
    printf("--- test_estimation ---\n");

    const int64_t n_layers = 3;
    const int64_t outs[] = { 64, 64, 64 };
    const int64_t ins[]  = { 32, 32, 32 };
    const float alpha_true[] = { 1.5f, 3.0f, 0.5f };

    // build synthetic weights
    std::vector<std::vector<float>> wbuf(n_layers);
    std::vector<const float *> wptrs(n_layers);
    for (int64_t l = 0; l < n_layers; l++) {
        int64_t n = outs[l] * ins[l];
        wbuf[l].resize(n);
        for (int64_t i = 0; i < n; i++) {
            wbuf[l][i] = 0.05f * (float)(i % 17) - 0.02f * (float)(i % 7);
        }
        wptrs[l] = wbuf[l].data();
    }

    mock_ctx mc;
    mc.alpha_true = alpha_true;
    mc.J = 5;
    mc.t_min = 0.01f;
    mc.t_max = 0.10f;
    mc.call_idx = 0;

    ts_higgs_params params = {};
    params.J = 5;
    params.t_min = 0.01f;
    params.t_max = 0.10f;
    params.r2_threshold = 0.90f;
    params.alpha_floor = 1e-6f;
    params.seed = 42u;
    params.verbose = false;

    ts_higgs_result result;
    int rc = ts_higgs_estimate(wptrs.data(), outs, ins, n_layers,
                               mock_metric, &mc, &params, &result);
    check(rc == 0, "ts_higgs_estimate returns 0");
    check((int64_t)result.layers.size() == n_layers, "alpha vector has 3 entries");

    std::vector<float> alphas = ts_higgs_extract_alphas(&result);
    check((int64_t)alphas.size() == n_layers, "extract_alphas returns 3 entries");

    bool all_positive = true;
    bool all_finite = true;
    for (int64_t l = 0; l < n_layers; l++) {
        if (alphas[l] <= 0.0f) all_positive = false;
        if (!std::isfinite(alphas[l])) all_finite = false;
        printf("  layer %lld: alpha=%.6f (true=%.2f) r2=%.4f valid=%d\n",
               (long long)l, alphas[l], alpha_true[l],
               result.layers[l].r_squared, result.layers[l].valid);
    }
    check(all_positive, "all alpha positive");
    check(all_finite, "all alpha finite");

    return g_fail == 0;
}

// ---------------------------------------------------------------------------
// Test 2: cache round-trip
// ---------------------------------------------------------------------------

static bool test_cache_roundtrip() {
    printf("--- test_cache_roundtrip ---\n");

    // use a temp dir
    std::string cache_dir = "/tmp/tessera_higgs_test_cache";
    system(("rm -rf " + cache_dir).c_str());

    // create synthetic weights and compute key
    const int64_t n_layers = 3;
    const int64_t outs[] = { 64, 64, 64 };
    const int64_t ins[]  = { 32, 32, 32 };

    std::vector<std::vector<float>> wbuf(n_layers);
    std::vector<const float *> wptrs(n_layers);
    for (int64_t l = 0; l < n_layers; l++) {
        int64_t n = outs[l] * ins[l];
        wbuf[l].resize(n);
        for (int64_t i = 0; i < n; i++) {
            wbuf[l][i] = 0.1f * (float)(i + 1);
        }
        wptrs[l] = wbuf[l].data();
    }

    ts_higgs_cache_key key = ts_higgs_cache_compute_key(
        wptrs.data(), outs, ins, n_layers);
    check(key.hex.size() == 64, "cache key is 64 hex chars");
    printf("  key: %s\n", key.hex.c_str());

    // store
    float alpha_in[] = { 1.5f, 3.0f, 0.5f };
    int rc = ts_higgs_cache_store(&key, alpha_in, n_layers, &cache_dir);
    check(rc == 0, "cache store returns 0");

    // load
    auto loaded = ts_higgs_cache_load(&key, &cache_dir);
    check(loaded.has_value(), "cache load hits");

    if (loaded.has_value()) {
        check((int64_t)loaded->size() == n_layers, "loaded 3 alphas");
        bool identical = true;
        for (int64_t l = 0; l < n_layers; l++) {
            if (fabsf((*loaded)[l] - alpha_in[l]) > 1e-6f) {
                identical = false;
                printf("  mismatch at %lld: got %.10g expected %.10g\n",
                       (long long)l, (*loaded)[l], alpha_in[l]);
            }
        }
        check(identical, "cache round-trip produces identical alpha");
    }

    // wrong key -> miss
    ts_higgs_cache_key wrong_key = key;
    wrong_key.hash[0] ^= 0xff;
    wrong_key.hex = "0000000000000000000000000000000000000000000000000000000000000000";
    auto miss = ts_higgs_cache_load(&wrong_key, &cache_dir);
    check(!miss.has_value(), "wrong key -> cache miss");

    // cleanup
    system(("rm -rf " + cache_dir).c_str());

    return true;
}

// ---------------------------------------------------------------------------
// Test 3: fitness with alpha vs uniform
// ---------------------------------------------------------------------------

static bool test_fitness_weighted() {
    printf("--- test_fitness_weighted ---\n");

    const int64_t n_layers = 3;
    float t2[] = { 0.01f, 0.02f, 0.03f };
    float alpha[] = { 2.0f, 1.0f, 0.5f };

    ts_search_config cfg_alpha;
    cfg_alpha.layer_alpha = alpha;
    cfg_alpha.n_layers = n_layers;

    ts_search_config cfg_uniform;
    cfg_uniform.layer_alpha = nullptr;
    cfg_uniform.n_layers = n_layers;

    float fit_alpha   = ts_search_fitness(t2, &cfg_alpha);
    float fit_uniform = ts_search_fitness(t2, &cfg_uniform);

    // uniform: 0.01 + 0.02 + 0.03 = 0.06
    // alpha:   2*0.01 + 1*0.02 + 0.5*0.03 = 0.02 + 0.02 + 0.015 = 0.055
    printf("  fitness_alpha=%.6f fitness_uniform=%.6f\n", fit_alpha, fit_uniform);

    check(fabsf(fit_uniform - 0.06f) < 1e-6f, "uniform fitness = Sum t_l^2");
    check(fabsf(fit_alpha - 0.055f) < 1e-6f, "alpha fitness = Sum alpha_l * t_l^2");
    check(fabsf(fit_alpha - fit_uniform) > 1e-6f, "alpha fitness != uniform fitness");

    return true;
}

// ---------------------------------------------------------------------------
// Test 4: nullptr alpha == uniform (backward compat)
// ---------------------------------------------------------------------------

static bool test_fitness_backward_compat() {
    printf("--- test_fitness_backward_compat ---\n");

    const int64_t n_layers = 4;
    float t2[] = { 0.1f, 0.2f, 0.3f, 0.4f };

    ts_search_config cfg_null;
    cfg_null.layer_alpha = nullptr;
    cfg_null.n_layers = n_layers;

    float ones[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    ts_search_config cfg_ones;
    cfg_ones.layer_alpha = ones;
    cfg_ones.n_layers = n_layers;

    float fit_null = ts_search_fitness(t2, &cfg_null);
    float fit_ones = ts_search_fitness(t2, &cfg_ones);

    printf("  fitness_null=%.6f fitness_ones=%.6f\n", fit_null, fit_ones);

    check(fabsf(fit_null - fit_ones) < 1e-9f, "nullptr alpha == all-ones alpha");
    check(fabsf(fit_null - 1.0f) < 1e-6f, "uniform fitness = 1.0 for t2={0.1..0.4}");

    // edge cases
    check(ts_search_fitness(nullptr, &cfg_null) == 0.0f, "null t2 -> 0");
    check(ts_search_fitness(t2, nullptr) == 0.0f, "null cfg -> 0");

    ts_search_config cfg_zero;
    cfg_zero.layer_alpha = nullptr;
    cfg_zero.n_layers = 0;
    check(ts_search_fitness(t2, &cfg_zero) == 0.0f, "n_layers=0 -> 0");

    return true;
}

// ---------------------------------------------------------------------------

int main() {
    printf("HIGGS integration tests\n\n");

    struct { const char * name; bool (*fn)(); } tests[] = {
        { "estimation",          test_estimation },
        { "cache_roundtrip",     test_cache_roundtrip },
        { "fitness_weighted",    test_fitness_weighted },
        { "fitness_backward_compat", test_fitness_backward_compat },
    };

    bool all = true;
    for (auto & t : tests) {
        bool ok = t.fn();
        printf("[%s] %s\n\n", ok ? "PASS" : "FAIL", t.name);
        all = all && ok;
    }

    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return all ? 0 : 1;
}
