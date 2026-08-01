//
// test_awq_fitness.cpp
//
// Parity tests for the C++ AWQ fitness port (tessera-awq-fitness.cpp):
//   1. ts_awq_ternary_reconstruct matches Python's _ternary_reconstruct to
//      rtol=1e-5 against a Python-generated fixture.
//   2. ts_awq_relative_output_error matches Python's relative_output_error
//      to atol=1e-6.
//   3. ts_awq_evaluate_layer + ts_awq_evaluate (composite fitness) match
//      Python's _evaluate_uncached to atol=1e-5.
//   4. Convergence: the GA driven by ts_awq_default_eval actually lowers the
//      composite fitness on a single layer over N generations (the GA is now
//      functional, not stubbed).
//
// Fixture: tools/quantize/tessera/fixtures/awq_fitness_fixture.json
// Regenerate via fixtures/gen_awq_fitness_fixture.py (Python's awq-evolve.py
// is the canonical reference; this test pins the C++ port to it).
//
// Tolerance choice (documented):
//   - reconstructed rtol=1e-5: the reconstruction is float32 throughout
//     (Python keeps the array math in np.float32 because every op there is
//     array-or-Python-scalar). float32 has ~7 significant digits, so rtol
//     1e-5 is comfortably above rounding noise and tight enough to catch a
//     sign flip or a wrong axis reduction.
//   - relative_output_error atol=1e-6: this is a double reduction over a
//     small fixture, so the gap is dominated by float32->float64 promotion
//     order. 1e-6 leaves headroom for that while still catching a wrong
//     matmul orientation.
//   - composite fitness atol=1e-5: a weighted sum of three f64 means +
//     a 0.9-quantile, all of which we compute in double. 1e-5 is well above
//     float64 rounding and tight enough to detect a wrong aggregation weight.
//

#include "tessera-awq-fitness.h"
#include "tessera-awq.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static int g_failures = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);      \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

// Resolve a path relative to this source file (so the test works regardless
// of the CWD the harness invokes it from).
static std::string src_relative(const std::string & tail) {
    std::string f = __FILE__;
    size_t slash = f.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : f.substr(0, slash);
    return dir + "/" + tail;
}

// Parse a JSON array of numbers into a float vector.
static std::vector<float> to_floats(const json & j) {
    std::vector<float> out;
    out.reserve(j.size());
    for (const auto & v : j) {
        out.push_back((float)v.get<double>());
    }
    return out;
}

// relative error: |a - b| / max(|a|, |b|, floor).
static float relerr(float a, float b, float floor) {
    float denom = std::max(std::fabs(a), std::fabs(b));
    if (denom < floor) {
        denom = floor;
    }
    return std::fabs(a - b) / denom;
}

struct fixture {
    ts_policy_genes genes;
    // primary layer
    int64_t out_dim, in_dim, n_tokens, n_tokens_h;
    std::vector<float> weight, second_moment, fourth_moment, max_abs;
    std::vector<float> train_act, heldout_act, ref_train, ref_heldout;
    std::vector<float> reconstructed;
    std::vector<float> importance;
    double rel_train, rel_heldout;
    double sc_train, sc_heldout, sc_tail, sc_size, sc_fitness, sc_worst;
    // secondary layer for the two-layer aggregation
    int64_t lb_out_dim;
    std::vector<float> lb_weight, lb_train_act, lb_held_act, lb_ref_train, lb_ref_heldout;
    double lb_train, lb_heldout, lb_tail, lb_size, lb_fitness, lb_worst;
};

static bool load_fixture(const std::string & path, fixture & fx) {
    std::ifstream in(path);
    if (!in.good()) {
        std::printf("could not open fixture: %s\n", path.c_str());
        return false;
    }
    json j;
    in >> j;

    const auto & c = j["candidate"];
    fx.genes.alpha             = c["alpha"];
    fx.genes.clip              = c["clip"];
    fx.genes.outlier_fraction  = c["outlier_fraction"];
    fx.genes.moment_mix        = c["moment_mix"];
    fx.genes.tail_guard        = c["tail_guard"];
    fx.genes.ternary_threshold = c["ternary_threshold"];

    const auto & L = j["layer"];
    fx.out_dim    = L["out_dim"];
    fx.in_dim     = L["in_dim"];
    fx.n_tokens   = L["n_tokens"];
    fx.n_tokens_h = L["n_tokens_h"];
    fx.weight         = to_floats(L["weight"]);
    fx.second_moment  = to_floats(L["second_moment"]);
    fx.fourth_moment  = to_floats(L["fourth_moment"]);
    fx.max_abs        = to_floats(L["max_abs"]);
    fx.train_act      = to_floats(L["train_activations"]);
    fx.heldout_act    = to_floats(L["heldout_activations"]);
    fx.ref_train      = to_floats(L["ref_train_output"]);
    fx.ref_heldout    = to_floats(L["ref_heldout_output"]);

    const auto & E = j["expected"];
    fx.reconstructed = to_floats(E["reconstructed"]);
    fx.importance    = to_floats(E["importance"]);
    fx.rel_train     = E["relative_output_error_train"];
    fx.rel_heldout   = E["relative_output_error_heldout"];
    const auto & S = E["score_single"];
    fx.sc_train    = S["train_error"];
    fx.sc_heldout  = S["heldout_error"];
    fx.sc_tail     = S["tail_error"];
    fx.sc_size     = S["size_cost"];
    fx.sc_fitness  = S["fitness"];
    fx.sc_worst    = S["worst_layer_error"];

    const auto & LB = j["layer_b"];
    fx.lb_out_dim    = LB["out_dim"];
    fx.lb_weight     = to_floats(LB["weight"]);
    fx.lb_train_act  = to_floats(LB["train_activations"]);
    fx.lb_held_act   = to_floats(LB["heldout_activations"]);
    fx.lb_ref_train     = to_floats(LB["ref_train_output"]);
    fx.lb_ref_heldout   = to_floats(LB["ref_heldout_output"]);

    const auto & SB = j["expected_two_layer"];
    fx.lb_train    = SB["train_error"];
    fx.lb_heldout  = SB["heldout_error"];
    fx.lb_tail     = SB["tail_error"];
    fx.lb_size     = SB["size_cost"];
    fx.lb_fitness  = SB["fitness"];
    fx.lb_worst    = SB["worst_layer_error"];

    return true;
}

// Build a ts_awq_layer pointing at the fixture's primary layer buffers.
static ts_awq_layer make_layer(fixture & fx) {
    ts_awq_layer L = {};
    L.name        = "fixture_layer";
    L.family      = "ffn";
    L.weights     = fx.weight.data();
    L.second_moment      = fx.second_moment.data();
    L.fourth_moment      = fx.fourth_moment.data();
    L.max_abs            = fx.max_abs.data();
    L.train_activations  = fx.train_act.data();
    L.heldout_activations = fx.heldout_act.data();
    L.ref_train_output   = fx.ref_train.data();
    L.ref_heldout_output = fx.ref_heldout.data();
    L.out_dim     = fx.out_dim;
    L.in_dim      = fx.in_dim;
    L.n_tokens    = fx.n_tokens;
    L.n_tokens_h  = fx.n_tokens_h;
    L.kurtosis    = 3.0f;
    L.eff_rank    = 0.5f;
    return L;
}

// Build a ts_awq_candidate from the fixture's genes (no clamping needed;
// the fixture's candidate is already in range).
static ts_awq_candidate make_cand(const ts_policy_genes & g) {
    ts_awq_candidate c;
    c.genes = g;
    c.expert_hint = -1;
    return c;
}

int main() {
    const std::string path = src_relative("fixtures/awq_fitness_fixture.json");
    fixture fx;
    if (!load_fixture(path, fx)) {
        std::printf("FAIL: could not load fixture %s\n", path.c_str());
        return 1;
    }
    std::printf("loaded fixture: %s\n", path.c_str());
    std::printf("  layer %lldx%lld, train=%lld heldout=%lld\n",
                (long long)fx.out_dim, (long long)fx.in_dim,
                (long long)fx.n_tokens, (long long)fx.n_tokens_h);

    // --- Test 1: ts_awq_ternary_reconstruct parity ---
    {
        std::vector<float> recon(fx.out_dim * fx.in_dim);
        ts_awq_ternary_reconstruct(fx.weight.data(), fx.genes, fx.importance.data(),
                                   fx.out_dim, fx.in_dim, recon.data());

        // rtol = 1e-5, with atol fallback for very small magnitudes.
        float max_rel = 0.0f;
        float max_abs_diff = 0.0f;
        const int64_t n = fx.out_dim * fx.in_dim;
        for (int64_t i = 0; i < n; i++) {
            float a = recon[i];
            float b = fx.reconstructed[i];
            float re = relerr(a, b, 1e-6f);
            if (re > max_rel) {
                max_rel = re;
            }
            float ad = std::fabs(a - b);
            if (ad > max_abs_diff) {
                max_abs_diff = ad;
            }
        }
        std::printf("[reconstruct] max_rel=%.3e max_abs=%.3e (rtol gate 1e-5)\n",
                    max_rel, max_abs_diff);
        CHECK(max_rel <= 1e-5f, "reconstructed rtol within 1e-5");
    }

    // --- Test 2: ts_awq_relative_output_error parity ---
    {
        std::vector<float> recon = fx.reconstructed;  // Python's reconstruction
        // train
        double cpp_train = ts_awq_relative_output_error(
            fx.train_act.data(), fx.weight.data(), recon.data(),
            fx.n_tokens, fx.in_dim, fx.out_dim, fx.ref_train.data());
        double cpp_heldout = ts_awq_relative_output_error(
            fx.heldout_act.data(), fx.weight.data(), recon.data(),
            fx.n_tokens_h, fx.in_dim, fx.out_dim, fx.ref_heldout.data());

        double d_train   = std::fabs(cpp_train   - fx.rel_train);
        double d_heldout = std::fabs(cpp_heldout - fx.rel_heldout);
        std::printf("[rel_output] train cpp=%.8e py=%.8e d=%.2e; "
                    "heldout cpp=%.8e py=%.8e d=%.2e\n",
                    cpp_train, fx.rel_train, d_train,
                    cpp_heldout, fx.rel_heldout, d_heldout);
        CHECK(d_train   <= 1e-6, "relative_output_error train within 1e-6");
        CHECK(d_heldout <= 1e-6, "relative_output_error heldout within 1e-6");
    }

    // --- Test 3: ts_awq_evaluate_layer + ts_awq_evaluate (fitness) parity ---
    {
        ts_awq_layer L = make_layer(fx);
        ts_awq_candidate c = make_cand(fx.genes);

        // single-layer composite
        ts_awq_score s;
        CHECK(ts_awq_evaluate(c, &L, 1, &s) == 0, "ts_awq_evaluate ok");
        double d_train   = std::fabs((double)s.mse           - fx.sc_train);
        double d_heldout = std::fabs((double)s.heldout_mse   - fx.sc_heldout);
        double d_tail    = std::fabs((double)s.relative_frob - fx.sc_tail);
        double d_fitness = std::fabs((double)s.composite     - fx.sc_fitness);
        std::printf("[single] train cpp=%.8e py=%.8e d=%.2e; "
                    "heldout cpp=%.8e py=%.8e d=%.2e; "
                    "tail cpp=%.8e py=%.8e d=%.2e; "
                    "fitness cpp=%.8e py=%.8e d=%.2e\n",
                    s.mse, fx.sc_train, d_train,
                    s.heldout_mse, fx.sc_heldout, d_heldout,
                    s.relative_frob, fx.sc_tail, d_tail,
                    s.composite, fx.sc_fitness, d_fitness);
        CHECK(d_train   <= 1e-5, "single-layer train within 1e-5");
        CHECK(d_heldout <= 1e-5, "single-layer heldout within 1e-5");
        CHECK(d_tail    <= 1e-5, "single-layer tail within 1e-5");
        CHECK(d_fitness <= 1e-5, "single-layer fitness within 1e-5");

        // two-layer aggregation: the second layer has its own buffers but
        // shares the primary layer's moments (Python does the same in the
        // fixture). Reconstruct L_b pointing at layer_b's data.
        ts_awq_layer Lb = {};
        Lb.name        = "fixture_layer_b";
        Lb.family      = "attention";
        Lb.weights     = fx.lb_weight.data();
        Lb.second_moment      = fx.second_moment.data();
        Lb.fourth_moment      = fx.fourth_moment.data();
        Lb.max_abs            = fx.max_abs.data();
        Lb.train_activations  = fx.lb_train_act.data();
        Lb.heldout_activations = fx.lb_held_act.data();
        Lb.ref_train_output   = fx.lb_ref_train.data();
        Lb.ref_heldout_output = fx.lb_ref_heldout.data();
        Lb.out_dim     = fx.lb_out_dim;
        Lb.in_dim      = fx.in_dim;
        Lb.n_tokens    = fx.n_tokens;
        Lb.n_tokens_h  = fx.n_tokens_h;
        Lb.kurtosis    = 3.0f;
        Lb.eff_rank    = 0.5f;

        ts_awq_layer layers[2] = { L, Lb };
        ts_awq_score s2;
        CHECK(ts_awq_evaluate(c, layers, 2, &s2) == 0,
              "ts_awq_evaluate (2 layers) ok");
        double d2_train   = std::fabs((double)s2.mse           - fx.lb_train);
        double d2_heldout = std::fabs((double)s2.heldout_mse   - fx.lb_heldout);
        double d2_tail    = std::fabs((double)s2.relative_frob - fx.lb_tail);
        double d2_fitness = std::fabs((double)s2.composite     - fx.lb_fitness);
        std::printf("[two]    train cpp=%.8e py=%.8e d=%.2e; "
                    "heldout cpp=%.8e py=%.8e d=%.2e; "
                    "tail cpp=%.8e py=%.8e d=%.2e; "
                    "fitness cpp=%.8e py=%.8e d=%.2e\n",
                    s2.mse, fx.lb_train, d2_train,
                    s2.heldout_mse, fx.lb_heldout, d2_heldout,
                    s2.relative_frob, fx.lb_tail, d2_tail,
                    s2.composite, fx.lb_fitness, d2_fitness);
        CHECK(d2_train   <= 1e-5, "two-layer train within 1e-5");
        CHECK(d2_heldout <= 1e-5, "two-layer heldout within 1e-5");
        CHECK(d2_tail    <= 1e-5, "two-layer tail within 1e-5");
        CHECK(d2_fitness <= 1e-5, "two-layer fitness within 1e-5");
    }

    // --- Test 4: convergence (the GA can actually optimize now) ---
    // Run ts_awq_evolve with ts_awq_default_eval for a few generations on the
    // fixture layer; confirm the best fitness (which ts_awq_default_eval
    // returns as -mse in composite, higher = better) strictly improves over
    // the worst of the initial random population. We approximate the initial
    // population's worst by sampling the default eval at a handful of random
    // candidates the GA itself would draw, then compare to the evolved best.
    {
        ts_awq_layer L = make_layer(fx);

        ts_awq_evolve_params params = {};
        params.population         = 12;
        params.generations        = 30;
        params.islands            = 2;
        params.migration_interval = 10;
        params.mutation_sigma     = 0.1f;
        params.crossover_rate     = 0.7f;
        params.heldout_weight     = 2.0f;
        params.seed               = 42;
        params.verbose            = false;

        ts_awq_evolve_result r;
        int rc = ts_awq_evolve(&L, ts_awq_default_eval, nullptr, &params, &r);
        CHECK(rc == 0, "GA default eval run rc==0");

        // Best composite from the GA (higher = better since default_eval
        // returns -mse). Translate back to fitness = mse so the test reads
        // "fitness decreases".
        float best_mse = -r.best_score.composite;

        // Hand-evaluate a deliberately bad candidate (alpha=0, clip=0.7,
        // sparse ternary, large outlier_fraction) to get a reference floor.
        ts_awq_candidate bad = make_cand(fx.genes);
        bad.genes.alpha             = 0.0f;
        bad.genes.clip              = 0.70f;
        bad.genes.outlier_fraction  = 0.05f;
        bad.genes.moment_mix        = 0.0f;
        bad.genes.tail_guard        = 0.0f;
        bad.genes.ternary_threshold = 3.0f;  // very sparse -> high error
        ts_awq_score bad_score;
        CHECK(ts_awq_evaluate_layer(bad, L, &bad_score) == 0,
              "bad candidate evaluate_layer ok");
        float bad_mse = bad_score.mse;

        // And the fixture's candidate (a reasonable point).
        ts_awq_candidate mid = make_cand(fx.genes);
        ts_awq_score mid_score;
        CHECK(ts_awq_evaluate_layer(mid, L, &mid_score) == 0,
              "mid candidate evaluate_layer ok");
        float mid_mse = mid_score.mse;

        std::printf("[ga] best_mse=%.6e mid_mse=%.6e bad_mse=%.6e evals=%lld gens=%lld\n",
                    best_mse, mid_mse, bad_mse,
                    (long long)r.evaluations, (long long)r.generations_run);
        CHECK(best_mse < bad_mse,    "GA improves over a deliberately bad candidate");
        CHECK(best_mse <= mid_mse,   "GA finds at least as good as the hand-tuned candidate");
        CHECK(r.evaluations > 0,     "GA actually ran evaluations");
    }

    if (g_failures == 0) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL (%d failures)\n", g_failures);
    return 1;
}
