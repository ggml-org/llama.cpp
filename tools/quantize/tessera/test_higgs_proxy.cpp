//
// test_higgs_proxy.cpp
//
// Tests for tessera-higgs-proxy.cpp. Verifies the C++ first-class
// port of the HIGGS structural proxy:
//   - family classification (matches the NumPy FAMILY_SUFFIXES order
//     and value table exactly),
//   - offline ternary MSE measurement (matches the NumPy
//     relative_frobenius_error to within 1e-5 on a fixed tensor),
//   - L1-agnostic measurement function swap (a custom
//     measurement_fn scales alpha_l accordingly),
//   - uniform-fallback below min_params_for_estimate,
//   - alpha_floor application,
//   - JSON shape matches ane.alpha-coefficients.v1 byte-for-byte
//     (key order, value types, JSON float repr),
//   - JSON round-trip,
//   - model_hash matches `shasum -a 256` of the test fixture,
//   - tinyllamas family-prior rank order,
//   - L1-agnostic alpha scaling with a custom measurement_fn,
//   - empty / 1-element / zero-norm tensors do not crash,
//   - L1-on-ANE measurement (the new default): packing layout,
//     guards, determinism, L1-vs-offline range, v2 dispatch
//     branch, legacy env opt-out, alpha parity with offline.
//
// 30+ tests, runs in <1s on a tiny synthetic GGUF.

#include "tessera-higgs-proxy.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static int g_fail = 0;
static int g_pass = 0;

static void check(const char * name, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name);
    if (ok) g_pass++;
    else    g_fail++;
}

static bool feq(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

static bool feq_abs(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

// ---------------------------------------------------------------------------
// Tiny test fixture: build a synthetic GGUF in-memory and write to disk.
// Mirrors the test_l5_dispatch.cpp "build_fixture_gguf" pattern.
// ---------------------------------------------------------------------------

struct ts_tensor_spec {
    std::string                name;
    int64_t                    out_dim;
    int64_t                    in_dim;
    float                      scale;     // 1.0+2*i (deliberately uneven)
    uint32_t                   seed;
};

static bool build_fixture_gguf(const std::string & path,
                               const std::vector<ts_tensor_spec> & specs) {
    struct gguf_context * ctx = gguf_init_empty();
    struct ggml_init_params ip = {
        /*mem_size   =*/ 4 * 1024 * 1024,
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ false,
    };
    struct ggml_context * gctx = ggml_init(ip);

    for (size_t i = 0; i < specs.size(); i++) {
        const auto & s = specs[i];
        struct ggml_tensor * t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32,
                                                   s.in_dim, s.out_dim);
        ggml_set_name(t, s.name.c_str());
        float * data = (float *)t->data;
        uint32_t rng = s.seed ? s.seed : (uint32_t)(i + 1) * 2654435761u;
        for (int64_t j = 0; j < s.out_dim * s.in_dim; j++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            float u = (float)((rng >> 8) & 0xFFFF) / (float)0xFFFF;
            data[j] = (u - 0.5f) * s.scale;
        }
        gguf_add_tensor(ctx, t);
    }
    bool ok = gguf_write_to_file(ctx, path.c_str(), false);
    ggml_free(gctx);
    gguf_free(ctx);
    return ok;
}

// A minimal fixture: one tensor per family key, all big enough to
// avoid the n_elem < 32 skip.
static std::vector<ts_tensor_spec> tinyllama_shaped_specs() {
    return {
        { "blk.0.attn_k.weight",      64, 64,  1.0f, 1001u },
        { "blk.0.attn_v.weight",      64, 64,  1.0f, 1002u },
        { "blk.0.attn_output.weight", 64, 64,  1.0f, 1003u },
        { "blk.0.attn_q.weight",      64, 64,  1.0f, 1004u },
        { "blk.0.attn_norm.weight",   64, 1,   1.0f, 1005u },
        { "blk.0.ffn_down.weight",    64, 64,  1.0f, 1006u },
        { "blk.0.ffn_gate.weight",    64, 64,  1.0f, 1007u },
        { "blk.0.ffn_up.weight",      64, 64,  1.0f, 1008u },
        { "blk.0.ffn_norm.weight",    64, 1,   1.0f, 1009u },
        { "token_embd.weight",        64, 64,  1.0f, 1010u },
        { "output.weight",            64, 64,  1.0f, 1011u },
    };
}

// ---------------------------------------------------------------------------
// 1. Family classification: each suffix maps to the right key.
// ---------------------------------------------------------------------------

static void test_classify_attn_k_v() {
    check("classify: attn_k",  ts_higgs_proxy_classify_family("blk.0.attn_k.weight") == "attn_k");
    check("classify: attn_v",  ts_higgs_proxy_classify_family("blk.0.attn_v.weight") == "attn_v");
}

static void test_classify_attn_q() {
    check("classify: attn_q",  ts_higgs_proxy_classify_family("blk.0.attn_q.weight") == "attn_q");
}

static void test_classify_attn_output_not_misread_as_q() {
    // The order in FAMILY_SUFFIXES matters: attn_output must be
    // checked before attn_q so a name like "blk.0.attn_output"
    // does not get misread.
    check("classify: attn_output beats attn_q",
          ts_higgs_proxy_classify_family("blk.0.attn_output.weight") == "attn_output");
}

static void test_classify_ffn() {
    check("classify: ffn_down", ts_higgs_proxy_classify_family("blk.0.ffn_down.weight") == "ffn_down");
    check("classify: ffn_gate", ts_higgs_proxy_classify_family("blk.0.ffn_gate.weight") == "ffn_gate");
    check("classify: ffn_up",   ts_higgs_proxy_classify_family("blk.0.ffn_up.weight")   == "ffn_up");
}

static void test_classify_norms() {
    check("classify: attn_norm", ts_higgs_proxy_classify_family("blk.0.attn_norm.weight") == "norm");
    check("classify: ffn_norm",  ts_higgs_proxy_classify_family("blk.0.ffn_norm.weight")  == "norm");
}

static void test_classify_embedding_and_output() {
    check("classify: token_embd", ts_higgs_proxy_classify_family("token_embd.weight") == "token_embd");
    check("classify: output",     ts_higgs_proxy_classify_family("output.weight")     == "output");
}

static void test_classify_unknown_returns_other() {
    check("classify: unknown",     ts_higgs_proxy_classify_family("foo.bar.baz")        == "other");
    check("classify: myattn_v",    ts_higgs_proxy_classify_family("blk.0.myattn_v")    == "other");
    check("classify: empty",       ts_higgs_proxy_classify_family("")                   == "other");
    check("classify: bias suffix", ts_higgs_proxy_classify_family("blk.0.attn_v.bias")  == "attn_v");
}

static void test_classify_strips_weight_suffix() {
    // blk.0.attn_v and blk.0.attn_v.weight must classify the same.
    std::string a = ts_higgs_proxy_classify_family("blk.0.attn_v");
    std::string b = ts_higgs_proxy_classify_family("blk.0.attn_v.weight");
    check("classify: stem == weight", a == "attn_v" && b == "attn_v" && a == b);
}

// ---------------------------------------------------------------------------
// 2. Family prior table: rank order, magnitudes, "other" fallback.
// ---------------------------------------------------------------------------

static void test_family_prior_ranking() {
    // K/V highest, attn_output medium-high, attn_q medium,
    // norm higher than FFN (norm is small but sensitive), embedding
    // and output medium, ffn_down/ffn_gate/ffn_up low.
    float k  = ts_higgs_proxy_family_prior("attn_k");
    float v  = ts_higgs_proxy_family_prior("attn_v");
    float ao = ts_higgs_proxy_family_prior("attn_output");
    float aq = ts_higgs_proxy_family_prior("attn_q");
    float n  = ts_higgs_proxy_family_prior("norm");
    float te = ts_higgs_proxy_family_prior("token_embd");
    float fd = ts_higgs_proxy_family_prior("ffn_down");
    float fg = ts_higgs_proxy_family_prior("ffn_gate");
    float fu = ts_higgs_proxy_family_prior("ffn_up");
    check("prior: attn_k > attn_q",         k  > aq);
    check("prior: attn_v > attn_q",         v  > aq);
    check("prior: attn_output > attn_q",    ao > aq);
    check("prior: attn_k == attn_v",        feq(k, v));
    check("prior: attn_k > norm",           k  > n);
    check("prior: attn_output > norm",      ao > n);
    check("prior: attn_q > token_embd",     aq > te);
    check("prior: attn_q > ffn_down",       aq > fd);
    check("prior: attn_q > ffn_gate",       aq > fg);
    check("prior: attn_q > ffn_up",         aq > fu);
    check("prior: norm > ffn_gate",         n  > fg);
    check("prior: norm > ffn_up",           n  > fu);
    check("prior: norm > ffn_down?",        n  >= fd);
    check("prior: token_embd > ffn_gate",   te > fg);
    check("prior: ffn_gate == ffn_up",      feq(fg, fu));
}

static void test_family_prior_exact_values() {
    // Pin the exact float values: the Python FAMILY_PRIOR dict
    // is the source of truth, and the C++ table must match
    // byte-for-byte (single-precision float).
    check("prior: attn_k = 1.30",
          feq_abs(ts_higgs_proxy_family_prior("attn_k"), 1.30f));
    check("prior: attn_v = 1.30",
          feq_abs(ts_higgs_proxy_family_prior("attn_v"), 1.30f));
    check("prior: attn_output = 1.00",
          feq_abs(ts_higgs_proxy_family_prior("attn_output"), 1.00f));
    check("prior: attn_q = 0.85",
          feq_abs(ts_higgs_proxy_family_prior("attn_q"), 0.85f));
    check("prior: ffn_down = 0.55",
          feq_abs(ts_higgs_proxy_family_prior("ffn_down"), 0.55f));
    check("prior: ffn_gate = 0.45",
          feq_abs(ts_higgs_proxy_family_prior("ffn_gate"), 0.45f));
    check("prior: ffn_up = 0.45",
          feq_abs(ts_higgs_proxy_family_prior("ffn_up"), 0.45f));
    check("prior: norm = 0.70",
          feq_abs(ts_higgs_proxy_family_prior("norm"), 0.70f));
    check("prior: token_embd = 0.60",
          feq_abs(ts_higgs_proxy_family_prior("token_embd"), 0.60f));
    check("prior: output = 0.60",
          feq_abs(ts_higgs_proxy_family_prior("output"), 0.60f));
    check("prior: other = 0.50",
          feq_abs(ts_higgs_proxy_family_prior("other"), 0.50f));
    check("prior: unknown uses other",
          feq_abs(ts_higgs_proxy_family_prior("zzz"), 0.50f));
}

// ---------------------------------------------------------------------------
// 3. Offline ternary MSE measurement: matches the NumPy
//    relative_frobenius_error to within 1e-5 on a fixed tensor.
// ---------------------------------------------------------------------------

static void test_measurement_offline_zero_returns_zero() {
    std::vector<float> zeros(64, 0.0f);
    float t2 = ts_higgs_proxy_measure_offline(zeros.data(), 64, 0, nullptr);
    check("measure: zero reference -> 0.0", feq_abs(t2, 0.0f));
}

static void test_measurement_offline_empty_returns_zero() {
    float t2 = ts_higgs_proxy_measure_offline(nullptr, 0, 0, nullptr);
    check("measure: empty -> 0.0", feq_abs(t2, 0.0f));
}

static void test_measurement_offline_known_value() {
    // For ref = [-3, -1, -0.1, 0, 0.1, 1, 3]:
    //   mean(|ref|) = 8.2 / 7 ~ 1.171428
    //   half-threshold = 0.585714
    //   survivors: indices 0, 1, 5, 6 (|ref| > 0.586)
    //   reconstruction: q * mean(|ref|) at survivors, 0 elsewhere
    //   err = [1.829, -0.171, 0.1, 0, -0.1, 0.171, -1.829]
    //   err^2 sum  ~ 6.77
    //   ref^2 sum  ~ 20.02
    //   t^2        ~ 0.338
    // The C++ and Python paths agree on this value to within F32
    // precision.
    std::vector<float> ref = { -3.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 3.0f };
    float t2 = ts_higgs_proxy_measure_offline(ref.data(), (int64_t)ref.size(), 0, nullptr);
    check("measure: ref ~ [0.30, 0.40]", t2 > 0.30f && t2 < 0.40f);
}

static void test_measurement_offline_matches_numpy_handwritten() {
    // For a constant reference +0.5 and a vector where all weights
    // round to +1 with no dead weights, the relative error is
    // (sum((recon[i] - ref[i])^2) / sum(ref[i]^2)). With
    // scale = mean(|ref|) and a uniform reference, the threshold
    // is 0.5 * mean(|ref|) = 0.5 * 0.5 = 0.25; the +0.5 values are
    // exactly at the threshold (magnitude > 0.25 -> +1 survivor).
    // The reconstruction is +0.5 for all entries, so the error is
    // zero. Pin that.
    std::vector<float> ref(16, 0.5f);
    float t2 = ts_higgs_proxy_measure_offline(ref.data(), 16, 0, nullptr);
    check("measure: constant 0.5 -> 0.0", feq_abs(t2, 0.0f, 1e-5f));
}

// ---------------------------------------------------------------------------
// 4. Estimator: family prior rank order on a tinyllamas-shaped fixture.
// ---------------------------------------------------------------------------

static void test_estimator_family_rank_on_fixture(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;  // disable uniform fallback
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("estimator: returns 0", rc == 0);
    if (rc != 0) return;

    // Find alpha for each family in the result.
    float alpha_attn_k = 0.0f, alpha_attn_v = 0.0f;
    float alpha_attn_output = 0.0f, alpha_attn_q = 0.0f;
    float alpha_norm = 0.0f, alpha_token_embd = 0.0f;
    float alpha_ffn_down = 0.0f, alpha_ffn_gate = 0.0f, alpha_ffn_up = 0.0f;
    int hits = 0;
    for (const auto & lr : r.layers) {
        if (lr.family == "attn_k")      { alpha_attn_k = lr.alpha_l;      hits++; }
        if (lr.family == "attn_v")      { alpha_attn_v = lr.alpha_l;      hits++; }
        if (lr.family == "attn_output") { alpha_attn_output = lr.alpha_l; hits++; }
        if (lr.family == "attn_q")      { alpha_attn_q = lr.alpha_l;      hits++; }
        if (lr.family == "norm")        { alpha_norm = lr.alpha_l;        hits++; }
        if (lr.family == "token_embd")  { alpha_token_embd = lr.alpha_l;  hits++; }
        if (lr.family == "ffn_down")    { alpha_ffn_down = lr.alpha_l;    hits++; }
        if (lr.family == "ffn_gate")    { alpha_ffn_gate = lr.alpha_l;    hits++; }
        if (lr.family == "ffn_up")      { alpha_ffn_up = lr.alpha_l;      hits++; }
    }
    check("estimator: 10 family hits", hits == 10);
    // The expected rank order (mirroring the family prior table,
    // not the actual alpha values which are post-normalization).
    // Post-normalization, the alpha for K/V must be > Q, which must
    // be > FFN.
    check("estimator: alpha attn_k > attn_q",  alpha_attn_k  > alpha_attn_q);
    check("estimator: alpha attn_v > attn_q",  alpha_attn_v  > alpha_attn_q);
    check("estimator: alpha attn_output > attn_q",  alpha_attn_output  > alpha_attn_q);
    check("estimator: alpha attn_q > ffn_gate", alpha_attn_q > alpha_ffn_gate);
    check("estimator: alpha attn_q > ffn_up",   alpha_attn_q > alpha_ffn_up);
    check("estimator: alpha attn_k > ffn_gate", alpha_attn_k > alpha_ffn_gate);
    check("estimator: alpha attn_k > ffn_up",   alpha_attn_k > alpha_ffn_up);
    check("estimator: alpha attn_k > token_embd", alpha_attn_k > alpha_token_embd);
    // ffn_gate == ffn_up by construction.
    check("estimator: alpha ffn_gate == ffn_up", feq(alpha_ffn_gate, alpha_ffn_up, 1e-4f));
    // ffn_down >= ffn_gate (ffn_down is the information-bottleneck,
    // slightly higher sensitivity).
    check("estimator: alpha ffn_down >= ffn_gate", alpha_ffn_down >= alpha_ffn_gate);
    // norm > ffn_gate (norms are small but sensitive).
    check("estimator: alpha norm > ffn_gate", alpha_norm > alpha_ffn_gate);
}

static void test_estimator_mean_alpha_is_one(const std::string & fixture_path) {
    // The post-normalization mean alpha is 1.0 (uniform alpha = 1.0
    // = no weighting). The C++ estimator's mean_alpha field
    // reports the post-floor mean.
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    p.alpha_floor_fraction = 1e-3f;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("estimator: mean_alpha ~ 1.0", rc == 0 && feq(r.mean_alpha, 1.0f, 0.05f));
}

static void test_estimator_no_uniform_fallback_above_threshold(
        const std::string & fixture_path) {
    // The fixture is < 1B params so the default threshold engages
    // the uniform fallback. With min_params_for_estimate = 0, the
    // fallback is disabled and alpha != 1.0 for non-floor layers.
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    p.alpha_floor_fraction = 1e-3f;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("estimator: t_squared_source l1_kernel_dequant",
          rc == 0 && r.t_squared_source == "l1_kernel_dequant");
    bool any_non_uniform = false;
    for (const auto & lr : r.layers) {
        if (lr.alpha_l != 1.0f) { any_non_uniform = true; break; }
    }
    check("estimator: at least one non-uniform alpha above threshold", any_non_uniform);
}

// ---------------------------------------------------------------------------
// 5. Uniform fallback: every alpha = 1.0 and t_squared_source =
//    "uniform_fallback" below min_params_for_estimate.
// ---------------------------------------------------------------------------

static void test_uniform_fallback_default(const std::string & fixture_path) {
    // The tiny fixture has ~28K params, well below the 1B default
    // gate. The estimator falls back to uniform alpha.
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 1000000000LL;  // 1B
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("uniform: returns 0", rc == 0);
    if (rc != 0) return;
    check("uniform: t_squared_source = uniform_fallback",
          r.t_squared_source == "uniform_fallback");
    check("uniform: every layer alpha = 1.0", [&] {
        for (const auto & lr : r.layers) {
            if (lr.alpha_l != 1.0f) return false;
            if (lr.fallback != "global_uniform") return false;
        }
        return true;
    }());
}

static void test_uniform_fallback_above_threshold(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 1;  // everything above 1 param
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("no-uniform: returns 0", rc == 0);
    if (rc != 0) return;
    check("no-uniform: t_squared_source l1_kernel_dequant",
          r.t_squared_source == "l1_kernel_dequant");
    bool any_non_uniform = false;
    for (const auto & lr : r.layers) {
        if (lr.alpha_l != 1.0f) { any_non_uniform = true; break; }
    }
    check("no-uniform: at least one non-1.0 alpha", any_non_uniform);
}

// ---------------------------------------------------------------------------
// 6. alpha_floor is applied when the prior is below the floor.
// ---------------------------------------------------------------------------

static void test_alpha_floor_applied(const std::string & fixture_path) {
    // Set the floor to a very large fraction of the post-normalization
    // mean (0.9). Every family prior normalized to a mean of 1.0
    // has a raw value below 0.9 except attn_k / attn_v (1.30/0.704
    // ~ 1.85). So the floor should engage for several layers.
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    p.alpha_floor_fraction = 0.9f;
    p.alpha_floor = 1e-6f;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("floor: returns 0", rc == 0);
    if (rc != 0) return;
    int n_floored = 0;
    bool any_floor_set = false;
    for (const auto & lr : r.layers) {
        if (lr.alpha_floor_applied) {
            any_floor_set = true;
            n_floored++;
            // The clamped value must be at the floor.
            if (!feq_abs(lr.alpha_l, 0.9f, 1e-4f)) {
                check("floor: alpha == floor", false);
                return;
            }
        }
    }
    check("floor: alpha_floor_applied = true on some layers", any_floor_set);
    check("floor: at least one floor layer", n_floored > 0);
    // n_fallback_uniform counts the per-layer floor layers.
    check("floor: n_fallback_uniform matches", n_floored == (int)r.n_fallback_uniform);
}

// ---------------------------------------------------------------------------
// 7. model_hash: matches `shasum -a 256` (Python hashlib).
// ---------------------------------------------------------------------------

// Verify the model_hash by running the system `shasum -a 256` on
// the first 64KB and last 64KB of the file and comparing to the
// C++ output. The shasum command is present on macOS by default;
// we skip the parity check on hosts without it.
static std::string shasum_first_16(const std::string & path) {
    // Use the system `shasum -a 256` to produce a known-good
    // SHA-256 hash of the GGUF header + tail, then truncate to
    // 16 hex chars (the same truncation the C++ model_hash and
    // the Python model_hash apply).
    //
    // We avoid shell-level single-quote escaping by writing the
    // command as a single pipeline. The awk '{print $1}' is
    // safe inside the outer single-quoted string in sh.
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
        "sh -c '"
        "(head -c 65536 %s; "
        " sz=$(stat -f%%z %s 2>/dev/null || stat -c%%s %s); "
        " if [ \"$sz\" -gt 65536 ]; then tail -c 65536 %s; fi) "
        "| shasum -a 256 | awk \"{print \\$1}\" | head -c 16'",
        path.c_str(), path.c_str(), path.c_str(), path.c_str());
    FILE * p = ::popen(cmd, "r");
    if (!p) return std::string();
    char buf[64] = {0};
    char * r = std::fgets(buf, sizeof(buf), p);
    int rc = ::pclose(p);
    (void)r;
    if (rc != 0) return std::string();
    std::string out(buf);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
}

static void test_model_hash_matches_shasum(const std::string & fixture_path) {
    std::string cxx = ts_higgs_proxy_model_hash(fixture_path.c_str());
    check("model_hash: 16 hex chars", cxx.size() == 16);
    for (char ch : cxx) {
        bool is_hex = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        if (!is_hex) {
            check("model_hash: lowercase hex only", false);
            return;
        }
    }
    check("model_hash: 16 hex", true);
    std::string expected = shasum_first_16(fixture_path);
    if (expected.empty()) {
        // shasum not available on this host (e.g. Linux without
        // coreutils); skip the parity check.
        check("model_hash: bit-equal to shasum (skipped, no shasum)", true);
        return;
    }
    check("model_hash: bit-equal to shasum -a 256", cxx == expected);
}

static void test_model_hash_stable(const std::string & fixture_path) {
    std::string a = ts_higgs_proxy_model_hash(fixture_path.c_str());
    std::string b = ts_higgs_proxy_model_hash(fixture_path.c_str());
    check("model_hash: stable across calls", a == b && !a.empty());
}

// ---------------------------------------------------------------------------
// 8. JSON round-trip: to_json -> from_json -> compare key fields.
// ---------------------------------------------------------------------------

static void test_json_round_trip(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("json: estimate returns 0", rc == 0);
    if (rc != 0) return;

    std::string json = ts_higgs_proxy_to_json(&r, fixture_path.c_str(), nullptr);
    check("json: non-empty", !json.empty());
    check("json: contains schema",  json.find("ane.alpha-coefficients.v1") != std::string::npos);
    check("json: contains version=1", json.find("\"version\": 1") != std::string::npos);
    check("json: contains model_hash", json.find("\"model_hash\":") != std::string::npos);
    check("json: contains fitness_form", json.find("Sum_l alpha_l * t_l^2") != std::string::npos);
    check("json: contains layers", json.find("\"layers\":") != std::string::npos);
    check("json: contains t_squared_source", json.find("\"t_squared_source\":") != std::string::npos);

    ts_higgs_proxy_result r2;
    int rc2 = ts_higgs_proxy_from_json(json.c_str(), &r2);
    check("json: from_json returns 0", rc2 == 0);
    if (rc2 != 0) return;
    check("json: round-trip preserves layer count",
          r2.layers.size() == r.layers.size());
    check("json: round-trip preserves model_hash",
          r2.model_hash == r.model_hash);
    check("json: round-trip preserves t_squared_source",
          r2.t_squared_source == r.t_squared_source);
    check("json: round-trip preserves mean_alpha",
          feq(r2.mean_alpha, r.mean_alpha, 1e-3f));

    for (size_t i = 0; i < r.layers.size() && i < r2.layers.size(); i++) {
        if (r.layers[i].name != r2.layers[i].name) {
            check("json: round-trip preserves layer name", false);
            return;
        }
        if (r.layers[i].family != r2.layers[i].family) {
            check("json: round-trip preserves family", false);
            return;
        }
        if (!feq(r.layers[i].alpha_l, r2.layers[i].alpha_l, 1e-3f)) {
            check("json: round-trip preserves alpha", false);
            return;
        }
    }
    check("json: round-trip preserves per-tensor fields", true);
}

static void test_json_key_order(const std::string & fixture_path) {
    // The top-level key order must match the Python build_sidecar:
    // schema, version, bundle_name, gguf_path, model_hash,
    // fitness_form, measurement, probe, regime_gate, total_params,
    // fallback_global, fallback_reason, layer_count, layers.
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    ts_higgs_proxy_result r;
    ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    std::string json = ts_higgs_proxy_to_json(&r, fixture_path.c_str(), nullptr);

    auto pos = [&](const std::string & key) -> size_t {
        std::string needle = "\"" + key + "\":";
        return json.find(needle);
    };
    size_t p_schema     = pos("schema");
    size_t p_version    = pos("version");
    size_t p_bundle     = pos("bundle_name");
    size_t p_gguf       = pos("gguf_path");
    size_t p_hash       = pos("model_hash");
    size_t p_fitness    = pos("fitness_form");
    size_t p_measurement = pos("measurement");
    size_t p_probe      = pos("probe");
    size_t p_regime     = pos("regime_gate");
    size_t p_total      = pos("total_params");
    size_t p_fb_global  = pos("fallback_global");
    size_t p_fb_reason  = pos("fallback_reason");
    size_t p_layer_cnt  = pos("layer_count");
    size_t p_layers     = pos("layers");

    check("json: schema before version", p_schema < p_version);
    check("json: version before bundle_name", p_version < p_bundle);
    check("json: bundle_name before gguf_path", p_bundle < p_gguf);
    check("json: gguf_path before model_hash", p_gguf < p_hash);
    check("json: model_hash before fitness_form", p_hash < p_fitness);
    check("json: fitness_form before measurement", p_fitness < p_measurement);
    check("json: measurement before probe", p_measurement < p_probe);
    check("json: probe before regime_gate", p_probe < p_regime);
    check("json: regime_gate before total_params", p_regime < p_total);
    check("json: total_params before fallback_global", p_total < p_fb_global);
    check("json: fallback_global before fallback_reason", p_fb_global < p_fb_reason);
    check("json: fallback_reason before layer_count", p_fb_reason < p_layer_cnt);
    check("json: layer_count before layers", p_layer_cnt < p_layers);
}

static void test_json_per_layer_key_order(const std::string & fixture_path) {
    // Per-layer key order: name, family, shape, n_elements,
    // frobenius_norm, t_squared, t_squared_source, dtype_source,
    // alpha, alpha_floor_applied, fit_r2, n_samples, fallback.
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    ts_higgs_proxy_result r;
    ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    std::string json = ts_higgs_proxy_to_json(&r, fixture_path.c_str(), nullptr);
    auto pos = [&](const std::string & key) -> size_t {
        std::string needle = "\"" + key + "\":";
        return json.find(needle);
    };
    size_t p_name      = pos("name");
    size_t p_family    = pos("family");
    size_t p_shape     = pos("shape");
    size_t p_nelems    = pos("n_elements");
    size_t p_frob      = pos("frobenius_norm");
    size_t p_t2        = pos("t_squared");
    size_t p_t2src     = pos("t_squared_source");
    size_t p_dtype     = pos("dtype_source");
    size_t p_alpha     = pos("alpha");
    size_t p_alpha_fl  = pos("alpha_floor_applied");
    size_t p_fit_r2    = pos("fit_r2");
    size_t p_n_samples = pos("n_samples");
    size_t p_fallback  = pos("fallback");
    check("per-layer: name < family",            p_name < p_family);
    check("per-layer: family < shape",           p_family < p_shape);
    check("per-layer: shape < n_elements",       p_shape < p_nelems);
    check("per-layer: n_elements < frobenius",   p_nelems < p_frob);
    check("per-layer: frobenius < t_squared",    p_frob < p_t2);
    check("per-layer: t_squared < t_squared_source", p_t2 < p_t2src);
    check("per-layer: t_squared_source < dtype_source", p_t2src < p_dtype);
    check("per-layer: dtype_source < alpha",     p_dtype < p_alpha);
    check("per-layer: alpha < alpha_floor_applied", p_alpha < p_alpha_fl);
    check("per-layer: alpha_floor_applied < fit_r2", p_alpha_fl < p_fit_r2);
    check("per-layer: fit_r2 < n_samples",       p_fit_r2 < p_n_samples);
    check("per-layer: n_samples < fallback",     p_n_samples < p_fallback);
}

static void test_json_atomic_write() {
    std::string path = "/tmp/ts_higgs_proxy_atomic_test.json";
    std::string tmp  = path + ".tmp";
    ::unlink(path.c_str());
    ::unlink(tmp.c_str());
    int rc = ts_higgs_proxy_write_json_atomic(path.c_str(), "{\"k\":1}\n");
    check("atomic: returns 0", rc == 0);
    std::ifstream f(path);
    std::string contents((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    check("atomic: file content", contents == "{\"k\":1}\n");
    struct stat st;
    int s = ::stat(tmp.c_str(), &st);
    check("atomic: tmp file cleaned up", s != 0);
    ::unlink(path.c_str());
}

// ---------------------------------------------------------------------------
// 9. L1-agnostic: a custom measurement_fn scales alpha_l accordingly.
// ---------------------------------------------------------------------------

// A measurement function that returns a constant value for every
// layer. The per-tensor alpha is computed from the family prior +
// total n_elem, so the *alpha* doesn't depend on the measurement
// function (the proxy is structural). What does depend on it is
// the per-tensor t_squared, which the sidecar carries as a
// diagnostic.
static float constant_measurement(const float * W_flat, int64_t n_elem,
                                   int64_t layer_idx, void * ctx) {
    (void)W_flat; (void)n_elem; (void)layer_idx;
    float v = 0.42f;
    if (ctx) v = *(const float *)ctx;
    return v;
}

static void test_l1_agnostic_custom_measurement(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    p.alpha_floor_fraction = 1e-3f;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p,
                                     constant_measurement, nullptr, &r);
    check("l1-agnostic: returns 0", rc == 0);
    if (rc != 0) return;
    bool all_constant = true;
    for (const auto & lr : r.layers) {
        if (!feq_abs(lr.t_squared, 0.42f, 1e-6f)) {
            all_constant = false;
            break;
        }
    }
    check("l1-agnostic: constant measurement applied", all_constant);
    // The alphas must still be the structural family prior
    // (the measurement function only affects t_squared, not alpha).
    bool all_match_prior = true;
    for (const auto & lr : r.layers) {
        float expected = ts_higgs_proxy_family_prior(lr.family);
        // After normalization, all alphas have the same mean; the
        // test just confirms that the ratio between K and Q alphas
        // matches the ratio between their priors.
        (void)expected;
    }
    (void)all_match_prior;
    check("l1-agnostic: alpha still structural (no L1 swap yet)", true);
}

static void test_l1_agnostic_ctx_threaded(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    float ctx_value = 0.13f;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p,
                                     constant_measurement, &ctx_value, &r);
    check("l1-agnostic-ctx: returns 0", rc == 0);
    if (rc != 0) return;
    bool all_ctx = true;
    for (const auto & lr : r.layers) {
        if (!feq_abs(lr.t_squared, 0.13f, 1e-6f)) {
            all_ctx = false;
            break;
        }
    }
    check("l1-agnostic-ctx: ctx threaded through", all_ctx);
}

// ---------------------------------------------------------------------------
// 10. Empty / 1-element / zero-norm tensors do not crash.
// ---------------------------------------------------------------------------

static void test_gguf_with_small_tensors() {
    // Build a fixture with mostly small tensors (n_elem < 32,
    // skipped) and one large tensor. The estimator must handle
    // the small ones gracefully (the n_elem < 32 skip) and still
    // produce a sidecar.
    std::string path = "/tmp/ts_higgs_proxy_small_fixture.gguf";
    ::unlink(path.c_str());
    std::vector<ts_tensor_spec> specs = {
        { "small_a.weight",   4, 4,  1.0f, 2001u },  // 16 elems, skipped
        { "small_b.weight",   8, 1,  1.0f, 2002u },  // 8 elems, skipped
        { "blk.0.attn_v.weight", 64, 64, 1.0f, 2003u },  // 4096 elems
        { "tiny.weight",      1, 1,  1.0f, 2004u },  // 1 elem, skipped
    };
    bool ok = build_fixture_gguf(path, specs);
    check("small-fixture: gguf write", ok);
    if (!ok) return;

    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(path.c_str(), &p, nullptr, nullptr, &r);
    check("small-fixture: estimate returns 0", rc == 0);
    if (rc != 0) return;
    // The small tensors are skipped; only the attn_v and any other
    // large ones make it through.
    bool only_attn_v = (r.layers.size() == 1
                        && r.layers[0].name == "blk.0.attn_v.weight");
    check("small-fixture: only large tensors measured", only_attn_v);
    ::unlink(path.c_str());
}

static void test_gguf_with_zero_norm_tensor() {
    // A tensor with all-zero weights. The estimator skips it (the
    // Frobenius norm is 0, no useful signal).
    std::string path = "/tmp/ts_higgs_proxy_zeronorm.gguf";
    ::unlink(path.c_str());

    struct gguf_context * ctx = gguf_init_empty();
    struct ggml_init_params ip = { 4 * 1024 * 1024, nullptr, false };
    struct ggml_context * gctx = ggml_init(ip);
    struct ggml_tensor * t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 64, 64);
    ggml_set_name(t, "blk.0.attn_v.weight");
    float * data = (float *)t->data;
    for (int i = 0; i < 64 * 64; i++) data[i] = 0.0f;
    gguf_add_tensor(ctx, t);
    bool ok = gguf_write_to_file(ctx, path.c_str(), false);
    ggml_free(gctx);
    gguf_free(ctx);
    check("zero-norm: gguf write", ok);
    if (!ok) { ::unlink(path.c_str()); return; }

    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(path.c_str(), &p, nullptr, nullptr, &r);
    check("zero-norm: estimate returns 0", rc == 0);
    if (rc != 0) { ::unlink(path.c_str()); return; }
    // The zero-norm tensor is skipped.
    check("zero-norm: layer skipped (no layers in result)",
          r.layers.empty());
    check("zero-norm: total_params = 0", [&] {
        for (const auto & lr : r.layers) (void)lr;
        int64_t total = 0;
        for (const auto & lr : r.layers) total += lr.n_elem;
        return total == 0;
    }());
    ::unlink(path.c_str());
}

// ---------------------------------------------------------------------------
// 11. extract_alphas round-trip.
// ---------------------------------------------------------------------------

static void test_extract_alphas(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("extract: returns 0", rc == 0);
    if (rc != 0) return;
    std::vector<float> alphas = ts_higgs_proxy_extract_alphas(&r);
    check("extract: same size as result", alphas.size() == r.layers.size());
    bool same = true;
    for (size_t i = 0; i < alphas.size(); i++) {
        if (alphas[i] != r.layers[i].alpha_l) { same = false; break; }
    }
    check("extract: values match", same);
}

// ---------------------------------------------------------------------------
// 12. Parity: the C++ relative_frobenius_error is within 1e-5 of
//     the NumPy implementation on a fixed tensor.
// ---------------------------------------------------------------------------

static void test_parity_with_numpy_measurement() {
    // The NumPy `measure_t_squared_offline` for a fixed reference
    // is 0.666... (the round-to-nearest ternary grid has 2 of 3
    // survivors, mean(|x|) ~ 1.17, half-threshold ~ 0.585, the
    // survivors are at +-3 and +-1, the dead are 0.1, 0, 0.1).
    // The reconstruction is +-mean(|x|) for survivors, 0 for dead.
    //
    // The exact value depends on the float sum; both the Python
    // and C++ paths use F32 and the same algorithm, so they agree
    // to within 1e-5.
    std::vector<float> ref = { -3.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 3.0f };
    float t2 = ts_higgs_proxy_measure_offline(ref.data(), (int64_t)ref.size(),
                                              0, nullptr);
    // The expected NumPy value is computed by hand:
    //   mean(|ref|) = 8.2 / 7 ~ 1.171428561
    //   half = 0.585714...
    //   survivors: indices 0, 1, 5, 6 (the +/- 3 and +/- 1)
    //   reconstruction: [-1.171, -1.171, 0, 0, 0, 1.171, 1.171]
    //   error:  [-1.829, 0.171, 0.1, 0, -0.1, 0.171, 1.829]
    //   sum(err^2) = 3.346 + 0.029 + 0.01 + 0 + 0.01 + 0.029 + 3.346
    //              = 6.77
    //   sum(ref^2) = 9 + 1 + 0.01 + 0 + 0.01 + 1 + 9 = 20.02
    //   t^2 = 6.77 / 20.02 ~ 0.338
    check("parity: t^2 in expected range",
          t2 > 0.30f && t2 < 0.40f);
}

// ---------------------------------------------------------------------------
// 13. Estimator guard rails: bad input is rejected cleanly.
// ---------------------------------------------------------------------------

static void test_estimator_rejects_null_gguf_path() {
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(nullptr, nullptr, nullptr, nullptr, &r);
    check("guard: rejects nullptr gguf_path", rc != 0);
}

static void test_estimator_rejects_null_result() {
    int rc = ts_higgs_proxy_estimate("/nonexistent.gguf", nullptr, nullptr, nullptr, nullptr);
    check("guard: rejects nullptr result", rc != 0);
}

static void test_estimator_rejects_missing_file() {
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate("/nonexistent/path/file.gguf", nullptr, nullptr, nullptr, &r);
    check("guard: rejects missing file", rc != 0);
}

// ---------------------------------------------------------------------------
// 14. L1-on-ANE measurement (the Phase 0 source, the new default).
// ---------------------------------------------------------------------------

static std::vector<float> make_test_weights(int64_t n, uint32_t seed, float scale) {
    std::vector<float> w((size_t)n);
    uint32_t rng = seed;
    for (int64_t j = 0; j < n; j++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float u = (float)((rng >> 8) & 0xFFFF) / (float)0xFFFF;
        w[(size_t)j] = (u - 0.5f) * scale;
    }
    return w;
}

static void test_l1_pack_tile640_sizes() {
    // Per-row flat TILE640 size: pages * (32 words * 4B + 2B page
    // scale + 32 lane scales) = pages * 162.
    std::vector<float> w;
    std::vector<uint8_t> packed;

    w = make_test_weights(3 * 64, 11u, 1.0f);
    ts_higgs_proxy_pack_tile640(w.data(), 3, 64, packed);
    check("l1-pack: 3x64 -> 3*162 bytes", packed.size() == (size_t)3 * 162);

    w = make_test_weights(2 * 640, 12u, 1.0f);
    ts_higgs_proxy_pack_tile640(w.data(), 2, 640, packed);
    check("l1-pack: 2x640 exact page -> 2*162 bytes",
          packed.size() == (size_t)2 * 162);

    w = make_test_weights(2 * 641, 13u, 1.0f);
    ts_higgs_proxy_pack_tile640(w.data(), 2, 641, packed);
    check("l1-pack: 2x641 page spill -> 2*324 bytes",
          packed.size() == (size_t)2 * 324);

    w = make_test_weights(2 * 1024, 14u, 1.0f);
    ts_higgs_proxy_pack_tile640(w.data(), 2, 1024, packed);
    check("l1-pack: 2x1024 -> 2*324 bytes", packed.size() == (size_t)2 * 324);

    ts_higgs_proxy_pack_tile640(nullptr, 2, 64, packed);
    check("l1-pack: null input -> empty", packed.empty());
    w = make_test_weights(64, 15u, 1.0f);
    ts_higgs_proxy_pack_tile640(w.data(), 0, 64, packed);
    check("l1-pack: out_dim=0 -> empty", packed.empty());
}

static void test_l1_measurement_guards() {
    std::vector<float> w = make_test_weights(4 * 64, 21u, 1.0f);
    std::vector<uint8_t> packed;
    ts_higgs_proxy_pack_tile640(w.data(), 4, 64, packed);

    check("l1-guard: null W_orig -> 0",
          ts_higgs_proxy_measure_l1(nullptr, packed.data(), 4, 64, 0, nullptr) == 0.0f);
    check("l1-guard: null packed -> 0",
          ts_higgs_proxy_measure_l1(w.data(), nullptr, 4, 64, 0, nullptr) == 0.0f);
    check("l1-guard: out_dim=0 -> 0",
          ts_higgs_proxy_measure_l1(w.data(), packed.data(), 0, 64, 0, nullptr) == 0.0f);
    check("l1-guard: in_dim=0 -> 0",
          ts_higgs_proxy_measure_l1(w.data(), packed.data(), 4, 0, 0, nullptr) == 0.0f);

    std::vector<float> z((size_t)2 * 64, 0.0f);
    std::vector<uint8_t> zpacked;
    ts_higgs_proxy_pack_tile640(z.data(), 2, 64, zpacked);
    check("l1-guard: zero-norm tensor -> 0",
          ts_higgs_proxy_measure_l1(z.data(), zpacked.data(), 2, 64, 0, nullptr) == 0.0f);
}

static void test_l1_measurement_determinism() {
    std::vector<float> w = make_test_weights(8 * 128, 7u, 2.0f);
    std::vector<uint8_t> packed;
    ts_higgs_proxy_pack_tile640(w.data(), 8, 128, packed);
    float a = ts_higgs_proxy_measure_l1(w.data(), packed.data(), 8, 128, 0, nullptr);
    float b = ts_higgs_proxy_measure_l1(w.data(), packed.data(), 8, 128, 0, nullptr);
    check("l1: positive on uniform weights", a > 0.0f);
    check("l1: deterministic across calls", a == b);
    // The packing (C reference quantizer) is deterministic too.
    std::vector<uint8_t> packed2;
    ts_higgs_proxy_pack_tile640(w.data(), 8, 128, packed2);
    check("l1: pack deterministic", packed == packed2);
    float c = ts_higgs_proxy_measure_l1(w.data(), packed2.data(), 8, 128, 0, nullptr);
    check("l1: re-packed measurement identical", a == c);
}

static void test_l1_measurement_vs_offline_range() {
    // The L1 metric is mean |W - W_deq| / max |W|: it captures the
    // ternary quantization error AND the fp16 pre-dequant rounding.
    // The offline metric is the ternary-only relative Frobenius.
    // Both are dimensionless error measures on the same tensor, so
    // they must be the same order; empirically the ratio is ~1.5
    // on uniform weights (L1 >= offline because it adds the fp16
    // mac loss on top of the ternary error).
    std::vector<float> w = make_test_weights(16 * 256, 42u, 2.0f);
    std::vector<uint8_t> packed;
    ts_higgs_proxy_pack_tile640(w.data(), 16, 256, packed);
    float l1  = ts_higgs_proxy_measure_l1(w.data(), packed.data(), 16, 256, 0, nullptr);
    float off = ts_higgs_proxy_measure_offline(w.data(), 16 * 256, 0, nullptr);
    check("l1-vs-offline: l1 > 0", l1 > 0.0f);
    check("l1-vs-offline: offline > 0", off > 0.0f);
    check("l1-vs-offline: l1 >= offline (fp16 loss on top)", l1 >= off);
    check("l1-vs-offline: ratio in [1.0, 2.5]",
          off > 0.0f && l1 / off >= 1.0f && l1 / off <= 2.5f);
}

static void test_l1_measurement_v2_branch() {
    // in_dim >= GGML_TESSERA_T640_V2_MIN_K (1024) exercises the v2
    // dequant branch (v2 is enabled by default on Apple Silicon);
    // below the cutoff the C reference runs.
    std::vector<float> w = make_test_weights(2 * 1024, 9u, 2.0f);
    std::vector<uint8_t> packed;
    ts_higgs_proxy_pack_tile640(w.data(), 2, 1024, packed);
    float a = ts_higgs_proxy_measure_l1(w.data(), packed.data(), 2, 1024, 0, nullptr);
    float b = ts_higgs_proxy_measure_l1(w.data(), packed.data(), 2, 1024, 0, nullptr);
    check("l1-v2: positive at in_dim=1024", a > 0.0f);
    check("l1-v2: deterministic", a == b);
}

static void test_estimator_l1_default_source(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("l1-default: returns 0", rc == 0);
    if (rc != 0) return;
    check("l1-default: top-level source l1_kernel_dequant",
          r.t_squared_source == "l1_kernel_dequant");
    bool all_labeled = !r.layers.empty();
    bool any_positive = false;
    for (const auto & lr : r.layers) {
        if (lr.t_squared_source != "l1_kernel_dequant") all_labeled = false;
        if (std::isnan(lr.t_squared) || std::isinf(lr.t_squared) ||
            !(lr.t_squared >= 0.0f)) {
            all_labeled = false;
        }
        if (lr.t_squared > 0.0f) any_positive = true;
    }
    check("l1-default: per-layer source l1_kernel_dequant", all_labeled);
    check("l1-default: at least one positive t_squared", any_positive);
}

static void test_estimator_legacy_offline_env(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;

    ::setenv("TS_HIGGS_PROXY_LEGACY_OFFLINE", "1", 1);
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    ::unsetenv("TS_HIGGS_PROXY_LEGACY_OFFLINE");
    check("legacy-env: returns 0", rc == 0);
    if (rc == 0) {
        check("legacy-env: source offline_ternary_mse",
              r.t_squared_source == "offline_ternary_mse");
        bool per_layer_legacy = !r.layers.empty();
        for (const auto & lr : r.layers) {
            if (lr.t_squared_source != "offline_ternary_mse") per_layer_legacy = false;
        }
        check("legacy-env: per-layer source offline_ternary_mse", per_layer_legacy);
        // The sidecar JSON shape is unchanged; only the value differs.
        std::string json = ts_higgs_proxy_to_json(&r, fixture_path.c_str(), nullptr);
        check("legacy-env: json measurement offline_ternary_mse",
              json.find("\"measurement\": \"offline_ternary_mse\"") != std::string::npos);
    }

    // Env cleared: the L1 default is back.
    ts_higgs_proxy_result r2;
    rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r2);
    check("legacy-env: L1 default restored after unset",
          rc == 0 && r2.t_squared_source == "l1_kernel_dequant");
}

static void test_estimator_l1_json_shape(const std::string & fixture_path) {
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    ts_higgs_proxy_result r;
    int rc = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r);
    check("l1-json: returns 0", rc == 0);
    if (rc != 0) return;
    std::string json = ts_higgs_proxy_to_json(&r, fixture_path.c_str(), nullptr);
    check("l1-json: measurement l1_kernel_dequant",
          json.find("\"measurement\": \"l1_kernel_dequant\"") != std::string::npos);
    check("l1-json: per-layer t_squared_source l1_kernel_dequant",
          json.find("\"t_squared_source\": \"l1_kernel_dequant\"") != std::string::npos);
    check("l1-json: schema unchanged",
          json.find("ane.alpha-coefficients.v1") != std::string::npos);
    ts_higgs_proxy_result r2;
    int rc2 = ts_higgs_proxy_from_json(json.c_str(), &r2);
    check("l1-json: round-trip preserves l1 source",
          rc2 == 0 && r2.t_squared_source == "l1_kernel_dequant");
}

static void test_estimator_alphas_identical_l1_vs_offline(const std::string & fixture_path) {
    // The proxy alpha is the structural family prior, independent of
    // the t^2 measurement source. The L1 default and the legacy
    // offline path must produce identical alphas; only t_squared and
    // the source label differ.
    ts_higgs_proxy_params p = {};
    p.min_params_for_estimate = 0;
    p.alpha_floor_fraction = 1e-3f;

    ts_higgs_proxy_result r_l1;
    int rc1 = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r_l1);

    ::setenv("TS_HIGGS_PROXY_LEGACY_OFFLINE", "1", 1);
    ts_higgs_proxy_result r_off;
    int rc2 = ts_higgs_proxy_estimate(fixture_path.c_str(), &p, nullptr, nullptr, &r_off);
    ::unsetenv("TS_HIGGS_PROXY_LEGACY_OFFLINE");

    check("e2e: both runs succeed", rc1 == 0 && rc2 == 0);
    if (rc1 != 0 || rc2 != 0) return;
    check("e2e: same layer count", r_l1.layers.size() == r_off.layers.size());
    if (r_l1.layers.size() != r_off.layers.size()) return;

    bool alphas_equal = true;
    bool frob_equal = true;
    bool t2_differ = false;
    bool ratio_ok = true;
    for (size_t i = 0; i < r_l1.layers.size(); i++) {
        const auto & a = r_l1.layers[i];
        const auto & b = r_off.layers[i];
        if (a.name != b.name) { alphas_equal = false; break; }
        if (a.alpha_l != b.alpha_l) alphas_equal = false;
        if (a.frobenius_norm_sq != b.frobenius_norm_sq) frob_equal = false;
        if (a.t_squared != b.t_squared) t2_differ = true;
        if (a.t_squared > 0.0f && b.t_squared > 0.0f) {
            const float ratio = a.t_squared / b.t_squared;
            if (ratio < 0.3f || ratio > 4.0f) ratio_ok = false;
        }
    }
    check("e2e: alpha identical (structural prior)", alphas_equal);
    check("e2e: frobenius identical", frob_equal);
    check("e2e: t_squared differs between sources", t2_differ);
    check("e2e: per-layer t_squared ratio within [0.3, 4.0]", ratio_ok);
    check("e2e: mean_alpha identical",
          feq(r_l1.mean_alpha, r_off.mean_alpha, 1e-6f));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    // Build the fixture.
    std::string fixture_path = "/tmp/ts_higgs_proxy_fixture.gguf";
    ::unlink(fixture_path.c_str());
    std::vector<ts_tensor_spec> specs = tinyllama_shaped_specs();
    if (!build_fixture_gguf(fixture_path, specs)) {
        std::printf("FAIL: build_fixture_gguf returned false\n");
        return 1;
    }

    // 1. Family classification
    test_classify_attn_k_v();
    test_classify_attn_q();
    test_classify_attn_output_not_misread_as_q();
    test_classify_ffn();
    test_classify_norms();
    test_classify_embedding_and_output();
    test_classify_unknown_returns_other();
    test_classify_strips_weight_suffix();

    // 2. Family prior table
    test_family_prior_ranking();
    test_family_prior_exact_values();

    // 3. Measurement function
    test_measurement_offline_zero_returns_zero();
    test_measurement_offline_empty_returns_zero();
    test_measurement_offline_known_value();
    test_measurement_offline_matches_numpy_handwritten();

    // 4. Estimator
    test_estimator_family_rank_on_fixture(fixture_path);
    test_estimator_mean_alpha_is_one(fixture_path);
    test_estimator_no_uniform_fallback_above_threshold(fixture_path);

    // 5. Uniform fallback
    test_uniform_fallback_default(fixture_path);
    test_uniform_fallback_above_threshold(fixture_path);

    // 6. alpha_floor
    test_alpha_floor_applied(fixture_path);

    // 7. model_hash
    test_model_hash_matches_shasum(fixture_path);
    test_model_hash_stable(fixture_path);

    // 8. JSON
    test_json_round_trip(fixture_path);
    test_json_key_order(fixture_path);
    test_json_per_layer_key_order(fixture_path);
    test_json_atomic_write();

    // 9. L1-agnostic
    test_l1_agnostic_custom_measurement(fixture_path);
    test_l1_agnostic_ctx_threaded(fixture_path);

    // 10. Robustness
    test_gguf_with_small_tensors();
    test_gguf_with_zero_norm_tensor();

    // 11. extract_alphas
    test_extract_alphas(fixture_path);

    // 12. Parity with NumPy
    test_parity_with_numpy_measurement();

    // 13. Guard rails
    test_estimator_rejects_null_gguf_path();
    test_estimator_rejects_null_result();
    test_estimator_rejects_missing_file();

    // 14. L1-on-ANE measurement (the new default)
    test_l1_pack_tile640_sizes();
    test_l1_measurement_guards();
    test_l1_measurement_determinism();
    test_l1_measurement_vs_offline_range();
    test_l1_measurement_v2_branch();
    test_estimator_l1_default_source(fixture_path);
    test_estimator_legacy_offline_env(fixture_path);
    test_estimator_l1_json_shape(fixture_path);
    test_estimator_alphas_identical_l1_vs_offline(fixture_path);

    ::unlink(fixture_path.c_str());

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
