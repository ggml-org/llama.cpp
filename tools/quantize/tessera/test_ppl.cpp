//
// test_ppl.cpp
//
// Smoke tests for tessera-ppl.cpp. Prints PASS/FAIL per test,
// returns 0 only if all pass.
//

#include "tessera-ppl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const int64_t V = 64;  // small vocab for tests
static const int64_t T = 16;  // small token count

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void fill_uniform(float * buf, int64_t n) {
    for (int64_t i = 0; i < n; i++) buf[i] = 1.0f;
}

static void fill_perturbed(float * buf, int64_t n, float eps) {
    for (int64_t i = 0; i < n; i++) {
        buf[i] = 1.0f + eps * ((i % 3) - 1);  // -eps, 0, +eps pattern
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

static bool test_kl_self_is_zero() {
    std::vector<float> logits(T * V);
    fill_uniform(logits.data(), T * V);

    float kl = ts_ppl_kl_divergence(logits.data(), logits.data(), T, V);
    printf("  kl_self: %.2e\n", kl);
    return fabsf(kl) < 1e-6f;
}

static bool test_kl_perturbed_positive() {
    std::vector<float> ref(T * V), quant(T * V);
    for (int64_t t = 0; t < T; t++) {
        fill_uniform(ref.data() + t * V, V);
        fill_perturbed(quant.data() + t * V, V, 0.1f);
    }

    float kl = ts_ppl_kl_divergence(ref.data(), quant.data(), T, V);
    printf("  kl_perturbed: %.6f\n", kl);
    return kl > 0.0f && kl < 1.0f;
}

static bool test_ppl_uniform() {
    std::vector<float> logits(T * V);
    fill_uniform(logits.data(), T * V);

    // targets don't matter for uniform: PPL should be V
    std::vector<int32_t> targets(T);
    for (int64_t t = 0; t < T; t++) targets[t] = (int32_t)(t % V);

    float ppl = ts_ppl_perplexity(logits.data(), targets.data(), T, V);
    printf("  ppl_uniform: %.4f (expect %lld)\n", ppl, (long long)V);
    return fabsf(ppl - (float)V) < 0.01f;
}

// mock forward functions for probe test
static void mock_forward_uniform(const int32_t * /*tokens*/, float * logits_out,
                                 int64_t n_tokens, int64_t vocab_size,
                                 void * /*ctx*/) {
    fill_uniform(logits_out, n_tokens * vocab_size);
}

static void mock_forward_perturbed(const int32_t * /*tokens*/, float * logits_out,
                                   int64_t n_tokens, int64_t vocab_size,
                                   void * /*ctx*/) {
    for (int64_t t = 0; t < n_tokens; t++) {
        fill_perturbed(logits_out + t * vocab_size, vocab_size, 0.05f);
    }
}

static bool test_probe() {
    ts_ppl_params params = {};
    params.n_tokens   = T;
    params.vocab_size = V;
    params.use_kl     = true;
    params.seed       = 123;

    ts_ppl_result result = {};
    int rc = ts_ppl_probe(mock_forward_uniform, nullptr,
                          mock_forward_perturbed, nullptr,
                          &params, &result);

    printf("  probe: rc=%d kl=%.6f ratio=%.4f delta=%.4f n=%lld\n",
           rc, result.kl_divergence, result.ppl_ratio,
           result.delta_ppl, (long long)result.n_tokens_used);

    return rc == 0
        && result.kl_divergence > 0.0f
        && result.kl_divergence < 0.1f
        && result.ppl_ratio > 1.0f
        && result.n_tokens_used == T;
}

static bool test_probe_null_rejected() {
    ts_ppl_params params = {};
    ts_ppl_result result = {};
    int rc = ts_ppl_probe(nullptr, nullptr, nullptr, nullptr, &params, &result);
    printf("  null_reject: rc=%d\n", rc);
    return rc == -1;
}

// ---------------------------------------------------------------------------

int main() {
    struct { const char * name; bool (*fn)(); } tests[] = {
        { "kl_self_is_zero",     test_kl_self_is_zero },
        { "kl_perturbed_positive", test_kl_perturbed_positive },
        { "ppl_uniform",         test_ppl_uniform },
        { "probe",               test_probe },
        { "probe_null_rejected", test_probe_null_rejected },
    };

    bool all = true;
    for (auto & t : tests) {
        bool ok = t.fn();
        printf("[%s] %s\n", ok ? "PASS" : "FAIL", t.name);
        all = all && ok;
    }
    return all ? 0 : 1;
}
