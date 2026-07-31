#include "tessera-dpace.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

#define CHECK_NEAR(a, b, eps, msg) do { \
    double _a = (a), _b = (b), _e = (eps); \
    if (std::fabs(_a - _b) <= _e) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s: got %.10f, expected %.10f (line %d)\n", msg, _a, _b, __LINE__); } \
} while (0)

// Tolerance for tests involving float->double promotion (~1e-7 relative error)
#define EPS_FLOAT 1e-5

static void test_continuation_values() {
    // block_size = 1: f_0 = 1
    {
        float a[] = { 0.9f };
        double f[1];
        ts_dpace_continuation_values(a, 1, f);
        CHECK_NEAR(f[0], 1.0, 1e-12, "f_0 = 1 for block_size=1");
    }

    // block_size = 3, a = [0.8, 0.6, 0.4]
    // f_2 = 1
    // f_1 = 1 + a_2 * f_2 = 1 + 0.4 = 1.4
    // f_0 = 1 + a_1 * f_1 = 1 + 0.6 * 1.4 = 1.84
    {
        float a[] = { 0.8f, 0.6f, 0.4f };
        double f[3];
        ts_dpace_continuation_values(a, 3, f);
        CHECK_NEAR(f[2], 1.0, 1e-12, "f_2 = 1");
        CHECK_NEAR(f[1], 1.4, EPS_FLOAT, "f_1 = 1 + 0.4");
        CHECK_NEAR(f[0], 1.84, EPS_FLOAT, "f_0 = 1 + 0.6*1.4");
    }

    // All probs = 1: f_j = B - j (each continuation is a chain of 1s)
    {
        float a[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        double f[4];
        ts_dpace_continuation_values(a, 4, f);
        CHECK_NEAR(f[3], 1.0, 1e-12, "f_3 = 1 (all-ones)");
        CHECK_NEAR(f[2], 2.0, 1e-12, "f_2 = 2 (all-ones)");
        CHECK_NEAR(f[1], 3.0, 1e-12, "f_1 = 3 (all-ones)");
        CHECK_NEAR(f[0], 4.0, 1e-12, "f_0 = 4 (all-ones)");
    }

    // All probs = 0: f_j = 1 for all j (no continuation possible)
    {
        float a[] = { 0.0f, 0.0f, 0.0f };
        double f[3];
        ts_dpace_continuation_values(a, 3, f);
        CHECK_NEAR(f[0], 1.0, 1e-12, "f_0 = 1 (all-zeros)");
        CHECK_NEAR(f[1], 1.0, 1e-12, "f_1 = 1 (all-zeros)");
        CHECK_NEAR(f[2], 1.0, 1e-12, "f_2 = 1 (all-zeros)");
    }
}

static void test_surrogate() {
    // block_size = 1, a = [0.9]: S = 0.9
    {
        float a[] = { 0.9f };
        CHECK_NEAR(ts_dpace_accepted_length_surrogate(a, 1), 0.9, EPS_FLOAT,
                   "S = 0.9 for single position");
    }

    // a = [1, 1, 1, 1]: S = 1 + 1 + 1 + 1 = 4 (perfect drafter)
    {
        float a[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        CHECK_NEAR(ts_dpace_accepted_length_surrogate(a, 4), 4.0, 1e-10,
                   "S = B for perfect drafter");
    }

    // a = [0.5, 0.5, 0.5]: S = 0.5 + 0.25 + 0.125 = 0.875
    {
        float a[] = { 0.5f, 0.5f, 0.5f };
        CHECK_NEAR(ts_dpace_accepted_length_surrogate(a, 3), 0.875, 1e-10,
                   "S = 0.875 for uniform 0.5");
    }

    // a = [0, ...]: S = 0 (anchor rejected, nothing accepted)
    {
        float a[] = { 0.0f, 0.9f, 0.9f };
        CHECK_NEAR(ts_dpace_accepted_length_surrogate(a, 3), 0.0, 1e-10,
                   "S = 0 when anchor prob = 0");
    }

    // S is bounded by [0, B]
    {
        float a[] = { 0.8f, 0.7f, 0.6f, 0.5f, 0.4f };
        double S = ts_dpace_accepted_length_surrogate(a, 5);
        CHECK(S >= 0.0 && S <= 5.0, "S in [0, B]");
    }
}

static void test_raw_weights() {
    // a = [0.8, 0.6, 0.4], B = 3
    // f = [1.84, 1.4, 1.0]
    // C_0 = 0.8, C_1 = 0.48, C_2 = 0.192
    // w_0 = 0.8 * 1.84 = 1.472
    // w_1 = 0.48 * 1.4 = 0.672
    // w_2 = 0.192 * 1.0 = 0.192
    {
        float a[] = { 0.8f, 0.6f, 0.4f };
        double w[3];
        ts_dpace_weights(a, 3, w);
        CHECK_NEAR(w[0], 0.8 * 1.84, EPS_FLOAT, "w_0 = C_0 * f_0");
        CHECK_NEAR(w[1], 0.48 * 1.4, EPS_FLOAT, "w_1 = C_1 * f_1");
        CHECK_NEAR(w[2], 0.192 * 1.0, EPS_FLOAT, "w_2 = C_2 * f_2");
    }

    // All probs = 1: w_j = 1 * (B - j) = B - j
    {
        float a[] = { 1.0f, 1.0f, 1.0f };
        double w[3];
        ts_dpace_weights(a, 3, w);
        CHECK_NEAR(w[0], 3.0, 1e-10, "w_0 = 3 (all-ones)");
        CHECK_NEAR(w[1], 2.0, 1e-10, "w_1 = 2 (all-ones)");
        CHECK_NEAR(w[2], 1.0, 1e-10, "w_2 = 1 (all-ones)");
    }

    // Weights are non-negative
    {
        float a[] = { 0.1f, 0.9f, 0.3f, 0.7f };
        double w[4];
        ts_dpace_weights(a, 4, w);
        bool all_nonneg = true;
        for (int j = 0; j < 4; ++j) {
            if (w[j] < 0.0) all_nonneg = false;
        }
        CHECK(all_nonneg, "weights are non-negative");
    }

    // First position weight >= last position weight (early positions matter more)
    {
        float a[] = { 0.5f, 0.5f, 0.5f, 0.5f };
        double w[4];
        ts_dpace_weights(a, 4, w);
        CHECK(w[0] >= w[3], "w_0 >= w_{B-1} for uniform probs");
    }
}

static void test_smoothed_weights() {
    // alpha = 0 should match raw weights
    {
        float a[] = { 0.8f, 0.6f, 0.4f };
        double raw[3], smoothed[3];
        ts_dpace_weights(a, 3, raw);
        ts_dpace_weights_smoothed(a, 3, 0.0f, smoothed);
        for (int j = 0; j < 3; ++j) {
            CHECK_NEAR(raw[j], smoothed[j], 1e-12, "alpha=0 matches raw");
        }
    }

    // alpha = 1: all smoothed probs = 1, so weights = [B, B-1, ..., 1]
    {
        float a[] = { 0.1f, 0.2f, 0.3f };
        double w[3];
        ts_dpace_weights_smoothed(a, 3, 1.0f, w);
        CHECK_NEAR(w[0], 3.0, 1e-10, "alpha=1: w_0 = 3");
        CHECK_NEAR(w[1], 2.0, 1e-10, "alpha=1: w_1 = 2");
        CHECK_NEAR(w[2], 1.0, 1e-10, "alpha=1: w_2 = 1");
    }

    // Smoothing raises the floor: low-confidence positions get more weight
    {
        float a[] = { 0.9f, 0.01f, 0.01f };
        double raw[3], smoothed[3];
        ts_dpace_weights(a, 3, raw);
        ts_dpace_weights_smoothed(a, 3, 0.1f, smoothed);
        // After normalization, the smoothed version should give relatively
        // more weight to positions 1 and 2 compared to raw
        ts_dpace_normalize_weights(raw, 3);
        ts_dpace_normalize_weights(smoothed, 3);
        CHECK(smoothed[2] > raw[2], "smoothing raises late-position weight");
    }
}

static void test_decay_weights() {
    // gamma = inf approximation: all weights ~ 1
    {
        double w[4];
        ts_dflash_decay_weights(4, 1000.0f, w);
        for (int j = 0; j < 4; ++j) {
            CHECK_NEAR(w[j], 1.0, 0.01, "large gamma -> uniform weights");
        }
    }

    // gamma = 1: w_k = exp(-k)
    {
        double w[3];
        ts_dflash_decay_weights(3, 1.0f, w);
        CHECK_NEAR(w[0], 1.0, 1e-10, "decay w_0 = 1");
        CHECK_NEAR(w[1], std::exp(-1.0), 1e-10, "decay w_1 = e^-1");
        CHECK_NEAR(w[2], std::exp(-2.0), 1e-10, "decay w_2 = e^-2");
    }

    // gamma = 0: only anchor gets weight
    {
        double w[3];
        ts_dflash_decay_weights(3, 0.0f, w);
        CHECK_NEAR(w[0], 1.0, 1e-12, "gamma=0: w_0 = 1");
        CHECK_NEAR(w[1], 0.0, 1e-12, "gamma=0: w_1 = 0");
        CHECK_NEAR(w[2], 0.0, 1e-12, "gamma=0: w_2 = 0");
    }

    // Monotonically decreasing
    {
        double w[8];
        ts_dflash_decay_weights(8, 3.0f, w);
        bool monotone = true;
        for (int j = 1; j < 8; ++j) {
            if (w[j] > w[j-1]) monotone = false;
        }
        CHECK(monotone, "decay weights are monotonically decreasing");
    }
}

static void test_normalize() {
    double w[] = { 2.0, 1.0, 0.5 };
    ts_dpace_normalize_weights(w, 3);
    double sum = w[0] + w[1] + w[2];
    CHECK_NEAR(sum, 3.0, 1e-10, "normalized weights sum to B");
    CHECK_NEAR(w[0] / w[1], 2.0, 1e-10, "ratio preserved after normalization");
}

static void test_loss_from_probs() {
    // Uniform weights, perfect predictions -> loss = 0
    {
        float probs[] = { 1.0f, 1.0f, 1.0f };
        double weights[] = { 1.0, 1.0, 1.0 };
        CHECK_NEAR(ts_dpace_loss_from_probs(probs, weights, 3), 0.0, 1e-8,
                   "perfect predictions -> zero loss");
    }

    // Uniform weights, prob = 0.5 -> loss = 3 * (-log(0.5)) = 3 * 0.693...
    {
        float probs[] = { 0.5f, 0.5f, 0.5f };
        double weights[] = { 1.0, 1.0, 1.0 };
        double expected = 3.0 * (-std::log(0.5));
        CHECK_NEAR(ts_dpace_loss_from_probs(probs, weights, 3), expected, 1e-8,
                   "uniform 0.5 -> 3*log(2)");
    }

    // Weighted: only first position matters
    {
        float probs[] = { 0.5f, 0.01f, 0.01f };
        double weights[] = { 3.0, 0.0, 0.0 };
        double expected = 3.0 * (-std::log(0.5));
        CHECK_NEAR(ts_dpace_loss_from_probs(probs, weights, 3), expected, 1e-8,
                   "zero-weight positions don't contribute");
    }

    // Very small prob is clamped
    {
        float probs[] = { 0.0f };
        double weights[] = { 1.0 };
        double loss = ts_dpace_loss_from_probs(probs, weights, 1);
        CHECK(loss > 0.0 && loss < 100.0, "clamped prob gives finite loss");
    }
}

static void test_loss_from_logits() {
    // 2 positions, vocab = 3
    // Position 0: logits = [10, 0, 0], target = 0 -> softmax(10) ~ 1.0
    // Position 1: logits = [0, 10, 0], target = 1 -> softmax(10) ~ 1.0
    {
        float logits[] = { 10.0f, 0.0f, 0.0f,
                            0.0f, 10.0f, 0.0f };
        int32_t targets[] = { 0, 1 };
        double weights[] = { 1.0, 1.0 };
        double loss = ts_dpace_loss(logits, targets, weights, 2, 3);
        CHECK(loss < 0.01, "high-confidence correct predictions -> near-zero loss");
    }

    // Wrong target: logits = [10, 0, 0], target = 1 -> loss is large
    {
        float logits[] = { 10.0f, 0.0f, 0.0f };
        int32_t targets[] = { 1 };
        double weights[] = { 1.0 };
        double loss = ts_dpace_loss(logits, targets, weights, 1, 3);
        CHECK(loss > 5.0, "wrong target -> large loss");
    }

    // Uniform logits -> loss = log(n_vocab) per position
    {
        float logits[] = { 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f };
        int32_t targets[] = { 0, 2 };
        double weights[] = { 1.0, 1.0 };
        double expected = 2.0 * std::log(4.0);
        CHECK_NEAR(ts_dpace_loss(logits, targets, weights, 2, 4), expected, 1e-6,
                   "uniform logits -> log(V) per position");
    }
}

static void test_end_to_end() {
    // Perfect drafter: acceptance_probs all 1.0, logits peaked on target
    {
        float logits[] = { 10.0f, 0.0f, 0.0f,
                            0.0f, 10.0f, 0.0f,
                            0.0f, 0.0f, 10.0f };
        int32_t targets[] = { 0, 1, 2 };
        float acc[] = { 1.0f, 1.0f, 1.0f };
        double loss = ts_dpace_loss_end_to_end(logits, targets, acc, 3, 3, 0.1f);
        CHECK(loss < 0.1, "end-to-end: perfect drafter -> near-zero loss");
    }

    // Loss is finite and positive for non-trivial inputs
    {
        float logits[] = { 1.0f, 2.0f, 0.5f,
                            0.3f, 1.5f, 2.0f };
        int32_t targets[] = { 1, 2 };
        float acc[] = { 0.6f, 0.4f };
        double loss = ts_dpace_loss_end_to_end(logits, targets, acc, 2, 3, 0.1f);
        CHECK(loss > 0.0 && std::isfinite(loss), "end-to-end: finite positive loss");
    }
}

static void test_ab_compare() {
    float logits[] = { 2.0f, 1.0f, 0.0f,
                        0.0f, 2.0f, 1.0f,
                        1.0f, 0.0f, 2.0f };
    int32_t targets[] = { 0, 1, 2 };
    float acc[] = { 0.7f, 0.5f, 0.3f };

    ts_dpace_ab_result r = ts_dpace_ab_compare(logits, targets, acc, 3, 3, 0.1f, 3.0f);

    CHECK(r.dpace_loss > 0.0 && std::isfinite(r.dpace_loss), "AB: dpace_loss finite");
    CHECK(r.decay_loss > 0.0 && std::isfinite(r.decay_loss), "AB: decay_loss finite");
    CHECK(r.dpace_surrogate > 0.0 && r.dpace_surrogate <= 3.0, "AB: surrogate in (0, B]");
    CHECK_NEAR(r.mean_dpace_weight, 1.0, 1e-10, "AB: normalized dpace mean = 1");
    CHECK_NEAR(r.mean_decay_weight, 1.0, 1e-10, "AB: normalized decay mean = 1");
}

static void test_edge_cases() {
    // block_size = 0: no crash
    {
        double w[1] = { -1.0 };
        ts_dpace_weights(nullptr, 0, w);
        ts_dpace_weights_smoothed(nullptr, 0, 0.1f, w);
        ts_dflash_decay_weights(0, 3.0f, w);
        ts_dpace_normalize_weights(w, 0);
        CHECK_NEAR(ts_dpace_accepted_length_surrogate(nullptr, 0), 0.0, 1e-12,
                   "block_size=0 -> surrogate = 0");
        CHECK_NEAR(ts_dpace_loss_from_probs(nullptr, nullptr, 0), 0.0, 1e-12,
                   "block_size=0 -> loss = 0");
        CHECK(true, "block_size=0: no crash");
    }

    // block_size = 1: single anchor position
    {
        float a[] = { 0.9f };
        double w[1];
        ts_dpace_weights(a, 1, w);
        // w_0 = a_0 * f_0 = 0.9 * 1.0 = 0.9
        CHECK_NEAR(w[0], 0.9, EPS_FLOAT, "single position: w = a * 1");
    }

    // Very small probabilities don't cause NaN
    {
        float a[] = { 1e-8f, 1e-8f, 1e-8f, 1e-8f };
        double w[4];
        ts_dpace_weights_smoothed(a, 4, 0.1f, w);
        bool all_finite = true;
        for (int j = 0; j < 4; ++j) {
            if (!std::isfinite(w[j])) all_finite = false;
        }
        CHECK(all_finite, "tiny probs with smoothing -> finite weights");
    }
}

int main() {
    test_continuation_values();
    test_surrogate();
    test_raw_weights();
    test_smoothed_weights();
    test_decay_weights();
    test_normalize();
    test_loss_from_probs();
    test_loss_from_logits();
    test_end_to_end();
    test_ab_compare();
    test_edge_cases();

    printf("dpace: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
