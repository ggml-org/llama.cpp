// test_lk_loss.cpp - offline tests for tessera-lk-loss (pure math, no model)
#include "tessera-lk-loss.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define APPROX(a, b, eps) (std::fabs((a) - (b)) < (eps))
// Float literals like 0.3f are not exact in double; use 1e-6 for all checks.
#define EPS 1e-6

// -------------------------------------------------------------------------

static void test_identical_distributions() {
    // alpha = 1 when p == q
    float p[] = {0.5f, 0.3f, 0.2f};
    float q[] = {0.5f, 0.3f, 0.2f};
    double alpha = ts_lk_acceptance_rate(p, q, 3);
    CHECK(APPROX(alpha, 1.0, EPS), "identical dists alpha=1");
}

static void test_disjoint_distributions() {
    // alpha = 0 when supports are disjoint
    float p[] = {1.0f, 0.0f, 0.0f};
    float q[] = {0.0f, 1.0f, 0.0f};
    double alpha = ts_lk_acceptance_rate(p, q, 3);
    CHECK(APPROX(alpha, 0.0, EPS), "disjoint dists alpha=0");
}

static void test_partial_overlap() {
    // p = [0.6, 0.4, 0.0], q = [0.5, 0.5, 0.0]
    // alpha = min(0.6,0.5) + min(0.4,0.5) + min(0,0) = 0.5 + 0.4 = 0.9
    float p[] = {0.6f, 0.4f, 0.0f};
    float q[] = {0.5f, 0.5f, 0.0f};
    double alpha = ts_lk_acceptance_rate(p, q, 3);
    CHECK(APPROX(alpha, 0.9, EPS), "partial overlap alpha=0.9");
}

static void test_loss_is_negative_alpha() {
    float p[] = {0.7f, 0.3f};
    float q[] = {0.7f, 0.3f};
    double loss = ts_lk_loss(p, q, 2);
    CHECK(APPROX(loss, -1.0, EPS), "loss = -alpha for identical");
}

static void test_token_contribution() {
    CHECK(APPROX(ts_lk_token_contribution(0.8f, 0.3f), 0.3, EPS), "contrib min(0.8,0.3)=0.3");
    CHECK(APPROX(ts_lk_token_contribution(0.2f, 0.9f), 0.2, EPS), "contrib min(0.2,0.9)=0.2");
}

static void test_topk_identical() {
    // Same top-k entries -> alpha ~ 1 (residual buckets also match)
    int32_t toks[] = {0, 1, 2};
    float   probs[] = {0.5f, 0.3f, 0.2f};
    double mass = 1.0;
    double alpha = ts_lk_acceptance_rate_topk(
        toks, probs, 3, mass,
        toks, probs, 3, mass,
        100);
    CHECK(APPROX(alpha, 1.0, 1e-6), "topk identical alpha~1");
}

static void test_topk_disjoint() {
    // Completely different top-k tokens, no residual overlap
    int32_t pt[] = {0, 1};
    float   pp[] = {0.9f, 0.1f};
    int32_t qt[] = {2, 3};
    float   qp[] = {0.9f, 0.1f};
    // n_vocab=4, all tokens in union -> no residual bucket
    double alpha = ts_lk_acceptance_rate_topk(
        pt, pp, 2, 1.0,
        qt, qp, 2, 1.0,
        4);
    CHECK(APPROX(alpha, 0.0, EPS), "topk disjoint alpha=0");
}

static void test_topk_partial() {
    // p: token0=0.8, token1=0.2; q: token0=0.6, token2=0.4; n_vocab=3
    // union: {0,1,2}; residual=0
    // alpha = min(0.8,0.6) + min(0.2,0) + min(0,0.4) = 0.6
    int32_t pt[] = {0, 1};
    float   pp[] = {0.8f, 0.2f};
    int32_t qt[] = {0, 2};
    float   qp[] = {0.6f, 0.4f};
    double alpha = ts_lk_acceptance_rate_topk(
        pt, pp, 2, 1.0,
        qt, qp, 2, 1.0,
        3);
    CHECK(APPROX(alpha, 0.6, EPS), "topk partial alpha=0.6");
}

static void test_topk_residual() {
    // p: token0=0.5 (mass=0.5, residual=0.5 over 99 tokens)
    // q: token0=0.5 (mass=0.5, residual=0.5 over 99 tokens)
    // alpha = min(0.5,0.5) + 99*min(0.5/99, 0.5/99) = 0.5 + 0.5 = 1.0
    int32_t pt[] = {0};
    float   pp[] = {0.5f};
    int32_t qt[] = {0};
    float   qp[] = {0.5f};
    double alpha = ts_lk_acceptance_rate_topk(
        pt, pp, 1, 0.5,
        qt, qp, 1, 0.5,
        100);
    CHECK(APPROX(alpha, 1.0, 1e-6), "topk residual symmetric alpha~1");
}

static void test_batch() {
    // Two positions, both identical -> mean alpha = 1
    int32_t t1[] = {0, 1};
    float   p1[] = {0.7f, 0.3f};
    int32_t t2[] = {2, 3};
    float   p2[] = {0.6f, 0.4f};

    const int32_t * pt[] = {t1, t2};
    const float   * pp[] = {p1, p2};
    int   pk[]  = {2, 2};
    double pm[] = {1.0, 1.0};

    double alpha = ts_lk_acceptance_rate_batch(
        pt, pp, pk, pm,
        pt, pp, pk, pm,
        2, 100);
    CHECK(APPROX(alpha, 1.0, 1e-6), "batch identical alpha~1");
}

static void test_batch_empty() {
    double alpha = ts_lk_acceptance_rate_batch(
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        0, 100);
    CHECK(APPROX(alpha, 0.0, EPS), "batch empty alpha=0");
}

// -------------------------------------------------------------------------

int main() {
    test_identical_distributions();
    test_disjoint_distributions();
    test_partial_overlap();
    test_loss_is_negative_alpha();
    test_token_contribution();
    test_topk_identical();
    test_topk_disjoint();
    test_topk_partial();
    test_topk_residual();
    test_batch();
    test_batch_empty();

    printf("lk_loss: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
