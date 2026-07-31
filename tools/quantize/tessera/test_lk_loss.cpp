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
// dense reconstruction from top-k

static double dense_sum(const float * d, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += d[i];
    return s;
}

static void test_dense_validity() {
    int32_t toks[] = {3, 7, 1};
    float   probs[] = {0.5f, 0.3f, 0.1f}; // mass 0.9, residual 0.1
    const int n_vocab = 20;
    std::vector<float> d(n_vocab);
    int rc = ts_lk_dense_from_topk(toks, probs, 3, n_vocab, d.data());
    CHECK(rc == 0, "dense rc==0");
    bool nonneg = true;
    for (int i = 0; i < n_vocab; ++i) nonneg = nonneg && d[i] >= 0.0f;
    CHECK(nonneg, "dense all non-negative");
    CHECK(APPROX(dense_sum(d.data(), n_vocab), 1.0, 1e-5), "dense sums to 1");
}

static void test_dense_topk_fidelity() {
    int32_t toks[] = {3, 7, 1};
    float   probs[] = {0.5f, 0.3f, 0.1f};
    const int n_vocab = 20;
    std::vector<float> d(n_vocab);
    ts_lk_dense_from_topk(toks, probs, 3, n_vocab, d.data());
    // High-probability support is preserved exactly.
    CHECK(APPROX(d[3], 0.5, EPS), "dense slot 3 == 0.5");
    CHECK(APPROX(d[7], 0.3, EPS), "dense slot 7 == 0.3");
    CHECK(APPROX(d[1], 0.1, EPS), "dense slot 1 == 0.1");
}

static void test_dense_residual_uniform() {
    int32_t toks[] = {3, 7, 1};
    float   probs[] = {0.5f, 0.3f, 0.1f}; // residual 0.1 over 17 slots
    const int n_vocab = 20;
    std::vector<float> d(n_vocab);
    ts_lk_dense_from_topk(toks, probs, 3, n_vocab, d.data());
    const float per = 0.1f / 17.0f;
    bool ok = true;
    for (int i = 0; i < n_vocab; ++i) {
        if (i == 3 || i == 7 || i == 1) continue;
        ok = ok && APPROX(d[i], per, 1e-6);
    }
    CHECK(ok, "dense residual spread uniformly over unfilled slots");
}

static void test_dense_full_coverage() {
    // k covers the whole vocab -> residual 0, dense == exact probs.
    int32_t toks[] = {0, 1, 2};
    float   probs[] = {0.6f, 0.3f, 0.1f};
    const int n_vocab = 3;
    std::vector<float> d(n_vocab);
    ts_lk_dense_from_topk(toks, probs, 3, n_vocab, d.data());
    CHECK(APPROX(d[0], 0.6, EPS) && APPROX(d[1], 0.3, EPS) && APPROX(d[2], 0.1, EPS),
          "dense full coverage == exact probs");
    CHECK(APPROX(dense_sum(d.data(), n_vocab), 1.0, 1e-6), "dense full coverage sums to 1");
}

static void test_dense_k0_uniform() {
    const int n_vocab = 8;
    std::vector<float> d(n_vocab);
    int rc = ts_lk_dense_from_topk(nullptr, nullptr, 0, n_vocab, d.data());
    CHECK(rc == 0, "dense k=0 rc==0");
    bool ok = true;
    for (int i = 0; i < n_vocab; ++i) ok = ok && APPROX(d[i], 1.0/8.0, 1e-6);
    CHECK(ok, "dense k=0 -> uniform");
}

static void test_dense_duplicate_tokens() {
    // Token 5 appears twice; mass accumulates, distinct count stays correct.
    int32_t toks[] = {5, 5, 2};
    float   probs[] = {0.2f, 0.3f, 0.1f}; // slot5 = 0.5, slot2 = 0.1, mass 0.6
    const int n_vocab = 10;
    std::vector<float> d(n_vocab);
    int rc = ts_lk_dense_from_topk(toks, probs, 3, n_vocab, d.data());
    CHECK(rc == 0, "dense dup rc==0");
    CHECK(APPROX(d[5], 0.5, EPS), "dense dup slot5 accumulates to 0.5");
    CHECK(APPROX(d[2], 0.1, EPS), "dense dup slot2 == 0.1");
    // 2 distinct slots filled -> residual 0.4 over 8 slots.
    const float per = 0.4f / 8.0f;
    CHECK(APPROX(d[0], per, 1e-6), "dense dup residual over 8 unfilled slots");
    CHECK(APPROX(dense_sum(d.data(), n_vocab), 1.0, 1e-5), "dense dup sums to 1");
}

static void test_dense_invalid_input() {
    const int n_vocab = 10;
    std::vector<float> d(n_vocab);
    int32_t bad_tok[] = {10};       // out of range
    float   p1[] = {0.5f};
    CHECK(ts_lk_dense_from_topk(bad_tok, p1, 1, n_vocab, d.data()) == -1, "dense rejects out-of-range token");
    int32_t neg_tok[] = {-1};
    CHECK(ts_lk_dense_from_topk(neg_tok, p1, 1, n_vocab, d.data()) == -1, "dense rejects negative token");
    int32_t tok[] = {0};
    float   neg_p[] = {-0.5f};
    CHECK(ts_lk_dense_from_topk(tok, neg_p, 1, n_vocab, d.data()) == -1, "dense rejects negative prob");
    CHECK(ts_lk_dense_from_topk(tok, p1, 11, n_vocab, d.data()) == -1, "dense rejects k > n_vocab");
    CHECK(ts_lk_dense_from_topk(tok, p1, 1, n_vocab, nullptr) == -1, "dense rejects null out");
}

static void test_dense_topk_consistency_exact() {
    // Zero-residual case: dense full-vocab alpha must equal topk alpha exactly.
    // p: token0=0.8, token1=0.2 ; q: token0=0.6, token2=0.4 ; n_vocab=3.
    int32_t pt[] = {0, 1};
    float   pp[] = {0.8f, 0.2f};
    int32_t qt[] = {0, 2};
    float   qp[] = {0.6f, 0.4f};
    const int n_vocab = 3;

    const double alpha_topk = ts_lk_acceptance_rate_topk(
        pt, pp, 2, 1.0, qt, qp, 2, 1.0, n_vocab);

    std::vector<float> pd(n_vocab), qd(n_vocab);
    ts_lk_dense_from_topk(pt, pp, 2, n_vocab, pd.data());
    ts_lk_dense_from_topk(qt, qp, 2, n_vocab, qd.data());
    const double alpha_dense = ts_lk_acceptance_rate(pd.data(), qd.data(), n_vocab);

    CHECK(APPROX(alpha_topk, 0.6, EPS), "consistency-exact topk alpha==0.6");
    CHECK(APPROX(alpha_dense, alpha_topk, 1e-6), "consistency-exact dense alpha == topk alpha");
}

static void test_dense_topk_consistency_residual() {
    // With residual mass the independent dense reconstruction and the joint
    // top-k model partition the long tail differently, so the two acceptance
    // rates differ - but only by an amount bounded by the residual mass.
    int32_t pt[] = {0};
    float   pp[] = {0.5f};                 // p_residual = 0.5
    int32_t qt[] = {0, 1};
    float   qp[] = {0.5f, 0.3f};           // q_residual = 0.2
    const int n_vocab = 10;
    const double p_residual = 0.5, q_residual = 0.2;

    const double alpha_topk = ts_lk_acceptance_rate_topk(
        pt, pp, 1, 0.5, qt, qp, 2, 0.8, n_vocab);

    std::vector<float> pd(n_vocab), qd(n_vocab);
    ts_lk_dense_from_topk(pt, pp, 1, n_vocab, pd.data());
    ts_lk_dense_from_topk(qt, qp, 2, n_vocab, qd.data());
    CHECK(APPROX(dense_sum(pd.data(), n_vocab), 1.0, 1e-5), "consistency-residual p sums to 1");
    CHECK(APPROX(dense_sum(qd.data(), n_vocab), 1.0, 1e-5), "consistency-residual q sums to 1");

    const double alpha_dense = ts_lk_acceptance_rate(pd.data(), qd.data(), n_vocab);
    CHECK(APPROX(alpha_topk, 0.7, EPS), "consistency-residual topk alpha==0.7");
    CHECK(std::fabs(alpha_dense - alpha_topk) <= p_residual + q_residual + 1e-9,
          "consistency-residual |dense - topk| bounded by residual mass");
}

static void test_dense_labels_batch() {
    // Two positions -> matrix [n_vocab, 2] laid out as out[pos*n_vocab + tok].
    int32_t t0[] = {0, 1};
    float   p0[] = {0.7f, 0.3f};
    int32_t t1[] = {2};
    float   p1[] = {0.4f};
    const int32_t * toks[] = {t0, t1};
    const float   * probs[] = {p0, p1};
    const int k[] = {2, 1};
    const int n_vocab = 4, n_pos = 2;

    std::vector<float> m(n_vocab * n_pos);
    int rc = ts_lk_dense_labels_batch(toks, probs, k, n_pos, n_vocab, m.data());
    CHECK(rc == 0, "labels batch rc==0");
    // position 0: exact (full mass), position 1: residual over 3 slots.
    CHECK(APPROX(m[0*n_vocab + 0], 0.7, EPS), "labels batch pos0 tok0");
    CHECK(APPROX(m[0*n_vocab + 1], 0.3, EPS), "labels batch pos0 tok1");
    CHECK(APPROX(m[1*n_vocab + 2], 0.4, EPS), "labels batch pos1 tok2");
    CHECK(APPROX(m[1*n_vocab + 0], 0.6f/3.0f, 1e-6), "labels batch pos1 residual");
    CHECK(APPROX(dense_sum(m.data() + 0*n_vocab, n_vocab), 1.0, 1e-5), "labels batch pos0 sums to 1");
    CHECK(APPROX(dense_sum(m.data() + 1*n_vocab, n_vocab), 1.0, 1e-5), "labels batch pos1 sums to 1");
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
    test_dense_validity();
    test_dense_topk_fidelity();
    test_dense_residual_uniform();
    test_dense_full_coverage();
    test_dense_k0_uniform();
    test_dense_duplicate_tokens();
    test_dense_invalid_input();
    test_dense_topk_consistency_exact();
    test_dense_topk_consistency_residual();
    test_dense_labels_batch();

    printf("lk_loss: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
