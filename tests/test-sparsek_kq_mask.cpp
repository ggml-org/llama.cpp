// tests/test-sparsek_kq_mask.cpp
// (comments in English only)

#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>
#include <iostream>

// ----- helpers -----

static float neg_inf() {
    return -std::numeric_limits<float>::infinity();
}

static void assert_is_neginf(float x) {
#ifndef NDEBUG
    assert(std::isinf(x) && x < 0.0f && "expected -INF");
#else
    (void) x; // silence unused parameter in release builds
#endif
}

static void assert_is_zero(float x) {
#ifndef NDEBUG
    constexpr float eps = 1e-6f;
    assert(std::fabs(x) <= eps && "expected zero");
#else
    (void) x; // silence unused parameter in release builds
#endif
}

// Naive matmul: scores = K [n_kv x d]  *  Q [d x n_cols]
static std::vector<float> matmul_KxQ(
        const std::vector<float> & K,
        const std::vector<float> & Q,
        int n_kv,
        int d,
        int n_cols) {
    std::vector<float> scores(n_kv * n_cols, 0.0f);

    for (int row = 0; row < n_kv; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < d; ++k) {
                float kval = K[row * d + k];
                float qval = Q[k * n_cols + col];
                acc += kval * qval;
            }
            scores[row * n_cols + col] = acc;
        }
    }

    return scores;
}

// For each column, return indices of top-k rows (by descending score).
static std::vector<int> topk_indices_per_column(
        const std::vector<float> & scores,
        int n_kv,
        int n_cols,
        int topk) {
    if (topk < 0) {
        topk = 0;
    }
    if (topk > n_kv) {
        topk = n_kv;
    }

    std::vector<int> topk_idx(n_cols * topk, -1);

    for (int col = 0; col < n_cols; ++col) {
        struct Entry {
            float score;
            int   row;
        };
        std::vector<Entry> entries;
        entries.reserve(n_kv);

        for (int row = 0; row < n_kv; ++row) {
            float s = scores[row * n_cols + col];
            entries.push_back(Entry{s, row});
        }

        std::sort(entries.begin(), entries.end(),
                  [](const Entry & a, const Entry & b) {
                      return a.score > b.score;
                  });

        for (int k = 0; k < topk; ++k) {
            topk_idx[col * topk + k] = entries[k].row;
        }
    }

    return topk_idx;
}

// Base mask helpers.
static std::vector<float> make_base_mask_zero(int n_kv, int n_cols) {
    return std::vector<float>(n_kv * n_cols, 0.0f);
}

static std::vector<float> make_base_mask_neginf(int n_kv, int n_cols) {
    return std::vector<float>(n_kv * n_cols, neg_inf());
}

// SparseK reference:
// - If topk <= 0: passthrough -> return base_mask as-is.
// - If topk > 0: build a pure SparseK mask, independent of base:
//       0      for rows in top-k per column
//       -INF   otherwise
static std::vector<float> build_sparsek_mask_reference(
        const std::vector<float> & base_mask,
        const std::vector<int>   & topk_idx,
        int n_kv,
        int n_cols,
        int topk) {

    const int N = n_kv * n_cols;

    // Passthrough when SparseK is effectively disabled.
    if (topk <= 0) {
        return base_mask;
    }

    std::vector<float> final_mask(N, neg_inf());

    for (int col = 0; col < n_cols; ++col) {
        for (int k = 0; k < topk; ++k) {
            int row = topk_idx[col * topk + k];
            if (row >= 0 && row < n_kv) {
                final_mask[row * n_cols + col] = 0.0f;
            }
        }
    }

    return final_mask;
}

// ----- tests -----

static void test_sparsek_topk_basic() {
    const int n_kv   = 8;
    const int d      = 4;
    const int n_cols = 3;
    const int topk   = 2;

    std::vector<float> K(n_kv * d);
    std::vector<float> Q(d * n_cols);

    // Deterministic but arbitrary values for K and Q.
    for (int row = 0; row < n_kv; ++row) {
        for (int k = 0; k < d; ++k) {
            K[row * d + k] = 0.1f * (row + 1) * (k + 1);
        }
    }
    for (int k = 0; k < d; ++k) {
        for (int col = 0; col < n_cols; ++col) {
            Q[k * n_cols + col] = 0.05f * (k + 1) * (col + 2);
        }
    }

    auto base      = make_base_mask_zero(n_kv, n_cols);
    auto scores    = matmul_KxQ(K, Q, n_kv, d, n_cols);
    auto topk_idx  = topk_indices_per_column(scores, n_kv, n_cols, topk);
    auto final_m   = build_sparsek_mask_reference(base, topk_idx, n_kv, n_cols, topk);

    // Check: in each column exactly topk entries are finite (0), the rest are -INF.
    for (int col = 0; col < n_cols; ++col) {
#ifndef NDEBUG
        int finite_count = 0;
#endif
        for (int row = 0; row < n_kv; ++row) {
            float v = final_m[row * n_cols + col];
            if (std::isinf(v) && v < 0.0f) {
                assert_is_neginf(v);
            } else {
                assert_is_zero(v);
#ifndef NDEBUG
                finite_count++;
#endif
            }
        }
#ifndef NDEBUG
        assert(finite_count == topk && "Expected exactly topk finite entries per column");
#endif
    }
}

static void test_sparsek_topk_with_base_neginf() {
    const int n_kv   = 8;
    const int d      = 4;
    const int n_cols = 2;
    const int topk   = 3;

    std::vector<float> K(n_kv * d);
    std::vector<float> Q(d * n_cols);

    // Deterministic values again.
    for (int row = 0; row < n_kv; ++row) {
        for (int k = 0; k < d; ++k) {
            K[row * d + k] = 0.2f * (row + 1) + 0.01f * (k + 1);
        }
    }
    for (int k = 0; k < d; ++k) {
        for (int col = 0; col < n_cols; ++col) {
            Q[k * n_cols + col] = 0.03f * (k + 1) * (col + 1);
        }
    }

    auto base      = make_base_mask_neginf(n_kv, n_cols);
    auto scores    = matmul_KxQ(K, Q, n_kv, d, n_cols);
    auto topk_idx  = topk_indices_per_column(scores, n_kv, n_cols, topk);
    auto final_m   = build_sparsek_mask_reference(base, topk_idx, n_kv, n_cols, topk);

    // Even with base = -INF everywhere, SparseK should unmask exactly topk entries per column.
    for (int col = 0; col < n_cols; ++col) {
#ifndef NDEBUG
        int finite_count = 0;
#endif
        for (int row = 0; row < n_kv; ++row) {
            float v = final_m[row * n_cols + col];
            if (std::isinf(v) && v < 0.0f) {
                assert_is_neginf(v);
            } else {
                assert_is_zero(v);
#ifndef NDEBUG
                finite_count++;
#endif
            }
        }
#ifndef NDEBUG
        assert(finite_count == topk && "Expected exactly topk finite entries per column");
#endif
    }
}

static void test_sparsek_topk_zero_passthrough() {
    const int n_kv   = 6;
    const int n_cols = 4;
    const int topk   = 0;  // SparseK disabled → passthrough.

    std::vector<float> base(n_kv * n_cols);

    // Build a deterministic pattern: even indices -> 0, odd indices -> -INF.
    for (int i = 0; i < n_kv * n_cols; ++i) {
        if (i % 2 == 0) {
            base[i] = 0.0f;
        } else {
            base[i] = neg_inf();
        }
    }

    // Scores and topk_idx are unused in this case, but we must pass something.
    std::vector<float> dummy_scores; // not used
    std::vector<int> dummy_topk_idx; // not used

    auto final_m = build_sparsek_mask_reference(base, dummy_topk_idx, n_kv, n_cols, topk);

    // Must be exactly equal (by type) to base: 0 stays 0, -INF stays -INF.
    for (int i = 0; i < n_kv * n_cols; ++i) {
        float v_base  = base[i];
        float v_final = final_m[i];

        if (std::isinf(v_base) && v_base < 0.0f) {
            assert_is_neginf(v_final);
        } else {
            assert_is_zero(v_base);
            assert_is_zero(v_final);
        }
    }
}

// ----- main entry -----

int main() {
    test_sparsek_topk_basic();
    test_sparsek_topk_with_base_neginf();
    test_sparsek_topk_zero_passthrough();
    return 0;
}
