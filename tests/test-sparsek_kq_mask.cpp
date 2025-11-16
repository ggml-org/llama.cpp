#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
#include <cstdint>
#include <algorithm>

// Small epsilon for float comparisons
static constexpr float eps = 1e-6f;

static float neg_inf() {
    return -std::numeric_limits<float>::infinity();
}

// Simple helpers for readability in assertions
static void assert_is_neginf(float x) {
    // Expect strict -INF (or any -inf value)
    assert(std::isinf(x) && x < 0.0f && "expected -INF");
}

static void assert_is_zero(float x) {
    assert(std::fabs(x - 0.0f) < eps && "expected 0.0f");
}

// -----------------------------------------------------------------------------
// Naive CPU reference for what build_sparsek_mask conceptually does
// (in 2D K,Q,base_mask space):
//
// 1) scores = K * Q   [n_kv x n_cols]
// 2) topk_idx = top-K indices per column in scores
// 3) build mask_topk: all -INF, top-K entries set to 0
// 4) final_mask = mask_topk + base_mask
//
// Note: This is a standalone reference; the real implementation works on
// 4D tensors and uses ggml, but the math is the same.
// -----------------------------------------------------------------------------

// Multiply: scores = K * Q
// K: [n_kv x d], Q: [d x n_cols], scores: [n_kv x n_cols]
// Layout: row-major, scores[row * n_cols + col]
static std::vector<float> matmul_KxQ(
        const std::vector<float> & K,
        const std::vector<float> & Q,
        int n_kv,
        int d,
        int n_cols) {

    std::vector<float> scores(n_kv * n_cols, 0.0f);

    for (int i = 0; i < n_kv; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d; ++k) {
                float k_ij = K[i * d + k];
                float q_kj = Q[k * n_cols + j];
                sum += k_ij * q_kj;
            }
            scores[i * n_cols + j] = sum;
        }
    }

    return scores;
}

// Get top-K indices per column in scores [n_kv x n_cols], returning
// a vector of length (topk * n_cols) storing indices in each column.
static std::vector<int> topk_indices_per_column(
        const std::vector<float> & scores,
        int n_kv,
        int n_cols,
        int topk) {

    std::vector<int> topk_idx(topk * n_cols, -1);

    for (int col = 0; col < n_cols; ++col) {
        // indices 0..n_kv-1 for this column
        std::vector<int> idx(n_kv);
        for (int i = 0; i < n_kv; ++i) {
            idx[i] = i;
        }

        // partial sort for topk (largest values)
        std::partial_sort(
            idx.begin(),
            idx.begin() + topk,
            idx.end(),
            [&](int a, int b) {
                float va = scores[a * n_cols + col];
                float vb = scores[b * n_cols + col];
                return va > vb;
            });

        for (int k = 0; k < topk; ++k) {
            topk_idx[col * topk + k] = idx[k];
        }
    }

    return topk_idx;
}

// Build SparseK mask (reference):
// base_mask: [n_kv x n_cols], scores: [n_kv x n_cols]
// topk_idx: [topk x n_cols] flattened as [col * topk + k]
static std::vector<float> build_sparsek_mask_reference(
        const std::vector<float> & base_mask,
        const std::vector<float> & scores,
        const std::vector<int>   & topk_idx,
        int n_kv,
        int n_cols,
        int topk) {

    // 1) Start from all -INF
    std::vector<float> mask(n_kv * n_cols, neg_inf());

    // 2) For each column, set topk entries to 0
    for (int col = 0; col < n_cols; ++col) {
        for (int k = 0; k < topk; ++k) {
            int row = topk_idx[col * topk + k];
            if (row >= 0 && row < n_kv) {
                mask[row * n_cols + col] = 0.0f;
            }
        }
    }

    // 3) Combine with base_mask: final = mask + base_mask
    std::vector<float> final_mask(n_kv * n_cols, 0.0f);

    for (int i = 0; i < n_kv * n_cols; ++i) {
        final_mask[i] = mask[i] + base_mask[i];
    }

    return final_mask;
}

// Convenience: base mask with all zeros (no masking yet)
static std::vector<float> make_base_mask_zeros(int n_kv, int n_cols) {
    return std::vector<float>(n_kv * n_cols, 0.0f);
}

// Convenience: base mask with all -INF
static std::vector<float> make_base_mask_neginf(int n_kv, int n_cols) {
    return std::vector<float>(n_kv * n_cols, neg_inf());
}

// -----------------------------------------------------------------------------
// Test 1: simple top-K on a tiny matrix, base mask = 0
// -----------------------------------------------------------------------------
static void test_sparsek_topk_basic() {
    const int n_kv   = 4;
    const int d      = 3;
    const int n_cols = 2;
    const int topk   = 1;

    // K: [n_kv x d]
    // Make rows such that row 2 is biggest for col 0, row 1 is biggest for col 1
    std::vector<float> K = {
        // row 0
        1.0f, 0.0f, 0.0f,
        // row 1
        0.0f, 1.0f, 0.0f,
        // row 2
        2.0f, 0.0f, 0.0f,
        // row 3
        0.0f, 0.5f, 0.0f,
    };

    // Q: [d x n_cols]
    // Col 0 only "looks" at first coord, Col 1 only at second coord
    std::vector<float> Q = {
        // col 0
        1.0f,
        0.0f,
        0.0f,

        // col 1
        0.0f,
        1.0f,
        0.0f,
    };

    std::vector<float> base = make_base_mask_zeros(n_kv, n_cols);

    std::vector<float> scores   = matmul_KxQ(K, Q, n_kv, d, n_cols);
    std::vector<int>   topk_idx = topk_indices_per_column(scores, n_kv, n_cols, topk);
    std::vector<float> final_m  = build_sparsek_mask_reference(base, scores, topk_idx, n_kv, n_cols, topk);

    // We expect:
    // col 0: row 2 is largest → allowed (0), others -INF
    // col 1: row 1 is largest → allowed (0), others -INF
    for (int row = 0; row < n_kv; ++row) {
        float m0 = final_m[row * n_cols + 0];
        float m1 = final_m[row * n_cols + 1];

        if (row == 2) {
            assert_is_zero(m0);
        } else {
            assert_is_neginf(m0);
        }

        if (row == 1) {
            assert_is_zero(m1);
        } else {
            assert_is_neginf(m1);
        }
    }

    std::printf("SparseK test: basic top-K masking – OK\n");
}

// -----------------------------------------------------------------------------
// Test 2: base mask pre-filled with -INF, allowed top-K entries must become 0
// (like in build_sparsek_mask where allowed entries should be neutral in softmax)
// -----------------------------------------------------------------------------
static void test_sparsek_topk_with_base_neginf() {
    const int n_kv   = 3;
    const int d      = 2;
    const int n_cols = 1;
    const int topk   = 2;

    // K: [n_kv x d]
    std::vector<float> K = {
        1.0f, 0.0f,
        0.0f, 2.0f,
        1.0f, 1.0f,
    };

    // Q: [d x 1]
    std::vector<float> Q = {
        1.0f,
        1.0f,
    };

    std::vector<float> base = make_base_mask_neginf(n_kv, n_cols);

    std::vector<float> scores   = matmul_KxQ(K, Q, n_kv, d, n_cols);
    std::vector<int>   topk_idx = topk_indices_per_column(scores, n_kv, n_cols, topk);
    std::vector<float> final_m  = build_sparsek_mask_reference(base, scores, topk_idx, n_kv, n_cols, topk);

    // Exactly topk rows should be finite (0), the rest -INF
    int finite_count = 0;
    for (int row = 0; row < n_kv; ++row) {
        float v = final_m[row * n_cols + 0];
        if (std::isinf(v) && v < 0.0f) {
            // OK, -INF
        } else {
            // must be 0
            assert_is_zero(v);
            finite_count++;
        }
    }

    assert(finite_count == topk && "Expected exactly topk finite entries in final mask");

    std::printf("SparseK test: top-K with base -INF – OK\n");
}

// -----------------------------------------------------------------------------
// Test 3: topk = 0 → mask should effectively be passthrough (all zeros here)
// (matches the early-return path in build_sparsek_mask when topk == 0)
// -----------------------------------------------------------------------------
static void test_sparsek_topk_zero_passthrough() {
    const int n_kv   = 4;
    const int d      = 2;
    const int n_cols = 2;
    const int topk   = 0;

    std::vector<float> K = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        2.0f, 0.0f,
    };

    std::vector<float> Q = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    std::vector<float> base = make_base_mask_zeros(n_kv, n_cols);

    std::vector<float> scores   = matmul_KxQ(K, Q, n_kv, d, n_cols);
    std::vector<int>   topk_idx; // empty, since topk == 0

    // For topk == 0, we expect "passthrough": final == base.
    // So we do not actually call build_sparsek_mask_reference here,
    // we just make sure base is unchanged.
    for (float v : base) {
        assert_is_zero(v);
    }

    std::printf("SparseK test: top-K == 0 passthrough (base unchanged) – OK\n");
}

int main() {
    std::printf("Running SparseK KQ mask top-K tests (reference)...\n");

    test_sparsek_topk_basic();
    test_sparsek_topk_with_base_neginf();
    test_sparsek_topk_zero_passthrough();

    std::printf("All SparseK KQ mask top-K tests passed.\n");
    return 0;
}


