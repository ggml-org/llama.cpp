#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
#include <cstdint>


// Simple helpers for readability in assertions
static void assert_is_neginf(float x) {
    // Expect strict -INF (or any -inf value)
    assert(std::isinf(x) && x < 0.0f && "expected -INF");
}

static void assert_is_zero(float x) {
    assert(std::fabs(x - 0.0f) < eps && "expected 0.0f");
}

// This helper mirrors the SparseK block inside llama_kv_cache::set_input_kq_mask:
//
//     if (!SPARSEK_ENABLE || (!SPARSEK_EN_LOCAL && !SPARSEK_EN_STRIDE)) {
//         // do nothing – keep original KQ mask
//     } else {
//         for each row i:
//             std::vector<uint8_t> allow(n_kv, 0);
//             if (SPARSEK_EN_LOCAL && SPARSEK_WIN_LOCAL > 0) { ... allow[j] = 1; }
//             if (SPARSEK_EN_STRIDE && SPARSEK_STRIDE > 0) { ... allow[j] = 1; }
//             for j:
//                 if (!allow[j]) {
//                     row[j] = -INFINITY;
//                 } else if (std::isinf(row[j]) && row[j] < 0.0f) {
//                     row[j] = 0.0f;
//                 }
//         }
//     }
//
// כאן אנחנו בודקים את הלוגיקה הזו על שורה אחת ("row i") במטריצה של KQ-mask.
static std::vector<float> apply_sparsek_to_base_row(
        const std::vector<float> & base_row,
        bool   enable_sparsek,
        bool   causal_attn,
        int    win_local,
        int    stride,
        bool   en_local,
        bool   en_stride,
        int    i,          // row index (token index within stream)
        int    n_kv) {

    std::vector<float> row = base_row;

    if (!enable_sparsek || (!en_local && !en_stride)) {
        // When SparseK is disabled, we must return the base mask unchanged.
        return row;
    }

    std::vector<uint8_t> allow(n_kv, 0);

    // Local window: mark tokens in [i - win_local, i + win_local] as allowed
    if (en_local && win_local > 0) {
        const int j0 = std::max(0,          i - win_local);
        const int j1 = std::min(n_kv - 1,   i + win_local);
        for (int j = j0; j <= j1; ++j) {
            allow[j] = 1;
        }
    }

    // Stride: mark tokens every "stride" steps backward, and optionally forward if non-causal
    if (en_stride && stride > 0) {
        for (int j = i; j >= 0; j -= stride) {
            allow[j] = 1;
        }
        if (!causal_attn) {
            for (int j = i; j < n_kv; j += stride) {
                allow[j] = 1;
            }
        }
    }

    // Final SparseK rule:
    // - if allow[j] == 0 → force -INF
    // - else if row[j] is already -INF → reset to 0 (so "allowed" entries are neutral in softmax)
    for (int j = 0; j < n_kv; ++j) {
        if (!allow[j]) {
            row[j] = -INFINITY;
        } else if (std::isinf(row[j]) && row[j] < 0.0f) {
            row[j] = 0.0f;
        }
    }

    return row;
}

// Convenience: build a base row with all zeros (no masking yet).
static std::vector<float> make_base_row(int n_kv) {
    return std::vector<float>(n_kv, 0.0f);
}

// --- Test cases ----------------------------------------------------------

// 1) Local window only: verify that only the band around i remains non -INF
static void test_local_window_only() {
    const int n_kv = 8;
    const int i    = 4;
    const int win  = 2;

    std::vector<float> base = make_base_row(n_kv);

    std::vector<float> row = apply_sparsek_to_base_row(
        base,
        /*enable_sparsek=*/true,
        /*causal_attn=*/true,
        /*win_local=*/win,
        /*stride=*/0,
        /*en_local=*/true,
        /*en_stride=*/false,
        /*i=*/i,
        /*n_kv=*/n_kv);

    // Expected allowed indices: [i - win, ..., i + win] → [2,3,4,5,6]
    for (int j = 0; j < n_kv; ++j) {
        bool should_be_allowed = (j >= i - win && j <= i + win);
        if (should_be_allowed) {
            assert_is_zero(row[j]);
        } else {
            assert_is_neginf(row[j]);
        }
    }

    std::printf("SparseK test: local window only – OK\n");
}

// 2) Stride only: verify symmetric backward steps, forward only if non-causal == false here
static void test_stride_only_causal() {
    const int n_kv   = 10;
    const int i      = 7;
    const int stride = 3;

    std::vector<float> base(n_kv, 0.0f);

    std::vector<float> row = apply_sparsek_to_base_row(
        base,
        /*enable_sparsek=*/true,
        /*causal_attn=*/true,
        /*win_local=*/0,
        /*stride=*/stride,
        /*en_local=*/false,
        /*en_stride=*/true,
        /*i=*/i,
        /*n_kv=*/n_kv);

    // For causal_attn = true we only walk backwards: i, i-stride, i-2*stride,...
    std::vector<uint8_t> expected_allow(n_kv, 0);
    for (int j = i; j >= 0; j -= stride) {
        expected_allow[j] = 1;
    }

    for (int j = 0; j < n_kv; ++j) {
        if (expected_allow[j]) {
            assert_is_zero(row[j]);
        } else {
            assert_is_neginf(row[j]);
        }
    }

    std::printf("SparseK test: stride only (causal) – OK\n");
}

// 3) Combined: local window + stride – any "allowed" wins, others must be -INF
static void test_local_plus_stride() {
    const int n_kv   = 16;
    const int i      = 8;
    const int win    = 2;
    const int stride = 5;

    std::vector<float> base(n_kv, 0.0f);

    std::vector<float> row = apply_sparsek_to_base_row(
        base,
        /*enable_sparsek=*/true,
        /*causal_attn=*/false,
        /*win_local=*/win,
        /*stride=*/stride,
        /*en_local=*/true,
        /*en_stride=*/true,
        /*i=*/i,
        /*n_kv=*/n_kv);

    // Build expected "allow" mask exactly like the production logic
    std::vector<uint8_t> expected_allow(n_kv, 0);

    // local window
    {
        const int j0 = std::max(0,         i - win);
        const int j1 = std::min(n_kv - 1,  i + win);
        for (int j = j0; j <= j1; ++j) {
            expected_allow[j] = 1;
        }
    }

    // stride (non-causal: both directions)
    {
        for (int j = i; j >= 0; j -= stride) {
            expected_allow[j] = 1;
        }
        for (int j = i; j < n_kv; j += stride) {
            expected_allow[j] = 1;
        }
    }

    for (int j = 0; j < n_kv; ++j) {
        if (expected_allow[j]) {
            assert_is_zero(row[j]);
        } else {
            assert_is_neginf(row[j]);
        }
    }

    std::printf("SparseK test: local + stride (non-causal) – OK\n");
}

// 4) Disabled: when SparseK is not enabled, base mask must remain unchanged.
static void test_sparsek_disabled() {
    const int n_kv = 6;
    std::vector<float> base(n_kv, 0.0f);

    // We intentionally pass enable_sparsek = false
    std::vector<float> row = apply_sparsek_to_base_row(
        base,
        /*enable_sparsek=*/false,
        /*causal_attn=*/true,
        /*win_local=*/4,
        /*stride=*/3,
        /*en_local=*/true,
        /*en_stride=*/true,
        /*i=*/3,
        /*n_kv=*/n_kv);

    // Must be identical to base: all zeros, no -INF introduced.
    for (int j = 0; j < n_kv; ++j) {
        assert_is_zero(row[j]);
    }

    std::printf("SparseK test: disabled path keeps base mask – OK\n");
}

// 5) Base row pre-filled with -INF on allowed positions: SparseK must reset them to 0
//    so that "allowed" entries are neutral in softmax.
static void test_reset_inf_to_zero_for_allowed() {
    const int n_kv = 8;
    const int i    = 3;
    const int win  = 1;

    // Base row has -INF everywhere
    std::vector<float> base(n_kv, -INFINITY);

    std::vector<float> row = apply_sparsek_to_base_row(
        base,
        /*enable_sparsek=*/true,
        /*causal_attn=*/true,
        /*win_local=*/win,
        /*stride=*/0,
        /*en_local=*/true,
        /*en_stride=*/false,
        /*i=*/i,
        /*n_kv=*/n_kv);

    for (int j = 0; j < n_kv; ++j) {
        bool should_be_allowed = (j >= i - win && j <= i + win);
        if (should_be_allowed) {
            // allowed entries must be reset to 0 even if they started as -INF
            assert_is_zero(row[j]);
        } else {
            assert_is_neginf(row[j]);
        }
    }

    std::printf("SparseK test: allowed positions reset -INF → 0 – OK\n");
}

int main() {
    std::printf("Running SparseK KQ mask CPU tests...\n");

    test_local_window_only();
    test_stride_only_causal();
    test_local_plus_stride();
    test_sparsek_disabled();
    test_reset_inf_to_zero_for_allowed();

    std::printf("All SparseK KQ mask tests passed.\n");
    return 0;
}
