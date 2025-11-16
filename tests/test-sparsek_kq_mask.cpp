#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Small helper: assert that a value is -INF
static void assert_is_neginf(float x) {
    assert(std::isinf(x) && x < 0.0f && "expected -INF");
}

// Small helper: assert that a value is exactly 0.0f
static void assert_is_zero(float x) {
    const float eps = 1e-8f;
    assert(std::fabs(x - 0.0f) < eps && "expected 0.0f");
}

// This helper mirrors the SparseK row logic used at the end of
// llama_kv_cache::set_input_kq_mask in src/llama-kv-cache.cpp.
//
// It operates on a single mask row of length n_kv for a specific token index i.
static void apply_sparsek_row(float * row, int64_t n_kv, int token_index, bool causal_attn) {
    // Read SparseK configuration from environment, similar to the production code.
    const char * s = nullptr;

    bool  SPARSEK_ENABLE    = false;
    int   SPARSEK_WIN_LOCAL = 64;
    int   SPARSEK_STRIDE    = 128;
    bool  SPARSEK_EN_LOCAL  = true;
    bool  SPARSEK_EN_STRIDE = true;

    if ((s = std::getenv("LLAMA_SPARSEK_ENABLE"))) {
        SPARSEK_ENABLE = std::atoi(s) != 0;
    }
    if ((s = std::getenv("LLAMA_SPARSEK_WIN"))) {
        SPARSEK_WIN_LOCAL = std::max(0, std::atoi(s));
    }
    if ((s = std::getenv("LLAMA_SPARSEK_STRIDE"))) {
        SPARSEK_STRIDE = std::max(0, std::atoi(s));
    }
    if ((s = std::getenv("LLAMA_SPARSEK_ENABLE_LOCAL"))) {
        SPARSEK_EN_LOCAL = std::atoi(s) != 0;
    }
    if ((s = std::getenv("LLAMA_SPARSEK_ENABLE_STRIDE"))) {
        SPARSEK_EN_STRIDE = std::atoi(s) != 0;
    }

    // Same intended gating as in the SparseK block:
    // if SparseK is disabled, or all patterns are disabled, leave the row unchanged.
    if (!SPARSEK_ENABLE || (!SPARSEK_EN_LOCAL && !SPARSEK_EN_STRIDE)) {
        return;
    }

    std::vector<uint8_t> allow(n_kv, 0);

    // Local window pattern (symmetric around the current token index)
    if (SPARSEK_EN_LOCAL && SPARSEK_WIN_LOCAL > 0) {
        const int j0 = std::max<int>(0, token_index - SPARSEK_WIN_LOCAL);
        const int j1 = std::min<int>(static_cast<int>(n_kv) - 1, token_index + SPARSEK_WIN_LOCAL);
        for (int j = j0; j <= j1; ++j) {
            allow[j] = 1;
        }
    }

    // Stride pattern (backward only for causal, both directions for non-causal)
    if (SPARSEK_EN_STRIDE && SPARSEK_STRIDE > 0) {
        for (int j = token_index; j >= 0; j -= SPARSEK_STRIDE) {
            allow[j] = 1;
        }
        if (!causal_attn) {
            for (int j = token_index; j < static_cast<int>(n_kv); j += SPARSEK_STRIDE) {
                allow[j] = 1;
            }
        }
    }

    // Final mask update: disallowed positions get -INF,
    // allowed positions reset any negative infinity back to 0.0f.
    for (int64_t j = 0; j < n_kv; ++j) {
        if (!allow[j]) {
            row[j] = -INFINITY;
        } else if (std::isinf(row[j]) && row[j] < 0.0f) {
            row[j] = 0.0f;
        }
    }
}

// Pretty-print helper for debugging, not strictly required but useful.
static void dump_row(const char * name, const std::vector<float> & row) {
    std::cout << name << ":";
    for (float v : row) {
        if (std::isinf(v) && v < 0.0f) {
            std::cout << " -INF";
        } else {
            std::cout << " " << v;
        }
    }
    std::cout << "\n";
}

// Scenario 1: SparseK disabled -> row must remain unchanged.
static void test_sparsek_disabled_keeps_row() {
    const int64_t n_kv = 8;
    std::vector<float> row(n_kv, 0.0f);

    // Configure environment: disabled SparseK.
    setenv("LLAMA_SPARSEK_ENABLE", "0", 1);
    setenv("LLAMA_SPARSEK_WIN", "2", 1);
    setenv("LLAMA_SPARSEK_STRIDE", "2", 1);
    setenv("LLAMA_SPARSEK_ENABLE_LOCAL", "1", 1);
    setenv("LLAMA_SPARSEK_ENABLE_STRIDE", "1", 1);

    apply_sparsek_row(row.data(), n_kv, /*token_index=*/3, /*causal_attn=*/true);

    for (int64_t j = 0; j < n_kv; ++j) {
        assert_is_zero(row[j]);
    }
}

// Scenario 2: Local window only, causal attention.
// With n_kv = 8, token_index = 3 and window = 1, we expect positions {2,3,4} to be allowed.
static void test_sparsek_local_window_only() {
    const int64_t n_kv = 8;
    std::vector<float> row(n_kv, -INFINITY);

    setenv("LLAMA_SPARSEK_ENABLE", "1", 1);
    setenv("LLAMA_SPARSEK_WIN", "1", 1);
    setenv("LLAMA_SPARSEK_STRIDE", "0", 1);
    setenv("LLAMA_SPARSEK_ENABLE_LOCAL", "1", 1);
    setenv("LLAMA_SPARSEK_ENABLE_STRIDE", "0", 1);

    const int token_index = 3;
    apply_sparsek_row(row.data(), n_kv, token_index, /*causal_attn=*/true);

    // Optional debug print:
    // dump_row("local_window_only", row);

    for (int64_t j = 0; j < n_kv; ++j) {
        bool should_allow = (j == 2 || j == 3 || j == 4);
        if (should_allow) {
            assert_is_zero(row[j]);
        } else {
            assert_is_neginf(row[j]);
        }
    }
}

// Scenario 3: Stride only, causal attention.
// With n_kv = 8, token_index = 5, stride = 2, causal:
// allowed positions should be {5, 3, 1}.
static void test_sparsek_stride_causal() {
    const int64_t n_kv = 8;
    std::vector<float> row(n_kv, -INFINITY);

    setenv("LLAMA_SPARSEK_ENABLE", "1", 1);
    setenv("LLAMA_SPARSEK_WIN", "0", 1);
    setenv("LLAMA_SPARSEK_STRIDE", "2", 1);
    setenv("LLAMA_SPARSEK_ENABLE_LOCAL", "0", 1);
    setenv("LLAMA_SPARSEK_ENABLE_STRIDE", "1", 1);

    const int token_index = 5;
    apply_sparsek_row(row.data(), n_kv, token_index, /*causal_attn=*/true);

    // dump_row("stride_causal", row);

    for (int64_t j = 0; j < n_kv; ++j) {
        bool should_allow = (j == 1 || j == 3 || j == 5);
        if (should_allow) {
            assert_is_zero(row[j]);
        } else {
            assert_is_neginf(row[j]);
        }
    }
}

// Scenario 4: Stride only, non-causal.
// With n_kv = 8, token_index = 5, stride = 2, non-causal:
// allowed positions should be {1, 3, 5, 7}.
static void test_sparsek_stride_noncausal() {
    const int64_t n_kv = 8;
    std::vector<float> row(n_kv, -INFINITY);

    setenv("LLAMA_SPARSEK_ENABLE", "1", 1);
    setenv("LLAMA_SPARSEK_WIN", "0", 1);
    setenv("LLAMA_SPARSEK_STRIDE", "2", 1);
    setenv("LLAMA_SPARSEK_ENABLE_LOCAL", "0", 1);
    setenv("LLAMA_SPARSEK_ENABLE_STRIDE", "1", 1);

    const int token_index = 5;
    apply_sparsek_row(row.data(), n_kv, token_index, /*causal_attn=*/false);

    // dump_row("stride_noncausal", row);

    for (int64_t j = 0; j < n_kv; ++j) {
        bool should_allow = (j == 1 || j == 3 || j == 5 || j == 7);
        if (should_allow) {
            assert_is_zero(row[j]);
        } else {
            assert_is_neginf(row[j]);
        }
    }
}

// Scenario 5: Combined local window + stride.
// This checks that both patterns are OR'ed together.
static void test_sparsek_combined_patterns() {
    const int64_t n_kv = 16;
    std::vector<float> row(n_kv, -INFINITY);

    setenv("LLAMA_SPARSEK_ENABLE", "1", 1);
    setenv("LLAMA_SPARSEK_WIN", "1", 1);
    setenv("LLAMA_SPARSEK_STRIDE", "4", 1);
    setenv("LLAMA_SPARSEK_ENABLE_LOCAL", "1", 1);
    setenv("LLAMA_SPARSEK_ENABLE_STRIDE", "1", 1);

    const int token_index = 8;
    apply_sparsek_row(row.data(), n_kv, token_index, /*causal_attn=*/true);

    // Local window (radius 1) -> {7,8,9}
    // Stride (4, causal, backward) from 8 -> {8,4,0}
    // Union -> {0,4,7,8,9}
    for (int64_t j = 0; j < n_kv; ++j) {
        bool should_allow = (j == 0 || j == 4 || j == 7 || j == 8 || j == 9);
        if (should_allow) {
            assert_is_zero(row[j]);
        } else {
            assert_is_neginf(row[j]);
        }
    }
}

int main() {
    std::cout << "Running SparseK KQ mask row tests...\n";

    test_sparsek_disabled_keeps_row();
    test_sparsek_local_window_only();
    test_sparsek_stride_causal();
    test_sparsek_stride_noncausal();
    test_sparsek_combined_patterns();

    std::cout << "All SparseK KQ mask tests passed.\n";
    return 0;
}
