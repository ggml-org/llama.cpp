// test-quant-q2dpt.cpp
// Quantization accuracy test for Q2_DPT (2-bit with per-tensor learned int8 levels).
// Compares multiple per-block scale-search strategies against Q2_K baseline.

#include "ggml-backend.h"
#include "ggml.h"
#include <cmath>
#include <cstdint>

extern "C" {
void    q2dpt_train_levels(const float * data, int64_t nrow, int64_t n_per_row,
                            const float * imatrix, int8_t levels_out[4]);
void    q2dpt_set_levels(const int8_t * levels);
void    q2dpt_set_quant_strategy(int s);
size_t  quantize_q2_dpt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
}

#define Q2DPT_N_LEVELS 4

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float rmse(const float * a, const float * b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double) a[i] - (double) b[i];
        s += d * d;
    }
    return (float) std::sqrt(s / (double) n);
}

static float std_quant_rmse(ggml_type type, const float * data, size_t nrow, size_t n_per_row) {
    const size_t         rs = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float>   dq(nrow * n_per_row);
    ggml_quantize_chunk(type, data, qb.data(), 0, nrow, n_per_row, nullptr);
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t) n_per_row, nullptr);
    }
    return rmse(data, dq.data(), nrow * n_per_row);
}

// Run Q2_DPT with a given strategy: train levels, set, quantize, dequantize, return RMSE
static float q2dpt_rmse_with_strategy(const float * data, size_t nrow, size_t n_per_row, int strategy) {
    std::vector<float> imatrix_train(nrow * n_per_row, 1.0f);
    std::vector<float> imatrix_quant(n_per_row, 1.0f);

    int8_t levels[Q2DPT_N_LEVELS];
    q2dpt_train_levels(data, (int64_t) nrow, (int64_t) n_per_row, imatrix_train.data(), levels);
    q2dpt_set_levels(levels);
    q2dpt_set_quant_strategy(strategy);

    const size_t         rs = ggml_row_size(GGML_TYPE_Q2_DPT, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float>   dq(nrow * n_per_row);
    quantize_q2_dpt(data, qb.data(), (int64_t) nrow, (int64_t) n_per_row, imatrix_quant.data());
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_DPT);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t) n_per_row, levels);
    }
    return rmse(data, dq.data(), nrow * n_per_row);
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------
struct TestCase {
    std::string        name;
    std::vector<float> data;
    size_t             nrow, n_per_row;
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(void) {
    ggml_backend_load_all();

    std::vector<TestCase> cases;
    std::mt19937 gen(42);

    // Test case 1: Gaussian(0, 0.02) 64x4096
    {
        auto & tc = cases.emplace_back();
        tc.name = "Gauss(0,0.02) 64x4096";
        tc.nrow = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto & v : tc.data) v = dist(gen);
    }

    // Test case 2: Laplace(0, 0.01) 64x4096
    {
        auto & tc = cases.emplace_back();
        tc.name = "Laplace(0,0.01) 64x4096";
        tc.nrow = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::piecewise_constant_distribution<float> dist(
            2, -1.0f, 1.0f, [](float x){ return std::exp(-std::abs(x)/0.01f); });
        for (auto & v : tc.data) v = dist(gen);
    }

    // Test case 3: Uniform(-0.1, 0.1) 64x4096
    {
        auto & tc = cases.emplace_back();
        tc.name = "Uniform(-0.1,0.1) 64x4096";
        tc.nrow = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto & v : tc.data) v = dist(gen);
    }

    // Strategies to test: { bitmask, label }
    struct Strategy { int mask; const char * label; };
    Strategy strategies[] = {
        { 0x1,  "A:lvl-anchor"   },  // 4 level-anchored CD starts
        { 0x2,  "B:bnd-sweep"    },  // boundary-crossing sweep + CD
        { 0x4,  "C:dual-extreme" },  // max_val + min_val anchors + CD
        { 0x8,  "D:elem-anchor"  },  // element-anchor scan + CD
        { 0x10, "E:brute-force"  },  // exhaustive monotone partition
        { 0x3,  "A+B"            },  // best of A and B combined
        { 0x1F, "A+B+C+D+E"     },  // everything
    };
    const int n_strat = (int)(sizeof(strategies) / sizeof(strategies[0]));

    // Header
    printf("Q2_DPT per-block strategy comparison (ratio vs Q2_K; lower=better)\n\n");
    printf("%-26s  %8s", "Test", "Q2_K");
    for (int s = 0; s < n_strat; ++s)
        printf("  %14s", strategies[s].label);
    printf("\n");
    printf("%-26s  %8s", "--------------------------", "--------");
    for (int s = 0; s < n_strat; ++s)
        printf("  %14s", "--------------");
    printf("\n");

    for (size_t i = 0; i < cases.size(); ++i) {
        auto & tc = cases[i];
        printf("%-26s", tc.name.c_str());
        fflush(stdout);

        float rmse_q2_k = std_quant_rmse(GGML_TYPE_Q2_K, tc.data.data(), tc.nrow, tc.n_per_row);
        printf("  %8.6f", rmse_q2_k);
        fflush(stdout);

        for (int s = 0; s < n_strat; ++s) {
            float r = q2dpt_rmse_with_strategy(tc.data.data(), tc.nrow, tc.n_per_row, strategies[s].mask);
            printf("  %14.4f", r / rmse_q2_k);
            fflush(stdout);
        }
        printf("\n");
    }

    printf("\n");
    return 0;
}
