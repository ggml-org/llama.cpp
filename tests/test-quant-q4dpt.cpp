// test-quant-q4dpt.cpp
// Quantization accuracy test for Q4_DPT (IQ4_NL with per-tensor learned int8 levels).

#include "ggml-backend.h"
#include "ggml.h"
#include <cmath>
#include <cstdint>

extern "C" {
void    q4dpt_train_levels(const float * data, int64_t nrow, int64_t n_per_row,
                            const float * imatrix, int8_t levels_out[16]);
void    q4dpt_set_levels(const int8_t * levels);
size_t  quantize_q4_dpt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
}

#define Q4DPT_N_LEVELS 16

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
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t) n_per_row);
    }
    return rmse(data, dq.data(), nrow * n_per_row);
}

// Run Q4_DPT: train levels, set, quantize, dequantize, return RMSE
static float q4dpt_rmse_actual(const float * data, size_t nrow, size_t n_per_row) {
    int8_t levels[Q4DPT_N_LEVELS];
    q4dpt_train_levels(data, (int64_t) nrow, (int64_t) n_per_row, nullptr, levels);
    q4dpt_set_levels(levels);
    const size_t         rs = ggml_row_size(GGML_TYPE_Q4_DPT, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float>   dq(nrow * n_per_row);
    quantize_q4_dpt(data, qb.data(), (int64_t) nrow, (int64_t) n_per_row, nullptr);
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q4_DPT);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t) n_per_row);
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
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    ggml_backend_load_all();

    std::mt19937          rng(0xdeadbeef);
    std::vector<TestCase> cases;

    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.02) 64x4096";
        tc.nrow      = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.02f);
        for (auto & v : tc.data) { v = nd(rng); }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.05) 32x8192";
        tc.nrow      = 32;
        tc.n_per_row = 8192;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto & v : tc.data) { v = nd(rng); }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.01) 128x2048";
        tc.nrow      = 128;
        tc.n_per_row = 2048;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.01f);
        for (auto & v : tc.data) { v = nd(rng); }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Laplace(0,0.01) 64x4096";
        tc.nrow      = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::exponential_distribution<float> ed(100.f);
        std::bernoulli_distribution          sgnd(0.5f);
        for (auto & v : tc.data) {
            v = ed(rng);
            if (sgnd(rng)) { v = -v; }
        }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Uniform(-0.1,0.1) 64x4096";
        tc.nrow      = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::uniform_real_distribution<float> ud(-0.1f, 0.1f);
        for (auto & v : tc.data) { v = ud(rng); }
        cases.push_back(std::move(tc));
    }

    printf("Q4_DPT quantization accuracy vs IQ4_NL (lower=better; 1.00=IQ4_NL baseline)\n\n");
    printf("%-28s  %7s  %7s  %7s\n", "Test", "IQ4_NL", "Q4_DPT", "Ratio");
    printf("%-28s  %7s  %7s  %7s\n", "----------------------------", "-------", "-------", "-------");

    int  tc_idx   = 0;
    bool any_fail = false;
    for (auto & tc : cases) {
        fprintf(stderr, "[%u/%zu] %s... ", ++tc_idx, cases.size(), tc.name.c_str());
        fflush(stderr);

        float iq4nl_rmse = std_quant_rmse(GGML_TYPE_IQ4_NL, tc.data.data(), tc.nrow, tc.n_per_row);
        float q4dpt_rmse = q4dpt_rmse_actual(tc.data.data(), tc.nrow, tc.n_per_row);

        fprintf(stderr, "done\n");

        float ratio = q4dpt_rmse / iq4nl_rmse;
        // Q4_DPT should be competitive with or better than IQ4_NL
        bool  ok    = (ratio < 1.2f);
        if (!ok) { any_fail = true; }
        printf("%-28s  %7.6f  %7.6f  %7.4f%s\n",
               tc.name.c_str(), iq4nl_rmse, q4dpt_rmse, ratio, ok ? "" : "  FAIL");
        fflush(stdout);
    }

    if (any_fail) {
        fprintf(stderr, "\nFAIL: Q4_DPT RMSE significantly worse than IQ4_NL on some test cases\n");
        return 1;
    }

    printf("\nPASS\n");
    return 0;
}
