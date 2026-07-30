//
// test_moe_shapes.cpp
//
// Shape-validation tests for ts_quantize_3d (MoE expert path). Exercises
// realistic expert counts and matrix dimensions, including edge cases
// around Tile640 page boundaries, and verifies that n_experts == 1
// produces output identical to ts_quantize_2d.
//

#include "tessera-quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            g_failures++;                                    \
        }                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// deterministic PRNG (xorshift32) + Box-Muller Gaussian
// ---------------------------------------------------------------------------

static uint32_t rng_state;

static void rng_seed(uint32_t s) {
    rng_state = s ? s : 1;
}

static uint32_t rng_next(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static float rng_uniform(void) {
    return (float)(rng_next() >> 8) / (float)(1u << 24);  // [0, 1)
}

static float rng_gaussian(void) {
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * (float)M_PI * u2);
}

// Fill a buffer with N(0, sigma^2) samples from the current PRNG state.
static void fill_gaussian(std::vector<float> & v, float sigma) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = rng_gaussian() * sigma;
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static int64_t pages_per_row(int64_t in_dim) {
    return (in_dim + 639) / 640;
}

static ts_quant_params_2d default_params(void) {
    ts_quant_params_2d p = {};
    p.alpha          = 0.0f;
    p.clip           = 0.0f;
    p.max_outliers   = 0;
    p.outlier_thresh = 0.0f;
    p.use_imatrix    = false;
    p.use_septq      = false;
    p.awq_grid       = 20;
    p.seed           = 42;
    return p;
}

// Verify all size invariants and MSE sanity for one expert result.
static void check_result(const ts_quant_result_2d & r,
                         int64_t out_dim, int64_t in_dim,
                         const char * label) {
    const int64_t pages = pages_per_row(in_dim);
    char buf[256];

    snprintf(buf, sizeof(buf), "%s: packed size", label);
    CHECK(r.packed.size() == (size_t)(out_dim * pages * 32), buf);

    snprintf(buf, sizeof(buf), "%s: page_scales size", label);
    CHECK(r.page_scales.size() == (size_t)(out_dim * pages), buf);

    snprintf(buf, sizeof(buf), "%s: lane_scales size", label);
    CHECK(r.lane_scales.size() == (size_t)(out_dim * pages * 32), buf);

    snprintf(buf, sizeof(buf), "%s: outlier_row_offsets size", label);
    CHECK(r.outlier_row_offsets.size() == (size_t)(out_dim + 1), buf);

    snprintf(buf, sizeof(buf), "%s: MSE finite and < 1.0", label);
    CHECK(std::isfinite(r.mse) && r.mse < 1.0f, buf);
}

// Byte-for-byte comparison of two quantize results.
static void check_results_identical(const ts_quant_result_2d & a,
                                    const ts_quant_result_2d & b,
                                    const char * label) {
    char buf[256];

    snprintf(buf, sizeof(buf), "%s: packed identical", label);
    CHECK(a.packed.size() == b.packed.size() &&
          std::memcmp(a.packed.data(), b.packed.data(),
                      a.packed.size() * sizeof(uint32_t)) == 0, buf);

    snprintf(buf, sizeof(buf), "%s: page_scales identical", label);
    CHECK(a.page_scales.size() == b.page_scales.size() &&
          std::memcmp(a.page_scales.data(), b.page_scales.data(),
                      a.page_scales.size() * sizeof(uint16_t)) == 0, buf);

    snprintf(buf, sizeof(buf), "%s: lane_scales identical", label);
    CHECK(a.lane_scales.size() == b.lane_scales.size() &&
          std::memcmp(a.lane_scales.data(), b.lane_scales.data(),
                      a.lane_scales.size() * sizeof(int8_t)) == 0, buf);

    snprintf(buf, sizeof(buf), "%s: outlier_row_offsets identical", label);
    CHECK(a.outlier_row_offsets.size() == b.outlier_row_offsets.size() &&
          std::memcmp(a.outlier_row_offsets.data(), b.outlier_row_offsets.data(),
                      a.outlier_row_offsets.size() * sizeof(int32_t)) == 0, buf);

    snprintf(buf, sizeof(buf), "%s: outlier_cols identical", label);
    CHECK(a.outlier_cols.size() == b.outlier_cols.size() &&
          (a.outlier_cols.empty() ||
           std::memcmp(a.outlier_cols.data(), b.outlier_cols.data(),
                       a.outlier_cols.size() * sizeof(int32_t)) == 0), buf);

    snprintf(buf, sizeof(buf), "%s: outlier_vals identical", label);
    CHECK(a.outlier_vals.size() == b.outlier_vals.size() &&
          (a.outlier_vals.empty() ||
           std::memcmp(a.outlier_vals.data(), b.outlier_vals.data(),
                       a.outlier_vals.size() * sizeof(uint16_t)) == 0), buf);

    snprintf(buf, sizeof(buf), "%s: mse identical", label);
    CHECK(a.mse == b.mse, buf);
}

// Run ts_quantize_3d on a generated weight tensor and validate all experts.
static void run_moe_test(int64_t n_experts, int64_t out_dim, int64_t in_dim,
                         uint32_t seed, const char * label) {
    const int64_t stride = out_dim * in_dim;
    const int64_t total  = n_experts * stride;

    rng_seed(seed);
    std::vector<float> weights((size_t)total);
    fill_gaussian(weights, 0.02f);

    ts_quant_params_2d params = default_params();
    std::vector<ts_quant_result_2d> results;

    int rc = ts_quantize_3d(weights.data(), nullptr, nullptr, nullptr, nullptr,
                            n_experts, out_dim, in_dim, 0, &params, &results);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s: rc == 0", label);
    CHECK(rc == 0, buf);

    snprintf(buf, sizeof(buf), "%s: results.size() == n_experts", label);
    CHECK((int64_t)results.size() == n_experts, buf);

    for (int64_t e = 0; e < n_experts; e++) {
        snprintf(buf, sizeof(buf), "%s expert %lld", label, (long long)e);
        check_result(results[(size_t)e], out_dim, in_dim, buf);
    }

    std::printf("%s: %lld experts %lldx%lld, pages/row=%lld, mse[0]=%g\n",
                label, (long long)n_experts, (long long)out_dim, (long long)in_dim,
                (long long)pages_per_row(in_dim), results.empty() ? -1.0f : results[0].mse);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void) {
    // (a) small MoE: 4 experts, 512x256 (1 page per row)
    run_moe_test(4, 512, 256, 100, "small_moe");

    // (b) medium MoE: 8 experts, 256x1280 (2 pages per row)
    run_moe_test(8, 256, 1280, 200, "medium_moe");

    // (c) single expert: must match quantize_2d byte-for-byte
    {
        const int64_t out_dim = 128;
        const int64_t in_dim  = 640;
        const int64_t n       = out_dim * in_dim;

        rng_seed(300);
        std::vector<float> weights((size_t)n);
        fill_gaussian(weights, 0.02f);

        ts_quant_params_2d params = default_params();

        // 3D path with n_experts == 1
        std::vector<ts_quant_result_2d> results_3d;
        int rc3 = ts_quantize_3d(weights.data(), nullptr, nullptr, nullptr, nullptr,
                                 1, out_dim, in_dim, 0, &params, &results_3d);
        CHECK(rc3 == 0, "single_expert_3d: rc == 0");
        CHECK(results_3d.size() == 1, "single_expert_3d: 1 result");

        // 2D path on the same data
        ts_quant_result_2d result_2d;
        int rc2 = ts_quantize_2d(weights.data(), nullptr, nullptr, nullptr, nullptr,
                                 out_dim, in_dim, 0, &params, &result_2d);
        CHECK(rc2 == 0, "single_expert_2d: rc == 0");

        check_result(results_3d[0], out_dim, in_dim, "single_expert_3d");
        check_results_identical(results_3d[0], result_2d, "single_expert_match");

        std::printf("single_expert: 3d/2d byte-for-byte match, mse=%g\n", result_2d.mse);
    }

    // (d) in_dim not divisible by 640: 8 experts, 128x700
    run_moe_test(8, 128, 700, 400, "nondiv_640");

    // (e) in_dim < 640: 4 experts, 64x320
    run_moe_test(4, 64, 320, 500, "sub_page");

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
