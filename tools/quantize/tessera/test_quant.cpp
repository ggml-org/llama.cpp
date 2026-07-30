//
// test_quant.cpp
//
// Smoke tests for tessera-quant.cpp: ternarization range, Tile640 packing
// size + round-trip, and an end-to-end quantize_2d.
//

#include "tessera-quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            g_failures++;                                    \
        }                                                    \
    } while (0)

int main(void) {
    const int64_t out_dim = 4;
    const int64_t in_dim  = 640; // exactly one Tile640 page wide
    const int64_t n       = out_dim * in_dim;
    const int64_t pages   = (in_dim + 640 - 1) / 640; // == 1

    // known, varied weights in ~[-5, 5]
    std::vector<float> weights((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        weights[(size_t)i] = (float)((i % 100) - 50) / 10.0f;
    }

    // --- 1. ternarize (alpha = 0, no AWQ) ---
    std::vector<int8_t> ternary((size_t)n, 0);
    float scale = ts_ternarize_with_acts(weights.data(), nullptr, 0.0f, 0.0f,
                                         ternary.data(), out_dim, in_dim);
    CHECK(std::isfinite(scale) && scale >= 0.0f, "ternarize returns finite scale");

    int n_pos = 0, n_neg = 0, n_zero = 0;
    for (int64_t i = 0; i < n; i++) {
        int8_t t = ternary[(size_t)i];
        CHECK(t == -1 || t == 0 || t == 1, "ternary value in {-1,0,+1}");
        if (t > 0) n_pos++;
        else if (t < 0) n_neg++;
        else n_zero++;
    }
    CHECK(n_pos > 0 && n_neg > 0, "ternary has both signs");
    CHECK(n_zero > 0, "ternary has zeros (sparse)");
    std::printf("ternary: +%d / -%d / 0x%d (scale=%g)\n", n_pos, n_neg, n_zero, scale);

    // --- 2. pack: size + trit round-trip ---
    const int64_t expected_words = out_dim * pages * 32; // 32 words per page
    std::vector<uint32_t> packed((size_t)expected_words, 0);
    std::vector<uint16_t> pscale((size_t)(out_dim * pages), 0);
    std::vector<int8_t>   lscale((size_t)(out_dim * pages * 32), 0);
    ts_pack_tile640(ternary.data(), packed.data(), pscale.data(), lscale.data(),
                    out_dim, in_dim);
    CHECK(packed.size() == (size_t)expected_words, "packed size matches expected");

    uint32_t pow3[20];
    pow3[0] = 1;
    for (int i = 1; i < 20; i++) pow3[i] = pow3[i - 1] * 3u;

    // decode every lane word and compare against the ternary source
    bool roundtrip_ok = true;
    for (int64_t o = 0; o < out_dim; o++) {
        for (int64_t p = 0; p < pages; p++) {
            for (int l = 0; l < 32; l++) {
                uint32_t word = packed[(size_t)((o * pages + p) * 32 + l)];
                for (int k = 0; k < 20; k++) {
                    int64_t col = p * 640 + l * 20 + k;
                    if (col >= in_dim) continue;
                    uint32_t trit = (word / pow3[k]) % 3u;
                    int8_t expect = ternary[(size_t)(o * in_dim + col)];
                    int8_t decoded = (trit == 1) ? 1 : ((trit == 2) ? -1 : 0);
                    if (decoded != expect) roundtrip_ok = false;
                }
            }
        }
    }
    CHECK(roundtrip_ok, "pack/unpack trit round-trip");

    // --- 3. quantize_2d end-to-end (no AWQ) ---
    ts_quant_params_2d params = {};
    params.alpha          = 0.0f;
    params.clip           = 1.0f;
    params.max_outliers   = 8;
    params.outlier_thresh = 2.0f;
    params.awq_grid       = 5;

    ts_quant_result_2d result;
    int rc = ts_quantize_2d(weights.data(), nullptr, nullptr, nullptr, nullptr,
                            out_dim, in_dim, 0, &params, &result);
    CHECK(rc == 0, "quantize_2d returns 0");
    CHECK(result.packed.size() == (size_t)expected_words, "result.packed size");
    CHECK(result.page_scales.size() == (size_t)(out_dim * pages), "result.page_scales size");
    CHECK(result.lane_scales.size() == (size_t)(out_dim * pages * 32), "result.lane_scales size");
    CHECK(result.outlier_row_offsets.size() == (size_t)(out_dim + 1), "outlier_row_offsets size");
    CHECK(result.outlier_row_offsets.back() == (int32_t)result.outlier_cols.size(),
          "outlier CSR consistent");
    CHECK(std::isfinite(result.mse) && result.mse >= 0.0f, "mse finite and non-negative");
    CHECK(result.act_scale.empty(), "act_scale empty when alpha == 0");
    std::printf("quantize_2d: mse=%g outliers=%zu alpha=%g\n",
                result.mse, result.outlier_cols.size(), result.best_alpha);

    // --- 4. quantize_2d with activations (exercises AWQ per-row search) ---
    std::vector<float> act((size_t)in_dim);
    for (int64_t c = 0; c < in_dim; c++) {
        act[(size_t)c] = 0.1f + 0.01f * (float)(c % 50);
    }
    ts_quant_result_2d result_awq;
    rc = ts_quantize_2d(weights.data(), act.data(), nullptr, nullptr, nullptr,
                        out_dim, in_dim, 0, &params, &result_awq);
    CHECK(rc == 0, "quantize_2d (AWQ) returns 0");
    CHECK(result_awq.act_scale.size() == (size_t)(result_awq.best_alpha > 0.0f ? in_dim : 0),
          "act_scale present iff alpha > 0");
    std::printf("quantize_2d (AWQ): mse=%g alpha=%g\n", result_awq.mse, result_awq.best_alpha);

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
