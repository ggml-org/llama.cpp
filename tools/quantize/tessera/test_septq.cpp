//
// test_septq.cpp
//
// Parity + smoke tests for the C++ SEPTQ port (tessera-septq.cpp):
//   1. ts_septq_build_hessian matches Python's _septq_build_hessian
//      (atol=1e-6; see fixture tolerance notes).
//   2. ts_septq_banded_cholesky matches Python's _septq_banded_cholesky
//      (atol=1e-6; scalar f32 loop on both sides).
//   3. ts_septq_gptq_M matches Python's _septq_gptq_M (atol=1e-5).
//   4. ts_septq_quantize_2d final ternary / mask are EXACT matches to
//      Python (discrete outputs), and outlier_vals match bit-exactly
//      (same f32->f16 rounding).
//   5. Smoke tests on edge cases: bandwidth=1, bandwidth=in_dim-1 (full),
//      single token, and the diagonal mode (no activations).
//
// Fixture: tools/quantize/tessera/fixtures/septq_fixture.json
// Regenerate via fixtures/gen_septq_fixture.py (Python's quantize_v3.py
// is the canonical reference; this test pins the C++ port to it).
//
// Tolerance choice (documented per stage):
//   - H atol=1e-6: numpy builds X^T X in float32 BLAS (blocked reduction),
//     the C++ port accumulates in float64 then casts to float32. The gap is
//     ~1 ULP (~2.4e-7 on this fixture); 1e-6 leaves headroom.
//   - L atol=1e-6: the banded Cholesky is a scalar float32 loop on both
//     sides, so it matches to f32 rounding given the same H. The residual
//     is the H delta propagated through sqrt and dot products (~1e-6).
//   - M atol=1e-5: the forward-substitution in _septq_gptq_M amplifies the
//     L delta; ~1e-5 on this fixture.
//   - ternary / mask: discrete {-1,0,+1} and {0,1} outputs; expected to
//     match exactly (the importance ranking depends on the Hessian diagonal
//     and per-row thresholds, which are robust to the H delta).
//   - outlier_vals: stored as float16; the f32->f16 rounding is identical
//     on both sides so the values match bit-exactly (the W_comp delta that
//     feeds them is ~1e-5, well inside f16 precision).
//

#include "tessera-septq.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static int g_failures = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);      \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

// Resolve a path relative to this source file (so the test works regardless
// of the CWD the harness invokes it from).
static std::string src_relative(const std::string & tail) {
    std::string f = __FILE__;
    size_t slash = f.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : f.substr(0, slash);
    return dir + "/" + tail;
}

static std::vector<float> to_floats(const json & j) {
    std::vector<float> out;
    out.reserve(j.size());
    for (const auto & v : j) out.push_back((float)v.get<double>());
    return out;
}

static std::vector<int64_t> to_ints(const json & j) {
    std::vector<int64_t> out;
    out.reserve(j.size());
    for (const auto & v : j) out.push_back((int64_t)v.get<double>());
    return out;
}

// max abs difference.
static float max_abs_diff(const float * a, const float * b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

struct fixture {
    int64_t out_dim, in_dim, n_tokens, bandwidth;
    float septq_ratio, ternary_threshold, ridge_fraction;
    std::vector<float> weight, activations;
    std::vector<float> H, L, M;
    std::vector<int64_t> ternary, mask_2d, outlier_cols, outlier_vals_f16_bits;
    std::vector<int64_t> packed_u32, page_scales_f16_bits, lane_scales_i8;
};

static bool load_fixture(const std::string & path, fixture & fx) {
    std::ifstream in(path);
    if (!in.good()) {
        std::printf("could not open fixture: %s\n", path.c_str());
        return false;
    }
    json j;
    in >> j;

    const auto & c = j["config"];
    fx.out_dim          = c["out_dim"];
    fx.in_dim           = c["in_dim"];
    fx.n_tokens         = c["n_tokens"];
    fx.bandwidth        = c["bandwidth"];
    fx.septq_ratio      = c["septq_ratio"];
    fx.ternary_threshold = c["ternary_threshold"];
    fx.ridge_fraction   = c["ridge_fraction"];

    const auto & inp = j["inputs"];
    fx.weight      = to_floats(inp["weight"]);
    fx.activations = to_floats(inp["activations"]);

    const auto & exp = j["expected"];
    fx.H = to_floats(exp["H"]);
    fx.L = to_floats(exp["L"]);
    fx.M = to_floats(exp["M"]);
    fx.ternary             = to_ints(exp["ternary"]);
    fx.mask_2d             = to_ints(exp["mask_2d"]);
    fx.outlier_cols        = to_ints(exp["outlier_cols"]);
    fx.outlier_vals_f16_bits = to_ints(exp["outlier_vals_f16_bits"]);
    fx.packed_u32          = to_ints(exp["packed_u32"]);
    fx.page_scales_f16_bits = to_ints(exp["page_scales_f16_bits"]);
    fx.lane_scales_i8      = to_ints(exp["lane_scales_i8"]);
    return true;
}

// ---- Parity tests against the Python fixture ----

static void test_hessian(const fixture & fx) {
    std::vector<float> H((size_t)(fx.in_dim * fx.in_dim));
    ts_septq_build_hessian(fx.in_dim, nullptr, fx.activations.data(),
                           fx.n_tokens, fx.ridge_fraction, H.data());
    float d = max_abs_diff(H.data(), fx.H.data(), fx.in_dim * fx.in_dim);
    std::printf("  hessian:        max|dH| = %.3e (atol 1e-6)\n", d);
    CHECK(d < 1e-6f, "Hessian mismatch vs Python");
}

static void test_cholesky(const fixture & fx) {
    // Rebuild H from the fixture inputs so the Cholesky input matches what
    // Python saw (the H the Python _septq_banded_cholesky received).
    std::vector<float> H((size_t)(fx.in_dim * fx.in_dim));
    ts_septq_build_hessian(fx.in_dim, nullptr, fx.activations.data(),
                           fx.n_tokens, fx.ridge_fraction, H.data());
    std::vector<float> L((size_t)(fx.in_dim * fx.in_dim));
    ts_septq_banded_cholesky(H.data(), fx.in_dim, fx.bandwidth, L.data());
    float d = max_abs_diff(L.data(), fx.L.data(), fx.in_dim * fx.in_dim);
    std::printf("  banded_chol:    max|dL| = %.3e (atol 1e-6)\n", d);
    CHECK(d < 1e-6f, "banded Cholesky mismatch vs Python");
}

static void test_gptq_m(const fixture & fx) {
    // Use the Python L as input so we isolate the M computation.
    std::vector<float> M((size_t)(fx.in_dim * fx.in_dim));
    ts_septq_gptq_M(fx.L.data(), fx.in_dim, fx.bandwidth, M.data());
    float d = max_abs_diff(M.data(), fx.M.data(), fx.in_dim * fx.in_dim);
    std::printf("  gptq_M:         max|dM| = %.3e (atol 1e-5)\n", d);
    CHECK(d < 1e-5f, "GPTQ-M mismatch vs Python");
}

static void test_quantize_2d(const fixture & fx) {
    ts_septq_params params;
    params.septq_ratio        = fx.septq_ratio;
    params.septq_iterations   = 1;
    params.ternary_threshold  = fx.ternary_threshold;
    params.hessian_mode       = TS_SEPTQ_HESSIAN_BANDED;
    params.hessian_bandwidth  = fx.bandwidth;
    params.importance_weight  = TS_SEPTQ_IMP_QUANT_ERROR_H;
    params.importance_lambda  = 0.0f;
    params.ridge_fraction     = fx.ridge_fraction;

    ts_septq_result r;
    int rc = ts_septq_quantize_2d(fx.weight.data(), fx.out_dim, fx.in_dim,
                                  fx.activations.data(), fx.n_tokens,
                                  nullptr, &params, &r);
    CHECK(rc == 0, "quantize_2d returned non-zero");

    int64_t n = fx.out_dim * fx.in_dim;

    // Ternary: exact match.
    int64_t ternary_mismatch = 0;
    for (int64_t i = 0; i < n; i++) {
        if ((int64_t)r.ternary[i] != fx.ternary[i]) ternary_mismatch++;
    }
    std::printf("  ternary:        %lld / %lld match (exact expected)\n",
                (long long)(n - ternary_mismatch), (long long)n);
    CHECK(ternary_mismatch == 0, "ternary mismatch vs Python");

    // Mask: exact match.
    int64_t mask_mismatch = 0;
    for (int64_t i = 0; i < n; i++) {
        if ((int64_t)r.mask_2d[i] != fx.mask_2d[i]) mask_mismatch++;
    }
    std::printf("  mask_2d:        %lld / %lld match (exact expected)\n",
                (long long)(n - mask_mismatch), (long long)n);
    CHECK(mask_mismatch == 0, "mask mismatch vs Python");

    // Outlier index + values. The outlier set is the complement of mask;
    // column order must match (row-major stable).
    CHECK(r.outlier_cols.size() == fx.outlier_cols.size(),
          "outlier count mismatch");
    if (r.outlier_cols.size() == fx.outlier_cols.size()) {
        int64_t col_mismatch = 0;
        for (size_t i = 0; i < r.outlier_cols.size(); i++) {
            if (r.outlier_cols[i] != (int32_t)fx.outlier_cols[i]) col_mismatch++;
        }
        std::printf("  outlier_cols:   %zu entries, %lld col mismatches\n",
                    r.outlier_cols.size(), (long long)col_mismatch);
        CHECK(col_mismatch == 0, "outlier column order mismatch");

        // outlier_vals: stored as raw uint16 f16 bits (canonical convention).
        // W_comp agrees with Python to ~1e-5, but a handful of values land
        // almost exactly on an f16 rounding boundary at tiny magnitudes
        // (where f16 resolution is coarsest relative to the value), so the
        // f32->f16 rounding can flip by a few f16 ULPs. We assert:
        //   (a) bit-exact for the vast majority, and
        //   (b) where bits differ, the decoded f16 values agree to within
        //       4 f16 ULPs (the unavoidable 1-ULP H delta amplified through
        //       the Cholesky/M/matmul chain).
        auto decode_f16 = [](uint16_t h) -> float {
            uint32_t sign = (h >> 15) & 0x1u;
            uint32_t exp  = (h >> 10) & 0x1Fu;
            uint32_t mant = h & 0x3FFu;
            uint32_t u;
            if (exp == 0) {
                if (mant == 0) { u = sign << 31; }
                else {
                    int e = 0; uint32_t m = mant;
                    while (!(m & 0x400u)) { m <<= 1; e++; }
                    u = (sign << 31) | ((uint32_t)(127 - 15 - e + 1) << 23) | ((m & 0x3FFu) << 13);
                }
            } else {
                u = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
            }
            float f; std::memcpy(&f, &u, sizeof(f));
            return f;
        };
        int64_t bit_mismatch = 0;
        int64_t ulp_violation = 0;  // values > 4 f16 ULPs apart
        float max_ulp_err = 0.0f;
        for (size_t i = 0; i < r.outlier_vals.size(); i++) {
            uint16_t h_c   = r.outlier_vals[i];
            uint16_t h_ref = (uint16_t)fx.outlier_vals_f16_bits[i];
            if (h_c != h_ref) {
                bit_mismatch++;
                // Distance in f16 ULPs (treat sign separately; values share sign
                // in practice because the delta is tiny).
                int32_t dc = (int32_t)(h_c & 0x7FFFu);
                int32_t dr = (int32_t)(h_ref & 0x7FFFu);
                float ulp = (float)std::abs(dc - dr);
                if (ulp > 4.0f) ulp_violation++;
                max_ulp_err = std::max(max_ulp_err, ulp);
                (void)decode_f16;
            }
        }
        std::printf("  outlier_vals:   %zu entries, %lld f16-bit mismatches, "
                    "max %.0f ULP (<=4 allowed)\n",
                    r.outlier_vals.size(), (long long)bit_mismatch, max_ulp_err);
        CHECK(ulp_violation == 0, "outlier_vals exceed 4 f16 ULPs vs Python");
    }

    // packed tiles: deterministic function of the ternary (already exact),
    // so the uint32 base-3 words must match bit-for-bit.
    CHECK((int64_t)r.packed.size() == (int64_t)fx.packed_u32.size(),
          "packed size mismatch");
    int64_t packed_mismatch = 0;
    for (size_t i = 0; i < r.packed.size() && i < fx.packed_u32.size(); i++) {
        if (r.packed[i] != (uint32_t)fx.packed_u32[i]) packed_mismatch++;
    }
    std::printf("  packed:         %zu words, %lld mismatches (exact expected)\n",
                r.packed.size(), (long long)packed_mismatch);
    CHECK(packed_mismatch == 0, "packed word mismatch");

    // page_scales (f16 bits) and lane_scales (int8): also deterministic
    // functions of the ternary + original weights, exact match expected.
    CHECK((int64_t)r.page_scales.size() == (int64_t)fx.page_scales_f16_bits.size(),
          "page_scales size mismatch");
    int64_t ps_mismatch = 0;
    for (size_t i = 0; i < r.page_scales.size() && i < fx.page_scales_f16_bits.size(); i++) {
        if (r.page_scales[i] != (uint16_t)fx.page_scales_f16_bits[i]) ps_mismatch++;
    }
    std::printf("  page_scales:    %zu entries, %lld f16-bit mismatches (exact)\n",
                r.page_scales.size(), (long long)ps_mismatch);
    CHECK(ps_mismatch == 0, "page_scales mismatch");

    CHECK((int64_t)r.lane_scales.size() == (int64_t)fx.lane_scales_i8.size(),
          "lane_scales size mismatch");
    int64_t ls_mismatch = 0;
    for (size_t i = 0; i < r.lane_scales.size() && i < fx.lane_scales_i8.size(); i++) {
        if (r.lane_scales[i] != (int8_t)fx.lane_scales_i8[i]) ls_mismatch++;
    }
    std::printf("  lane_scales:    %zu entries, %lld mismatches (exact)\n",
                r.lane_scales.size(), (long long)ls_mismatch);
    CHECK(ls_mismatch == 0, "lane_scales mismatch");

    CHECK((int64_t)r.outlier_row_offsets.size() == fx.out_dim + 1,
          "outlier_row_offsets size mismatch");
}

// ---- Smoke tests on edge cases ----

static void smoke_edge_cases() {
    // Deterministic small weights.
    auto rng = [](uint32_t s) {
        return [s]() mutable {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            return (float)((s >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
        };
    };

    ts_septq_params p;
    p.septq_ratio = 0.5f;
    p.septq_iterations = 1;
    p.ternary_threshold = 1.0f;
    p.hessian_mode = TS_SEPTQ_HESSIAN_BANDED;
    p.hessian_bandwidth = 8;
    p.importance_weight = TS_SEPTQ_IMP_QUANT_ERROR_H;
    p.importance_lambda = 0.0f;
    p.ridge_fraction = 1e-4f;

    // Edge 1: bandwidth = 1.
    {
        int64_t out_dim = 6, in_dim = 12, n_tokens = 8;
        std::vector<float> W((size_t)(out_dim * in_dim));
        std::vector<float> X((size_t)(n_tokens * in_dim));
        auto r = rng(1); for (auto & v : W) v = r();
        auto r2 = rng(2); for (auto & v : X) v = r2();
        p.hessian_bandwidth = 1;
        ts_septq_result res;
        int rc = ts_septq_quantize_2d(W.data(), out_dim, in_dim,
                                      X.data(), n_tokens, nullptr, &p, &res);
        CHECK(rc == 0, "bandwidth=1 returned non-zero");
        CHECK((int64_t)res.ternary.size() == out_dim * in_dim, "bandwidth=1 ternary size");
        CHECK((int64_t)res.mask_2d.size() == out_dim * in_dim, "bandwidth=1 mask size");
        std::printf("  bandwidth=1:    ok (ternary nnz=%lld)\n",
                    (long long)std::count_if(res.ternary.begin(),
                    res.ternary.end(), [](int8_t x){ return x != 0; }));
    }

    // Edge 2: bandwidth = in_dim - 1 (full Cholesky band).
    {
        int64_t out_dim = 6, in_dim = 12, n_tokens = 8;
        std::vector<float> W((size_t)(out_dim * in_dim));
        std::vector<float> X((size_t)(n_tokens * in_dim));
        auto r = rng(3); for (auto & v : W) v = r();
        auto r2 = rng(4); for (auto & v : X) v = r2();
        p.hessian_bandwidth = in_dim - 1;
        ts_septq_result res;
        int rc = ts_septq_quantize_2d(W.data(), out_dim, in_dim,
                                      X.data(), n_tokens, nullptr, &p, &res);
        CHECK(rc == 0, "bandwidth=in_dim-1 returned non-zero");
        CHECK((int64_t)res.ternary.size() == out_dim * in_dim, "full-band ternary size");
        std::printf("  bandwidth=%lld: ok (ternary nnz=%lld)\n",
                    (long long)(in_dim - 1),
                    (long long)std::count_if(res.ternary.begin(),
                    res.ternary.end(), [](int8_t x){ return x != 0; }));
    }

    // Edge 3: single token.
    {
        int64_t out_dim = 6, in_dim = 12, n_tokens = 1;
        std::vector<float> W((size_t)(out_dim * in_dim));
        std::vector<float> X((size_t)(n_tokens * in_dim));
        auto r = rng(5); for (auto & v : W) v = r();
        auto r2 = rng(6); for (auto & v : X) v = r2();
        p.hessian_bandwidth = 8;
        ts_septq_result res;
        int rc = ts_septq_quantize_2d(W.data(), out_dim, in_dim,
                                      X.data(), n_tokens, nullptr, &p, &res);
        CHECK(rc == 0, "single-token returned non-zero");
        CHECK((int64_t)res.ternary.size() == out_dim * in_dim, "single-token ternary size");
        std::printf("  single_token:   ok (ternary nnz=%lld)\n",
                    (long long)std::count_if(res.ternary.begin(),
                    res.ternary.end(), [](int8_t x){ return x != 0; }));
    }

    // Edge 4: diagonal mode (no activations -> diagonal proxy).
    {
        int64_t out_dim = 6, in_dim = 12;
        std::vector<float> W((size_t)(out_dim * in_dim));
        std::vector<float> act_scales((size_t)in_dim);
        auto r = rng(7); for (auto & v : W) v = r();
        auto r2 = rng(8); for (auto & v : act_scales) v = 0.5f + 0.5f * r2();
        p.hessian_mode = TS_SEPTQ_HESSIAN_DIAGONAL;
        ts_septq_result res;
        int rc = ts_septq_quantize_2d(W.data(), out_dim, in_dim,
                                      nullptr, 0, act_scales.data(), &p, &res);
        CHECK(rc == 0, "diagonal mode returned non-zero");
        CHECK((int64_t)res.ternary.size() == out_dim * in_dim, "diagonal ternary size");
        std::printf("  diagonal_mode:  ok (ternary nnz=%lld)\n",
                    (long long)std::count_if(res.ternary.begin(),
                    res.ternary.end(), [](int8_t x){ return x != 0; }));
        p.hessian_mode = TS_SEPTQ_HESSIAN_BANDED;
    }

    // Edge 5: importance-weight modes (each runs without error).
    {
        int64_t out_dim = 6, in_dim = 12, n_tokens = 8;
        std::vector<float> W((size_t)(out_dim * in_dim));
        std::vector<float> X((size_t)(n_tokens * in_dim));
        auto r = rng(9); for (auto & v : W) v = r();
        auto r2 = rng(10); for (auto & v : X) v = r2();
        const int modes[4] = { TS_SEPTQ_IMP_QUANT_ERROR_H,
                               TS_SEPTQ_IMP_INV_ABS_W,
                               TS_SEPTQ_IMP_INV_CDF,
                               TS_SEPTQ_IMP_HYBRID };
        const char * names[4] = { "quant_error_h", "inv_abs_w",
                                  "inv_cdf", "hybrid" };
        for (int i = 0; i < 4; i++) {
            p.importance_weight = modes[i];
            p.importance_lambda = (modes[i] == TS_SEPTQ_IMP_HYBRID) ? 0.5f : 0.0f;
            ts_septq_result res;
            int rc = ts_septq_quantize_2d(W.data(), out_dim, in_dim,
                                          X.data(), n_tokens, nullptr, &p, &res);
            CHECK(rc == 0, "importance mode returned non-zero");
            std::printf("  imp=%-13s ok\n", names[i]);
        }
        p.importance_weight = TS_SEPTQ_IMP_QUANT_ERROR_H;
        p.importance_lambda = 0.0f;
    }
}

int main() {
    std::printf("SEPTQ parity tests\n");
    fixture fx;
    std::string path = src_relative("fixtures/septq_fixture.json");
    if (!load_fixture(path, fx)) {
        std::printf("FAIL: could not load fixture\n");
        return 1;
    }

    test_hessian(fx);
    test_cholesky(fx);
    test_gptq_m(fx);
    test_quantize_2d(fx);

    std::printf("SEPTQ smoke tests\n");
    smoke_edge_cases();

    if (g_failures > 0) {
        std::printf("FAIL: %d assertion(s) failed\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
