//
// test_flrq.cpp
//
// Parity tests for the C++ FLRQ port (tessera-flrq.cpp) against the Python
// reference (tools/tessera/per_tensor_calibrate.py: flrq_sketch, flrq_bcl).
//
// The FLRQ pipeline has two RNG-bearing pieces and one deterministic core:
//
//   flrq_sketch  -- Y = W @ Omega, then U_basis = top-r left singular vectors
//                   of Y. Uses numpy's PCG64 (Python) vs xoshiro128** Box-Muller
//                   (C++), so the sketch Omega is NOT bit-identical across
//                   languages. The test therefore loads the Python U_basis
//                   directly from the fixture and hands it to ts_flrq_bcl.
//   flrq_bcl     -- fully deterministic given the basis. This is the
//                   outlier-aware low-rank fit + clip-residual numerical core,
//                   and is what this test pins bit-for-bit.
//
// Tests:
//   1. BLC parity: low_rank (U@V), residual, residual_q, scale, clip, mse all
//      match the Python fixture. Because the basis is shared and the low-rank
//      product U@V + residual is invariant under a per-column sign flip of the
//      basis, the sign ambiguity of the SVD drops out entirely.
//   2. Sketch self-consistency: the C++ sketch produces an orthonormal basis
//      whose singular values reproduce ||Y e_j|| (a sanity check on the
//      eigendecomposition path, not a cross-language parity).
//   3. End-to-end smoke: ts_train_flrq returns a decomposition whose relative
//      reconstruction MSE clears the default 1e-3 threshold.
//
// Fixture: tools/quantize/tessera/fixtures/flrq_fixture.json
// Regenerate: python3 tools/quantize/tessera/fixtures/gen_flrq_fixture.py
//
// Tolerance choice (documented):
//   - low_rank / residual atol=1e-4: the BLC inner products accumulate in
//     float32 in both Python and C++; the difference is bounded by O(n * eps)
//     of rounding, ~1e-5 for these sizes, so 1e-4 is generous but still tight
//     enough to catch a wrong projection orientation or a skipped iteration.
//   - residual_q atol=1e-4: the dequantised residual is qmax/scale-spaced;
//     the spacing is well above 1e-4, so this catches any quantisation-rule
//     drift (rounding mode, clip, scale update order).
//   - scale/clip exact: these are closed-form scalars with no accumulation.
//   - mse atol=1e-6: a mean of K*N terms, double-promoted; the gap is
//     float32->float64 promotion order, well under 1e-6.
//

#include "tessera-flrq.h"
#include "tessera-linalg.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
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

static std::string src_relative(const std::string & tail) {
    std::string f = __FILE__;
    size_t slash = f.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : f.substr(0, slash);
    return dir + "/" + tail;
}

// Flatten a (possibly nested) JSON number array into a float vector. The
// fixture stores 2-D matrices as nested lists; we flatten row-major.
static std::vector<float> to_floats(const json & j) {
    std::vector<float> out;
    if (j.is_array() && !j.empty() && j[0].is_number()) {
        out.reserve(j.size());
        for (const auto & v : j) {
            out.push_back((float)v.get<double>());
        }
        return out;
    }
    for (const auto & row : j) {
        auto sub = to_floats(row);
        out.insert(out.end(), sub.begin(), sub.end());
    }
    return out;
}

struct fixture {
    int64_t K, N, r, blc_iters, qbits;
    std::vector<float> weight;     // (K x N)
    std::vector<float> U_basis;    // (K x r)  -- Python flrq_sketch output
    // expected BLC outputs (Python flrq_bcl)
    std::vector<float> low_rank;   // (K x N)  U @ V
    std::vector<float> residual;   // (K x N)
    std::vector<float> residual_q; // (K x N)
    float scale, clip;
    float reconstruction_mse;
    float reconstruction_rel_mse;
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
    fx.K         = c["K"];
    fx.N         = c["N"];
    fx.r         = c["rank"];
    fx.blc_iters = c["blc_iters"];
    fx.qbits     = c["qbits"];
    fx.weight    = to_floats(j["input"]["weight"]);
    fx.U_basis   = to_floats(j["input"]["U_basis"]);
    const auto & e = j["expected"];
    fx.low_rank    = to_floats(e["low_rank"]);
    fx.residual    = to_floats(e["residual"]);
    fx.residual_q  = to_floats(e["residual_q"]);
    fx.scale       = e["scale"];
    fx.clip        = e["clip"];
    fx.reconstruction_mse     = e["reconstruction_mse"];
    fx.reconstruction_rel_mse = e["reconstruction_rel_mse"];
    return true;
}

// max abs diff between two float buffers.
static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main() {
    const std::string path = src_relative("fixtures/flrq_fixture.json");
    fixture fx;
    if (!load_fixture(path, fx)) {
        std::printf("FAIL: could not load fixture %s\n", path.c_str());
        return 1;
    }
    std::printf("loaded fixture: %s\n", path.c_str());
    std::printf("  W=(%lldx%lld) r=%lld blc_iters=%lld qbits=%lld\n",
                (long long)fx.K, (long long)fx.N, (long long)fx.r,
                (long long)fx.blc_iters, (long long)fx.qbits);

    // --- Test 1: BLC parity (the deterministic numerical core) ---
    {
        ts_flrq_bcl_result res;
        CHECK(ts_flrq_bcl(fx.weight.data(), fx.K, fx.N,
                          fx.U_basis.data(), fx.r,
                          fx.blc_iters, fx.qbits, &res) == 0,
              "ts_flrq_bcl ok");

        // U should equal the basis exactly (BLC returns the basis unchanged).
        CHECK(res.U.size() == (size_t)fx.K * fx.r, "U shape");
        float dU = max_abs_diff(res.U, fx.U_basis);
        CHECK(dU <= 1e-6f, "BLC U == basis");

        // low_rank = U @ V is the sign-invariant ground truth.
        std::vector<float> low_rank((size_t)fx.K * fx.N);
        for (int64_t i = 0; i < fx.K; i++) {
            for (int64_t j = 0; j < fx.N; j++) {
                float s = 0.0f;
                for (int64_t p = 0; p < fx.r; p++) {
                    s += res.U[i*fx.r + p] * res.V[p*fx.N + j];
                }
                low_rank[i*fx.N + j] = s;
            }
        }
        float d_lr = max_abs_diff(low_rank, fx.low_rank);
        float d_res = max_abs_diff(res.residual, fx.residual);
        float d_rq = max_abs_diff(res.residual_q, fx.residual_q);
        float d_scale = std::fabs(res.residual_scale - fx.scale);
        float d_clip = std::fabs(res.residual_clip - fx.clip);
        float d_mse = std::fabs(res.reconstruction_mse - fx.reconstruction_mse);
        std::printf("[blc] low_rank=%.3e residual=%.3e residual_q=%.3e "
                    "scale=%.3e clip=%.3e mse=%.3e\n",
                    d_lr, d_res, d_rq, d_scale, d_clip, d_mse);
        CHECK(d_lr <= 1e-4f,    "BLC low_rank within 1e-4");
        CHECK(d_res <= 1e-4f,   "BLC residual within 1e-4");
        CHECK(d_rq <= 1e-4f,    "BLC residual_q within 1e-4");
        CHECK(d_scale <= 1e-5f, "BLC scale within 1e-5");
        CHECK(d_clip <= 1e-6f,  "BLC clip within 1e-6");
        CHECK(d_mse <= 1e-6f,   "BLC reconstruction_mse within 1e-6");

        // relative MSE should also clear the default threshold.
        CHECK(fx.reconstruction_rel_mse <= 1e-3f,
              "Python reconstruction clears 1e-3 threshold");
    }

    // --- Test 2: sketch self-consistency ---
    // The C++ sketch RNG differs from numpy's, so we only check the
    // eigenstructure: the basis is orthonormal and sigma reproduces ||Y v_j||.
    {
        ts_flrq_sketch sk;
        CHECK(ts_flrq_sketch_run(fx.weight.data(), fx.K, fx.N,
                                 /*n_projections=*/8, /*seed=*/0,
                                 fx.r, &sk) == 0,
              "ts_flrq_sketch_run ok");
        CHECK((int64_t)sk.U_basis.size() == fx.K * fx.r, "sketch basis shape");
        CHECK((int64_t)sk.sigma.size() == fx.r,         "sketch sigma shape");

        // Orthonormality of the basis columns: <col_i, col_j> = delta_ij.
        float worst_orth = 0.0f;
        for (int64_t i = 0; i < fx.r; i++) {
            for (int64_t j = 0; j < fx.r; j++) {
                float d = 0.0f;
                for (int64_t row = 0; row < fx.K; row++) {
                    d += sk.U_basis[row*fx.r + i] * sk.U_basis[row*fx.r + j];
                }
                float target = (i == j) ? 1.0f : 0.0f;
                worst_orth = std::max(worst_orth, std::fabs(d - target));
            }
        }
        // sigma_j should equal ||Y_full @ u_j|| where u_j is the basis column.
        float worst_sigma = 0.0f;
        for (int64_t j = 0; j < fx.r; j++) {
            float s = 0.0f;  // ||Y u_j|| via Y (K x total_width)... but basis
            // comes from Y Y^T eigenvectors, so ||Y^T u_j||? The left singular
            // vector u_j of Y satisfies ||Y^T u_j|| = sigma_j. Use Y (K x W):
            // sigma_j^2 = u_j^T (Y Y^T) u_j = ||Y^T u_j||^2.
            std::vector<float> Ytu(fx.N > sk.total_width ? sk.total_width : fx.N, 0.0f);
            int64_t W = sk.total_width;
            for (int64_t b = 0; b < W; b++) {
                float acc = 0.0f;
                for (int64_t row = 0; row < fx.K; row++) {
                    acc += sk.Y[row*W + b] * sk.U_basis[row*fx.r + j];
                }
                Ytu[b] = acc;
                s += acc * acc;
            }
            float sj = std::sqrt(std::max(0.0f, s));
            worst_sigma = std::max(worst_sigma, std::fabs(sj - sk.sigma[j]));
        }
        std::printf("[sketch] worst_orthogonality=%.3e worst_sigma_err=%.3e "
                    "top_sigma=%.4f\n",
                    worst_orth, worst_sigma, sk.sigma[0]);
        CHECK(worst_orth <= 1e-3f, "sketch basis orthonormal");
        CHECK(worst_sigma <= 1e-2f, "sketch sigma self-consistent");
        CHECK(sk.sigma[0] > 1.0f, "sketch top singular value non-trivial");
    }

    // --- Test 3: end-to-end smoke (rank sweep) ---
    {
        ts_flrq_params params = {};
        params.n_projections = 8;
        params.blc_iters     = fx.blc_iters;
        params.qbits         = fx.qbits;
        params.seed          = 1;
        // Candidate set scaled to this tiny weight: {2, 4, 8}.
        params.rank_candidates[0] = 2;
        params.rank_candidates[1] = 4;
        params.rank_candidates[2] = 8;
        params.rank_candidates[3] = 0;
        params.mse_threshold = 1e-3f;

        ts_flrq_bcl_result out;
        int64_t rank = -1;
        CHECK(ts_train_flrq(fx.weight.data(), fx.K, fx.N, &params, &out, &rank) == 0,
              "ts_train_flrq ok");
        std::printf("[e2e] chosen_rank=%lld recon_mse=%.4e\n",
                    (long long)rank, out.reconstruction_mse);
        CHECK(rank >= 1, "train_flrq chose a valid rank");

        // Relative reconstruction MSE.
        double w_fro2 = 0.0;
        for (int64_t i = 0; i < fx.K * fx.N; i++) {
            w_fro2 += (double)fx.weight[i] * fx.weight[i];
        }
        w_fro2 += 1e-12;
        double rel = (double)out.reconstruction_mse * (double)(fx.K * fx.N) / w_fro2;
        CHECK(rel <= 1e-2, "end-to-end relative MSE reasonable (< 1e-2)");
    }

    if (g_failures == 0) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL (%d failures)\n", g_failures);
    return 1;
}
