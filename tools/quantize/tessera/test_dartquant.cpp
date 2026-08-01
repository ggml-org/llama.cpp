//
// test_dartquant.cpp
//
// Parity tests for the C++ DartQuant port (tessera-dartquant.cpp):
//   1. Loss-trajectory parity: per-iter combined loss matches Python's
//      dartquant_qr_orth history to loss_combined_atol.
//   2. Whip-loss / quant-MSE / output-MSE parity at the final iterate.
//   3. Rotation parity: max|R_cpp - R_py| reported; asserted loosely
//      (rotation_max_abs_atol) because the QR-Orth iterate drifts between
//      float32 Householder (C++) and float64 LAPACK QR (Python). We also
//      accept -R column-flips and assert orthogonality tightly.
//   4. Functional sanity: rotating W by the learned R lowers the output
//      MSE vs the identity baseline (the deployment metric DartQuant
//      optimises for).
//
// Fixture: tools/quantize/tessera/fixtures/dartquant_fixture.json
// Regenerate via fixtures/gen_dartquant_fixture.py.
//
// Tolerance rationale (documented):
//   - loss_combined_atol = 5e-3: combined loss is qmse + 0.1*whip, both
//     float32 reductions. The QR-Orth iterate itself is computed in
//     float64 + LAPACK on the Python side and float32 + Householder in
//     C++; after 30 iterations the per-step round-off compounds to ~1e-3
//     in the loss, so 5e-3 leaves comfortable headroom while still
//     catching a wrong gradient sign or missing X_hat weighting.
//   - whip_atol / quant_mse_atol / output_mse_atol = 5e-3: same float32
//     drift bound; the per-metric values are O(0.1) to O(60), so 5e-3 is
//     a tight relative bar on the small metrics and a very tight
//     relative bar on the output MSE.
//   - rotation_max_abs_atol = 5e-2: the rotation matrix elements drift
//     the most because each Householder step is a different rounding than
//     LAPACK. We assert orthogonality tightly (1e-3) and report the
//     achieved max-abs gap; 5e-2 catches a sign-flipped or transposed R
//     without failing on legitimate float32 round-off.
//

#include "tessera-dartquant.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);      \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

// Resolve a path relative to this source file (works regardless of CWD).
static std::string src_relative(const std::string & tail) {
    std::string f = __FILE__;
    size_t slash = f.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : f.substr(0, slash);
    return dir + "/" + tail;
}

static std::vector<float> to_floats(const json & j) {
    std::vector<float> out;
    out.reserve(j.size());
    for (const auto & v : j) {
        out.push_back((float)v.get<double>());
    }
    return out;
}

static double to_double(const json & j) {
    return (double)j.get<double>();
}

struct fixture {
    // params
    int64_t out_dim, in_dim, block_size, num_samples, n_iters, seed;
    double lr, whip_weight;
    // inputs
    std::vector<float> weight, X, X_hat, x_scale;
    // expected
    std::vector<float> rotation, history;
    double initial_whip, final_whip;
    double initial_quant_mse, final_quant_mse;
    double initial_output_mse, final_output_mse;
    int64_t iterations;
    // tolerance
    double loss_combined_atol, whip_atol, quant_mse_atol, output_mse_atol;
    double rotation_max_abs_atol, rotation_orthogonality_atol;
};

static bool load_fixture(const std::string & path, fixture & fx) {
    std::ifstream in(path);
    if (!in.good()) {
        std::printf("could not open fixture: %s\n", path.c_str());
        return false;
    }
    json j;
    in >> j;

    const auto & P = j["params"];
    fx.out_dim      = P["out_dim"];
    fx.in_dim       = P["in_dim"];
    fx.block_size   = P["block_size"];
    fx.num_samples  = P["num_samples"];
    fx.n_iters      = P["n_iters"];
    fx.seed         = P["seed"];
    fx.lr           = P["lr"];
    fx.whip_weight  = P["whip_weight"];

    const auto & I = j["inputs"];
    fx.weight = to_floats(I["weight"]);
    fx.X      = to_floats(I["X"]);
    fx.X_hat  = to_floats(I["X_hat"]);
    fx.x_scale = to_floats(I["x_scale"]);

    const auto & E = j["expected"];
    fx.rotation           = to_floats(E["rotation"]);
    fx.history            = to_floats(E["history"]);
    fx.initial_whip       = to_double(E["initial_whip"]);
    fx.final_whip         = to_double(E["final_whip"]);
    fx.initial_quant_mse  = to_double(E["initial_quant_mse"]);
    fx.final_quant_mse    = to_double(E["final_quant_mse"]);
    fx.initial_output_mse = to_double(E["initial_output_mse"]);
    fx.final_output_mse   = to_double(E["final_output_mse"]);
    fx.iterations         = E["iterations"];

    const auto & T = j["tolerance"];
    fx.loss_combined_atol        = to_double(T["loss_combined_atol"]);
    fx.whip_atol                 = to_double(T["whip_atol"]);
    fx.quant_mse_atol            = to_double(T["quant_mse_atol"]);
    fx.output_mse_atol           = to_double(T["output_mse_atol"]);
    fx.rotation_max_abs_atol     = to_double(T["rotation_max_abs_atol"]);
    fx.rotation_orthogonality_atol = to_double(T["rotation_orthogonality_atol"]);
    return true;
}

// max |A - B| over a flat float buffer.
static float max_abs_diff(const float * A, const float * B, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        m = std::max(m, std::fabs(A[i] - B[i]));
    }
    return m;
}

// max |A + B| over a flat buffer (used for the -R sign-ambiguity check).
static float max_abs_sum(const float * A, const float * B, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        m = std::max(m, std::fabs(A[i] + B[i]));
    }
    return m;
}

// Frobenius deviation of R^T R from I (orthogonality check).
static float orthogonality_err(const float * R, int64_t K) {
    float worst = 0.0f;
    for (int64_t i = 0; i < K; i++) {
        for (int64_t j = 0; j < K; j++) {
            double s = 0.0;
            for (int64_t r = 0; r < K; r++) {
                s += (double)R[r*K + i] * (double)R[r*K + j];
            }
            float target = (i == j) ? 1.0f : 0.0f;
            worst = std::max(worst, std::fabs((float)s - target));
        }
    }
    return worst;
}

int main() {
    fixture fx;
    std::string fxpath = src_relative("fixtures/dartquant_fixture.json");
    if (!load_fixture(fxpath, fx)) {
        return 1;
    }

    ts_dartquant_params params = {};
    params.block_size  = fx.block_size;
    params.max_iters   = fx.n_iters;
    params.lr          = (float)fx.lr;
    params.whip_weight = (float)fx.whip_weight;
    params.seed        = (uint32_t)fx.seed;

    ts_dartquant_result result;
    int rc = ts_dartquant_qr_orth(fx.weight.data(), fx.out_dim, fx.in_dim,
                                  fx.X.data(), fx.num_samples,
                                  fx.X_hat.data(),
                                  &params, &result);
    CHECK(rc == 0, "ts_dartquant_qr_orth returned non-zero");
    if (rc != 0) {
        std::printf("Results: %d checks, %d failures\n", g_checks, g_failures);
        return g_failures ? 1 : 0;
    }

    int64_t K = fx.in_dim;
    CHECK((int64_t)result.R.size() == K * K, "R size mismatch");
    CHECK((int64_t)result.history.size() == fx.n_iters + 1, "history length mismatch");
    CHECK(result.n_iters == fx.n_iters, "n_iters mismatch");

    // --- Loss trajectory parity ---
    {
        float hist_max = 0.0f;
        for (size_t i = 0; i < result.history.size() && i < fx.history.size(); i++) {
            float d = std::fabs(result.history[i] - (float)fx.history[i]);
            hist_max = std::max(hist_max, d);
        }
        // Report the worst step; assert each is within atol.
        size_t worst_i = 0;
        for (size_t i = 0; i < result.history.size() && i < fx.history.size(); i++) {
            if (std::fabs(result.history[i] - (float)fx.history[i]) ==
                hist_max) worst_i = i;
        }
        std::printf("  history: max_abs=%.3e (step %zu, cpp=%.6f py=%.6f)\n",
                    hist_max, worst_i,
                    result.history[worst_i], (float)fx.history[worst_i]);
        CHECK(hist_max < fx.loss_combined_atol, "combined loss trajectory out of tol");
    }

    // --- Per-metric final parity ---
    {
        float d_init_whip = std::fabs(result.initial_whip - (float)fx.initial_whip);
        float d_final_whip = std::fabs(result.final_whip - (float)fx.final_whip);
        float d_init_qmse = std::fabs(result.initial_quant_mse - (float)fx.initial_quant_mse);
        float d_final_qmse = std::fabs(result.final_quant_mse - (float)fx.final_quant_mse);
        float d_init_omse = std::fabs(result.initial_output_mse - (float)fx.initial_output_mse);
        float d_final_omse = std::fabs(result.final_output_mse - (float)fx.final_output_mse);
        std::printf("  whip:   init d=%.3e  final d=%.3e  (py init=%.4e final=%.4e)\n",
                    d_init_whip, d_final_whip, fx.initial_whip, fx.final_whip);
        std::printf("  qmse:   init d=%.3e  final d=%.3e  (py init=%.4e final=%.4e)\n",
                    d_init_qmse, d_final_qmse, fx.initial_quant_mse, fx.final_quant_mse);
        std::printf("  omse:   init d=%.3e  final d=%.3e  (py init=%.4e final=%.4e)\n",
                    d_init_omse, d_final_omse, fx.initial_output_mse, fx.final_output_mse);
        CHECK(d_init_whip  < fx.whip_atol,      "initial whip out of tol");
        CHECK(d_final_whip < fx.whip_atol,      "final whip out of tol");
        CHECK(d_init_qmse  < fx.quant_mse_atol, "initial quant_mse out of tol");
        CHECK(d_final_qmse < fx.quant_mse_atol, "final quant_mse out of tol");
        CHECK(d_init_omse  < fx.output_mse_atol, "initial output_mse out of tol");
        CHECK(d_final_omse < fx.output_mse_atol, "final output_mse out of tol");
    }

    // --- Rotation parity ---
    //
    // The QR-Orth iterate is float64+LAPACK on the Python side and
    // float32+Householder in C++; the rotation matrix elements are the
    // most sensitive to per-step round-off, so we assert orthogonality
    // tightly (1e-3) and check the gap to the Python R loosely. We also
    // accept a global -R sign flip (R is only defined up to a column
    // sign if the underlying QR sign canonicalisation disagrees).
    {
        const float * Rcpp = result.R.data();
        const float * Rpy  = fx.rotation.data();
        float ortho_cpp = orthogonality_err(Rcpp, K);
        float ortho_py  = orthogonality_err(Rpy, K);
        float gap_pos   = max_abs_diff(Rcpp, Rpy, K * K);   // |R_cpp - R_py|
        float gap_neg   = max_abs_sum(Rcpp, Rpy, K * K);    // |R_cpp + R_py|
        float gap       = std::min(gap_pos, gap_neg);
        std::printf("  rotation: ortho_cpp=%.3e ortho_py=%.3e\n", ortho_cpp, ortho_py);
        std::printf("           |R-Rpy|=%.3e  |R+Rpy|=%.3e  (min=%.3e, tol=%.3e)\n",
                    gap_pos, gap_neg, gap, fx.rotation_max_abs_atol);
        CHECK(ortho_cpp < fx.rotation_orthogonality_atol, "C++ R not orthogonal");
        CHECK(ortho_py  < fx.rotation_orthogonality_atol, "Python R not orthogonal (fixture corrupted?)");
        CHECK(gap < fx.rotation_max_abs_atol,
              "rotation max-abs gap exceeds tol (report achieved; see sign/rotation note)");
    }

    // --- Functional sanity: rotation lowers output MSE vs identity ---
    //
    // The combined loss has whip_weight = 0.1, so the optimiser trades a
    // small whip regression for an output-MSE gain. Confirm that trade
    // actually happened: final_output_mse must be < initial_output_mse.
    {
        float d_omse = result.final_output_mse - result.initial_output_mse;
        std::printf("  functional: omse delta = %.4e (final-init, negative = improved)\n",
                    d_omse);
        CHECK(result.final_output_mse < result.initial_output_mse,
              "rotation failed to lower output MSE");
    }

    // --- ts_dartquant_apply round-trip sanity ---
    //
    // Apply the learned R block-diagonally to W; the result must equal
    // W @ R (which is exactly what the loss was computed on). This guards
    // against a W @ R vs W @ R^T convention slip in the apply helper.
    {
        std::vector<float> W_rot(fx.out_dim * fx.in_dim);
        ts_dartquant_apply(fx.weight.data(), result.R.data(),
                           W_rot.data(), fx.out_dim, fx.in_dim, fx.block_size);
        // Recompute W @ R directly and compare.
        std::vector<float> W_ref(fx.out_dim * fx.in_dim, 0.0f);
        for (int64_t i = 0; i < fx.out_dim; i++) {
            for (int64_t j = 0; j < fx.in_dim; j++) {
                double s = 0.0;
                for (int64_t k = 0; k < fx.in_dim; k++) {
                    s += (double)fx.weight[i*fx.in_dim + k] *
                         (double)result.R[k*fx.in_dim + j];
                }
                W_ref[i*fx.in_dim + j] = (float)s;
            }
        }
        float apply_err = max_abs_diff(W_rot.data(), W_ref.data(),
                                       fx.out_dim * fx.in_dim);
        std::printf("  apply:    max|ts_dartquant_apply - W@R| = %.3e\n", apply_err);
        CHECK(apply_err < 1e-4f, "ts_dartquant_apply disagrees with W @ R");
    }

    std::printf("\nResults: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
