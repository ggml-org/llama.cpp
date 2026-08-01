//
// tessera-dartquant.cpp
//
// DartQuant (distribution-aware rotation) regime expert.
// Port of tools/tessera/per_tensor_calibrate.py::dartquant_qr_orth.
//
// Optimizes an orthogonal rotation R on the Stiefel manifold that minimizes
//   L = L_quant + whip_weight * L_whip
// using QR-Orth retraction. L_quant is the asymmetric-STE output MSE
// (active when calibration activations X are supplied); L_whip is the
// coefficient of variation of per-row L2 norms (optionally activation-
// weighted by X_hat). QR / Stiefel primitives are reused from
// tessera-linalg.cpp; this file holds the DartQuant-specific loss,
// gradient, and iteration driver.
//
// Convention (matches Python): W_rot = W @ R, R is (in_dim x in_dim).
// Row-major float32 throughout. The optimization runs on the first
// K-wide column block of W (K = block_size, defaults to in_dim) and the
// returned R is applied block-diagonally by ts_dartquant_apply.
//

#include "tessera-dartquant.h"
#include "tessera-linalg.h"

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define TS_HAS_CBLAS 1
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#define TS_HAS_CBLAS 1
#endif

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Shared reconstruction primitive: ternarize to {-1,0,+1} then rescale by
// the global mean(|W|). Mirrors Python's ternarize + ternary_recon in
// per_tensor_calibrate.py. Also reused by tessera-lrq.cpp.
// ---------------------------------------------------------------------------

static float ts_mean_abs(const float * x, int64_t n) {
    double s = 0.0;  // accumulate in double for parity with numpy
    for (int64_t i = 0; i < n; i++) {
        s += std::fabs((double)x[i]);
    }
    return (float)(s / (double)n);
}

// ternarize_recon: sign(x) * (|x| >= mean(|x|)) * mean(|x|)
static void ts_ternarize_recon(const float * x, float * out, int64_t n) {
    float scale = ts_mean_abs(x, n);
    if (scale <= 0.0f) {
        memset(out, 0, n * sizeof(float));
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        out[i] = (std::fabs(v) >= scale)
                 ? ((v > 0.0f ? 1.0f : -1.0f) * scale)
                 : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Matmul helpers (row-major). Kept local so this file is self-contained;
// tessera-linalg has its own static copies which are file-local.
// ---------------------------------------------------------------------------

// C(m x n) = A(m x k) @ B(k x n). Double accumulation to match
// Python numpy matmul (which promotes f32 inputs to f64 accumulation).
static void ts_dq_matmul(const float * A, const float * B, float * C,
                         int64_t m, int64_t k, int64_t n) {
    if (m <= 0 || n <= 0) return;
#if defined(TS_HAS_CBLAS)
    // Promote to double for parity with the scalar f64 accumulation path.
    std::vector<double> Ad((size_t)m * k), Bd((size_t)k * n), Cd((size_t)m * n);
    for (int64_t i = 0; i < m * k; i++) Ad[i] = (double)A[i];
    for (int64_t i = 0; i < k * n; i++) Bd[i] = (double)B[i];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                1.0, Ad.data(), (int)k, Bd.data(), (int)n, 0.0, Cd.data(), (int)n);
    for (int64_t i = 0; i < m * n; i++) C[i] = (float)Cd[i];
#else
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            double s = 0.0;
            for (int64_t p = 0; p < k; p++) {
                s += (double)A[i*k + p] * (double)B[p*n + j];
            }
            C[i*n + j] = (float)s;
        }
    }
#endif
}

// C(n x k) = A(m x n)^T @ B(m x k)
static void ts_dq_matmul_atb(const float * A, const float * B, float * C,
                             int64_t m, int64_t n, int64_t k) {
    if (n <= 0 || k <= 0) return;
#if defined(TS_HAS_CBLAS)
    std::vector<double> Ad((size_t)m * n), Bd((size_t)m * k), Cd((size_t)n * k);
    for (int64_t i = 0; i < m * n; i++) Ad[i] = (double)A[i];
    for (int64_t i = 0; i < m * k; i++) Bd[i] = (double)B[i];
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)n, (int)k, (int)m,
                1.0, Ad.data(), (int)n, Bd.data(), (int)k, 0.0, Cd.data(), (int)k);
    for (int64_t i = 0; i < n * k; i++) C[i] = (float)Cd[i];
#else
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < k; j++) {
            double s = 0.0;
            for (int64_t r = 0; r < m; r++) {
                s += (double)A[r*n + i] * (double)B[r*k + j];
            }
            C[i*k + j] = (float)s;
        }
    }
#endif
}

// C(m x n) = A(m x k) @ B(n x k)^T, i.e. C[i,j] = sum_p A[i,p] * B[j,p].
// Used for residual(out,K) @ X(n_tokens,K)^T -> err(out,n_tokens).
static void ts_dq_matmul_abt(const float * A, const float * B, float * C,
                             int64_t m, int64_t k, int64_t n) {
    if (m <= 0 || n <= 0) return;
#if defined(TS_HAS_CBLAS)
    std::vector<double> Ad((size_t)m * k), Bd((size_t)n * k), Cd((size_t)m * n);
    for (int64_t i = 0; i < m * k; i++) Ad[i] = (double)A[i];
    for (int64_t i = 0; i < n * k; i++) Bd[i] = (double)B[i];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)m, (int)n, (int)k,
                1.0, Ad.data(), (int)k, Bd.data(), (int)k, 0.0, Cd.data(), (int)n);
    for (int64_t i = 0; i < m * n; i++) C[i] = (float)Cd[i];
#else
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            double s = 0.0;
            for (int64_t p = 0; p < k; p++) {
                s += (double)A[i*k + p] * (double)B[j*k + p];
            }
            C[i*n + j] = (float)s;
        }
    }
#endif
}

// Per-row L2 norms with a 1e-24 floor (matches Python _row_l2).
static void ts_row_l2(const float * W, float * norms, int64_t out_dim, int64_t in_dim) {
    for (int64_t i = 0; i < out_dim; i++) {
        double s = 0.0;
        const float * row = W + i*in_dim;
        for (int64_t j = 0; j < in_dim; j++) {
            s += (double)row[j] * (double)row[j];
        }
        norms[i] = sqrtf(std::max((float)s, 1e-24f));
    }
}

// ---------------------------------------------------------------------------
// Whip loss: std(||W[o,:]||_2) / mean(||W[o,:]||_2). Optionally weight the
// rows by per-input-channel sqrt(E[x^2]) == X_hat (SmoothQuant convention).
// Matches Python dartquant_whip_loss.
// ---------------------------------------------------------------------------

static float ts_whip_loss(const float * W_rot, int64_t out_dim, int64_t in_dim,
                          const float * X_hat) {
    std::vector<float> Ww;
    const float * W = W_rot;
    if (X_hat) {
        Ww.resize(out_dim * in_dim);
        for (int64_t i = 0; i < out_dim; i++) {
            for (int64_t j = 0; j < in_dim; j++) {
                Ww[i*in_dim + j] = W_rot[i*in_dim + j] * X_hat[j];
            }
        }
        W = Ww.data();
    }
    std::vector<float> norms(out_dim);
    ts_row_l2(W, norms.data(), out_dim, in_dim);
    double mean = 0.0;
    for (int64_t i = 0; i < out_dim; i++) mean += norms[i];
    mean /= (double)out_dim;
    if (mean < 1e-12) {
        return 0.0f;
    }
    double var = 0.0;
    for (int64_t i = 0; i < out_dim; i++) {
        double d = (double)norms[i] - mean;
        var += d*d;
    }
    var /= (double)out_dim;
    return (float)(std::sqrt(var) / mean);
}

// Relative tile-quant MSE: ||W - Q(W)||_F^2 / (||W||_F^2 + 1e-12).
// Matches Python _relative_quant_mse.
static float ts_relative_quant_mse(const float * W, int64_t out_dim, int64_t in_dim) {
    std::vector<float> Wq(out_dim * in_dim);
    ts_ternarize_recon(W, Wq.data(), out_dim * in_dim);
    double num = 0.0, den = 0.0;
    int64_t n = out_dim * in_dim;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)Wq[i] - (double)W[i];
        num += d*d;
        den += (double)W[i] * (double)W[i];
    }
    den += 1e-12;
    return (float)(num / den);
}

// ---------------------------------------------------------------------------
// Whip-loss gradient w.r.t. R. Matches Python _whip_grad_impl:
//   Wp      = W @ R                         (out x in)
//   Wp     *= X_hat[None,:]  if X_hat       (activation-weighted)
//   f_o     = ||Wp[o,:]||_2
//   diff_o  = (f_o - mean(f)) / max(f_o, 1e-12)
//   dWp     = diff_o[:,None] * Wp           (out x in)
//   grad    = (1/(out*mean(f))) * W^T @ dWp (in x in)
// ---------------------------------------------------------------------------

static void ts_whip_grad(const float * W, const float * R, float * grad,
                         int64_t out_dim, int64_t K,
                         const float * X_hat) {
    std::vector<float> Wp(out_dim * K);
    ts_dq_matmul(W, R, Wp.data(), out_dim, K, K);
    if (X_hat) {
        for (int64_t i = 0; i < out_dim; i++) {
            for (int64_t j = 0; j < K; j++) {
                Wp[i*K + j] *= X_hat[j];
            }
        }
    }
    std::vector<float> norms(out_dim);
    ts_row_l2(Wp.data(), norms.data(), out_dim, K);
    double mean_n = 0.0;
    for (int64_t i = 0; i < out_dim; i++) mean_n += norms[i];
    mean_n /= (double)out_dim;
    if (mean_n < 1e-12) {
        memset(grad, 0, K*K * sizeof(float));
        return;
    }
    std::vector<float> dWp(out_dim * K);
    for (int64_t o = 0; o < out_dim; o++) {
        float diff = (float)(((double)norms[o] - mean_n) /
                             std::max((double)norms[o], 1e-12));
        for (int64_t j = 0; j < K; j++) {
            dWp[o*K + j] = diff * Wp[o*K + j];
        }
    }
    // grad = W^T @ dWp / (out * mean)
    ts_dq_matmul_atb(W, dWp.data(), grad, out_dim, K, K);
    float inv = (float)(1.0 / ((double)out_dim * mean_n));
    for (int64_t i = 0; i < K*K; i++) {
        grad[i] *= inv;
    }
}

// ---------------------------------------------------------------------------
// Output-MSE loss + gradient. Matches Python _output_mse_and_grad.
//
// Forward (with X):
//   Wp = W @ R ; Wq = Q(Wp)
//   residual = Wq - W                          (asymmetric STE; -W constant)
//   err      = residual @ X^T                  (out x n_tokens)
//   L        = mean(err * err)
//
// Backward (asymmetric STE; gradient flows only through Q -> Wp):
//   dL/dWp = (2/N) * err @ X                   (out x in)
//   dL/dR  = W^T @ dL/dWp                      (in x in)
//
// Without X: L = mean((Wp - Wq)^2), gradient is exactly zero (STE collapses).
// ---------------------------------------------------------------------------

static float ts_output_mse_and_grad(const float * W, const float * R,
                                    const float * X, int64_t X_count,
                                    int64_t out_dim, int64_t K,
                                    float * grad_out) {
    std::vector<float> Wp(out_dim * K);
    ts_dq_matmul(W, R, Wp.data(), out_dim, K, K);
    std::vector<float> Wq(out_dim * K);
    ts_ternarize_recon(Wp.data(), Wq.data(), out_dim * K);

    if (X == nullptr || X_count == 0) {
        double loss = 0.0;
        int64_t n = out_dim * K;
        for (int64_t i = 0; i < n; i++) {
            double d = (double)Wq[i] - (double)Wp[i];
            loss += d*d;
        }
        loss /= (double)n;
        if (grad_out) {
            memset(grad_out, 0, K*K * sizeof(float));
        }
        return (float)loss;
    }

    // residual = Wq - W (out x K); err = residual @ X^T (out x n_tokens)
    std::vector<float> residual(out_dim * K);
    for (int64_t i = 0; i < out_dim * K; i++) {
        residual[i] = Wq[i] - W[i];
    }
    std::vector<float> err(out_dim * X_count);
    // err[o, t] = sum_j residual[o, j] * X[t, j]  ->  err = residual @ X^T
    ts_dq_matmul_abt(residual.data(), X, err.data(), out_dim, K, X_count);

    double loss = 0.0;
    int64_t n_err = out_dim * X_count;
    for (int64_t i = 0; i < n_err; i++) {
        loss += (double)err[i] * (double)err[i];
    }
    loss /= (double)n_err;

    if (grad_out) {
        // dL/dWp = (2/N) * err @ X  (out x K)
        std::vector<float> dL_dWp(out_dim * K);
        double inv = 2.0 / (double)n_err;
#if defined(TS_HAS_CBLAS)
        // err is (out_dim x X_count), X is (X_count x K). C = err @ X.
        // Double accumulation to match the scalar path.
        std::vector<double> err_d((size_t)out_dim * X_count);
        std::vector<double> X_d((size_t)X_count * K);
        std::vector<double> dL_dWp_d((size_t)out_dim * K);
        for (int64_t i = 0; i < out_dim * X_count; i++) err_d[i] = (double)err[i];
        for (int64_t i = 0; i < X_count * K; i++) X_d[i] = (double)X[i];
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)out_dim, (int)K, (int)X_count,
                    inv, err_d.data(), (int)X_count, X_d.data(), (int)K,
                    0.0, dL_dWp_d.data(), (int)K);
        for (int64_t i = 0; i < out_dim * K; i++)
            dL_dWp[i] = (float)dL_dWp_d[i];
#else
        for (int64_t o = 0; o < out_dim; o++) {
            for (int64_t j = 0; j < K; j++) {
                double s = 0.0;
                for (int64_t t = 0; t < X_count; t++) {
                    s += (double)err[o*X_count + t] * (double)X[t*K + j];
                }
                dL_dWp[o*K + j] = (float)(inv * s);
            }
        }
#endif
        // grad = W^T @ dL/dWp (in x in)
        ts_dq_matmul_atb(W, dL_dWp.data(), grad_out, out_dim, K, K);
    }
    return (float)loss;
}

// ---------------------------------------------------------------------------
// Full driver. Matches Python dartquant_qr_orth (one column-block of width K).
// ---------------------------------------------------------------------------

int ts_dartquant_qr_orth(const float * weights, int64_t out_dim, int64_t in_dim,
                         const float * X, int64_t X_count,
                         const float * X_hat,
                         const ts_dartquant_params * params,
                         ts_dartquant_result * result) {
    if (!weights || !params || !result || out_dim <= 0 || in_dim <= 0) {
        return -1;
    }

    int64_t K = params->block_size > 0 ? params->block_size : in_dim;
    if (K > in_dim) K = in_dim;
    if (K <= 0) return -1;

    int64_t max_iters    = params->max_iters > 0 ? params->max_iters : 50;
    float    lr          = params->lr > 0.0f ? params->lr : 1.0e-2f;
    float    whip_weight = params->whip_weight;
    uint32_t seed        = params->seed;

    // Work on the first K-wide column block of W (same simplification as
    // the Python demo: a single shared R for every column block).
    std::vector<float> W_block(out_dim * K);
    for (int64_t i = 0; i < out_dim; i++) {
        memcpy(&W_block[i*K], &weights[i*in_dim], K * sizeof(float));
    }

    // Initial rotation: identity (seed==0) or Haar-uniform orthogonal.
    std::vector<float> R(K * K);
    if (seed) {
        ts_linalg_random_orthogonal(R.data(), K, seed);
    } else {
        memset(R.data(), 0, K*K * sizeof(float));
        for (int64_t i = 0; i < K; i++) {
            R[i*K + i] = 1.0f;
        }
    }

    // Initial metrics (mirror Python: Wp = W @ R, whip, qmse, omse).
    std::vector<float> Wp(out_dim * K);
    ts_dq_matmul(W_block.data(), R.data(), Wp.data(), out_dim, K, K);
    float initial_whip     = ts_whip_loss(Wp.data(), out_dim, K, X_hat);
    float initial_qmse     = ts_relative_quant_mse(Wp.data(), out_dim, K);
    float initial_omse     = ts_output_mse_and_grad(W_block.data(), R.data(),
                                                    X, X_count, out_dim, K, nullptr);

    result->history.clear();
    result->history.reserve((size_t)max_iters + 1);
    result->history.push_back(initial_qmse + whip_weight * initial_whip);

    // Best-so-far tracking on output MSE (the deployment metric). When X is
    // absent, output MSE == tile-quant-style reconstruction MSE and we still
    // track it; otherwise we track the X-projected error as Python does.
    std::vector<float> best_R = R;
    float best_omse  = initial_omse;
    float best_qmse  = initial_qmse;
    float best_whip  = initial_whip;

    std::vector<float> q_grad(K * K), w_grad(K * K), total_grad(K * K);

    for (int64_t it = 0; it < max_iters; it++) {
        ts_output_mse_and_grad(W_block.data(), R.data(), X, X_count,
                               out_dim, K, q_grad.data());
        ts_whip_grad(W_block.data(), R.data(), w_grad.data(), out_dim, K, X_hat);
        for (int64_t i = 0; i < K*K; i++) {
            total_grad[i] = q_grad[i] + whip_weight * w_grad[i];
            total_grad[i] = -total_grad[i];  // descent: qr_orth_step adds lr*G
        }
        ts_linalg_qr_orth_step(R.data(), total_grad.data(), lr, K, K);

        ts_dq_matmul(W_block.data(), R.data(), Wp.data(), out_dim, K, K);
        float cur_whip = ts_whip_loss(Wp.data(), out_dim, K, X_hat);
        float cur_qmse = ts_relative_quant_mse(Wp.data(), out_dim, K);
        float cur_omse = ts_output_mse_and_grad(W_block.data(), R.data(),
                                                X, X_count, out_dim, K, nullptr);
        result->history.push_back(cur_qmse + whip_weight * cur_whip);
        if (cur_omse < best_omse) {
            best_omse = cur_omse;
            best_qmse = cur_qmse;
            best_whip = cur_whip;
            best_R    = R;
        }
    }

    // Python returns the FINAL rotation (not best). We match that: the
    // reported final_* metrics are on the final R; best_* metrics are not
    // surfaced (kept internally for parity audit only).
    ts_dq_matmul(W_block.data(), R.data(), Wp.data(), out_dim, K, K);
    float final_whip = ts_whip_loss(Wp.data(), out_dim, K, X_hat);
    float final_qmse = ts_relative_quant_mse(Wp.data(), out_dim, K);
    float final_omse = ts_output_mse_and_grad(W_block.data(), R.data(),
                                              X, X_count, out_dim, K, nullptr);
    (void)best_R; (void)best_omse; (void)best_qmse; (void)best_whip;

    result->R                  = std::move(R);
    result->initial_whip       = initial_whip;
    result->final_whip         = final_whip;
    result->initial_quant_mse  = initial_qmse;
    result->final_quant_mse    = final_qmse;
    result->initial_output_mse = initial_omse;
    result->final_output_mse   = final_omse;
    result->n_iters            = max_iters;
    result->whip_loss          = final_whip;
    // Legacy mse field: reconstruction MSE (not relative) on W @ R, kept so
    // test_search.cpp's "mse should drop" assertion still has a value to read.
    {
        std::vector<float> Wq(out_dim * K);
        ts_ternarize_recon(Wp.data(), Wq.data(), out_dim * K);
        double mse = 0.0;
        int64_t n = out_dim * K;
        for (int64_t i = 0; i < n; i++) {
            double d = (double)Wp[i] - (double)Wq[i];
            mse += d*d;
        }
        result->mse = (float)(mse / (double)n);
    }
    return 0;
}

// Back-compat shim: drop X / X_hat and call the full driver.
int ts_dartquant_qr_orth(const float * weights, int64_t out_dim, int64_t in_dim,
                         const ts_dartquant_params * params,
                         ts_dartquant_result * result) {
    return ts_dartquant_qr_orth(weights, out_dim, in_dim,
                                nullptr, 0, nullptr, params, result);
}

// ---------------------------------------------------------------------------
// Apply rotation block-diagonally along in_dim. R is (block_size x block_size),
// applied as W_rot[:, b*K:(b+1)*K] = W[:, b*K:(b+1)*K] @ R for each block b.
// Convention W_rot = W @ R matches Python dartquant_apply_rotation when
// block_size == in_dim (single block).
// ---------------------------------------------------------------------------

void ts_dartquant_apply(const float * W, const float * R,
                        float * W_rot, int64_t out_dim, int64_t in_dim,
                        int64_t block_size) {
    int64_t n_blocks = in_dim / block_size;
    for (int64_t b = 0; b < n_blocks; b++) {
        int64_t col_off = b * block_size;
#if defined(TS_HAS_CBLAS)
        // Double accumulation to match the scalar path.
        std::vector<double> Wd((size_t)out_dim * block_size);
        std::vector<double> Rd((size_t)block_size * block_size);
        std::vector<double> Cd((size_t)out_dim * block_size);
        for (int64_t i = 0; i < out_dim; i++)
            for (int64_t k = 0; k < block_size; k++)
                Wd[i*block_size + k] = (double)W[i*in_dim + col_off + k];
        for (int64_t i = 0; i < block_size * block_size; i++)
            Rd[i] = (double)R[i];
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)out_dim, (int)block_size, (int)block_size,
                    1.0, Wd.data(), (int)block_size, Rd.data(), (int)block_size,
                    0.0, Cd.data(), (int)block_size);
        for (int64_t i = 0; i < out_dim; i++)
            for (int64_t j = 0; j < block_size; j++)
                W_rot[i*in_dim + col_off + j] = (float)Cd[i*block_size + j];
#else
        for (int64_t i = 0; i < out_dim; i++) {
            const float * w_row = W + i*in_dim + col_off;
            float * out_row = W_rot + i*in_dim + col_off;
            for (int64_t j = 0; j < block_size; j++) {
                double s = 0.0;
                for (int64_t k = 0; k < block_size; k++) {
                    s += (double)w_row[k] * (double)R[k*block_size + j];
                }
                out_row[j] = (float)s;
            }
        }
#endif
    }
    // Remainder columns (in_dim not a multiple of block_size) are copied
    // verbatim, matching the no-op behaviour of an identity tail block.
    int64_t rem = in_dim - n_blocks * block_size;
    if (rem > 0) {
        int64_t col_off = n_blocks * block_size;
        for (int64_t i = 0; i < out_dim; i++) {
            memcpy(&W_rot[i*in_dim + col_off],
                   &W[i*in_dim + col_off],
                   rem * sizeof(float));
        }
    }
}
