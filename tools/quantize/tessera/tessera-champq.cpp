#include "tessera-champq.h"
#include "tessera-linalg.h"
#include "tessera-lbfgs.h"

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
#include <numeric>
#include <vector>

// --- CHAMP-Q ---
//
// Ported from tools/tessera/champq_permute.py. The continuous relaxation
// parameterizes the permutation as a doubly-stochastic matrix M (K x K);
// the relaxed weight is W_perm = W @ M. The smoothness loss is the discrete
// second-difference proxy applied to W_perm, optionally activation-weighted.
// L-BFGS descends in M-space; Sinkhorn re-projects M onto the doubly-stochastic
// polytope between steps; Hungarian/greedy recovers a permutation at the end.
//
// All math is float32 (the Python reference keeps the closure in float64 but
// the matmul inputs are f32; the parity test documents the resulting gap).

void ts_champq_sinkhorn(float * M, int64_t n, int64_t n_iters, float eps) {
    for (int64_t i = 0; i < n * n; i++) {
        M[i] = std::max(M[i], eps);
    }
    for (int64_t iter = 0; iter < n_iters; iter++) {
        // row normalization
        for (int64_t i = 0; i < n; i++) {
            float s = 0.0f;
            for (int64_t j = 0; j < n; j++) {
                s += M[i * n + j];
            }
            if (s > eps) {
                float inv = 1.0f / s;
                for (int64_t j = 0; j < n; j++) {
                    M[i * n + j] *= inv;
                }
            }
        }
        // column normalization
        for (int64_t j = 0; j < n; j++) {
            float s = 0.0f;
            for (int64_t i = 0; i < n; i++) {
                s += M[i * n + j];
            }
            if (s > eps) {
                float inv = 1.0f / s;
                for (int64_t i = 0; i < n; i++) {
                    M[i * n + j] *= inv;
                }
            }
        }
    }
}

// Hungarian algorithm (Kuhn-Munkres), O(n^3).
// `cost` is (n x n) row-major; lower cost = preferred.
// assignment[i] = column assigned to row i.
static void ts_hungarian(const float * cost, int32_t * assignment, int64_t n) {
    if (n == 0) {
        return;
    }
    const float INF = 1e30f;
    std::vector<float> u(n + 1, 0.0f), v(n + 1, 0.0f), minv(n + 1);
    std::vector<int64_t> p(n + 1, 0), way(n + 1, 0);
    std::vector<bool> used(n + 1);

    for (int64_t i = 1; i <= n; i++) {
        p[0] = i;
        int64_t j0 = 0;
        std::fill(minv.begin(), minv.end(), INF);
        std::fill(used.begin(), used.end(), false);

        while (true) {
            used[j0] = true;
            int64_t i0 = p[j0];
            float delta = INF;
            int64_t j1 = 0;
            for (int64_t j = 1; j <= n; j++) {
                if (used[j]) continue;
                float cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int64_t j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if (p[j0] == 0) break;
        }
        while (true) {
            int64_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) break;
        }
    }

    std::fill(assignment, assignment + n, 0);
    for (int64_t j = 1; j <= n; j++) {
        if (p[j] != 0) {
            assignment[p[j] - 1] = (int32_t)(j - 1);
        }
    }
}

// greedy assignment: sort all (row, col) by descending M value
static void ts_greedy_assignment(const float * M, int32_t * assignment, int64_t n) {
    std::vector<int64_t> order(n * n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return M[a] > M[b];
    });

    std::vector<bool> col_used(n, false), row_used(n, false);
    std::fill(assignment, assignment + n, -1);

    for (int64_t idx : order) {
        int64_t r = idx / n;
        int64_t c = idx % n;
        if (!col_used[c] && !row_used[r]) {
            assignment[r] = (int32_t)c;
            col_used[c] = true;
            row_used[r] = true;
        }
    }
    // fill leftovers
    int64_t next_col = 0;
    for (int64_t i = 0; i < n; i++) {
        if (assignment[i] == -1) {
            while (next_col < n && col_used[next_col]) next_col++;
            assignment[i] = (int32_t)next_col;
            if (next_col < n) col_used[next_col] = true;
        }
    }
}

// Smoothness proxy on a (already permuted) 2-D weight.
// d2[r, c] = W[r, c-1] - 2*W[r, c] + W[r, c+1] for c in [1, K-2].
// Weighted by act_scales[c] at the stencil center; act_scales may be null.
float ts_champq_smoothness(const float * W, const int32_t * perm,
                           const float * act_scales,
                           int64_t out_dim, int64_t in_dim) {
    float s = 0.0f;
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 1; c < in_dim - 1; c++) {
            float v0 = W[r * in_dim + perm[c - 1]];
            float v1 = W[r * in_dim + perm[c]];
            float v2 = W[r * in_dim + perm[c + 1]];
            float d2 = v0 - 2.0f * v1 + v2;
            float term = d2 * d2;
            if (act_scales) {
                term *= act_scales[c];
            }
            s += term;
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Objective: smoothness loss + gradient on the relaxed M (K x K).
// Mirrors champq_permute.smoothness_loss_grad.
// ---------------------------------------------------------------------------

struct ts_champq_ctx {
    const float * W;
    int64_t       out_dim;
    int64_t       in_dim;
    float         binariness;
    const float * act_scales;  // length in_dim, may be null
};

static float ts_champq_eval(const float * x, float * grad, int64_t n, void * ctx) {
    auto * c = (ts_champq_ctx *)ctx;
    int64_t K = c->in_dim;
    int64_t out_dim = c->out_dim;
    const float * W = c->W;
    float binariness = c->binariness;
    const float * act = c->act_scales;

    // Trivial interior case: only the binariness penalty contributes.
    if (K < 3) {
        float loss = 0.0f;
        if (binariness > 0.0f) {
            for (int64_t i = 0; i < n; i++) {
                loss += binariness * x[i] * (1.0f - x[i]);
                grad[i] = binariness * (1.0f - 2.0f * x[i]);
            }
        } else {
            std::memset(grad, 0, (size_t)n * sizeof(float));
        }
        return loss;
    }

    // W_perm = W @ M : (out_dim x K)
    std::vector<float> W_perm(out_dim * K);
#if defined(TS_HAS_CBLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)out_dim, (int)K, (int)K,
                1.0f, W, (int)K, x, (int)K, 0.0f, W_perm.data(), (int)K);
#else
    for (int64_t r = 0; r < out_dim; r++) {
        const float * Wr = W + r * K;
        for (int64_t col = 0; col < K; col++) {
            float s = 0.0f;
            for (int64_t i = 0; i < K; i++) {
                s += Wr[i] * x[i * K + col];
            }
            W_perm[r * K + col] = s;
        }
    }
#endif

    // d2[r, c] = W_perm[r, c] - 2*W_perm[r, c+1] + W_perm[r, c+2], c in [0, K-3].
    // h[r, c] = d2[r, c] * weight, where weight is act[c+1] (the stencil
    // center) or 1.0. Stored shifted into a length-K buffer so h[r, k]
    // corresponds to center column k in [1, K-2] (matches the Python
    // h_pad[:, 1:K-1] = h layout).
    std::vector<float> h(out_dim * K, 0.0f);
    float loss = 0.0f;
    for (int64_t r = 0; r < out_dim; r++) {
        const float * wp = W_perm.data() + r * K;
        for (int64_t c = 0; c < K - 2; c++) {
            float d2 = wp[c] - 2.0f * wp[c + 1] + wp[c + 2];
            float d2sq = d2 * d2;
            if (act) {
                // center column is c+1
                float w = act[c + 1];
                d2sq *= w;
                h[r * K + (c + 1)] = d2 * w;
            } else {
                h[r * K + (c + 1)] = d2;
            }
            loss += d2sq;
        }
    }

    // g[r, k] = d L / d W_perm[r, k] = 2 * (h[r, k-1] - 2*h[r, k] + h[r, k+1])
    // for interior k in [1, K-2]. Boundary k=0, k=K-1 stays 0 (the missing
    // neighbour in the padded h buffer is zero).
    std::vector<float> g(out_dim * K, 0.0f);
    for (int64_t r = 0; r < out_dim; r++) {
        const float * hr = h.data() + r * K;
        float * gr = g.data() + r * K;
        for (int64_t k = 1; k < K - 1; k++) {
            gr[k] = 2.0f * (hr[k - 1] - 2.0f * hr[k] + hr[k + 1]);
        }
    }

    // grad_M[i, k] = sum_r W[r, i] * g[r, k] = (W^T @ g)[i, k]
#if defined(TS_HAS_CBLAS)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)K, (int)K, (int)out_dim,
                1.0f, W, (int)K, g.data(), (int)K,
                0.0f, grad, (int)K);
#else
    for (int64_t i = 0; i < K; i++) {
        for (int64_t k = 0; k < K; k++) {
            float s = 0.0f;
            for (int64_t r = 0; r < out_dim; r++) {
                s += W[r * K + i] * g[r * K + k];
            }
            grad[i * K + k] = s;
        }
    }
#endif

    if (binariness > 0.0f) {
        for (int64_t i = 0; i < n; i++) {
            loss += binariness * x[i] * (1.0f - x[i]);
            grad[i] += binariness * (1.0f - 2.0f * x[i]);
        }
    }

    return loss;
}

// ---------------------------------------------------------------------------
// Search driver.
// ---------------------------------------------------------------------------

struct ts_champq_project_ctx {
    int64_t K;
    int64_t sinkhorn_iters;
};

static void ts_champq_project(float * x, int64_t n, void * ctx) {
    auto * c = (ts_champq_project_ctx *)ctx;
    // clamp to non-negative, then Sinkhorn-normalize.
    for (int64_t i = 0; i < n; i++) {
        x[i] = std::max(x[i], 1e-12f);
    }
    ts_champq_sinkhorn(x, c->K, c->sinkhorn_iters, 1e-12f);
}

// argsort descending with stable tie-break on index (matches numpy's
// argsort(-magnitudes, kind="stable") which keeps ascending index order
// among equal magnitudes). When `magnitudes` is null the per-column L2 norm
// of W is used; otherwise the supplied per-channel magnitudes are used
// (Python's compute_champq_permutation switches to act_scales when given).
static void ts_argsort_desc(const float * W, const float * magnitudes,
                            int64_t out_dim, int64_t in_dim,
                            std::vector<int64_t> & perm) {
    std::vector<double> mag(in_dim);
    if (magnitudes) {
        for (int64_t j = 0; j < in_dim; j++) {
            mag[j] = (double)magnitudes[j];
        }
    } else {
        for (int64_t j = 0; j < in_dim; j++) {
            double s = 0.0;
            for (int64_t i = 0; i < out_dim; i++) {
                float v = W[i * in_dim + j];
                s += (double)v * (double)v;
            }
            mag[j] = std::sqrt(s);
        }
    }
    perm.resize(in_dim);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        if (mag[a] != mag[b]) {
            return mag[a] > mag[b];
        }
        return a < b;
    });
}

// Minimal deterministic RNG (xorshift32) for the random init mode, so the
// C++ init reproduces numpy default_rng(seed).permutation-free behaviour:
// we only need a reproducible non-negative matrix to feed Sinkhorn. The
// parity test uses init=l2rank (the default), so this path is exercised
// only by the convergence smoke test.
static uint32_t ts_xs32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int ts_champq_compute(const float * weights, int64_t out_dim, int64_t in_dim,
                      const ts_champq_params * params, ts_champq_result * result) {
    if (!weights || !params || !result || out_dim <= 0 || in_dim <= 0) {
        return 1;
    }
    int64_t max_iters      = params->max_iters > 0 ? params->max_iters : 100;
    int64_t sinkhorn_iters = params->sinkhorn_iters > 0 ? params->sinkhorn_iters : 25;
    int64_t history        = params->history > 0 ? params->history : 8;
    float   binariness     = params->binariness;          // 0 by value-init
    int32_t projection     = params->projection;          // 0 = hungarian
    int32_t init           = params->init;                // 0 = l2rank

    int64_t K = in_dim;
    int64_t n = K * K;

    // Init M.
    std::vector<float> M(n, 0.0f);
    if (init == 2) {
        // random: Sinkhorn(uniform). Deterministic given seed.
        uint32_t st = params->seed ? params->seed : 1u;
        for (int64_t i = 0; i < n; i++) {
            M[i] = (float)((double)(ts_xs32(st) >> 8) / (double)(1u << 24));
        }
        ts_champq_project_ctx pctx = { K, std::max<int64_t>(sinkhorn_iters, 50) };
        ts_champq_project(M.data(), n, &pctx);
    } else if (init == 1) {
        // identity
        for (int64_t i = 0; i < K; i++) {
            M[i * K + i] = 1.0f;
        }
    } else {
        // l2rank: embed the rank permutation as a permutation matrix.
        // When act_scales is provided, rank by act_scales (matches Python's
        // compute_champq_permutation); otherwise rank by per-column L2 norm.
        std::vector<int64_t> rank_perm;
        ts_argsort_desc(weights, params->act_scales, out_dim, in_dim, rank_perm);
        for (int64_t i = 0; i < K; i++) {
            M[i * K + rank_perm[i]] = 1.0f;
        }
    }

    // baseline smoothness at the identity perm (for the improvement ratio).
    std::vector<int32_t> id_perm(K);
    std::iota(id_perm.begin(), id_perm.end(), 0);
    float baseline_smooth =
        ts_champq_smoothness(weights, id_perm.data(), params->act_scales, out_dim, in_dim);

    if (params->use_lbfgs) {
        ts_champq_ctx eval_ctx;
        eval_ctx.W           = weights;
        eval_ctx.out_dim     = out_dim;
        eval_ctx.in_dim      = in_dim;
        eval_ctx.binariness  = binariness;
        eval_ctx.act_scales  = params->act_scales;

        ts_champq_project_ctx proj_ctx = { K, sinkhorn_iters };

        // L-BFGS project-between-steps driver. Mirrors
        // champq_permute.lbfgs_permutation:
        //   1. eval(M) -> loss, grad
        //   2. step (Armijo line search) -> M_new, loss_new, grad_new
        //   3. Sinkhorn-project M_new
        //   4. recompute loss/grad at the projected point
        // The trajectory (loss at the projected point each iteration) is
        // recorded so the parity test can compare against Python.
        ts_lbfgs_state * st = ts_lbfgs_state_create(n, history);

        std::vector<float> grad(n), grad_new(n), x_new(n);
        float loss = ts_champq_eval(M.data(), grad.data(), n, &eval_ctx);
        result->loss_history.push_back(loss);

        // Python: max_ls = 12 for the CHAMP-Q closure.
        ts_lbfgs_step_params sp = { 1e-4f, 12, 0.5f, 1e-12f };

        for (int64_t it = 0; it < max_iters; it++) {
            int done = 0;
            ts_lbfgs_step(st, M.data(), loss, grad.data(),
                          ts_champq_eval, &eval_ctx, &sp,
                          x_new.data(), grad_new.data(), &done);

            // Re-project onto the doubly-stochastic manifold.
            ts_champq_project(x_new.data(), n, &proj_ctx);

            // Recompute loss/grad at the projected point so the next step
            // has consistent values (the projection invalidates grad_new).
            std::memcpy(M.data(), x_new.data(), (size_t)n * sizeof(float));
            loss = ts_champq_eval(M.data(), grad.data(), n, &eval_ctx);
            result->loss_history.push_back(loss);

            // NOTE: champq_permute.lbfgs_permutation does NOT stop on a
            // line-search failure (done); it only stops on the gradient-norm
            // tol. We match that here: keep iterating to max_iters. The
            // L-BFGS ring buffer self-clears on curvature violations, so a
            // failed step just degrades gracefully to steepest descent next.
            (void)done;
        }
        ts_lbfgs_state_destroy(st);
    }

    // Final projection to a permutation. Hungarian minimizes cost = -M.
    std::vector<int32_t> perm(K);
    bool use_greedy = (projection == 1) || (K > 512);
    if (use_greedy) {
        ts_greedy_assignment(M.data(), perm.data(), K);
    } else {
        std::vector<float> cost(n);
        for (int64_t i = 0; i < n; i++) {
            cost[i] = -M[i];
        }
        ts_hungarian(cost.data(), perm.data(), K);
    }

    float final_smooth =
        ts_champq_smoothness(weights, perm.data(), params->act_scales, out_dim, in_dim);

    result->perm = std::move(perm);
    result->smoothness = final_smooth;
    result->mse_improvement = baseline_smooth > 0.0f
        ? (baseline_smooth - final_smooth) / baseline_smooth
        : 0.0f;
    return 0;
}

void ts_champq_apply(const float * W, const int32_t * perm,
                     float * W_perm, int64_t out_dim, int64_t in_dim) {
    for (int64_t i = 0; i < out_dim; i++) {
        for (int64_t j = 0; j < in_dim; j++) {
            W_perm[i * in_dim + j] = W[i * in_dim + perm[j]];
        }
    }
}

void ts_champq_invert(const int32_t * perm, int32_t * inv, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        inv[perm[i]] = (int32_t)i;
    }
}
