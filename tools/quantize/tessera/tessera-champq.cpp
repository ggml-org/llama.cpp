#include "tessera-champq.h"
#include "tessera-linalg.h"
#include "tessera-lbfgs.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

// --- CHAMP-Q ---

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

// Hungarian algorithm (Kuhn-Munkres), O(n^3)
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
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
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

struct ts_champq_ctx {
    const float * W;
    int64_t       out_dim;
    int64_t       in_dim;
    float         binariness;
};

static float ts_champq_eval(const float * x, float * grad, int64_t n, void * ctx) {
    auto * c = (ts_champq_ctx *)ctx;
    int64_t K = c->in_dim;
    int64_t out_dim = c->out_dim;
    const float * W = c->W;
    float binariness = c->binariness;

    // M is (K x K), x is flattened M
    // W_perm = W @ M: (out_dim x K)
    std::vector<float> W_perm(out_dim * K);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t col = 0; col < K; col++) {
            float s = 0.0f;
            for (int64_t i = 0; i < K; i++) {
                s += W[r * K + i] * x[i * K + col];
            }
            W_perm[r * K + col] = s;
        }
    }

    float loss = 0.0f;
    if (K < 3) {
        // trivial: no interior columns
        if (binariness > 0.0f) {
            for (int64_t i = 0; i < n; i++) {
                loss += binariness * x[i] * (1.0f - x[i]);
                grad[i] = binariness * (1.0f - 2.0f * x[i]);
            }
        } else {
            memset(grad, 0, n * sizeof(float));
        }
        return loss;
    }

    // d2[r, c] = W_perm[r, c] - 2*W_perm[r, c+1] + W_perm[r, c+2], c in [0, K-3]
    // loss = sum d2^2
    std::vector<float> h(out_dim * K, 0.0f);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 0; c < K - 2; c++) {
            float d2 = W_perm[r * K + c] - 2.0f * W_perm[r * K + c + 1] + W_perm[r * K + c + 2];
            loss += d2 * d2;
            // h[r, c+1] += d2 (the center of the stencil)
            h[r * K + c + 1] = d2;
        }
    }

    // gradient: g[r, c] = 2 * (h[r, c-1] - 2*h[r, c] + h[r, c+1]) for interior
    // grad_M = W^T @ g
    std::vector<float> g(out_dim * K, 0.0f);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 1; c < K - 1; c++) {
            g[r * K + c] = 2.0f * (h[r * K + c - 1] - 2.0f * h[r * K + c] + h[r * K + c + 1]);
        }
    }

    // grad_M[i, k] = sum_r W[r, i] * g[r, k]
    for (int64_t i = 0; i < K; i++) {
        for (int64_t k = 0; k < K; k++) {
            float s = 0.0f;
            for (int64_t r = 0; r < out_dim; r++) {
                s += W[r * K + i] * g[r * K + k];
            }
            grad[i * K + k] = s;
        }
    }

    if (binariness > 0.0f) {
        for (int64_t i = 0; i < n; i++) {
            loss += binariness * x[i] * (1.0f - x[i]);
            grad[i] += binariness * (1.0f - 2.0f * x[i]);
        }
    }

    return loss;
}

struct ts_champq_project_ctx {
    int64_t K;
    int64_t sinkhorn_iters;
};

static void ts_champq_project(float * x, int64_t n, void * ctx) {
    auto * c = (ts_champq_project_ctx *)ctx;
    // clamp to non-negative
    for (int64_t i = 0; i < n; i++) {
        x[i] = std::max(x[i], 1e-12f);
    }
    ts_champq_sinkhorn(x, c->K, c->sinkhorn_iters, 1e-12f);
}

int ts_champq_compute(const float * weights, int64_t out_dim, int64_t in_dim,
                      const ts_champq_params * params, ts_champq_result * result) {
    int64_t max_iters = params->max_iters > 0 ? params->max_iters : 100;
    int64_t sinkhorn_iters = params->sinkhorn_iters > 0 ? params->sinkhorn_iters : 25;

    int64_t K = in_dim;
    int64_t n = K * K;

    // init M from L2-rank permutation
    std::vector<float> col_norms(K);
    for (int64_t j = 0; j < K; j++) {
        float s = 0.0f;
        for (int64_t i = 0; i < out_dim; i++) {
            float v = weights[i * in_dim + j];
            s += v * v;
        }
        col_norms[j] = sqrtf(s);
    }
    // argsort descending
    std::vector<int64_t> rank_perm(K);
    std::iota(rank_perm.begin(), rank_perm.end(), 0);
    std::sort(rank_perm.begin(), rank_perm.end(), [&](int64_t a, int64_t b) {
        return col_norms[a] > col_norms[b];
    });

    std::vector<float> M(n, 0.0f);
    for (int64_t i = 0; i < K; i++) {
        M[i * K + rank_perm[i]] = 1.0f;
    }

    // baseline smoothness (identity perm)
    float baseline_smooth = 0.0f;
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 1; c < K - 1; c++) {
            float d2 = weights[r * K + c - 1] - 2.0f * weights[r * K + c] + weights[r * K + c + 1];
            baseline_smooth += d2 * d2;
        }
    }

    if (params->use_lbfgs) {
        ts_champq_ctx eval_ctx = { weights, out_dim, in_dim, 1.0f };
        ts_champq_project_ctx proj_ctx = { K, sinkhorn_iters };

        ts_pgd_minimize(M.data(), n, ts_champq_eval, &eval_ctx,
                        ts_champq_project, &proj_ctx,
                        max_iters, 0.1f, 1e-6f);
    }

    // project to permutation
    std::vector<int32_t> perm(K);
    if (K <= 512) {
        // Hungarian on negative M (minimize cost = maximize M)
        std::vector<float> cost(n);
        for (int64_t i = 0; i < n; i++) {
            cost[i] = -M[i];
        }
        ts_hungarian(cost.data(), perm.data(), K);
    } else {
        ts_greedy_assignment(M.data(), perm.data(), K);
    }

    // compute final smoothness
    float final_smooth = 0.0f;
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 1; c < K - 1; c++) {
            float v0 = weights[r * K + perm[c - 1]];
            float v1 = weights[r * K + perm[c]];
            float v2 = weights[r * K + perm[c + 1]];
            float d2 = v0 - 2.0f * v1 + v2;
            final_smooth += d2 * d2;
        }
    }

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
