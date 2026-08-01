#pragma once

//
// tessera-champq.h
//
// CHAMP-Q channel-permutation regime expert. Finds a column permutation
// that minimizes ternary reconstruction MSE (second-difference
// smoothness of the permuted weights), via L-BFGS + Sinkhorn projection
// onto the doubly-stochastic polytope, then Hungarian/greedy assignment.
// Ported from tools/tessera/champq_permute.py.
//

#include <cstdint>
#include <vector>

struct ts_champq_result {
    std::vector<int32_t> perm;      // (in_dim,) permutation vector
    float                smoothness;
    float                mse_improvement;
};

struct ts_champq_params {
    int64_t max_iters;      // L-BFGS iterations, default 100
    int64_t sinkhorn_iters; // Sinkhorn projection iterations, default 25
    bool      use_lbfgs;    // true = L-BFGS, false = greedy
    uint32_t  seed;
};

// Compute channel permutation minimizing ternary reconstruction MSE.
int ts_champq_compute(const float * weights, int64_t out_dim, int64_t in_dim,
                      const ts_champq_params * params, ts_champq_result * result);

// Apply permutation to columns: W_perm[:, j] = W[:, perm[j]].
void ts_champq_apply(const float * W, const int32_t * perm,
                     float * W_perm, int64_t out_dim, int64_t in_dim);

// Invert permutation.
void ts_champq_invert(const int32_t * perm, int32_t * inv, int64_t n);

// Sinkhorn projection: project (n x n) matrix onto doubly-stochastic.
void ts_champq_sinkhorn(float * M, int64_t n, int64_t n_iters, float eps);
