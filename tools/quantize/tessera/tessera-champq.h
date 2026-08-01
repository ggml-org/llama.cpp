#pragma once

//
// tessera-champq.h
//
// CHAMP-Q channel-permutation regime expert. Finds an input-channel
// permutation that minimizes a smoothness proxy (sum of squared second
// differences of the permuted weight rows, optionally activation-weighted),
// which is a cheap stand-in for ternary reconstruction MSE.
//
// Search: L-BFGS on a doubly-stochastic relaxation M (Sinkhorn-projected
// between steps), then Hungarian/greedy assignment to recover a permutation.
// Ported from tools/tessera/champq_permute.py (lbfgs_permutation) and
// tools/tessera/_champq_lbfgs.py (the L-BFGS solver lives in tessera-lbfgs).
//

#include <cstdint>
#include <vector>

struct ts_champq_result {
    std::vector<int32_t> perm;      // (in_dim,) permutation vector
    float                smoothness;       // final proxy at the recovered perm
    float                mse_improvement; // (baseline - final) / baseline
    // Loss trajectory across the L-BFGS iterations. loss_history[0] is the
    // loss at the initial M; loss_history[i+1] is the loss after iteration i
    // (taken at the projected point). Empty when use_lbfgs is false.
    std::vector<float>   loss_history;
};

// Per-input-channel activation magnitude (shape in_dim,). May be null:
// when null the smoothness proxy is unweighted. Matches the `act_scales`
// argument of champq_permute.lbfgs_permutation.
struct ts_champq_params {
    int64_t max_iters;      // L-BFGS iterations, default 100
    int64_t sinkhorn_iters; // Sinkhorn projection iterations, default 25
    bool      use_lbfgs;    // true = L-BFGS, false = greedy/L2-rank init only
    uint32_t  seed;
    // Optional activation magnitudes; may be null.
    const float * act_scales;
    // Binariness penalty weight (lambda * sum M*(1-M)). 0 disables.
    // Default 1e-3, matches champq_permute.py CLI default.
    float    binariness;
    // L-BFGS history length. Default 8 (matches champq_permute.py).
    int64_t  history;
    // Final projection: 0 = hungarian (K <= 512), 1 = greedy.
    int32_t  projection;
    // Init mode: 0 = l2rank (default), 1 = identity, 2 = random.
    int32_t  init;
};

// Compute channel permutation minimizing the smoothness proxy.
int ts_champq_compute(const float * weights, int64_t out_dim, int64_t in_dim,
                      const ts_champq_params * params, ts_champq_result * result);

// Smoothness proxy (sum of squared second differences of the permuted rows),
// optionally weighted by per-channel act_scales. act_scales may be null.
// Matches champq_permute.smoothness_proxy.
float ts_champq_smoothness(const float * W, const int32_t * perm,
                           const float * act_scales,
                           int64_t out_dim, int64_t in_dim);

// Apply permutation to columns: W_perm[:, j] = W[:, perm[j]].
void ts_champq_apply(const float * W, const int32_t * perm,
                     float * W_perm, int64_t out_dim, int64_t in_dim);

// Invert permutation.
void ts_champq_invert(const int32_t * perm, int32_t * inv, int64_t n);

// Sinkhorn projection: project (n x n) matrix onto doubly-stochastic.
void ts_champq_sinkhorn(float * M, int64_t n, int64_t n_iters, float eps);
