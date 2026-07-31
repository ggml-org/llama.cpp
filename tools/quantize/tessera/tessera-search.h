#pragma once

//
// tessera-search.h
//
// Regime experts: LRQ, DartQuant, FLRQ, CHAMP-Q permutation.
// Each expert produces a policy entry (U/V factors, rotation matrix,
// or permutation vector) consumed by the GA and the quantizer.
// Ports tools/tessera/per_tensor_calibrate.py and champq_permute.py.
//

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// --- LRQ (Low-Rank Quantization) ---

struct ts_lrq_result {
    std::vector<float> U;       // (out_dim x rank)
    std::vector<float> V;       // (rank x in_dim)
    float              mse;
    int64_t            rank;
    int64_t            n_iters;
};

struct ts_lrq_params {
    int64_t rank;
    int64_t max_iters;      // default 200
    float     lr;           // Adam learning rate, default 1e-3
    float     tol;          // convergence tolerance
    uint32_t  seed;
};

int ts_train_lrq(const float * weights, int64_t out_dim, int64_t in_dim,
                 const ts_lrq_params * params, ts_lrq_result * result);

// --- DartQuant (distribution-aware rotation) ---

struct ts_dartquant_result {
    std::vector<float> R;       // (K x K) rotation matrix, K = block size
    float              whip_loss;
    float              mse;
    int64_t            n_iters;
};

struct ts_dartquant_params {
    int64_t block_size;     // rotation block size K (default 64)
    int64_t max_iters;      // QR-Orth iterations, default 50
    float     lr;           // step size, default 0.1
    uint32_t  seed;
};

// Optimize rotation R minimizing whip loss (ternary reconstruction MSE
// after rotation). weights is (out_dim x in_dim); rotation applied
// block-wise along in_dim.
int ts_dartquant_qr_orth(const float * weights, int64_t out_dim, int64_t in_dim,
                         const ts_dartquant_params * params,
                         ts_dartquant_result * result);

// Apply rotation to weights: W_rot = W @ R^T (block-wise).
void ts_dartquant_apply(const float * W, const float * R,
                        float * W_rot, int64_t out_dim, int64_t in_dim,
                        int64_t block_size);

// --- FLRQ (Factored Low-Rank Quantization) ---

struct ts_flrq_result {
    std::vector<float> U;       // (out_dim x rank)
    std::vector<float> V;       // (rank x in_dim)
    int64_t            rank;    // selected rank
    float              mse;
    float              spectral_compactness;  // rho
};

struct ts_flrq_params {
    int64_t max_rank;       // upper bound on rank search
    float     rho_thresh;   // spectral compactness gate (default 0.8)
    int64_t   sketch_oversample;  // default 10
    uint32_t  seed;
};

// Select rank via spectral compactness, then compute low-rank factors.
int ts_train_flrq(const float * weights, int64_t out_dim, int64_t in_dim,
                  const ts_flrq_params * params, ts_flrq_result * result);

// --- CHAMP-Q permutation ---

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

// --- GA search config and fitness ---

struct ts_search_config {
    const float * layer_alpha;  // per-layer HIGGS weights (n_layers,), nullptr = uniform
    int64_t       n_layers;
};

// Composite fitness: Sum_l alpha_l * t_l^2.
// t2: per-layer relative Frobenius errors (n_layers,).
// cfg: search config with optional layer_alpha (nullptr = all alpha_l = 1).
float ts_search_fitness(const float * t2, const ts_search_config * cfg);

// --- Regime router ---

enum ts_expert_id {
    TS_EXPERT_AWQ       = 0,
    TS_EXPERT_LRQ       = 1,
    TS_EXPERT_DARTQUANT = 2,
    TS_EXPERT_FLRQ      = 3,
    TS_EXPERT_CHAMPQ    = 4,
    TS_EXPERT_SEPTQ     = 5,
    TS_EXPERT_COUNT     = 6,
};

// Route a tensor to its best expert based on regime descriptors.
// kurtosis: activation kurtosis from imatrix.
// eff_rank: effective rank (spectral entropy).
// family: tensor family string ("attn_q", "attn_k", "ffn_gate", etc).
ts_expert_id ts_route_expert(float kurtosis, float eff_rank,
                             const char * family);
