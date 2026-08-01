#pragma once

//
// tessera-lrq.h
//
// LRQ (Low-Rank Quantization) regime expert. Trains low-rank factors
// U/V whose product approximates an identity scaling of the weights,
// so the scaled weights ternarize with minimal reconstruction error.
// Ported from tools/tessera/per_tensor_calibrate.py.
//

#include <cstdint>
#include <vector>

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
