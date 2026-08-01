#pragma once

//
// tessera-dartquant.h
//
// DartQuant (distribution-aware rotation) regime expert. Optimizes a
// block-wise rotation R that minimizes the whip loss (coefficient of
// variation of per-row L2 norms), making the rotated weights more
// ternary-friendly. Ported from tools/tessera/per_tensor_calibrate.py.
//

#include <cstdint>
#include <vector>

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
