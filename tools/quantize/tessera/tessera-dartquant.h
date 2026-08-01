#pragma once

//
// tessera-dartquant.h
//
// DartQuant (distribution-aware rotation) regime expert. Optimizes a
// rotation R on the Stiefel manifold that minimizes the combined loss
//   L = L_quant + whip_weight * L_whip
// where L_quant is the tile-quant output MSE (asymmetric STE) and L_whip
// is the coefficient of variation of per-row L2 norms (the "whip" loss
// from the DartQuant paper). Ported from tools/tessera/per_tensor_calibrate.py
// (dartquant_qr_orth). QR / Stiefel primitives come from tessera-linalg.
//
// Convention: W_rot = W @ R (matches Python). R is (in_dim x in_dim),
// orthogonal, row-major float32.
//

#include <cstdint>
#include <vector>

struct ts_dartquant_result {
    std::vector<float> R;                 // (in_dim x in_dim) rotation, FP32
    std::vector<float> history;           // combined loss per iteration
    float              initial_whip;
    float              final_whip;
    float              initial_quant_mse;  // relative tile-quant MSE: ||W'-Q(W')||^2 / ||W'||^2
    float              final_quant_mse;
    float              initial_output_mse; // asymmetric-STE output MSE (needs X)
    float              final_output_mse;
    int64_t            n_iters;

    // Legacy fields kept for backward compatibility with test_search.cpp.
    // whip_loss mirrors final_whip; mse mirrors final_quant_mse * ||W'||^2 / N.
    float              whip_loss;
    float              mse;
};

struct ts_dartquant_params {
    int64_t block_size;     // rotation block size K (0 -> full in_dim)
    int64_t max_iters;      // QR-Orth iterations, default 50
    float     lr;           // step size, default 1.0e-2
    float     whip_weight;  // whip-loss weight in combined loss, default 0.1
    uint32_t  seed;         // 0 -> identity init, else Haar-uniform R
};

// Optimize rotation R for `weights` (out_dim x in_dim) using QR-Orth.
// Optional calibration hooks:
//   - X (n_tokens x in_dim): activation rows. When non-null the loss
//     includes the asymmetric-STE output MSE term (matches Python's
//     _output_mse_and_grad). Pass X=NULL/X_count=0 to drop that term.
//   - X_hat (length in_dim): per-input-channel sqrt(E[x^2]) scale applied
//     inside the whip loss and its gradient (SmoothQuant-style). Pass
//     NULL to ignore.
// K (block size) must divide in_dim and be <= in_dim. R is square (K x K)
// and shared across all in_dim/K column blocks. The full rotation applied
// to `weights` is block-diagonal with this R repeated. The optimization
// uses the first column block (the same simplification as the Python demo).
int ts_dartquant_qr_orth(const float * weights, int64_t out_dim, int64_t in_dim,
                         const float * X, int64_t X_count,
                         const float * X_hat,
                         const ts_dartquant_params * params,
                         ts_dartquant_result * result);

// Back-compat overload: whip-only optimization (X / X_hat dropped). Kept so
// existing callers (test_search.cpp) compile unchanged.
int ts_dartquant_qr_orth(const float * weights, int64_t out_dim, int64_t in_dim,
                         const ts_dartquant_params * params,
                         ts_dartquant_result * result);

// Apply rotation block-diagonally: W_rot = W @ R (block-wise along in_dim).
// R is (block_size x block_size); in_dim must be a multiple of block_size.
void ts_dartquant_apply(const float * W, const float * R,
                        float * W_rot, int64_t out_dim, int64_t in_dim,
                        int64_t block_size);
