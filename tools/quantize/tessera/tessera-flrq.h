#pragma once

//
// tessera-flrq.h
//
// FLRQ (Factored Low-Rank Quantization, Jan 2026) regime expert.
// Calibration-free decomposition of a 2-D weight W as
//
//     W ~= U @ V + Q(R; scale, clip),   R = W - U @ V
//
// where (U, V) are FP16 low-rank factors identified by the R1-Sketch (a
// handful of Gaussian projections of W, no calibration data needed) and
// Q is a per-tensor symmetric uniform quantiser that BLC iteratively fits.
//
// Mirrors tools/tessera/per_tensor_calibrate.py: flrq_sketch, flrq_bcl,
// flrq_select_rank, train_flrq. The BLC step is fully deterministic given
// the sketch basis, so the C++ port pins to the Python reference bit-for-bit
// (up to per-column sign of the basis, which leaves U@V / residual invariant).
//

#include <cstdint>
#include <vector>

// Output of the R1-Sketch step. Y is (K x total_width); U_basis is (K x r)
// with orthonormal columns (top-r left singular vectors of Y); sigma holds
// the top-r singular values, descending.
struct ts_flrq_sketch {
    std::vector<float> Y;        // (K x total_width)
    std::vector<float> U_basis;  // (K x r)
    std::vector<float> sigma;    // (r,)
    int64_t            total_width;
    int64_t            r;
};

// Output of the BLC (Best Low-rank approximation under Clipping) step.
// U is the basis passed in (== U_basis); V is (r x N); residual / residual_q
// are (K x N); residual_q is the dequantised residual (the int8 payload is
// implicit, recomordable from residual + scale).
struct ts_flrq_bcl_result {
    std::vector<float> U;           // (K x r)
    std::vector<float> V;           // (r x N)
    std::vector<float> residual;    // (K x N), = W - U @ V after the final step
    std::vector<float> residual_q;  // (K x N), dequantised residual
    float residual_scale;
    float residual_clip;
    float reconstruction_mse;       // mean((W - (U@V + residual_q))^2)
};

// R1-Sketch with Gaussian projection (calibration-free).
//   weight:       (K x N) row-major
//   n_projections: Gaussian projection count (16-64 operating range)
//   seed:          RNG seed (C++ xorshift Box-Muller; deterministic but NOT
//                  bit-identical to numpy -- the parity test loads the basis
//                  from the fixture instead)
//   target_rank:   r, the number of sketch basis vectors (0 => min(K,N,16))
int ts_flrq_sketch_run(const float * weight, int64_t K, int64_t N,
                       int64_t n_projections, uint32_t seed,
                       int64_t target_rank, ts_flrq_sketch * out);

// Best Low-Rank Approximation under Clipping. Fully deterministic given the
// basis; this is the numerical core that the parity test pins.
//   weight:  (K x N)
//   U_basis: (K x r) with orthonormal columns (from the sketch step)
//   n_iters: BLC iterations (default 4)
//   qbits:   residual quantiser bit-width (1-12; 3-4 is the operating range)
int ts_flrq_bcl(const float * weight, int64_t K, int64_t N,
                const float * U_basis, int64_t r,
                int64_t n_iters, int qbits, ts_flrq_bcl_result * out);

// Default operating constants (match FLRQ_DEFAULT_* in Python).
#define TS_FLRQ_DEFAULT_PROJECTIONS   32
#define TS_FLRQ_DEFAULT_BLC_ITERS     4
#define TS_FLRQ_DEFAULT_QBITS         4
#define TS_FLRQ_DEFAULT_MSE_THRESHOLD 1.0e-3f

struct ts_flrq_params {
    int64_t  rank_candidates[8];  // 0-terminated; empty => {4,8,16,32,64}
    int64_t  n_projections;       // 0 => 32
    uint32_t seed;                // 0 => 1 (xoshiro needs a non-zero seed)
    int64_t  blc_iters;           // 0 => 4
    int64_t  qbits;               // 0 => 4
    float    mse_threshold;       // 0 => 1e-3
};

// End-to-end FLRQ driver for one tensor: select the smallest rank whose
// relative reconstruction MSE clears mse_threshold, then return the chosen
// decomposition. result_rank is -1 if the sketch step fails.
int ts_train_flrq(const float * weight, int64_t K, int64_t N,
                  const ts_flrq_params * params,
                  ts_flrq_bcl_result * out, int64_t * result_rank);
