#pragma once

//
// tessera-septq.h
//
// SEPTQ (banded-Cholesky Hessian) weight quantizer. Ported from
// tools/tile640/quantize_v3.py: quantize_2d_septq and its helpers
// _septq_banded_cholesky / _septq_gptq_M / _septq_build_hessian.
//
// Algorithm overview (see the Python docstring for the full writeup):
//   1. Build the Hessian H of the calibration activations
//      (H = X^T X / n, ridge-regularized on the diagonal).
//   2. Banded Cholesky factorization L of H (band radius = bandwidth),
//      with a full-Cholesky fallback when the band is non-positive-definite.
//   3. GPTQ update matrix M = banded portion of L^{-1}, rescaled by
//      diag(L), so M is strictly upper-triangular with the same band.
//   4. Per-element importance score, static top-k mask, vectorized
//      ternarization of the masked elements to {-1, 0, +1}, and the
//      cross-column error-compensation update W' = W - E @ M.
//
// Dtype policy (must match the Python reference, which is float32
// end-to-end for the Hessian/Cholesky/M path):
//   - Weight I/O is float32 (the on-disk Tessera format).
//   - The Hessian H is accumulated in float64 (X^T X is a long reduction)
//     and stored as float32; this lands within 1 ULP of numpy's float32
//     BLAS Hessian and is the most accurate choice for a scalar port.
//   - The banded Cholesky L and the GPTQ matrix M are computed in
//     float32 with scalar arithmetic, matching Python's scalar f32 loops
//     bit-for-bit given the same H (the only residual divergence vs the
//     reference is the unavoidable 1-ULP H difference).
//   - The full-Cholesky fallback also runs in float32 to match numpy's
//     float32 np.linalg.cholesky.
//

#include <cstdint>
#include <vector>

// Hessian mode: banded uses the activations + banded Cholesky; diagonal
// uses act_scales^2 as a diagonal proxy (M is identically zero, so the
// cross-column update is a no-op).
enum ts_septq_hessian_mode {
    TS_SEPTQ_HESSIAN_DIAGONAL = 0,
    TS_SEPTQ_HESSIAN_BANDED    = 1,
};

// Importance score variant. "quant_error_h" is the canonical SEPTQ score
// (W - Q(W))^2 * h_diag. The other modes downweight heavy-tailed outliers
// so the mask focuses on the bulk.
enum ts_septq_importance_weight {
    TS_SEPTQ_IMP_QUANT_ERROR_H = 0,
    TS_SEPTQ_IMP_INV_ABS_W     = 1,
    TS_SEPTQ_IMP_INV_CDF       = 2,
    TS_SEPTQ_IMP_HYBRID        = 3,
};

struct ts_septq_params {
    float     septq_ratio;                 // fraction quantized, (0, 1]
    int64_t   septq_iterations;            // mask passes (>= 1); default 1
    float     ternary_threshold;          // multiplier on per-row mean|W|; [0.3, 3]
    int       hessian_mode;               // ts_septq_hessian_mode
    int64_t   hessian_bandwidth;          // band radius (>= 0); clamped to in_dim-1
    int       importance_weight;          // ts_septq_importance_weight
    float     importance_lambda;          // hybrid mode weight (>= 0)
    float     ridge_fraction;             // Hessian ridge; default 1e-4
};

struct ts_septq_result {
    // Tessera packed outputs (the on-disk format). Types match the canonical
    // ts_quant_result_2d convention (tessera-quant.h): f16 fields are stored
    // as raw uint16 bit patterns, lane_scales as int8, packed as uint32.
    std::vector<uint32_t> packed;             // packed ternary tiles
    std::vector<uint16_t> page_scales;        // f16 bits
    std::vector<int8_t>   lane_scales;        // per-lane scales in [1, 127]
    std::vector<int32_t>  outlier_row_offsets;// (out_dim + 1,) CSR row ptrs
    std::vector<int32_t>  outlier_cols;       // column index per outlier
    std::vector<uint16_t> outlier_vals;       // f16 bits
    // Diagnostic / parity extras (mirror the Python "_ternary" / "_mask_2d" keys).
    std::vector<int8_t>   ternary;            // (out_dim * in_dim,) in {-1, 0, +1}
    std::vector<uint8_t>  mask_2d;            // (out_dim * in_dim,) 0/1
};

// --- Internal helpers (exposed for parity testing) ---

// Build the symmetric Hessian H (in_dim x in_dim) of the calibration
// activations. activations is (n_tokens x in_dim), may be null (then the
// diagonal act_scales^2 proxy is used; if act_scales is also null H = I).
// Output H must have room for in_dim*in_dim floats.
void ts_septq_build_hessian(int64_t in_dim,
                            const float * act_scales,
                            const float * activations, int64_t n_tokens,
                            float ridge_fraction,
                            float * H);

// Banded Cholesky: factor H = L L^T with L lower-triangular, restricting
// the factorization to |i - j| <= bandwidth. Falls back to a full
// Cholesky when the banded factorization hits a non-positive pivot.
// L_out must have room for n*n floats (row-major).
void ts_septq_banded_cholesky(const float * H, int64_t n, int64_t bandwidth,
                              float * L_out);

// GPTQ update matrix M from L: strictly upper-triangular banded portion
// of L^{-1} rescaled by diag(L). M_out must have room for n*n floats.
void ts_septq_gptq_M(const float * L, int64_t n, int64_t bandwidth,
                     float * M_out);

// --- Top-level entry point ---

// Quantize a 2D weight matrix [out_dim, in_dim] (row-major float32) with
// the SEPTQ recipe. activations is (n_tokens x in_dim), may be null.
// act_scales is (in_dim,), may be null (uniform column importance).
// Returns 0 on success, non-zero on invalid arguments.
int ts_septq_quantize_2d(const float * weights, int64_t out_dim, int64_t in_dim,
                         const float * activations, int64_t n_tokens,
                         const float * act_scales,
                         const ts_septq_params * params,
                         ts_septq_result * result);
