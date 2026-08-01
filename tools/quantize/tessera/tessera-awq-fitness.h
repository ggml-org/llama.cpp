#pragma once

//
// tessera-awq-fitness.h
//
// C++ port of the AWQ GA fitness (Python: tools/tessera/awq-evolve.py).
// Faithful numeric port of _ternary_reconstruct, relative_output_error,
// _evaluate_layer (incl. _layer_features) and _evaluate_uncached.
//
// Parity is verified by test_awq_fitness.cpp against a Python-generated
// fixture (see fixtures/gen_awq_fitness_fixture.py). The port keeps the
// hot path in float (Python's np.power(..., dtype=np.float32) for the
// scale, and the rest of the reconstruction inherits float32 from the
// weight/importance inputs) and uses double for the matmul reductions
// and fitness aggregation to match numpy's default float64 reductions.
//

#include "tessera-awq.h"

#include <cstdint>

// Reconstruct a ternarized + sparse-residual weight matrix.
// Faithful port of awq-evolve.py:_ternary_reconstruct (lines 219-252).
//
// W:                (out_dim x in_dim) row-major original weights
// g:                the candidate's 6 genes (alpha, clip, outlier_fraction,
//                   moment_mix, tail_guard, ternary_threshold)
// importance:       (in_dim,) importance vector (Python computes this from
//                   second/fourth moments and max_abs before calling)
// out_dim, in_dim:  matrix shape
// reconstructed_out:(out_dim x in_dim) caller-owned output buffer
//
// All internal buffers are allocated inside this call.
void ts_awq_ternary_reconstruct(const float * W,
                                const ts_policy_genes & g,
                                const float * importance,
                                int64_t out_dim, int64_t in_dim,
                                float * reconstructed_out);

// Relative output error: ||A @ R^T - A @ W^T||_F^2 / (||A @ W^T||_F^2 + 1e-12).
// Faithful port of awq-evolve.py:relative_output_error (lines 658-680,
// numpy branch - the MLX branch computes the identical math).
//
// activations:       (n_tokens x in_dim)
// W:                 (out_dim x in_dim)
// reconstructed:     (out_dim x in_dim)
// reference_or_null: (n_tokens x out_dim) precomputed A @ W^T, or nullptr to
//                    compute it here
double ts_awq_relative_output_error(const float * activations,
                                    const float * W,
                                    const float * reconstructed,
                                    int64_t n_tokens,
                                    int64_t in_dim,
                                    int64_t out_dim,
                                    const float * reference_or_null);

// Per-layer score (port of _evaluate_layer + _layer_features).
// Computes the importance vector from the layer's moments/max_abs on the fly,
// reconstructs via _ternary_reconstruct, then fills out->mse (train_error),
// out->relative_frob (tail_error), out->heldout_mse (heldout_error). composite
// is left at 0 (the per-layer score has no fitness; only the aggregate does).
//
// Returns 0 on success, non-zero on invalid arguments.
int ts_awq_evaluate_layer(const ts_awq_candidate & c,
                          const ts_awq_layer & layer,
                          ts_awq_score * out);

// Composite aggregation across layers (port of _evaluate_uncached).
// Computes the per-layer scores, then the train/heldout/tail means, the 0.9
// quantile of the heldout errors, and the composite fitness
//   train + 2.0 * heldout + 0.25 * worst_layer + 0.05 * tail + 0.15 * size
// (size = candidate.genes.outlier_fraction).
//
// n_layers >= 1. out is overwritten with the aggregate: mse = mean train,
// heldout_mse = mean heldout, relative_frob = mean tail, composite = fitness.
// (worst_layer_error is not stored in ts_awq_score; the per-layer worst-case
// is recoverable from the heldout quantile but is not exposed here - it only
// feeds the composite, mirroring how Python folds it into fitness.)
//
// Returns 0 on success, non-zero on invalid arguments.
int ts_awq_evaluate(const ts_awq_candidate & c,
                    const ts_awq_layer * layers, int64_t n_layers,
                    ts_awq_score * out);
