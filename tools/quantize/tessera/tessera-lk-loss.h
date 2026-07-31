#pragma once

//
// tessera-lk-loss.h
//
// LK loss: direct acceptance-rate optimization for speculative decoding
// drafter training (Samarin et al., 2026). The standard training objective
// minimizes KL(q || p) as a proxy; LK loss directly maximizes the acceptance
// rate alpha = sum_x min(p(x), q(x)), which is what actually governs
// inference speedup. Drop-in replacement for KL with no computational
// overhead; +5-10% acceptance length gains reported across 8B-685B models.
//
// Two modes:
//   full  - both distributions available over the full vocabulary
//   topk  - only top-k entries available (the llama.spec_calib.v2 case)
//
// The topk mode computes alpha over the union of the two top-k sets,
// treating out-of-set mass as a single residual bucket. This is the
// correct approximation when k is large enough to cover the high-probability
// support (k >= 8 is typical).
//

#include <cstdint>

// Full-vocabulary acceptance rate: alpha = sum_x min(p[x], q[x]).
// p and q must be valid probability distributions (non-negative, sum to ~1).
// Returns alpha in [0, 1]. alpha = 1 means identical distributions.
double ts_lk_acceptance_rate(const float * p, const float * q, int n_vocab);

// Top-k acceptance rate. p_tokens/p_probs and q_tokens/q_probs are the
// top-k entries (sorted high-to-low) for the verifier and drafter
// distributions respectively. p_mass/q_mass are the total probability mass
// covered by the top-k entries (sum of probs); the residual (1 - mass) is
// treated as a single uniform bucket over the remaining vocabulary.
//
// n_vocab is needed to size the residual bucket correctly.
// Returns alpha in [0, 1].
double ts_lk_acceptance_rate_topk(const int32_t * p_tokens, const float * p_probs, int p_k,
                                  double p_mass,
                                  const int32_t * q_tokens, const float * q_probs, int q_k,
                                  double q_mass,
                                  int n_vocab);

// Batch acceptance rate over n_positions spec-step positions.
// Each position has its own top-k distributions. Returns the mean alpha.
double ts_lk_acceptance_rate_batch(const int32_t * const * p_tokens,
                                   const float * const * p_probs,
                                   const int * p_k,
                                   const double * p_mass,
                                   const int32_t * const * q_tokens,
                                   const float * const * q_probs,
                                   const int * q_k,
                                   const double * q_mass,
                                   int n_positions,
                                   int n_vocab);

// LK-1 loss: negative acceptance rate (to be minimized).
// loss = -alpha. Gradient descent on this directly maximizes acceptance.
double ts_lk_loss(const float * p, const float * q, int n_vocab);

// Per-token contribution to alpha: min(p[x], q[x]).
// Useful for diagnostics: which tokens drive acceptance vs rejection.
double ts_lk_token_contribution(float p_x, float q_x);
