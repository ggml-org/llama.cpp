#pragma once

//
// tessera-dpace.h
//
// D-PACE: Dynamic Position-Aware Cross-Entropy for parallel speculative
// drafting (Wu et al., 2026, arXiv:2605.18810). Replaces DFlash's fixed
// exponential position decay with per-position weights derived from the
// gradient of an accepted-length surrogate, tying the drafter's training
// signal to the actual expected accepted draft length.
//
// For a block of B positions (0 = anchor, 1..B-1 = drafted), given
// per-position acceptance probabilities a_i (the drafter's softmax
// probability on the target token at position i):
//
//   Surrogate:  S = sum_{k=0}^{B-1} prod_{i=0}^{k} a_i
//   Weight:     w_j = (prod_{i<=j} a_i) * f_j
//   Continuation: f_j = 1 + sum_{m=j+1}^{B-1} prod_{i=j+1}^{m} a_i
//
// Asymmetric smoothing (applied to weights only, not the CE term):
//   a_tilde_i = (1 - alpha) * a_i + alpha
//
// The smoothed weights w_bar_j are detached from the gradient (treated
// as constants during backprop). The final loss is:
//   L = sum_j w_bar_j * (-log q(y_j))
//
// Reported gains: +8-10% wall-clock speedup, +8.5-10.7% accepted length
// over DFlash baseline, with 2.3% training-time overhead.
//

#include <cstdint>

// Default asymmetric smoothing floor (paper Sec. 3.3).
#define TS_DPACE_DEFAULT_ALPHA 0.1f

// Default DFlash exponential decay gamma (paper Sec. 4.1 baseline).
#define TS_DFLASH_DEFAULT_GAMMA 3.0f

//
// Continuation values: f_j for each position in a block.
// f_j = 1 + sum_{m=j+1}^{B-1} prod_{i=j+1}^{m} a_i
// For the last position, f_{B-1} = 1.
// out must hold block_size doubles.
//
void ts_dpace_continuation_values(const float * acceptance_probs,
                                  int block_size,
                                  double * out);

//
// Accepted-length surrogate: S = sum_{k=0}^{B-1} prod_{i=0}^{k} a_i.
// This is the expected number of accepted tokens under longest-prefix
// verification, using the drafter's own confidence as a proxy.
//
double ts_dpace_accepted_length_surrogate(const float * acceptance_probs,
                                          int block_size);

//
// Raw D-PACE weights: w_j = (prod_{i<=j} a_i) * f_j.
// out must hold block_size doubles. Weights are NOT normalized.
//
void ts_dpace_weights(const float * acceptance_probs,
                      int block_size,
                      double * out);

//
// Smoothed D-PACE weights: applies asymmetric smoothing to the
// acceptance probabilities before computing weights.
//   a_tilde_i = (1 - alpha) * a_i + alpha
// alpha = 0 disables smoothing (equivalent to ts_dpace_weights).
// out must hold block_size doubles.
//
void ts_dpace_weights_smoothed(const float * acceptance_probs,
                               int block_size,
                               float alpha,
                               double * out);

//
// DFlash baseline exponential decay weights: w_k = exp(-k / gamma).
// Position 0 (anchor) gets weight 1.0.
// out must hold block_size doubles.
//
void ts_dflash_decay_weights(int block_size,
                             float gamma,
                             double * out);

//
// Normalize weights in-place so they sum to block_size (preserving
// the mean weight = 1.0 convention used by DFlash training).
//
void ts_dpace_normalize_weights(double * weights, int block_size);

//
// D-PACE loss from per-position target-token probabilities.
// probs[j] = q(y_j), the drafter's softmax probability on the target
// token at position j. weights[j] are the (detached) D-PACE weights.
// Returns sum_j weights[j] * (-log(probs[j])).
// Probabilities are clamped to [1e-10, 1] before the log.
//
double ts_dpace_loss_from_probs(const float * probs,
                                const double * weights,
                                int block_size);

//
// D-PACE loss from logits. For each position j, extracts the softmax
// probability of target_tokens[j] from the logit row at
// logits + j * n_vocab, then computes the weighted CE.
// weights[j] are the (detached) D-PACE weights.
//
double ts_dpace_loss(const float * logits,
                     const int32_t * target_tokens,
                     const double * weights,
                     int block_size,
                     int n_vocab);

//
// End-to-end D-PACE loss: computes smoothed weights from the drafter's
// own per-position acceptance probabilities, then evaluates the weighted
// CE against the target tokens. This is the full training objective.
//
// acceptance_probs[j] = drafter's softmax prob on target_tokens[j]
// logits + j * n_vocab = raw logit row for position j
// target_tokens[j] = ground-truth token at position j
//
double ts_dpace_loss_end_to_end(const float * logits,
                                const int32_t * target_tokens,
                                const float * acceptance_probs,
                                int block_size,
                                int n_vocab,
                                float alpha);

//
// Diagnostics: compute both D-PACE and DFlash-decay losses for the
// same block, for A/B comparison during training.
//
struct ts_dpace_ab_result {
    double dpace_loss;
    double decay_loss;
    double dpace_surrogate;    // accepted-length surrogate S
    double mean_dpace_weight;  // mean of the smoothed weights
    double mean_decay_weight;  // mean of the decay weights (always 1.0 after norm)
};

ts_dpace_ab_result ts_dpace_ab_compare(const float * logits,
                                       const int32_t * target_tokens,
                                       const float * acceptance_probs,
                                       int block_size,
                                       int n_vocab,
                                       float alpha,
                                       float gamma);
