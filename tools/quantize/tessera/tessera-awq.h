#pragma once

//
// tessera-awq.h
//
// Evolutionary AWQ policy search (GA). Ports tools/tessera/awq-evolve.py.
// Island model with MAP-Elites archive keyed on regime descriptors.
// Deterministic given seed.
//
// The searched parameter space is the 6-gene ts_policy_genes payload shared
// with the policy reader (B4): alpha, clip, outlier_fraction, moment_mix,
// tail_guard, ternary_threshold. This is the same space Python's Candidate
// (awq-evolve.py) searches, so a policy written by Python seeds the GA and
// vice-versa.
//

#include "tessera-policy.h"

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>

// A candidate policy for one tensor family. The searched genes are the
// embedded ts_policy_genes (the same type carried by ts_policy_tensor), so
// the GA and the policy reader speak one type. expert_hint is metadata, not
// searched. LRQ/DartQuant knobs live in a separate ts_awq_lrq_params (they
// are a sub-policy in Python, not part of Candidate).
struct ts_awq_candidate {
    ts_policy_genes genes;     // the 6 searched genes (alpha, clip, ...)
    int64_t         expert_hint;  // ts_expert_id or -1 for auto
};

// LRQ/DartQuant sub-policy, kept separate from the candidate so the GA's
// searched space matches Python's Candidate exactly. Carried alongside a
// candidate by the experts that need it (tessera-lrq, tessera-dartquant).
struct ts_awq_lrq_params {
    float lrq_rank_frac = 0.0f;  // fraction of max rank for LRQ residual [0, 1]
    float rotation_lr   = 0.0f;  // DartQuant step size hint
};

// Construct a candidate from a gene payload + expert hint. Bridges the
// policy reader (B4) into the GA: read a policy, pick a family, feed its
// genes here to seed evolution.
struct ts_awq_candidate ts_awq_candidate_from_genes(ts_policy_genes genes,
                                                    int64_t expert_hint);

// Fitness score for a candidate.
struct ts_awq_score {
    float mse;              // layer-output MSE (primary)
    float relative_frob;    // t_l^2 = ||W_hat - W||_F^2 / ||W||_F^2
    float heldout_mse;      // held-out token MSE
    float composite;        // alpha_l-weighted composite (uniform for now)
};

// Layer data passed to the evaluator.
struct ts_awq_layer {
    std::string name;
    std::string family;         // tensor family
    const float * weights;      // (out_dim x in_dim)
    const float * act_scales;   // (in_dim,) or nullptr
    const float * calib_X;     // (n_tokens x in_dim) or nullptr
    const float * ref_output;  // (n_tokens x out_dim) or nullptr
    const float * imatrix;     // (in_dim,) or nullptr
    int64_t out_dim;
    int64_t in_dim;
    int64_t n_tokens;
    float   kurtosis;           // from imatrix regime stats
    float   eff_rank;           // effective rank
};

// Evaluator callback: quantize with candidate, return score.
// The GA calls this; the implementation dispatches to ts_quantize_2d
// or the kernel-direct path (L6) depending on what's available.
typedef ts_awq_score (*ts_awq_eval_fn)(const ts_awq_candidate * cand,
                                       const ts_awq_layer * layer,
                                       void * ctx);

// MAP-Elites archive cell descriptor.
struct ts_awq_archive_cell {
    int32_t kurtosis_bucket;    // quantized kurtosis
    int32_t eff_rank_bucket;    // quantized effective rank
    int32_t family_bucket;      // tensor family index
};

struct ts_awq_evolve_params {
    int64_t population;     // island population size, default 32
    int64_t generations;    // total generations, default 100
    int64_t islands;        // number of islands, default 4
    int64_t migration_interval;  // generations between migration, default 10
    float     mutation_sigma;    // Gaussian mutation std, default 0.1
    float     crossover_rate;    // default 0.7
    float     heldout_weight;    // weight for held-out MSE in composite, default 2.0
    uint32_t  seed;              // determinism
    bool      verbose;
};

// Result of evolution for one tensor family.
struct ts_awq_evolve_result {
    ts_awq_candidate best;
    ts_awq_score     best_score;
    int64_t          generations_run;
    int64_t          evaluations;
    // archive: best candidate per regime cell
    std::vector<std::pair<ts_awq_archive_cell, ts_awq_candidate>> archive;
};

// Run evolutionary search for one layer.
int ts_awq_evolve(const ts_awq_layer * layer,
                  ts_awq_eval_fn eval, void * eval_ctx,
                  const ts_awq_evolve_params * params,
                  ts_awq_evolve_result * result);

// Run evolutionary search across all layers (the full pipeline).
// layers: array of n_layers. results: one per layer.
int ts_awq_evolve_all(const ts_awq_layer * layers, int64_t n_layers,
                      ts_awq_eval_fn eval, void * eval_ctx,
                      const ts_awq_evolve_params * params,
                      std::vector<ts_awq_evolve_result> * results);

// Utility: compute archive cell from layer regime descriptors.
ts_awq_archive_cell ts_awq_make_cell(float kurtosis, float eff_rank,
                                     int32_t family_idx);

// Utility: serialize candidate to JSON fragment (for policy output).
std::string ts_awq_candidate_json(const ts_awq_candidate * cand);
