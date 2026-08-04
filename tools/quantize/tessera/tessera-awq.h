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

// Fitness score for a candidate. Mirrors Python's Score (awq-evolve.py):
//   mse            = train_error  (per-layer relative output error if
//                  train_activations present, else diagonal second-moment error)
//   relative_frob  = tail_error   (mean(error^2 * fourth_moment))
//   heldout_mse    = heldout_error (relative output error on held-out tokens,
//                  falls back to the diagonal/train error when no heldout split)
//   composite      = fitness = train + 2*heldout + 0.25*worst + 0.05*tail
//                  + 0.15*outlier_fraction (computed only by ts_awq_evaluate,
//                  the per-layer aggregator leaves it at 0)
// The 1:1 field mapping keeps the Python parity test honest: each field is
// checked against the Python Score computed on the same fixture.
struct ts_awq_score {
    float mse;              // train_error (Python Score.train_error)
    float relative_frob;    // tail_error  (Python Score.tail_error)
    float heldout_mse;      // heldout_error (Python Score.heldout_error)
    float composite;        // fitness (Python Score.fitness) or 0 per-layer
};

// Layer data passed to the evaluator. The new fields (second_moment,
// fourth_moment, max_abs, *_activations, ref_*_output) carry exactly what
// Python's Layer carries (awq-evolve.py:85); they are nullptr when the host
// has not computed them, in which case the evaluator falls back to the
// same paths Python falls back to (diagonal error, train==heldout, etc.).
struct ts_awq_layer {
    std::string name;
    std::string family;            // tensor family
    const float * weights;         // (out_dim x in_dim)
    const float * act_scales;      // (in_dim,) or nullptr
    const float * calib_X;        // (n_tokens x in_dim) or nullptr
    const float * ref_output;     // (n_tokens x out_dim) or nullptr
    const float * imatrix;        // (in_dim,) or nullptr
    // Observer telemetry used to build the importance vector (Python's
    // _layer_features): second_moment and fourth_moment are E[x^2]/E[x^4]
    // per input channel, max_abs is the per-channel peak. All length in_dim.
    const float * second_moment;   // (in_dim,) or nullptr (defaults to ones)
    const float * fourth_moment;   // (in_dim,) or nullptr (defaults to second^2)
    const float * max_abs;         // (in_dim,) or nullptr (defaults to sqrt(second))
    // Activation splits. train_activations is (n_tokens x in_dim),
    // heldout_activations is (n_tokens_h x in_dim). When present, the
    // evaluator uses relative_output_error (matmul-based) instead of the
    // diagonal second-moment proxy, matching Python's _evaluate_layer.
    const float * train_activations;    // (n_tokens x in_dim) or nullptr
    const float * heldout_activations;  // (n_tokens_h x in_dim) or nullptr
    // Precomputed reference outputs (activations @ weight.T). Optional: when
    // null the evaluator computes them on the fly, exactly like Python's
    // _reference_output lazy cache. train shape is (n_tokens x out_dim),
    // heldout is (n_tokens_h x out_dim).
    const float * ref_train_output;     // (n_tokens x out_dim) or nullptr
    const float * ref_heldout_output;   // (n_tokens_h x out_dim) or nullptr
    int64_t out_dim;
    int64_t in_dim;
    int64_t n_tokens;    // rows in train_activations
    int64_t n_tokens_h;  // rows in heldout_activations
    float   kurtosis;    // from imatrix regime stats
    float   eff_rank;    // effective rank
    // Block index parsed from the tensor name (e.g. blk.12.ffn_gate -> 12).
    // 0 for non-block tensors (embeddings, norm, output). Set by the dispatch
    // from ts_tessera_db_layer_depth(); enables depth-aware family queries
    // like "best alpha for ffn_gate at depth 10-15" against the persistent
    // store. Default 0 keeps existing behaviour when the host does not set it.
    int32_t layer_depth = 0;
    // Streaming weight loading: when non-null, the GA worker calls this
    // callback to fetch the layer's weights on demand (instead of holding
    // all layers' weights in memory at once). The callback receives the
    // layer's user_data and returns a pointer to the f32 weight buffer
    // (out_dim * in_dim floats). The caller owns the buffer; it must stay
    // valid until weights_release_fn is called. When null, weights must be
    // pre-loaded into the weights pointer above.
    const float * (* weights_load_fn)(void * user_data);
    void          (* weights_release_fn)(void * user_data, const float * buf);
    void         * weights_user_data;
};

// Evaluator callback: quantize with candidate, return score.
// The GA calls this; the implementation dispatches to ts_quantize_2d
// or the kernel-direct path (L6) depending on what's available.
typedef ts_awq_score (*ts_awq_eval_fn)(const ts_awq_candidate * cand,
                                       const ts_awq_layer * layer,
                                       void * ctx);

// Concrete host-supplied evaluator that ports Python's _evaluate_uncached
// (the per-layer reconstruction + composite fitness). Use this when the host
// has populated ts_awq_layer.{second_moment,fourth_moment,max_abs,
// train_activations,heldout_activations,ref_*_output}. Layers without those
// fields fall back to the same paths Python falls back to (diagonal
// second-moment error, train==heldout). ctx is unused (pass nullptr).
ts_awq_score ts_awq_default_eval(const ts_awq_candidate * cand,
                                 const ts_awq_layer * layer,
                                 void * ctx);

// MAP-Elites archive cell descriptor.
struct ts_awq_archive_cell {
    int32_t kurtosis_bucket;    // quantized kurtosis
    int32_t eff_rank_bucket;    // quantized effective rank
    int32_t family_bucket;      // tensor family index
};

// Forward declaration: ts_awq_evolve_params.layer_skip_lookup takes a pointer
// to ts_awq_evolve_result, defined below.
struct ts_awq_evolve_result;

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
    // Parallel candidate evaluation: when > 1, the population evaluation
    // within each generation is fanned out across this many threads. This is
    // the "one layer, parallel candidates" model: the weight buffer is loaded
    // once and shared read-only across threads, each evaluating a different
    // candidate. Combined with ts_awq_evolve_all's n_threads=1 (serial layer
    // processing), this keeps peak memory at 1 weight buffer + n_eval_threads
    // scratch buffers instead of n_threads x (weight + scratch).
    int32_t   n_eval_threads;
    // One-shot family hypothesis test: when seed_candidate is non-null AND
    // seed_composite > 0, evaluate the seed once. If the score is >=
    // seed_composite * seed_accept_ratio, accept it directly (skip the GA).
    // seed_composite is the score the seed achieved on its original tensor.
    // seed_accept_ratio is the transfer threshold (e.g. 0.95 = accept if this
    // tensor scores within 95% of the original). Default 0.95.
    float     seed_composite;
    float     seed_accept_ratio;
    // Per-tensor parallelism for ts_awq_evolve_all: each layer's GA runs
    // independently (its population, archive and rng are local to one
    // ts_awq_evolve call), so the per-layer loop fans out across this many
    // threads. 0/1 = serial (the historical behaviour). The evaluator ctx is
    // shared across threads and must guard any mutable state itself.
    int32_t   n_threads;
    // Optional per-layer progress hook. Invoked once when each layer's GA
    // completes, with the layer index, the layer count, and the layer's
    // tensor name (may be NULL). The callback must be thread-safe since
    // layers run concurrently when n_threads > 1. May be NULL.
    void    (* on_layer_done)(int64_t idx, int64_t n_layers, const char * name,
                              void * user);
    void     * on_layer_done_user;
    // Optional phase-transition hook. Invoked when evolve_all enters the
    // screening phase (with the screen layer count) and again when it enters
    // the main evolution (with the full layer count). phase is one of
    // "screen" or "evolve". May be NULL.
    void    (* on_phase_change)(const char * phase, int64_t n_layers,
                                void * user);
    void     * on_phase_change_user;
    // Early termination: if the best composite score has not improved by more
    // than stagnation_epsilon for stagnation_limit consecutive generations,
    // the GA stops early and marks the tensor as converged. 0 = disabled
    // (run all generations). Default is set by the caller (the dispatch sets
    // a sensible default).
    int64_t   stagnation_limit;    // gens without improvement before stop, 0 = off
    float     stagnation_epsilon;  // min improvement to count as progress
    // Family warm-start seed: if non-null, injected into the initial population
    // of every island (replacing one random member). This lets the caller pass
    // the best candidate from a previously-converged tensor in the same family,
    // so the GA starts near-optimal instead of from scratch. May be NULL.
    const ts_awq_candidate * seed_candidate;
    // Persistent-store hooks (optional, used when the dispatch opens a
    // DuckDB-backed store via --quantize-db). All may be NULL.
    //
    // eval_recorder: invoked once per candidate evaluation. Called from
    // inside the GA's parallel eval_population, so the implementation MUST
    // be thread-safe (e.g. buffer into a per-call Appender). Used to stream
    // every GA candidate to the ga_evaluations table.
    void    (* eval_recorder)(const ts_awq_layer * layer,
                              int32_t generation, int32_t island,
                              int32_t candidate_idx,
                              const ts_awq_candidate * cand,
                              const ts_awq_score    * score,
                              void * user) = nullptr;
    void     * eval_recorder_user = nullptr;
    // family_seed_lookup: evolve_all calls this when seeding a layer's GA
    // from a same-family prior result. Returns true and fills *out when a
    // seed exists; false otherwise. Used to warm-start from prior runs
    // stored on disk (replaces the in-memory family_seeds map for the
    // first layer of each family). out_composite is the seed's score on its
    // original tensor (used by the one-shot hypothesis test); may be null.
    bool    (* family_seed_lookup)(const char * family,
                                   ts_awq_candidate * out,
                                   float * out_composite,
                                   void * user) = nullptr;
    void     * family_seed_lookup_user = nullptr;
    // layer_skip_lookup: evolve_all calls this once per layer before
    // running its GA. When it returns true, the layer's result is filled
    // from a prior converged run (the callback writes into *out_result)
    // and the GA is skipped entirely. Used for crash-resumability: a
    // restart picks up from the last completed tensor.
    bool    (* layer_skip_lookup)(const ts_awq_layer * layer,
                                  ts_awq_evolve_result * out_result,
                                  void * user) = nullptr;
    void     * layer_skip_lookup_user = nullptr;
};

// Result of evolution for one tensor family.
struct ts_awq_evolve_result {
    ts_awq_candidate best;
    ts_awq_score     best_score;
    int64_t          generations_run;
    int64_t          evaluations;
    // Convergence flags surfaced for the persistent store. converged is set
    // when the GA ended via stagnation early-termination (true) vs ran all
    // generations (false). warm_started is set when the initial population
    // was seeded from a family/persistent-store seed. The dispatch records
    // both into the ga_results row.
    bool             converged     = false;
    bool             warm_started = false;
    bool             family_matched = false;  // one-shot hypothesis test accepted
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
int ts_awq_evolve_all(ts_awq_layer * layers, int64_t n_layers,
                      ts_awq_eval_fn eval, void * eval_ctx,
                      const ts_awq_evolve_params * params,
                      std::vector<ts_awq_evolve_result> * results);

// Utility: compute archive cell from layer regime descriptors.
ts_awq_archive_cell ts_awq_make_cell(float kurtosis, float eff_rank,
                                     int32_t family_idx);

// Utility: serialize candidate to JSON fragment (for policy output).
std::string ts_awq_candidate_json(const ts_awq_candidate * cand);
