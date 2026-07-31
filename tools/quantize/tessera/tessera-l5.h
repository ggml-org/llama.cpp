#pragma once

//
// tessera-l5.h
//
// L5 sensitivity scoring and iterative requantization orchestrator.
// Ports tools/tessera/l5_metrics.py and l5_orchestrator.py.
//

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <map>

#include "tessera-l2-diff.h"    // ts_l2_report (adaptive requant input)

// --- Sensitivity metrics (l5_metrics.py) ---

// Per-component score map: tensor_name -> score.
using ts_score_map = std::map<std::string, float>;

// Imatrix magnitude scorer: score = mean(|act|) per tensor.
ts_score_map ts_l5_imatrix_magnitude(const float * imatrix_vals,
                                     const char ** tensor_names,
                                     const int64_t * tensor_dims,
                                     int64_t n_tensors);

// Gradient proxy: score ~ ||dL/dW|| estimated from output sensitivity.
ts_score_map ts_l5_gradient_proxy(const float * output_sensitivity,
                                  const char ** tensor_names,
                                  int64_t n_tensors);

// Layer position prior: earlier/later layers get different priors.
ts_score_map ts_l5_layer_position_prior(const char ** tensor_names,
                                        int64_t n_tensors,
                                        int64_t n_layers_total);

// Combine multiple scorers with weights.
ts_score_map ts_l5_combine(const ts_score_map ** scorers,
                           const float * weights,
                           int64_t n_scorers);

// Momentum EMA tracker for streaming sensitivity updates.
struct ts_l5_ema {
    ts_score_map state;
    float        beta;      // EMA decay, default 0.9
};

void ts_l5_ema_init(ts_l5_ema * ema, float beta);
void ts_l5_ema_update(ts_l5_ema * ema, const ts_score_map * new_scores);

// Percentile rank normalization: scores -> [0, 1] percentile.
ts_score_map ts_l5_percentile_rank(const ts_score_map * scores);

// Pick top fraction of tensors by score.
std::vector<std::string> ts_l5_pick_top(const ts_score_map * scores,
                                        float fraction);

// Expected MSE delta from requantizing a tensor.
float ts_l5_expected_mse_delta(const char * tensor_name,
                               float current_score, float target_score);

// Quantization ladder stepping.
int         ts_l5_ladder_index(const char * qtype);
const char * ts_l5_step_up(const char * qtype);    // higher precision
const char * ts_l5_step_down(const char * qtype);  // lower precision

// --- Orchestrator (l5_orchestrator.py) ---

enum ts_requant_action_type {
    TS_REQUANT_NONE     = 0,
    TS_REQUANT_STEP_UP  = 1,    // increase precision
    TS_REQUANT_STEP_DOWN = 2,   // decrease precision
};

struct ts_requant_action {
    std::string tensor_name;
    ts_requant_action_type type;
    std::string from_qtype;
    std::string to_qtype;
    float expected_delta;
};

struct ts_requant_plan {
    std::vector<ts_requant_action> actions;
    float total_expected_delta;
    int64_t generation;
};

struct ts_orchestrator_params {
    int64_t max_generations;    // default 10
    float     top_fraction;     // fraction of tensors to consider, default 0.1
    float     delta_threshold;  // minimum expected delta to act, default 0.01
    float     ema_beta;         // EMA decay for streaming scores
    bool      verbose;
};

// Run one orchestrator generation: score, plan, return actions.
int ts_l5_orchestrate_step(const ts_score_map * sensitivity,
                           const char ** current_qtypes,
                           int64_t n_tensors,
                           int64_t generation,
                           const ts_orchestrator_params * params,
                           ts_requant_plan * plan);

// --- Adaptive requantization (L5 closes the L2 loop) ---
//
// L2 flags tensors whose divergence overshoots their type baseline; this
// turns those flags into tightened requantization params. The worse the
// overshoot, the more alpha/clip are reduced. Applying the plan
// (re-quantize + GGUF rewrite) goes through the existing quantize /
// GGUF-writer path, matching ts_l5_orchestrate_step which also emits a
// plan for downstream application.

struct ts_l5_adaptive_params {
    float alpha_scale;    // base alpha multiplier (< 1 tightens), default 0.5
    float clip_scale;     // base clip multiplier (< 1 tightens), default 0.5
    float min_alpha;      // alpha floor, default 0.1
    float min_clip;       // clip floor, default 0.1
};

// Tightened requantization spec for one flagged tensor.
struct ts_l5_requant_spec {
    std::string tensor_name;
    std::string qtype;
    float divergence;    // observed relative_frobenius (from L2)
    float expected;      // type baseline
    float overshoot;     // divergence / expected (>= 1)
    float new_alpha;
    float new_clip;
};

struct ts_l5_adaptive_plan {
    std::vector<ts_l5_requant_spec> specs;
    int64_t n_requant;
    int64_t generation;
};

void ts_l5_adaptive_default_params(ts_l5_adaptive_params * p);

// Identify flagged tensors in an L2 report and compute tightened
// requantization params for each. params == nullptr uses defaults.
// Returns the number of specs (>= 0), or -1 on invalid plan.
int ts_l5_adaptive_requant(const ts_l2_report * report,
                           const ts_l5_adaptive_params * params,
                           int64_t generation,
                           ts_l5_adaptive_plan * plan);
