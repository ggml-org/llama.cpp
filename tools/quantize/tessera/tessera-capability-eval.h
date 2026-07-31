#pragma once

//
// tessera-capability-eval.h
//
// Multi-axis capability eval (docs/self-improving-loop-design.md section
// 4.7). The fitness has five mechanistically-independent behavioral
// axes. Four are OPTIMIZATION axes where trade-offs live; general
// competence is a GUARD axis - a hard regression constraint, not a
// trade-off weight. The score vector is the substrate; the weighted-sum
// scalar and the Pareto non-domination test are two lenses over the same
// numbers (both stay live and are A/B'd). This is the C++ analogue the
// Swift `evaluate` tool can shell out to.
//

#include <string>

struct ts_capability_score {
    double mechanical;          // failing-test + compiler/type-error instances
    double api_currency;        // deprecated-API migration instances
    double hard_tail;           // escalation / hard-reasoning instances
    double personal_style;      // trunk LoRA / personal-distribution fit
    double general_competence;  // broad held-out set; the collapse-guard axis
};

// Weighted-sum lens over the four optimization axes, in field order:
// weights[0]=mechanical, [1]=api_currency, [2]=hard_tail,
// [3]=personal_style. weights[4] (general_competence) is the guard axis
// and is deliberately NOT summed here.
double ts_capability_score_weighted_sum(const ts_capability_score * s, const double weights[5]);

// Pareto lens: a dominates b when it is >= on all five axes and > on at
// least one.
bool ts_capability_score_dominates(const ts_capability_score * a, const ts_capability_score * b);

// Guard lens: general_competence must not drop more than epsilon below
// baseline. A NULL baseline passes (nothing to regress against).
bool ts_capability_score_passes_guard(const ts_capability_score * s, const ts_capability_score * baseline, double epsilon);

// Load per-axis instance results from a versioned JSON file and reduce
// each axis to its pass fraction. Schema (schema_version 1):
//   { "schema_version": 1,
//     "axes": { "mechanical": {"pass":N,"fail":M}, ... all five ... },
//     "baseline": { "mechanical":0.8, ... all five ... }   // optional
//   }
// The optional "baseline" is a prior score vector (already fractions);
// when present and baseline/has_baseline are non-NULL they are filled.
// Validates schema_version and fails loudly on mismatch or malformed
// input (returns non-zero, leaves *out untouched); never silently
// defaults.
int ts_capability_score_load(const char * path,
                             ts_capability_score * out,
                             ts_capability_score * baseline,
                             bool * has_baseline,
                             std::string * err_msg);
