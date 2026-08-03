#pragma once

//
// tessera-regime.h
//
// Operative regime router: connects ts_route_expert to real imatrix
// data and tensor metadata. Infers tensor families from names, computes
// regime descriptors from activation statistics, and routes each tensor
// to its best expert (AWQ, LRQ, DartQuant, FLRQ, CHAMP-Q, SEPTQ).
//

#include <string>
#include <vector>
#include <cstdint>

#include "tessera-search.h"

struct ts_regime_descriptor {
    std::string tensor_name;
    std::string family;         // "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down"
    float       kurtosis;       // activation kurtosis from imatrix
    float       eff_rank;       // effective rank (spectral entropy)
    float       mean_magnitude; // mean |activation|
    float       p99;            // 99th percentile
    int64_t     out_dim;
    int64_t     in_dim;
    int32_t     modality;       // 0=text, 1=image, 2=audio (default 0)
    // Per-channel max-outlier concentration, derived from the imatrix
    // .in_maxabs field when available: ratio of the largest per-channel
    // max|activation| to the median across channels. 1.0 means outliers are
    // spread evenly (no single channel dominates); large values mean a small
    // set of channels carry the heavy tail and the rotation/permutation
    // experts should grow their per-row repair budget accordingly. 0.0 when
    // no per-channel max is available (legacy .npz, or producer run that did
    // not collect max) - experts then keep their default outlier budget.
    float       max_outlier_ratio;
};

struct ts_regime_routing {
    std::string  tensor_name;
    ts_expert_id expert;
    std::string  reason;        // human-readable routing reason
    float        confidence;    // routing confidence [0, 1]
};

// Infer tensor family from name (e.g. "blk.0.attn_q.weight" -> "attn_q").
std::string ts_regime_infer_family(const char * tensor_name);

// Infer modality from tensor name patterns (0=text, 1=image, 2=audio).
// Vision/audio embedder tensors map to their modality; everything else
// (the shared LM blocks) defaults to text.
int ts_regime_infer_modality(const char * tensor_name);

// Classify a tensor's regime from its descriptors.
// Returns the routing decision with reason.
ts_regime_routing ts_regime_classify(const ts_regime_descriptor * desc);

// Route all tensors in a model. Returns one routing per tensor.
std::vector<ts_regime_routing> ts_regime_route_all(
    const ts_regime_descriptor * descs, int64_t n_tensors);

// Compute regime descriptors from imatrix data + weight matrix.
// Fills kurtosis, eff_rank, mean_magnitude, p99 from activation stats.
// max_outlier_ratio is left at 0 (no per-channel max supplied).
ts_regime_descriptor ts_regime_compute_descriptor(
    const char * tensor_name,
    const float * weights, int64_t out_dim, int64_t in_dim,
    const float * imatrix_data, int64_t imatrix_dim);

// As above, but also fills max_outlier_ratio from the per-channel max|act|
// vector (the imatrix .in_maxabs field, in_dim floats). imatrix_max_abs may
// be null / imatrix_max_abs_dim 0 to signal "no per-channel max available";
// the descriptor then keeps max_outlier_ratio = 0 and the experts fall back
// to their default outlier budget. Pass the same imatrix_data as the 6-arg
// form; the two arrays are independent (mean squared act vs running max).
ts_regime_descriptor ts_regime_compute_descriptor(
    const char * tensor_name,
    const float * weights, int64_t out_dim, int64_t in_dim,
    const float * imatrix_data, int64_t imatrix_dim,
    const float * imatrix_max_abs, int64_t imatrix_max_abs_dim);

// Per-expert quantization parameter profile. Maps a routed expert to the
// concrete adjustments applied to ts_quant_params_2d before quantizing, so
// the routing decision actually changes the quantization code path.
struct ts_expert_profile {
    float alpha_scale;       // multiplier on AWQ alpha (1.0 = no change)
    float clip_scale;        // multiplier on AWQ clip
    bool  use_septq;         // enable SEPTQ Hessian compensation
    int   awq_grid;          // AWQ grid search resolution
    int   max_outliers;      // outlier budget (per row)
    float outlier_thresh;    // outlier threshold multiplier
};

// Return the default parameter profile for a routed expert. The optional
// modality_id (0=text, 1=image, 2=audio) layers per-modality adjustments on
// top of the expert baseline: audio tightens the clip, image widens the
// outlier budget, text is unchanged.
ts_expert_profile ts_expert_default_profile(ts_expert_id expert, int modality_id = 0);

// Human-readable expert name ("AWQ", "LRQ", "DartQuant", ...).
const char * ts_expert_name(ts_expert_id expert);

// Summary statistics for a routing plan.
struct ts_regime_summary {
    int64_t count_per_expert[TS_EXPERT_COUNT];
    float   mean_kurtosis;
    float   mean_eff_rank;
};

ts_regime_summary ts_regime_summarize(const std::vector<ts_regime_routing> * routings,
                                      const ts_regime_descriptor * descs,
                                      int64_t n_tensors);
