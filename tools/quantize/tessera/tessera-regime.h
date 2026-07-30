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
};

struct ts_regime_routing {
    std::string  tensor_name;
    ts_expert_id expert;
    std::string  reason;        // human-readable routing reason
    float        confidence;    // routing confidence [0, 1]
};

// Infer tensor family from name (e.g. "blk.0.attn_q.weight" -> "attn_q").
std::string ts_regime_infer_family(const char * tensor_name);

// Classify a tensor's regime from its descriptors.
// Returns the routing decision with reason.
ts_regime_routing ts_regime_classify(const ts_regime_descriptor * desc);

// Route all tensors in a model. Returns one routing per tensor.
std::vector<ts_regime_routing> ts_regime_route_all(
    const ts_regime_descriptor * descs, int64_t n_tensors);

// Compute regime descriptors from imatrix data + weight matrix.
// Fills kurtosis, eff_rank, mean_magnitude, p99 from activation stats.
ts_regime_descriptor ts_regime_compute_descriptor(
    const char * tensor_name,
    const float * weights, int64_t out_dim, int64_t in_dim,
    const float * imatrix_data, int64_t imatrix_dim);

// Summary statistics for a routing plan.
struct ts_regime_summary {
    int64_t count_per_expert[TS_EXPERT_COUNT];
    float   mean_kurtosis;
    float   mean_eff_rank;
};

ts_regime_summary ts_regime_summarize(const std::vector<ts_regime_routing> * routings,
                                      const ts_regime_descriptor * descs,
                                      int64_t n_tensors);
