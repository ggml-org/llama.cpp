#pragma once

//
// tessera-higgs.h
//
// HIGGS Algorithm 3: per-layer alpha_l estimation via Gaussian-noise
// perturbation sweep + through-origin least-squares fit.
//

#include <cstdint>
#include <vector>
#include <string>

struct ts_higgs_params {
    int64_t J;              // number of noise levels (default 15)
    float   t_min;          // minimum t value (default 0.01)
    float   t_max;          // maximum t value (default 0.10)
    float   r2_threshold;   // minimum R^2 for valid fit (default 0.95)
    float   alpha_floor;    // minimum alpha_l (default 1e-6)
    uint32_t seed;
    bool    verbose;
};

struct ts_higgs_layer_result {
    std::string name;
    float       alpha_l;        // fitted coefficient
    float       r_squared;      // goodness of fit
    bool        valid;          // true if R^2 >= threshold
    std::vector<float> t_grid;  // noise levels used
    std::vector<float> deltas;  // measured metric changes
};

struct ts_higgs_result {
    std::vector<ts_higgs_layer_result> layers;
    int64_t n_valid;
    int64_t n_fallback_uniform;   // layers that fell back to alpha=1
    float   mean_alpha;
};

// Metric callback: given perturbed weights for one layer, return the
// metric change (delta-KL or delta-PPL) vs unperturbed baseline.
typedef float (*ts_higgs_metric_fn)(const float * perturbed_weights,
                                    int64_t out_dim, int64_t in_dim,
                                    int64_t layer_idx, void * ctx);

// Estimate alpha_l for all layers.
// weights: array of L weight matrices (each out_dim[l] x in_dim[l]).
// metric_fn: called J times per layer with perturbed weights.
int ts_higgs_estimate(const float ** weights,
                      const int64_t * out_dims,
                      const int64_t * in_dims,
                      int64_t n_layers,
                      ts_higgs_metric_fn metric_fn, void * metric_ctx,
                      const ts_higgs_params * params,
                      ts_higgs_result * result);

// Serialize alpha_l vector to JSON (for caching in sidecar/policy).
std::string ts_higgs_to_json(const ts_higgs_result * result);

// Load cached alpha_l from JSON. Returns n_layers loaded, or -1.
int ts_higgs_from_json(const char * json_str, float * alphas_out, int64_t max_layers);

// Extract alpha_l values from a result into a flat vector.
std::vector<float> ts_higgs_extract_alphas(const ts_higgs_result * result);
