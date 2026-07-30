#pragma once

//
// tessera-mm-fitness.h
//
// Modality-aware GA fitness weighting. Composite objective is the
// alpha-weighted Linearity-Theorem form Sum_l alpha_l * t_l^2 extended
// across modality: per-modality losses are combined with GA-evolved
// weights (default text/image/audio = 0.5/0.3/0.2, M1).
//

#include <cstdint>
#include <vector>
#include <string>

struct ts_mm_fitness_params {
    float modality_weights[3];  // default {0.5, 0.3, 0.2}
    bool  per_family_breakdown; // report per-family per-modality breakdown
};

struct ts_mm_fitness_score {
    float composite;            // weighted sum across modalities
    float per_modality[3];      // individual modality losses
    float alpha_weighted;       // Sum_l alpha_l * t_l^2 (if alpha provided)
};

// Compute modality-weighted fitness from per-modality t_l^2 values.
// t2_per_modality: [3] arrays of per-layer t_l^2, each (n_layers,).
// alpha_l: per-layer HIGGS weights (n_layers,), or nullptr for uniform.
// present: [3] bools indicating which modalities have data.
ts_mm_fitness_score ts_mm_fitness_compute(
    const float * t2_per_modality[3],
    const float * alpha_l,
    const bool present[3],
    int64_t n_layers,
    const ts_mm_fitness_params * params);

// Per-family breakdown: group tensors by family, report per-modality loss.
struct ts_mm_family_score {
    std::string family;
    float       loss_per_modality[3];
    float       composite;
    int64_t     n_tensors;
};

std::vector<ts_mm_family_score> ts_mm_fitness_family_breakdown(
    const char ** tensor_names,
    const char ** tensor_families,
    const float * t2_per_modality[3],
    const bool present[3],
    int64_t n_tensors,
    const ts_mm_fitness_params * params);

// Default params.
ts_mm_fitness_params ts_mm_fitness_default_params();
