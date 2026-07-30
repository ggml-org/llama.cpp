#pragma once

//
// tessera-mm-awq.h
//
// Per-modality AWQ scale computation (text / image / audio). Extends the
// single-modality AWQ scale search to three modality-tagged activation
// scale arrays with a modality-weighted fitness. See
// docs/multimodal-calibration-design.md (M1, M2, M8).
//

#include <cstdint>
#include <vector>
#include <string>

struct ts_mm_awq_params {
    float   alpha[3];            // per-modality AWQ exponent (0 = auto-search)
    float   clip[3];             // per-modality clip threshold (applied at ternarization)
    float   modality_weights[3]; // fitness weights (default 0.5, 0.3, 0.2)
    bool    error_on_missing;    // true = error if a modality is absent (default true)
    int64_t awq_grid;            // grid points for alpha search (default 20)
};

struct ts_mm_awq_result {
    std::vector<uint16_t> act_scale_text;   // f16 (in_dim,)
    std::vector<uint16_t> act_scale_image;  // f16 (in_dim,)
    std::vector<uint16_t> act_scale_audio;  // f16 (in_dim,)
    float best_alpha[3];
    float weighted_mse;         // renormalized sum of modality_weights[i] * mse_i over present modalities
    float mse_per_modality[3];
};

// Compute per-modality AWQ scales.
// act_scales: array of 3 pointers (text, image, audio), each (in_dim,) or nullptr.
// weights: (out_dim x in_dim).
// calib_X: array of 3 calibration matrices per modality (or nullptr if absent).
// ref_output: array of 3 reference outputs per modality (or nullptr).
int ts_mm_awq_compute(const float * weights,
                      const float * act_scales[3],
                      const float * calib_X[3],
                      const float * ref_output[3],
                      const int64_t n_tokens[3],
                      int64_t out_dim, int64_t in_dim,
                      const ts_mm_awq_params * params,
                      ts_mm_awq_result * result,
                      std::string * err_msg);

// Default params: alpha=auto for all, weights 0.5/0.3/0.2, error_on_missing=true.
ts_mm_awq_params ts_mm_awq_default_params();
