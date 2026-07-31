#pragma once

//
// tessera-mm-imatrix.h
//
// Reader for multimodal imatrix v3 files. Extends the base imatrix
// with a per-modality breakdown (text / image / audio) per tensor,
// each carrying its own regime stats, in_sum2 and token counts.
//

#include <string>
#include <map>
#include <vector>
#include <cstdint>

#include "tessera-imatrix.h" // for ts_imatrix_regime_stats

enum ts_modality {
    TS_MODALITY_TEXT  = 0,
    TS_MODALITY_IMAGE = 1,
    TS_MODALITY_AUDIO = 2,
    TS_MODALITY_COUNT = 3,
};

struct ts_mm_imatrix_entry {
    ts_imatrix_regime_stats stats[3]; // per-modality regime stats
    float   in_sum2[3];               // sum of squared activations per modality
    int64_t counts[3];                // token counts per modality
    bool    has_modality[3];          // which modalities are present
    std::vector<float> act[3];        // per-modality per-channel activation
                                      // magnitudes (in_dim,), used as AWQ act_scales
};

struct ts_mm_imatrix {
    std::map<std::string, ts_mm_imatrix_entry> data;
    std::string source_path;
    int32_t version; // 2 or 3
};

// Load multimodal imatrix (v3 format with modality_breakdown).
int ts_mm_imatrix_load(const char * path, ts_mm_imatrix * out, std::string * err_msg);

// Get the full entry for a tensor (name normalization: strips ".weight").
// Returns nullptr if the tensor is absent from the imatrix.
const ts_mm_imatrix_entry * ts_mm_imatrix_entry_get(
    const ts_mm_imatrix * mm, const char * tensor_name);

// Get per-modality regime stats for a tensor.
const ts_imatrix_regime_stats * ts_mm_imatrix_modality_stats(
    const ts_mm_imatrix * mm, const char * tensor_name, ts_modality mod);

// Get per-modality per-channel activation scales for a tensor (in_dim floats).
// Returns nullptr if the tensor or modality is absent. *dim (optional) receives
// the array length; a modality is only usable as AWQ act_scales when its length
// matches the tensor in_dim.
const float * ts_mm_imatrix_act_scales(
    const ts_mm_imatrix * mm, const char * tensor_name, ts_modality mod, int64_t * dim);

// Compute joint kurtosis across modalities (weighted by counts).
float ts_mm_imatrix_joint_kurtosis(const ts_mm_imatrix_entry * entry);

// Check if a tensor has all 3 modalities. Returns missing mask (0 = all present).
int ts_mm_imatrix_missing_mask(const ts_mm_imatrix_entry * entry);
