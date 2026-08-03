#pragma once

//
// tessera-imatrix.h
//
// Reader for imatrix .npz files (per-tensor activation statistics).
// The imatrix provides per-channel activation magnitudes used for
// AWQ scaling and regime routing.
//

#include <string>
#include <map>
#include <vector>
#include <cstdint>

struct ts_imatrix {
    // tensor_name -> per-channel activation magnitudes (in_dim floats).
    // For GGUF imatrix this is the mean squared activation sums[i]/counts[i];
    // for .npz it is the raw stored vector.
    std::map<std::string, std::vector<float>> data;
    // tensor_name -> per-channel running max of |activation| (in_dim floats).
    // Populated only from GGUF imatrix (.in_maxabs); empty for the legacy
    // .npz path which does not carry per-channel max. Used by the outlier-
    // aware experts (DartQuant/CHAMP-Q) to localize which channels carry the
    // heavy tail. Length matches data[name] when present.
    std::map<std::string, std::vector<float>> max_abs;
    std::string source_path;
};

// Load imatrix from .npz file. Returns 0 on success.
int ts_imatrix_load_npz(const char * path, ts_imatrix * out, std::string * err_msg);

// Load imatrix from a GGUF imatrix file (the format emitted by llama-imatrix).
// Per-channel activation magnitudes are derived from the GGUF imatrix entry's
// sums/counts as sums[i] / max(counts[i], 1). Returns 0 on success.
int ts_imatrix_load_gguf(const char * path, ts_imatrix * out, std::string * err_msg);

// Lookup activation scales for a tensor. Returns nullptr if not found.
// Handles name normalization (strips ".weight" suffix, handles "blk.N." prefix).
const float * ts_imatrix_lookup(const ts_imatrix * imatrix,
                                const char * tensor_name,
                                int64_t * out_dim);

// Lookup per-channel max |activation| for a tensor (the .in_maxabs field of a
// GGUF imatrix). Returns nullptr if the imatrix has no max_abs entry for this
// tensor (e.g. legacy .npz inputs, or a GGUF produced by an older producer
// that did not collect max). out_dim receives the vector length on success.
const float * ts_imatrix_lookup_max_abs(const ts_imatrix * imatrix,
                                        const char * tensor_name,
                                        int64_t * out_dim);

// Compute regime statistics from imatrix data for one tensor.
struct ts_imatrix_regime_stats {
    float kurtosis;         // excess kurtosis of activation distribution
    float eff_rank;         // effective rank (spectral entropy proxy)
    float mean_magnitude;   // mean |activation|
    float p99;              // 99th percentile
};

ts_imatrix_regime_stats ts_imatrix_regime(const float * act_data, int64_t dim);
