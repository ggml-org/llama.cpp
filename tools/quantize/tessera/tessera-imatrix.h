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
    // tensor_name -> per-channel activation magnitudes (in_dim floats)
    std::map<std::string, std::vector<float>> data;
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

// Compute regime statistics from imatrix data for one tensor.
struct ts_imatrix_regime_stats {
    float kurtosis;         // excess kurtosis of activation distribution
    float eff_rank;         // effective rank (spectral entropy proxy)
    float mean_magnitude;   // mean |activation|
    float p99;              // 99th percentile
};

ts_imatrix_regime_stats ts_imatrix_regime(const float * act_data, int64_t dim);
