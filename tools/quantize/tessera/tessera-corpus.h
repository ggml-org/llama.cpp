#pragma once

//
// tessera-corpus.h
//
// Built-in synthetic calibration corpus. Generated at runtime from a
// fixed-seed PRNG (xorshift32 + Box-Muller), so nothing is baked into
// the binary as a static array. Deterministic: same params -> same
// output, always. Used as the default calibration input when no
// --calib-corpus is provided.
//

#include <cstdint>
#include <vector>
#include <string>

struct ts_corpus_params {
    int64_t n_tokens;       // number of calibration tokens (default 256)
    int64_t in_dim;         // activation dimension (default 4096)
    uint32_t seed;          // fixed seed for reproducibility (default 12345)
    float   outlier_rate;   // fraction of outlier channels (default 0.01)
    float   outlier_scale;  // multiplier for outlier channels (default 10.0)
};

// Default params.
ts_corpus_params ts_corpus_default_params();

// Generate the built-in mini-corpus deterministically.
// Output: (n_tokens x in_dim) row-major float matrix.
// Same params -> same output, always.
std::vector<float> ts_corpus_generate(const ts_corpus_params * params);

// Generate and optionally write to a .npy file (for --calib-corpus-out).
// Returns 0 on success.
int ts_corpus_generate_to_file(const ts_corpus_params * params,
                               const char * output_path,
                               std::string * err_msg);

// Load calibration activations from a directory of .npy/.npz files.
// Returns (n_tokens x in_dim) matrix, or empty on error.
std::vector<float> ts_corpus_load_directory(const char * dir_path,
                                            int64_t * out_n_tokens,
                                            int64_t * out_in_dim,
                                            std::string * err_msg);
