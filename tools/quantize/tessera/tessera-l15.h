#pragma once

//
// tessera-l15.h
//
// L1.5 FP16 reference reader. At quantize time, loads the FP16
// dequantized weight captured by the runtime dequant hook (v3 sidecar)
// and provides kernel-direct fitness metrics (relative Frobenius error,
// layer-output MSE) for the AWQ scale search and GA evaluation.
//

#include <string>
#include <vector>
#include <cstdint>

// L1.5 reference data for one tensor: the FP16 dequantized output
// captured at runtime, stored as F32 for computation.
struct ts_l15_reference {
    std::string tensor_name;
    int64_t     rows;       // out_dim
    int64_t     cols;       // in_dim
    std::vector<float> data;  // (rows x cols) row-major F32
    float       outlier_threshold;
    int64_t     outlier_count;
};

// Load L1.5 reference from a v3 sidecar file.
// The sidecar contains the dequantized weight matrix captured at runtime.
// Returns 0 on success.
int ts_l15_load_reference(const char * sidecar_path,
                          ts_l15_reference * out,
                          std::string * err_msg);

// Load all L1.5 references from a directory of sidecar files.
// Files are named "<tensor_name>.tdqt" in the directory.
// Returns number of references loaded, or -1 on error.
int ts_l15_load_directory(const char * dir_path,
                          std::vector<ts_l15_reference> * refs,
                          std::string * err_msg);

// Compute the kernel-direct t_l^2 (relative Frobenius reconstruction error)
// between a quantized-then-dequantized weight and the L1.5 reference.
// t_l^2 = ||W_hat - W_ref||_F^2 / ||W_ref||_F^2
float ts_l15_relative_frob(const float * w_hat, const ts_l15_reference * ref);

// Compute layer-output MSE: ||W_hat @ X - W_ref @ X||_F^2 / (n_tokens * out_dim)
// for calibration activations X (cols x n_tokens, row-major).
float ts_l15_layer_output_mse(const float * w_hat,
                              const ts_l15_reference * ref,
                              const float * calib_X,
                              int64_t n_tokens);
