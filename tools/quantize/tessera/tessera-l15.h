#pragma once

//
// tessera-l15.h
//
// L1.5 FP16 reference reader. At quantize time, loads the FP16
// dequantized weight captured by the runtime dequant hook (v3 sidecar)
// and provides kernel-direct fitness metrics (relative Frobenius error,
// layer-output MSE) for the AWQ scale search and GA evaluation.
//
// The reader accepts both on-disk dtypes:
//   - `.act.dequant.f32` -> ts_l15_reference::file_dtype = 0
//     (legacy W4A4 mode, F32 reference; preserved for back-compat)
//   - `.act.dequant.f16` -> ts_l15_reference::file_dtype = 1
//     (default, the FP16 ground truth that distinguishes L1.5 from L1)
// Both decode into `data` as F32 so the metric functions can stay
// dtype-agnostic; use `file_dtype` if the caller needs the original
// on-disk dtype (e.g. for receipt provenance or to compute the
// reference quantization error vs the L1 F32 dequant).
//

#include <string>
#include <vector>
#include <cstdint>

// L1.5 reference data for one tensor: the FP16 ground truth (or F32
// legacy reference) captured at runtime, stored as F32 in `data` for
// downstream metrics. `file_dtype` records the on-disk dtype (0 = F32,
// 1 = F16) so callers can tell which path the reference came from.
struct ts_l15_reference {
    std::string tensor_name;
    int64_t     rows;       // out_dim
    int64_t     cols;       // in_dim
    std::vector<float> data;  // (rows x cols) row-major F32
                              // (FP16 is upcast on read; the upcast is
                              // exact, so this is the same numbers the
                              // writer would have stored for F32)
    uint32_t    file_dtype; // 0 = F32, 1 = F16 (DEQUANT_DTYPE_F16)
    float       outlier_threshold;
    int64_t     outlier_count;
};

// Load L1.5 reference from a v3 sidecar file. The path's suffix is
// not checked here - any v3 sidecar with a recognized dtype (F32 or
// F16) is accepted. The tensor name is derived from the filename
// (the longest trailing `.act.dequant.f32` / `.act.dequant.f16` is
// stripped, so a file named `foo.act.dequant.f16` loads as tensor
// `foo`).
// Returns 0 on success.
int ts_l15_load_reference(const char * sidecar_path,
                          ts_l15_reference * out,
                          std::string * err_msg);

// Load all L1.5 references from a directory of sidecar files.
// Files are matched on either L1.5 suffix
// (".act.dequant.f32" or ".act.dequant.f16") in the directory
// (matching DEQUANT_FILE_SUFFIX_L15_F32 / DEQUANT_FILE_SUFFIX_L15_F16
// in common/tessera-debug/tessera-debug.h, the suffixes the runtime
// hook actually writes). When both suffixes are present for the same
// tensor, the F16 file is preferred (it's the new default; the F32
// is a legacy duplicate).
// Returns number of references loaded, or -1 on error.
int ts_l15_load_directory(const char * dir_path,
                          std::vector<ts_l15_reference> * refs,
                          std::string * err_msg);

// Compute the kernel-direct t_l^2 (relative Frobenius reconstruction error)
// between a quantized-then-dequantized weight and the L1.5 reference.
// t_l^2 = ||W_hat - W_ref||_F^2 / ||W_ref||_F^2
// Works on both F32 and F16 L1.5 references (the F16 reference is
// already upcast to F32 in `ts_l15_reference::data`).
float ts_l15_relative_frob(const float * w_hat, const ts_l15_reference * ref);

// Compute layer-output MSE: ||W_hat @ X - W_ref @ X||_F^2 / (n_tokens * out_dim)
// for calibration activations X (cols x n_tokens, row-major).
// Works on both F32 and F16 L1.5 references.
float ts_l15_layer_output_mse(const float * w_hat,
                              const ts_l15_reference * ref,
                              const float * calib_X,
                              int64_t n_tokens);
