#pragma once

//
// tessera-l1-fitness.h
//
// L1 kernel-direct fitness: evaluates quantization candidates against
// the kernel's actual dequant output (from L1 sidecar files) rather than
// the offline relative-Frobenius proxy. This is the production t_l^2 of
// the stated fitness form Sum alpha_l * t_l^2 (see
// docs/research-alignment-2026-07-30.md and docs/runtime-aware-pipeline.md
// Section 6): per tensor, t_l^2 = ||W_hat - dequant_kernel(W_l)||_F^2 /
// ||W_l||_F^2, where dequant_kernel(W_l) is the L1 sidecar (what the
// kernel actually reconstructs), not the offline ternary reconstruction.
//
// The sidecar files are the v3 TDQT dumps written by the runtime dequant
// hook (common/tessera-debug) at <sidecar_dir>/<tensor_name>.dequant.f32.
//

#include <cstdint>
#include <vector>

struct ts_l1_fitness_config {
    char  sidecar_dir[1024];    // directory containing .dequant.f32 files
    bool  use_kernel_direct;    // true = kernel-direct, false = offline proxy
    float blend_factor;         // 0.0 = pure offline, 1.0 = pure kernel-direct
                                // intermediate values blend the two
};

// Initialize a config to the offline-proxy default (kernel-direct off,
// blend 1.0, empty sidecar dir).
void ts_l1_fitness_default_config(ts_l1_fitness_config * cfg);

// Load the L1 sidecar for a specific tensor. The file is expected at
// <sidecar_dir>/<tensor_name>.dequant.f32. On success returns 0 and fills
// out_data (row-major F32) plus out_rows / out_cols. Returns non-zero when
// the file is absent or malformed (out buffers left unchanged).
int ts_l1_load_sidecar(const char * sidecar_dir,
                       const char * tensor_name,
                       std::vector<float> * out_data,
                       int64_t * out_rows, int64_t * out_cols);

// Compute kernel-direct t_l^2 for a single tensor:
// t_l^2 = ||W_hat - dequant_kernel||_F^2 / ||W||_F^2
// where dequant_kernel is from the L1 sidecar. All arrays hold n elements
// in the same (row-major) order. Returns 0 when ||W||_F^2 == 0.
float ts_l1_kernel_direct_t2(const float * w_hat,
                             const float * w_original,
                             const float * kernel_dequant,
                             int64_t n);

// Compute blended fitness: (1-blend)*offline_t2 + blend*kernel_direct_t2.
// blend_factor is clamped to [0, 1].
float ts_l1_blended_t2(float offline_t2, float kernel_direct_t2,
                       float blend_factor);

// Batch: compute kernel-direct t_l^2 for all tensors that have sidecar
// files. w_hats[l] / w_originals[l] point at sizes[l] elements. Tensors
// without a readable sidecar fall back to the offline proxy
// (||W_hat - W||_F^2 / ||W||_F^2). Returns the number of tensors that had
// sidecar data.
int ts_l1_compute_all_t2(const char * sidecar_dir,
                         const float * const * w_hats,
                         const float * const * w_originals,
                         const char * const * tensor_names,
                         int64_t n_tensors,
                         const int64_t * sizes,
                         float * out_t2);
