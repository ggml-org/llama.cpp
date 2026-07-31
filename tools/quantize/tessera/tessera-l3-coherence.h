#pragma once

//
// tessera-l3-coherence.h
//
// L3 per-token (per-row) coherence (Layer 3 of the runtime-aware
// pipeline, see docs/runtime-aware-pipeline.md). The spec's L3 tracks
// per-token distribution divergence across a forward pass; at the
// quantize tool's weight level the equivalent is per-row coherence:
// for each tensor that has both an L1 kernel-dequant sidecar and an
// L1.5 reference sidecar, compute the cosine similarity between the
// kernel's reconstructed row and the reference row. Rows below the
// threshold are tokens where the kernel reconstruction diverges
// significantly from reference.
//
// Sidecar layout (written by common/tessera-debug):
//   L1   : <sidecar_dir>/<tensor>.dequant.f32
//   L1.5 : <reference_dir>/<tensor>.act.dequant.f32
//

#include <cstdint>
#include <string>
#include <vector>

struct ts_l3_config {
    char  sidecar_dir[1024];    // L1 kernel-dequant sidecars
    char  reference_dir[1024];  // L1.5 (FP16/BF16) reference sidecars
    float threshold;            // row cosine floor, default 0.99
};

// Per-tensor coherence summary plus the flagged row indices.
struct ts_l3_tensor_result {
    std::string tensor_name;
    int64_t     rows;
    int64_t     cols;
    float       mean_cosine;
    float       min_cosine;
    int64_t     n_flagged;
    std::vector<int64_t> flagged_rows;
};

struct ts_l3_report {
    std::vector<ts_l3_tensor_result> tensors;
    int64_t n_tensors;
    int64_t n_flagged_rows;     // sum across tensors
};

void ts_l3_default_config(ts_l3_config * cfg);

// Cosine similarity between two vectors of n elements. Returns 1.0 when
// both norms are zero (identical zero rows), 0.0 when exactly one is zero.
float ts_l3_row_cosine(const float * a, const float * b, int64_t n);

// Per-row coherence for one tensor. l1 and ref are (rows x cols) row-major.
// Flags rows whose cosine falls below threshold. Returns 0 on success,
// -1 on invalid args or shape mismatch.
int ts_l3_tensor_coherence(const float * l1, const float * ref,
                           int64_t rows, int64_t cols,
                           float threshold,
                           ts_l3_tensor_result * out);

// Run over all candidate tensors that have BOTH an L1 sidecar (in
// sidecar_dir) and an L1.5 reference sidecar (in reference_dir). Tensors
// missing either sidecar are skipped. Returns the number of tensors
// processed (>= 0), or -1 on invalid args.
int ts_l3_run(const ts_l3_config * cfg,
              const char * const * tensor_names,
              int64_t n_tensors,
              ts_l3_report * report);
