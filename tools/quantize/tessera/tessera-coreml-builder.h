#pragma once

//
// tessera-coreml-builder.h
//
// Builds a CoreML .mlpackage directory from Tessera-quantized tensors.
// Stock ops v1: pre-dequantizes Tile640 weights to fp16 and stores them
// as standard innerProduct layers. The .mlpackage is then compiled by
// the Objective-C layer via +compileModelAtURL: (C9).
//

#include <cstdint>
#include <string>
#include <vector>

struct ts_coreml_builder_tensor {
    std::string name;
    int64_t     out_dim;
    int64_t     in_dim;
    // Pre-dequantized fp16 weights (out_dim * in_dim), row-major.
    // Caller is responsible for dequantizing from Tile640 format.
    const uint16_t * weights_f16;
    bool        has_act_scale;
    const uint16_t * act_scale_f16;  // [in_dim] or nullptr
};

struct ts_coreml_builder_params {
    std::string output_path;    // path to write .mlpackage directory
    std::string model_name;     // display name in metadata
    int32_t     compute_units;  // 0=all, 1=cpu, 2=cpu+gpu, 3=cpu+ne
};

// Build a .mlpackage directory structure:
//   <output_path>/
//     Manifest.json
//     Data/
//       model.mlmodel   (JSON spec + weight blobs)
//     Metadata/
//       model.json
//
// Returns 0 on success, -1 on error (err_msg set if non-null).
int ts_coreml_build_package(const ts_coreml_builder_tensor * tensors,
                            int64_t n_tensors,
                            const ts_coreml_builder_params * params,
                            std::string * err_msg);

// Compute the total weight size in bytes (for progress reporting).
int64_t ts_coreml_total_weight_bytes(const ts_coreml_builder_tensor * tensors,
                                     int64_t n_tensors);
