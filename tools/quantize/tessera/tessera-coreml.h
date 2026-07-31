#pragma once

//
// tessera-coreml.h
//
// Tessera -> CoreML conversion scaffolding (stock ops v1). Stateless: reads
// tensor descriptors, writes a spec the Objective-C layer compiles into a
// .mlmodelc at quantize time. See docs/tessera-coreml-conversion-design.md
// (C1 stock ops, C2 mlmodelc at quantize time, C5 MMAP weights, C7 fallback,
// C8 one mlmodelc with runtime act_scale).
//

#include <cstdint>
#include <string>
#include <vector>

struct ts_coreml_params {
    std::string output_dir;         // where to write .mlmodelc
    bool        mmap_weights;       // C5: MMAP weights (default true)
    bool        ram_activations;    // C5: RAM activations (default true)
    int32_t     compute_units;      // 0=all, 1=cpu_only, 2=cpu_and_gpu, 3=cpu_and_ne
    bool        allow_fallback;     // C7: auto-fallback to Metal (default true)
};

struct ts_coreml_tensor_desc {
    std::string name;
    int64_t     out_dim;
    int64_t     in_dim;
    int32_t     ggml_type;          // GGML_TYPE_TESSERA_T640 etc.
    bool        has_act_scale;      // whether runtime act_scale is needed
};

struct ts_coreml_result {
    std::string mlmodelc_path;      // path to compiled model
    bool        used_custom_ops;    // false for v1 (stock ops only)
    bool        fell_back_to_metal; // C7: true if CoreML failed
    std::string fallback_reason;    // why fallback happened (empty if none)
    int64_t     n_tensors;
};

// Validate that a set of tensors can be represented with stock CoreML ops.
// Returns 0 if all pass, -1 if any tensor requires custom ops (v2).
int ts_coreml_validate_stock_ops(const ts_coreml_tensor_desc * tensors,
                                 int64_t n_tensors,
                                 std::string * err_msg);

// Generate the CoreML model specification (as a protobuf-ready struct).
// This does NOT compile the model - it produces the spec that the
// Objective-C layer will compile via +compileModelAtURL (C9).
int ts_coreml_generate_spec(const ts_coreml_tensor_desc * tensors,
                            int64_t n_tensors,
                            const ts_coreml_params * params,
                            const char * spec_output_path,
                            std::string * err_msg);

// Check if CoreML is available on this system (macOS only).
bool ts_coreml_available();

// Check whether the Xcode command-line tools (xcrun) are on PATH. Required for
// ts_coreml_compile. Always false off Apple.
bool ts_coreml_xcrun_available();

// Compile a .mlpackage into a .mlmodelc via `xcrun coremlcompiler compile`
// (design 1.6 / 4.5 step 8). Validates the compiled model has the expected
// structure. Returns 0 on success, -1 on error (err_msg set: missing tools,
// compilation failure, or malformed output).
int ts_coreml_compile(const char * mlpackage_path,
                      const char * output_dir,
                      std::string * err_msg);

// Default params.
ts_coreml_params ts_coreml_default_params();
