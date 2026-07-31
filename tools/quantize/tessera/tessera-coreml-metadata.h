#pragma once

//
// tessera-coreml-metadata.h
//
// Reads Tessera metadata from GGUF and sidecar JSON (C10: GGUF primary,
// sidecar override, warn on disagreement). Produces a CoreMLConfig struct
// for the conversion pipeline.
//

#include <cstdint>
#include <string>
#include <vector>

struct ts_coreml_config {
    int64_t n_layers;
    int64_t hidden_dim;
    int64_t n_heads;
    int64_t n_kv_heads;
    int64_t vocab_size;
    int32_t quant_type;       // GGML_TYPE_TESSERA_T640 etc.
    bool    has_act_scale;
    float   calibration_mse;
    std::string calibration_policy;
    std::string model_name;
};

// Read config from GGUF metadata keys (tessera.calibration.*, etc.)
// Returns 0 on success. If the GGUF has no tessera metadata, returns 1
// (not an error, just absent).
int ts_coreml_config_from_gguf(const char * gguf_path,
                               ts_coreml_config * config,
                               std::string * err_msg);

// Read config from sidecar JSON (overrides GGUF values when present).
// Returns 0 on success, 1 if file not found (not an error).
int ts_coreml_config_from_sidecar(const char * sidecar_path,
                                  ts_coreml_config * config,
                                  std::string * err_msg);

// Merge: GGUF primary, sidecar override. Warns on disagreement by
// appending to warnings vector.
int ts_coreml_config_merge(const ts_coreml_config * gguf_config,
                           const ts_coreml_config * sidecar_config,
                           ts_coreml_config * merged,
                           std::vector<std::string> * warnings);
