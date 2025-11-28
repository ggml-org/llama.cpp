#pragma once

#include "llama.h"
#include "llama-safetensors.h"
#include "llama-hf-config.h"
#include "llama-safetensors-loader.h"

#include <string>
#include <memory>
#include <map>

// Forward declarations
struct llama_model;
struct llama_model_params;
struct ggml_context;

// Main entry point for loading a model from safetensors
// model_path can be either:
//   - Directory containing model.safetensors + config.json
//   - Path to a single .safetensors file (config.json must be in same dir)
llama_model * llama_model_load_from_safetensors(
    const char * model_path,
    const llama_model_params & params
);

// Internal implementation class
class safetensors_model_builder {
public:
    safetensors_model_builder(
        const std::string & model_dir,
        const llama_model_params & params
    );

    ~safetensors_model_builder();

    // Main loading pipeline
    llama_model * build();

    // Get last error message
    const std::string & get_error() const { return error_msg; }

private:
    std::string model_dir;
    llama_model_params params;
    std::string error_msg;

    // Components
    std::unique_ptr<hf_config> config;
    std::unique_ptr<safetensors_loader> st_loader;
    std::unique_ptr<safetensors_tensor_mapper> mapper;

    // Model being built
    llama_model * model = nullptr;

    // GGML contexts and backend buffers (one per buffer type for GPU offloading)
    struct ggml_context * ctx_meta = nullptr;  // Legacy: now unused, kept for compatibility
    struct ggml_context * ctx_data = nullptr;
    struct ggml_backend_buffer * backend_buffer = nullptr;  // Legacy: now unused

    // Multi-device support: map from buffer type to context and buffer
    std::map<ggml_backend_buffer_type_t, struct ggml_context *> ctx_map;
    std::map<ggml_backend_buffer_type_t, struct ggml_backend_buffer *> buffer_map;

    // Pipeline steps
    bool load_config();
    bool load_safetensors_files();
    bool detect_architecture();
    bool create_model_structure();
    bool init_devices();
    bool allocate_tensors();
    bool load_tensor_data();
    bool link_tensors_to_model();
    bool register_buffers_with_model();
    bool init_vocabulary();
    bool finalize_model();
};
