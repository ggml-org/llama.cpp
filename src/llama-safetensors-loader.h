#pragma once

#include "llama-safetensors.h"
#include "llama-hf-config.h"
#include "llama-arch.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// Forward declarations
struct llama_model;
struct llama_model_params;

// Maps HuggingFace architecture name to llama.cpp architecture
llm_arch hf_arch_to_llm_arch(const std::string & hf_arch_name);

// Tensor name mapper - converts HF tensor names to llama.cpp internal names
class safetensors_tensor_mapper {
public:
    virtual ~safetensors_tensor_mapper() = default;

    // Map HF tensor name to internal name
    // Returns empty string if tensor should be skipped
    virtual std::string map_tensor_name(const std::string & hf_name) const = 0;

    // Get the architecture this mapper handles
    virtual llm_arch get_arch() const = 0;

    // Get expected tensor names (for validation)
    virtual std::vector<std::string> get_required_tensors(int n_layers) const = 0;
};

// Llama/Mistral architecture mapper
class llama_tensor_mapper : public safetensors_tensor_mapper {
public:
    std::string map_tensor_name(const std::string & hf_name) const override;
    llm_arch get_arch() const override { return LLM_ARCH_LLAMA; }
    std::vector<std::string> get_required_tensors(int n_layers) const override;
};

// Phi architecture mapper
class phi_tensor_mapper : public safetensors_tensor_mapper {
public:
    std::string map_tensor_name(const std::string & hf_name) const override;
    llm_arch get_arch() const override { return LLM_ARCH_PHI3; }
    std::vector<std::string> get_required_tensors(int n_layers) const override;
};

// Qwen2 architecture mapper
class qwen2_tensor_mapper : public safetensors_tensor_mapper {
public:
    std::string map_tensor_name(const std::string & hf_name) const override;
    llm_arch get_arch() const override { return LLM_ARCH_QWEN2; }
    std::vector<std::string> get_required_tensors(int n_layers) const override;
};

// Gemma architecture mapper
class gemma_tensor_mapper : public safetensors_tensor_mapper {
public:
    std::string map_tensor_name(const std::string & hf_name) const override;
    llm_arch get_arch() const override { return LLM_ARCH_GEMMA; }
    std::vector<std::string> get_required_tensors(int n_layers) const override;
};

// Factory function to create appropriate mapper
std::unique_ptr<safetensors_tensor_mapper> create_tensor_mapper(const std::string & hf_arch);

// Main safetensors model loader
class safetensors_model_loader {
public:
    safetensors_model_loader() = default;
    ~safetensors_model_loader() = default;

    // Load model from safetensors file(s)
    // Returns nullptr on error (check get_error())
    llama_model * load(
        const std::string & model_dir,
        const llama_model_params & params
    );

    // Get last error message
    const std::string & get_error() const { return error_msg; }

private:
    std::string error_msg;

    hf_config config;
    std::unique_ptr<safetensors_loader> st_loader;
    std::unique_ptr<safetensors_tensor_mapper> mapper;

    bool load_config(const std::string & model_dir);
    bool load_safetensors_files(const std::string & model_dir);
    bool create_mapper();
};
