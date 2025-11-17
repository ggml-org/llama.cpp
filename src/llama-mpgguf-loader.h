#pragma once

#include "llama-model-loader.h"
#include "gguf.h"
#include <vector>
#include <string>
#include <map>
#include <memory>

// MPGGUF v3 loader - loads mixed-precision GGUF files with high/low precision MoE weights
// Integrates with the existing llama_model_loader infrastructure

// MPGGUF loader - inherits from llama_model_loader
// Does NOT call parent constructor - reimplements initialization for MPGGUF format
struct llama_mpgguf_loader : public llama_model_loader {
    // MPGGUF specific data
    struct mpgguf_file * mpgguf;
    
    // Cache to map tensor names to their mpgguf records
    std::map<std::string, const struct mpgguf_tensor_rec *> mpgguf_tensor_map;

    std::vector<std::string> tensor_names;
    
    // Constructor: parses MPGGUF file and initializes like llama_model_loader
    // but WITHOUT calling parent constructor
    llama_mpgguf_loader(
        const std::string & fname,
        std::vector<std::string> & splits,
        bool use_mmap,
        bool check_tensors,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p,
        const std::vector<char> & activation_pattern);
    
    // Destructor to clean up mpgguf resources
    ~llama_mpgguf_loader();
    
    // Override load_data_for to handle MPGGUF precision selection with mmap support
    void load_data_for(struct ggml_tensor * cur) const override;
    
private:
    // Determine which precision to use for a given MoE expert tensor
    bool use_high_precision(const std::string & tensor_name) const;
    int  mpgguf_get_expert_index(const std::string & tensor_name) const;
};
