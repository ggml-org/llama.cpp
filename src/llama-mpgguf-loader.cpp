#include "llama-mpgguf-loader.h"
#include "llama-impl.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>

// Constructor for MPGGUF loader
// NOTE: For MPGGUF files, we need to parse the file structure ourselves
// because the parent llama_model_loader expects GGUF format with "GGUF" magic,
// but MPGGUF has "MPGGUF3" magic// MPGGUF Loader Constructor
// DOES NOT call parent constructor - reimplements initialization for MPGGUF format
// Copies logic from llama_model_loader but parses MPGGUF instead of GGUF
llama_mpgguf_loader::llama_mpgguf_loader(
    const std::string & fname,
    std::vector<std::string> & splits,
    bool use_mmap_param,
    bool check_tensors_param,
    const llama_model_kv_override * param_overrides_p,
    const llama_model_tensor_buft_override * param_tensor_buft_overrides_p,
    const std::vector<char> & activation_pattern)
    : llama_model_loader() // Call protected default constructor
{
    // Initialize parent members directly (from llama_model_loader constructor)
    this->use_mmap = use_mmap_param;
    this->check_tensors = check_tensors_param;
    this->tensor_buft_overrides = param_tensor_buft_overrides_p;
    
    // Set up KV overrides
    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            this->kv_overrides.insert({std::string(p->key), *p});
        }
    }
    
    // Step 1: Parse MPGGUF file structure
    LLAMA_LOG_INFO("%s: loading MPGGUF file from %s\n", __func__, fname.c_str());
    
    this->mpgguf = mpgguf_open(fname.c_str());
    if (!this->mpgguf) {
        throw std::runtime_error("Failed to open MPGGUF file: " + fname);
    }
    
    // Copy activation pattern to mpgguf structure
    if (!activation_pattern.empty()) {
        this->mpgguf->activation_pattern_size = activation_pattern.size();
        this->mpgguf->activation_pattern = (char*)malloc(activation_pattern.size());
        if (this->mpgguf->activation_pattern) {
            memcpy(this->mpgguf->activation_pattern, activation_pattern.data(), activation_pattern.size());
        }
    }

    // Save all MoE tensor names
    for (int i = 0; i < this->mpgguf->n_tensors; i++) {
        if (mpgguf_is_moe_expert_tensor(this->mpgguf->tensors[i].name))
            tensor_names.push_back(this->mpgguf->tensors[i].name);
    }
    
    // Build tensor map for fast lookup
    for (size_t i = 0; i < this->mpgguf->n_tensors; i++) {
        const struct mpgguf_tensor_rec * rec = &this->mpgguf->tensors[i];
        mpgguf_tensor_map[std::string(rec->name)] = rec;
    }
    
    LLAMA_LOG_INFO("%s: loaded MPGGUF with %zu tensors, %zu KB KV data\n", 
                   __func__, this->mpgguf->n_tensors, this->mpgguf->kv_size / 1024);
    LLAMA_LOG_INFO("%s: activation pattern size: %zu\n", __func__, activation_pattern.size());
    
    // Step 2: Initialize GGUF context from MPGGUF data using gguf_init_from_mpgguf
    LLAMA_LOG_INFO("%s: initializing GGUF context from MPGGUF embedded data\n", __func__);
    
    
    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };
    //
    this->meta.reset(gguf_init_from_mpgguf(this->mpgguf, params));

    if (!this->meta) {
        throw std::runtime_error(format("%s: failed to parse GGUF metadata from MPGGUF", __func__));
    }

    // Get architecture (copied from llama_model_loader)
    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), this->arch_name, false);
    this->llm_kv = LLM_KV(llm_arch_from_string(this->arch_name));
    
    // Open MPGGUF file for reading tensor data
    this->files.emplace_back(new llama_file(fname.c_str(), "rb"));
    this->contexts.emplace_back(ctx);
    
    // Build weights_map from tensors in the ggml context  
    // Each tensor gets its offset from the MPGGUF tensor directory
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string tensor_name = std::string(cur->name);
        
        // Check for duplicates
        if (this->weights_map.find(tensor_name) != this->weights_map.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", tensor_name.c_str()));
        }
        
        // Find this tensor in MPGGUF directory to get its offset
        auto it = mpgguf_tensor_map.find(tensor_name);
        if (it == mpgguf_tensor_map.end()) {
            throw std::runtime_error(format("tensor '%s' not found in MPGGUF directory", tensor_name.c_str()));
        }
        
        const struct mpgguf_tensor_rec * rec = it->second;

        bool b_high_precision = use_high_precision(tensor_name);
        if (!b_high_precision)
        {
            // let's set the type as low precision
           cur->type = static_cast<ggml_type>(rec->ggml_type_low);

           const size_t  type_size = ggml_type_size(cur->type);
           const int64_t blck_size = ggml_blck_size(cur->type);

           // calculate byte offsets given the tensor shape and type
           cur->nb[0] = type_size;
           cur->nb[1] = cur->nb[0] * (cur->ne[0] / blck_size);
           for (int j = 2; j < GGML_MAX_DIMS; ++j) {
               cur->nb[j] = cur->nb[j - 1] * cur->ne[j - 1];
           }
        }

        // Create tensor weight with high-precision offset by default
        // (will be swapped during load_data_for if low-precision is requested)
        this->n_elements += ggml_nelements(cur);
        this->n_bytes    += ggml_nbytes(cur);
        
        // Create tensor weight using constructor (like llama_model_loader does)
        // But we need to manually set the offset to MPGGUF location
        llama_tensor_weight weight(this->files.back().get(), 0, this->meta.get(), cur);
        
        // Override the offset to point to high-precision data in MPGGUF file
        weight.offs = this->mpgguf->data_offset + (b_high_precision ? rec->offset_high : rec->offset_low);
        
        this->weights_map.emplace(tensor_name, weight);
    }
    
    this->n_kv      = gguf_get_n_kv(this->meta.get());
    this->n_tensors = this->weights_map.size();
    this->fver      = (enum llama_fver) gguf_get_version(this->meta.get());
    
    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from MPGGUF\n",
                   __func__, this->n_kv, this->n_tensors);
    LLAMA_LOG_INFO("%s: will use low-precision for experts marked 'L' in activation pattern\n", __func__);
}

llama_mpgguf_loader::~llama_mpgguf_loader() {
    if (this->mpgguf) {
        mpgguf_close(this->mpgguf);
        this->mpgguf = nullptr;
    }
}

bool llama_mpgguf_loader::use_high_precision(const std::string & tensor_name) const {
    if (!mpgguf_is_moe_expert_tensor(tensor_name.c_str())) {
        // Non-MoE tensors always use high precision (or FP if that's all that's available)
        return true;
    }
    
    int expert_idx = mpgguf_get_expert_index(tensor_name);
    if (expert_idx < 0) {
        // Default to high precision if index is invalid
        return true;
    }

    expert_idx = expert_idx % this->mpgguf->activation_pattern_size;
    
    return this->mpgguf->activation_pattern[expert_idx] == 'H' || this->mpgguf->activation_pattern[expert_idx] == 'h';
}

int llama_mpgguf_loader::mpgguf_get_expert_index(const std::string & tensor_name) const {
    auto it = std::find_if(this->tensor_names.begin(), this->tensor_names.end(),
                 [&](const std::string & a) { return a == tensor_name; });

    if (it != this->tensor_names.end())
        return std::distance(this->tensor_names.begin(), it);

    return -1;
}

// Override load_data_for to handle MPGGUF precision selection
// Uses mmap through parent llama_model_loader by temporarily swapping offsets
void llama_mpgguf_loader::load_data_for(struct ggml_tensor * cur) const {
    const char * tensor_name = ggml_get_name(cur);
    
    // Check if this is a MoE expert tensor that needs low-precision
    if (mpgguf_is_moe_expert_tensor(tensor_name)) {
        int expert_idx = mpgguf_get_expert_index(tensor_name);
        
        if (expert_idx >= 0 && expert_idx < (int)this->mpgguf->activation_pattern_size) {
            char precision = this->mpgguf->activation_pattern[expert_idx];
            
            if (precision == 'L' || precision == 'l') {
                // Load low-precision version
                auto it = mpgguf_tensor_map.find(tensor_name);
                if (it != mpgguf_tensor_map.end()) {
                    const struct mpgguf_tensor_rec * rec = it->second;
                    
                    // Temporarily modify the weight offset to point to low-precision data
                    // Need to cast away const since we're in a const method but need to temporarily modify
                    std::map<std::string, llama_tensor_weight, weight_name_comparer> & weights_map_mut = 
                        const_cast<std::map<std::string, llama_tensor_weight, weight_name_comparer>&>(this->weights_map);
                    auto weight_it = weights_map_mut.find(tensor_name);
                    if (weight_it != weights_map_mut.end()) {
                        size_t original_offset = weight_it->second.offs;
                        
                        // Change to low-precision offset
                        weight_it->second.offs = this->mpgguf->data_offset + rec->offset_low;
                        
                        // Load using parent's load_data_for (with mmap support)
                        llama_model_loader::load_data_for(cur);
                        
                        // Restore original offset
                        weight_it->second.offs = original_offset;
                        
                        return;
                    }
                }
            }
        }
    }
    
    // Load high-precision (default path, uses mmap if enabled)
    llama_model_loader::load_data_for(cur);
}
