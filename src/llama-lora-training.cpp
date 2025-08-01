#include "llama-lora-training.h"

#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>


ggml_context * llama_lora_create_context(size_t mem_size) {
    struct ggml_init_params init_params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(init_params);
}

bool llama_lora_create_tensor_pair(
        struct ggml_context * lora_ctx,
        const char * base_name,
        const struct ggml_tensor * base_tensor,
        int32_t rank,
        struct ggml_tensor ** lora_a,
        struct ggml_tensor ** lora_b) {
    
    if (!lora_ctx || !base_name || !base_tensor || !lora_a || !lora_b) {
        return false;
    }
    
    const int64_t d0 = base_tensor->ne[0]; // input dim
    const int64_t d1 = base_tensor->ne[1]; // output dim
    
    char lora_a_name[256], lora_b_name[256];
    snprintf(lora_a_name, sizeof(lora_a_name), "%s.lora_a", base_name);
    snprintf(lora_b_name, sizeof(lora_b_name), "%s.lora_b", base_name);
    
    *lora_a = ggml_new_tensor_2d(lora_ctx, GGML_TYPE_F32, d0, rank);
    ggml_set_name(*lora_a, lora_a_name);
    
    *lora_b = ggml_new_tensor_2d(lora_ctx, GGML_TYPE_F32, rank, d1);
    ggml_set_name(*lora_b, lora_b_name);
    
    return true;
}

void llama_lora_init_tensor_weights(struct ggml_tensor * lora_a, struct ggml_tensor * lora_b, float init_std) {
    if (!lora_a || !lora_b) return;
    
    // Initialize A with Gaussian distribution
    const size_t a_elements = ggml_nelements(lora_a);
    float * a_data = (float *)lora_a->data;
    if (a_data) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, init_std);
        for (size_t i = 0; i < a_elements; i++) {
            a_data[i] = dist(gen);
        }
    }
    
    // Initialize B with zeros
    const size_t b_elements = ggml_nelements(lora_b);
    float * b_data = (float *)lora_b->data;
    if (b_data) {
        std::fill_n(b_data, b_elements, 0.0f);
    }
}

struct llama_adapter_lora * llama_lora_create_adapter(
        struct llama_model * model, 
        const struct llama_lora_training_params * params) {

    // Create a new LoRA adapter instance
    llama_adapter_lora * adapter = new llama_adapter_lora();
    adapter->alpha = params->alpha;
    
    // Create GGML context for LoRA tensors
    const size_t estimated_lora_mem = 256 * 1024 * 1024; // 256MB should be enough for most LoRA configs
    ggml_context * lora_ctx = llama_lora_create_context(estimated_lora_mem);
    if (!lora_ctx) {
        LLAMA_LOG_ERROR("Failed to create LoRA context\n");
        delete adapter;
        return nullptr;
    }
    
    adapter->ctxs.emplace_back(lora_ctx);
    
    int created_count = 0;
    
    for (const auto & tensor_pair : model->tensors_by_name) {
        const std::string & tensor_name = tensor_pair.first;
        struct ggml_tensor * base_tensor = tensor_pair.second;
        
        bool should_create_lora = false;
        
        // Apply LoRA to all layers (blk.0) that match target modules only for the first layer
        if (tensor_name.find("blk.0.") != std::string::npos) {
            if ((params->target_modules & LLAMA_LORA_TARGET_ATTN_Q) && tensor_name.find("attn_q") != std::string::npos) {
                should_create_lora = true;
            } else if ((params->target_modules & LLAMA_LORA_TARGET_ATTN_V) && tensor_name.find("attn_v") != std::string::npos) {
                should_create_lora = true;
            }
        }
        
        if (should_create_lora && base_tensor->ne[1] > 0) {
            struct ggml_tensor * lora_a = nullptr;
            struct ggml_tensor * lora_b = nullptr;
            
            if (llama_lora_create_tensor_pair(lora_ctx, tensor_name.c_str(), base_tensor, params->rank, &lora_a, &lora_b)) {
                created_count++;
                
                llama_lora_init_tensor_weights(lora_a, lora_b, params->init_std);
                adapter->ab_map[tensor_name] = llama_adapter_lora_weight(lora_a, lora_b);
                
            } else {
                delete adapter;
                return nullptr;
            }
        }
    }
    
    if (created_count == 0) {
        delete adapter;
        return nullptr;
    }
    
    LLAMA_LOG_INFO("Created LoRA adapter with %d tensor pairs\n", created_count);
    return adapter;
}

bool llama_lora_allocate_buffers(
        struct llama_adapter_lora * adapter, 
        struct llama_model * model) {
        
    if (!adapter || !model) {
        return false;
    }
    
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type(); // fallback to CPU
    
    for (const auto & tensor_pair : model->tensors_by_name) {
        const std::string & name = tensor_pair.first;
        struct ggml_tensor * tensor = tensor_pair.second;
        

        if (name.find("blk.0.") != std::string::npos && tensor && tensor->buffer) {
            buft = ggml_backend_buffer_get_type(tensor->buffer);
            break;
        }
    }
    
    if (adapter->ctxs.empty()) {
        return false;
    }
    ggml_context * lora_ctx = adapter->ctxs[0].get();
    
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(lora_ctx, buft) };
    if (!buf) {
        return false;
    }
    adapter->bufs.emplace_back(std::move(buf));
    
    return true;
}

bool llama_lora_register_adapter(
        struct llama_context * ctx,
        struct llama_adapter_lora * adapter) {
        
    if (!ctx || !adapter) {
        return false;
    }
    
    if (llama_set_adapter_lora(ctx, reinterpret_cast<struct llama_adapter_lora *>(adapter), 1.0f) < 0) {
        LLAMA_LOG_ERROR("Failed to register LoRA adapter with context\n");
        return false;
    }
    
    
    return true;
}

bool llama_lora_training_init(
        struct llama_context * ctx,
        struct llama_model * model,
        const struct llama_lora_training_params * params) {
    
    if (!ctx || !model || !params) {
        LLAMA_LOG_ERROR("LoRA training init: invalid parameters\n");
        return false;
    }

    LLAMA_LOG_INFO("LoRA training parameters validated successfully\n");
    
    // For now, always create new LoRA adapters
    // TODO: Add a method to check existing adapters
    bool has_existing_lora = false;
    
    if (has_existing_lora) {
        LLAMA_LOG_INFO("Found existing LoRA adapter(s) - will train those\n");
        return true;
    }
    
    struct llama_adapter_lora * adapter = llama_lora_create_adapter(model, params);
    if (!adapter) {
        return false;
    }
    
    if (!llama_lora_allocate_buffers(adapter, model)) {
        delete adapter;
        return false;
    }
    
    if (!llama_lora_register_adapter(ctx, adapter)) {
        delete adapter;
        return false;
    }
    
    LLAMA_LOG_INFO("Successfully created and registered LoRA adapter\n");
    
    return true;
}


bool llama_opt_param_filter_lora(const struct ggml_tensor * tensor, void * userdata) {
    (void) userdata; // Unused param

    if (!tensor || !tensor->name) {
        return false;
    }

    const char * name = tensor->name;
    
    // Check if tensor is LoRA A or B
    // LoRA tensor naming convention: blk.{layer}.{module}.lora_a or .lora_b
    if (strstr(name, ".lora_a") || strstr(name, ".lora_b")) {
        LLAMA_LOG_DEBUG("LoRA filter: including trainable params '%s'\n", name);
        return true;
    }

    return false;
}
