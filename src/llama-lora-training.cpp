#include "llama-lora-training.h"

#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <stdexcept>


ggml_context * llama_lora_create_context(size_t mem_size) {
    struct ggml_init_params init_params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(init_params);
}

bool llama_lora_validate_training_params(const struct llama_lora_training_params * params) {
    if (!params) {
        LLAMA_LOG_ERROR("LoRA training validation: params is null\n");
        return false;
    }
    
    if (params->rank <= 0 || params->rank > 1024) {
        LLAMA_LOG_ERROR("LoRA training validation: invalid rank %d (must be 1-1024)\n", params->rank);
        return false;
    }
    
    if (params->alpha <= 0.0f) {
        LLAMA_LOG_ERROR("LoRA training validation: invalid alpha %f (must be > 0)\n", params->alpha);
        return false;
    }
    
    if (params->dropout < 0.0f || params->dropout > 1.0f) {
        LLAMA_LOG_ERROR("LoRA training validation: invalid dropout %f (must be [0, 1])\n", params->dropout);
        return false;
    }
    
    if (params->init_std <= 0.0f || params->init_std > 1.0f) {
        LLAMA_LOG_ERROR("LoRA training validation: invalid init_std %f (must be (0, 1])\n", params->init_std);
        return false;
    }
    
    if (params->target_modules == 0) {
        LLAMA_LOG_ERROR("LoRA training validation: no target modules specified\n");
        return false;
    }
    
    return true;
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
    
    // Get base tensor dim
    const int64_t d0 = base_tensor->ne[0]; // input dim
    const int64_t d1 = base_tensor->ne[1]; // output dim
    
    char lora_a_name[256], lora_b_name[256];
    snprintf(lora_a_name, sizeof(lora_a_name), "%s.lora_a", base_name);
    snprintf(lora_b_name, sizeof(lora_b_name), "%s.lora_b", base_name);
    
    // LoRA A: [d0, rank] - projects input to low rank
    *lora_a = ggml_new_tensor_2d(lora_ctx, GGML_TYPE_F32, d0, rank);
    ggml_set_name(*lora_a, lora_a_name);
    
    // LoRA B: [rank, d1] - projects from low rank to output
    *lora_b = ggml_new_tensor_2d(lora_ctx, GGML_TYPE_F32, rank, d1);
    ggml_set_name(*lora_b, lora_b_name);
    
    return true;
}

static bool is_tensor_on_device(const struct ggml_tensor * tensor) {
    return tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer);
}

static void init_tensor_guassian(struct ggml_tensor * tensor, float std_dev) {
    const size_t n_elements = ggml_nelements(tensor);
    std::vector<float> data(n_elements);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dist(gen);
    }

    if (is_tensor_on_device(tensor)) {
        ggml_backend_tensor_set(tensor, data.data(), 0, n_elements * sizeof(float));
    } else {
        std::copy(data.begin(), data.end(), (float *)tensor->data);
    }
}

static void init_tensor_zeros(struct ggml_tensor * tensor) {
    const size_t n_elements = ggml_nelements(tensor);

    if (is_tensor_on_device(tensor)) {
        std::vector<float> zeros(n_elements, 0.0f);
        ggml_backend_tensor_set(tensor, zeros.data(), 0, n_elements * sizeof(float));
    } else {
        std::fill_n((float *)tensor->data, n_elements, 0.0f);
    }
}

void llama_lora_init_tensor_weights(struct ggml_tensor * lora_a, struct ggml_tensor * lora_b, float init_std) {
    if (!lora_a || !lora_b) return;
    
    // LoRA initialization: A ~ N(0, init_std), B = 0
    init_tensor_guassian(lora_a, init_std);
    init_tensor_zeros(lora_b);
}

bool llama_lora_allocate_buffers(
        struct llama_adapter_lora * adapter, 
        struct llama_model * model) {
        
    if (!adapter || !model) {
        return false;
    }
    
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type(); // fallback to CPU
    
    // Find any layer tensor to determine the correct backend  
    for (const auto & tensor_pair : model->tensors_by_name) {
        const std::string & name = tensor_pair.first;
        struct ggml_tensor * tensor = tensor_pair.second;
        
        if (name.find("blk.") != std::string::npos && tensor && tensor->buffer) {
            buft = ggml_backend_buffer_get_type(tensor->buffer);
            break;
        }
    }
    
    if (adapter->ctxs.empty()) {
        LLAMA_LOG_ERROR("No contexts found in adapter\n");
        return false;
    }
    ggml_context * lora_ctx = adapter->ctxs[0].get();

    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(lora_ctx, buft) };
    if (!buf) {
        LLAMA_LOG_ERROR("Failed to allocate buffer for LoRA adapter\n");
        return false;
    }
    LLAMA_LOG_INFO("LoRA buffer size = %.2f MiB\n", ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
    adapter->bufs.emplace_back(std::move(buf));
    
    return true;
}

struct llama_adapter_lora * llama_lora_create_adapter(
        struct llama_model * model, 
        const struct llama_lora_training_params * params) {

    // Create a new LoRA adapter instance
    llama_adapter_lora * adapter = new llama_adapter_lora();
    try {
        adapter->alpha = params->alpha;

        // Create LoRA tensors and populate ab_map
        // Create GGML context for LoRA tensors
        const size_t estimated_lora_mem = 256 * 1024 * 1024; // 256MB should be enough for most LoRA configs
        ggml_context * lora_ctx = llama_lora_create_context(estimated_lora_mem);
        if (!lora_ctx) {
            throw std::runtime_error("Failed to create LoRA context");
        }
        
        adapter->ctxs.emplace_back(lora_ctx);
        int created_count = 0;

        for (const auto & tensor_pair : model->tensors_by_name) {
            const std::string & tensor_name = tensor_pair.first;
            struct ggml_tensor * base_tensor = tensor_pair.second;

            if (!base_tensor) {
                continue;
            }

            bool should_create_lora = false;
            if (tensor_name.find("blk.") != std::string::npos) {
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
                    if (!lora_a || !lora_b) {
                        throw std::runtime_error("Created null LoRA tensors for " + tensor_name);
                    }                    
                    created_count++;
                    adapter->ab_map[tensor_name] = llama_adapter_lora_weight(lora_a, lora_b);
                } else {
                    throw std::runtime_error("Failed to create LoRA tensor pair for " + tensor_name);
                }
            }
        }

        if (created_count == 0) {
            throw std::runtime_error("No suitable tensors found for LoRA adaptation");
        }

        if (!llama_lora_allocate_buffers(adapter, model)) {
            throw std::runtime_error("Failed to allocate LoRA buffers");
        }

        for (const auto & ab_pair : adapter->ab_map) {
            const std::string & tensor_name = ab_pair.first;
            const llama_adapter_lora_weight & weight = ab_pair.second;

            if (weight.a && weight.b && weight.a->data && weight.b->data) {
                llama_lora_init_tensor_weights(weight.a, weight.b, params->init_std);
            } else {
                throw std::runtime_error("LoRA tensor initialization failed for " + tensor_name);
            }
        }
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("Failed to create LoRA adapter: %s\n", err.what());
        delete adapter;
        return nullptr;
    }
}

bool llama_lora_training_init(
        struct llama_context * ctx,
        struct llama_model * model,
        const struct llama_lora_training_params * params) {
    
    if (!ctx || !model || !params) {
        LLAMA_LOG_ERROR("LoRA training init: invalid parameters\n");
        return false;
    }

    if (!llama_lora_validate_training_params(params)) {
        return false;
    }
    
    struct llama_adapter_lora * adapter = llama_lora_create_adapter(model, params);
    if (!adapter) {
        return false;
    }

    llama_clear_adapter_lora(ctx);
    
    if (llama_set_adapter_lora(ctx, adapter, 1.0f) < 0) {
        LLAMA_LOG_ERROR("Failed to apply LoRA adapter to context\n");
        delete adapter;
        return false;
    }
    
    LLAMA_LOG_INFO("LoRA adapter contains %zu tensor pairs and is now registered with context\n", adapter->ab_map.size());

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
