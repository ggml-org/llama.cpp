#include "llama-lora-training.h"

#include <cstring>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

ggml_context * llama_lora_create_context(size_t mem_size) {
    ggml_init_params init_params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(init_params);
}

bool llama_lora_validate_training_params(const llama_adapter_lora_train_params * params) {
    if (!params) {
        LLAMA_LOG_ERROR("%s: params is null\n", __func__);
        return false;
    }

    if (params->rank <= 0 || params->rank > 1024) {
        LLAMA_LOG_ERROR("%s: invalid rank %d, must be 1-1024\n", __func__, params->rank);
        return false;
    }

    if (params->alpha <= 0.0f) {
        LLAMA_LOG_ERROR("%s: invalid alpha %f, must be > 0\n", __func__, params->alpha);
        return false;
    }

    if (params->init_std <= 0.0f || params->init_std > 1.0f) {
        LLAMA_LOG_ERROR("%s: invalid init_std %f, must be in (0, 1]\n", __func__, params->init_std);
        return false;
    }

    if (params->target_modules == 0) {
        LLAMA_LOG_ERROR("%s: no target modules\n", __func__);
        return false;
    }

    return true;
}

void llama_lora_create_tensor_pair(
        ggml_context * lora_ctx,
        const char * base_name,
        const ggml_tensor * base_tensor,
        int32_t rank,
        ggml_tensor ** lora_a,
        ggml_tensor ** lora_b) {
    const int64_t d0 = base_tensor->ne[0]; // input dim
    const int64_t d1 = base_tensor->ne[1]; // output dim

    // LoRA A: [d0, rank] - projects input to low rank
    *lora_a = ggml_new_tensor_2d(lora_ctx, GGML_TYPE_F32, d0, rank);
    ggml_format_name(*lora_a, "%s.lora_a", base_name);

    // LoRA B: [rank, d1] - projects from low rank to output
    *lora_b = ggml_new_tensor_2d(lora_ctx, GGML_TYPE_F32, rank, d1);
    ggml_format_name(*lora_b, "%s.lora_b", base_name);
}

static bool tensor_module_is(const std::string & name, const char * module) {
    std::string needle = ".";
    needle += module;
    needle += ".";
    return name.find(needle) != std::string::npos;
}

static bool tensor_is_output(const std::string & name) {
    static const std::string prefix = "output.";
    return name.compare(0, prefix.size(), prefix) == 0;
}

static bool lora_should_target(const std::string & tensor_name, uint32_t target_modules) {
    const bool is_blk = tensor_name.compare(0, 4, "blk.") == 0;

    if (is_blk) {
        if ((target_modules & LLAMA_ADAPTER_LORA_TARGET_ATTN_Q)            && tensor_module_is(tensor_name, "attn_q"))           return true;
        if ((target_modules & LLAMA_ADAPTER_LORA_TARGET_ATTN_K)            && tensor_module_is(tensor_name, "attn_k"))           return true;
        if ((target_modules & LLAMA_ADAPTER_LORA_TARGET_ATTN_V)            && tensor_module_is(tensor_name, "attn_v"))           return true;
        if ((target_modules & LLAMA_ADAPTER_LORA_TARGET_ATTN_O)            && tensor_module_is(tensor_name, "attn_output"))      return true;
        if ((target_modules & LLAMA_ADAPTER_LORA_TARGET_FFN_GATE)          && tensor_module_is(tensor_name, "ffn_gate"))         return true;
        if ((target_modules & LLAMA_ADAPTER_LORA_TARGET_FFN_UP)            && tensor_module_is(tensor_name, "ffn_up"))           return true;
        if ((target_modules & LLAMA_ADAPTER_LORA_TARGET_FFN_DOWN)          && tensor_module_is(tensor_name, "ffn_down"))         return true;
    }

    if ((target_modules & LLAMA_ADAPTER_LORA_TARGET_OUTPUT) && tensor_is_output(tensor_name)) return true;

    return false;
}

static void init_tensor_gaussian(ggml_tensor * tensor, float std_dev, uint32_t seed) {
    const size_t n_elements = ggml_nelements(tensor);
    std::vector<float> data(n_elements);

    std::mt19937 gen;
    if (seed != 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::normal_distribution<float> dist(0.0f, std_dev);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dist(gen);
    }

    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

// A ~ N(0, init_std) and B = 0, so the adapter starts as a no-op
static void llama_lora_init_tensor_weights(ggml_tensor * lora_a, ggml_tensor * lora_b, float init_std, uint32_t seed) {
    init_tensor_gaussian(lora_a, init_std, seed);
    ggml_backend_tensor_memset(lora_b, 0, 0, ggml_nbytes(lora_b));
}

bool llama_lora_allocate_buffers(
        llama_adapter_lora * adapter,
        const std::vector<ggml_backend_buffer_type_t> & bufts) {
    if (!adapter || adapter->ctxs.empty() || adapter->ctxs.size() != bufts.size()) {
        LLAMA_LOG_ERROR("%s: adapter contexts and buffer types do not match\n", __func__);
        return false;
    }

    for (size_t i = 0; i < adapter->ctxs.size(); i++) {
        ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(adapter->ctxs[i].get(), bufts[i]) };
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate the %s buffer\n", __func__, ggml_backend_buft_name(bufts[i]));
            return false;
        }
        LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(bufts[i]), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
        adapter->bufs.emplace_back(std::move(buf));
    }

    return true;
}

// optimizer updates are in-place views of the param, so the scheduler cannot move them off its buffer
static bool llama_lora_buft_supports_optimizer(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return false;
    }

    ggml_init_params ip = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(ip) };
    if (!ctx) {
        return false;
    }

    // the optimizer type is chosen later in llama_opt_init, so require both
    ggml_tensor * param = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
    ggml_set_param(param);
    ggml_tensor * grad  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
    ggml_tensor * m     = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
    ggml_tensor * v     = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
    ggml_tensor * adamw = ggml_opt_step_adamw(ctx.get(), param, grad, m, v, ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 7));
    ggml_tensor * sgd   = ggml_opt_step_sgd(ctx.get(), param, grad, ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2));

    return ggml_backend_dev_supports_op(dev, adamw) && ggml_backend_dev_supports_op(dev, sgd);
}

llama_adapter_lora * llama_lora_create_adapter(
        llama_model * model,
        const llama_adapter_lora_train_params * params) {

    llama_adapter_lora * adapter = new llama_adapter_lora(model);
    try {
        adapter->alpha = params->alpha;

        // count the pairs first to size the no_alloc contexts
        size_t lora_pair_count = 0;
        for (const auto & tensor_pair : model->tensors_by_name) {
            const std::string & tensor_name = tensor_pair.first;
            ggml_tensor * base_tensor = tensor_pair.second;
            if (!base_tensor || base_tensor->ne[1] <= 0) continue;
            if (lora_should_target(tensor_name, params->target_modules)) {
                lora_pair_count++;
            }
        }

        if (lora_pair_count == 0) {
            throw std::runtime_error("No suitable tensors found for LoRA");
        }

        const size_t tensor_overhead = ggml_tensor_overhead();
        const size_t margin          = 64 * 1024;
        const size_t estimated_lora_mem = 2 * lora_pair_count * tensor_overhead + margin;

        std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
        std::map<ggml_backend_buffer_type_t, ggml_backend_buffer_type_t> buft_map;
        std::vector<ggml_backend_buffer_type_t> ctx_bufts;
        auto ctx_for_base = [&](const ggml_tensor * base) -> ggml_context * {
            ggml_backend_buffer_type_t base_buft = ggml_backend_buffer_get_type(base->buffer);
            auto bit = buft_map.find(base_buft);
            if (bit == buft_map.end()) {
                ggml_backend_buffer_type_t lora_buft = base_buft;
                if (ggml_backend_buft_is_host(base_buft)) {
                    lora_buft = ggml_backend_cpu_buffer_type();
                } else if (!llama_lora_buft_supports_optimizer(base_buft)) {
                    LLAMA_LOG_WARN("%s: %s does not support the optimizer step, LoRA factors stay on the CPU\n",
                                   __func__, ggml_backend_buft_name(base_buft));
                    lora_buft = ggml_backend_cpu_buffer_type();
                }
                bit = buft_map.emplace(base_buft, lora_buft).first;
            }
            ggml_backend_buffer_type_t buft = bit->second;
            auto it = ctx_map.find(buft);
            if (it != ctx_map.end()) {
                return it->second;
            }
            ggml_context * ctx = llama_lora_create_context(estimated_lora_mem);
            if (!ctx) {
                throw std::runtime_error("Failed to create LoRA context");
            }
            adapter->ctxs.emplace_back(ctx);
            ctx_bufts.push_back(buft);
            ctx_map[buft] = ctx;
            return ctx;
        };
        for (const auto & tensor_pair : model->tensors_by_name) {
            const std::string & tensor_name = tensor_pair.first;
            ggml_tensor * base_tensor = tensor_pair.second;

            if (!base_tensor || base_tensor->ne[1] <= 0) {
                continue;
            }

            if (!lora_should_target(tensor_name, params->target_modules)) {
                continue;
            }

            ggml_tensor * lora_a = nullptr;
            ggml_tensor * lora_b = nullptr;

            llama_lora_create_tensor_pair(ctx_for_base(base_tensor), tensor_name.c_str(), base_tensor, params->rank, &lora_a, &lora_b);
            adapter->ab_map[tensor_name] = llama_adapter_lora_weight(lora_a, lora_b);
        }

        if (!llama_lora_allocate_buffers(adapter, ctx_bufts)) {
            throw std::runtime_error("Failed to allocate LoRA buffers");
        }

        for (const auto & ab_pair : adapter->ab_map) {
            const llama_adapter_lora_weight & weight = ab_pair.second;

            llama_lora_init_tensor_weights(weight.a, weight.b, params->init_std, params->seed);
        }

        model->loras.insert(adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        delete adapter;
        return nullptr;
    }
}

llama_adapter_lora * llama_adapter_lora_init_train(
        llama_model * model,
        const llama_adapter_lora_train_params * params) {
    if (!model || !params) {
        LLAMA_LOG_ERROR("%s: invalid parameters\n", __func__);
        return nullptr;
    }

    if (!llama_lora_validate_training_params(params)) {
        return nullptr;
    }

    llama_adapter_lora * adapter = llama_lora_create_adapter(model, params);
    if (!adapter) {
        return nullptr;
    }

    LLAMA_LOG_INFO("%s: created a LoRA adapter with %zu tensor pairs\n", __func__, adapter->ab_map.size());

    return adapter;
}

bool llama_opt_param_filter_lora(const ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(userdata);
    return strstr(tensor->name, ".lora_a") != nullptr || strstr(tensor->name, ".lora_b") != nullptr;
}

bool llama_adapter_lora_save(
        const llama_adapter_lora * adapter,
        const char * filename) {
    if (!adapter || !filename) {
        LLAMA_LOG_ERROR("%s: invalid parameters\n", __func__);
        return false;
    }

    gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) {
        LLAMA_LOG_ERROR("%s: failed to create a gguf context\n", __func__);
        return false;
    }

    std::string arch_name = adapter->model->arch_name();
    if (arch_name.empty()) {
        LLAMA_LOG_ERROR("%s: failed to get the model architecture\n", __func__);
        gguf_free(gguf_ctx);
        return false;
    }

    gguf_set_val_str(gguf_ctx, "general.architecture", arch_name.c_str());
    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "general.name", "LoRA Adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");
    gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", adapter->alpha);

    int tensor_count = 0;
    for (const auto & kv : adapter->ab_map) {
        const auto & lora_weight = kv.second;

        if (lora_weight.a && lora_weight.b) {
            gguf_add_tensor(gguf_ctx, lora_weight.a);
            gguf_add_tensor(gguf_ctx, lora_weight.b);
            tensor_count += 2;
        }
    }

    bool success = gguf_write_to_file(gguf_ctx, filename, false);
    if (success) {
        LLAMA_LOG_INFO("%s: saved %d tensors to %s\n", __func__, tensor_count, filename);
    } else {
        LLAMA_LOG_ERROR("%s: failed to write %s\n", __func__, filename);
    }

    gguf_free(gguf_ctx);
    return success;
}
