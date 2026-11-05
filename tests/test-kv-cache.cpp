#include "llama-kv-cache.h"
#include "llama-model.h"

#include <numeric>
#include <string>

static void init_model(llama_model & model, uint32_t n_layer) {
    model.hparams = {};
    model.hparams.n_layer_all = n_layer;
    model.hparams.n_embd_head_k_full = 4;
    model.hparams.n_embd_head_v_full = 4;
    model.hparams.n_head_arr.fill(1);
    model.hparams.n_head_kv_arr.fill(1);
    std::iota(model.hparams.kv_source_impl.begin(), model.hparams.kv_source_impl.end(), 0);
    model.layers.resize(n_layer);
}

int main() {
    llama_backend_init();

    {
        std::string errors;
        llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
            if (level == GGML_LOG_LEVEL_ERROR) {
                *static_cast<std::string *>(user_data) += text;
            }
        }, &errors);

        for (llm_arch arch : { LLM_ARCH_DEEPSEEK2, LLM_ARCH_DEEPSEEK4, LLM_ARCH_DEEPSEEK41 }) {
            std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
            init_model(*model, 1);
            if (arch == LLM_ARCH_DEEPSEEK2) {
                model->hparams.n_embd_head_k_mla_impl = 4;
                model->hparams.n_embd_head_v_mla_impl = 4;
            }
            for (bool fp8_k : { false, true }) {
                auto params = llama_context_default_params();
                params.type_k = fp8_k ? GGML_TYPE_F8_E4M3 : GGML_TYPE_F16;
                params.type_v = fp8_k ? GGML_TYPE_F16 : GGML_TYPE_F8_E4M3;
                errors.clear();
                GGML_ASSERT(llama_init_from_model(model.get(), params) == nullptr);
                GGML_ASSERT(errors.find("model does not support different K") != std::string::npos);
            }
        }
        llama_log_set(nullptr, nullptr);
    }

    {
        std::unique_ptr<llama_model> target(llama_model_create(LLM_ARCH_GEMMA4, llama_model_default_params()));
        std::unique_ptr<llama_model> assistant(llama_model_create(LLM_ARCH_GEMMA4_ASSISTANT, llama_model_default_params()));
        init_model(*target, 3);
        init_model(*assistant, 2);

        ggml_context_ptr ctx(ggml_init({ 4096, nullptr, false }));
        target->layers[2].k_cache_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        target->layers[2].v_cache_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].k_cache_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].v_cache_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[1].k_cache_scale = assistant->layers[0].k_cache_scale;
        assistant->layers[1].v_cache_scale = assistant->layers[0].v_cache_scale;
        ggml_set_f32(target->layers[2].k_cache_scale, 0.5f);
        ggml_set_f32(target->layers[2].v_cache_scale, 0.25f);
        ggml_set_f32(assistant->layers[0].k_cache_scale, 2.0f);
        ggml_set_f32(assistant->layers[0].v_cache_scale, 4.0f);

        llama_kv_cache source(*target, false, target->hparams, GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E4M3,
                false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr,
                [](uint32_t il) { return il == 0 ? 2 : -1; }, nullptr);
        llama_kv_cache shared(*assistant, false, assistant->hparams, GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E4M3,
                false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, &source, nullptr, nullptr,
                [](int32_t il) { return il == 0 ? 2 : 1; });

        GGML_ASSERT(source.get_k_scale(2) == target->layers[2].k_cache_scale);
        GGML_ASSERT(source.get_v_scale(2) == target->layers[2].v_cache_scale);
        GGML_ASSERT(source.get_k_scale(0) == source.get_k_scale(2));
        GGML_ASSERT(source.get_v_scale(0) == source.get_v_scale(2));
        GGML_ASSERT(shared.get_k_storage(0) == source.get_k_storage(2));
        GGML_ASSERT(shared.get_k_scale(0) == source.get_k_scale(2));
        GGML_ASSERT(shared.get_v_scale(0) == source.get_v_scale(2));
        GGML_ASSERT(shared.get_k_scale(1) == nullptr);
        GGML_ASSERT(shared.get_v_scale(1) == nullptr);
    }

    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_DEEPSEEK2, llama_model_default_params()));
        init_model(*model, 1);
        model->hparams.n_embd_head_k_mla_impl = 2;
        model->hparams.n_embd_head_v_mla_impl = 2;

        ggml_context_ptr ctx(ggml_init({ 4096, nullptr, false }));
        model->layers[0].k_cache_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        model->layers[0].v_cache_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_f32(model->layers[0].k_cache_scale, 0.5f);
        ggml_set_f32(model->layers[0].v_cache_scale, 0.25f);

        for (bool scaled : { true, false }) {
            if (!scaled) {
                model->layers[0].k_cache_scale = nullptr;
            }
            llama_kv_cache cache(*model, false, model->hparams, GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E4M3,
                    false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);
            GGML_ASSERT(cache.get_k_scale(0) == model->layers[0].k_cache_scale);
            GGML_ASSERT(cache.get_v_scale(0) == cache.get_k_scale(0));
            GGML_ASSERT(cache.type_v() == cache.type_k());
        }
    }

    llama_backend_free();
    return 0;
}
