#include "llama-kv-cache.h"
#include "llama-model.h"

#include <numeric>

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

    llama_backend_free();
    return 0;
}
