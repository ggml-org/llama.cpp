#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <numeric>
#include <string>
#include <vector>

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

// count find_slot position warnings for one ubatch of seq 0, with pos0 as the first position section
static int find_slot_pos_warns(llama_memory_recurrent & mem, uint32_t n_pos, const std::vector<llama_pos> & pos0) {
    const uint32_t n_tokens = pos0.size();
    std::vector<llama_pos> pos(n_tokens*n_pos, 0);
    std::copy(pos0.begin(), pos0.end(), pos.begin());
    llama_seq_id seq_id = 0;
    std::vector<int32_t> n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id *> seq_ids(n_tokens, &seq_id);

    llama_ubatch ubatch = {};
    ubatch.b_equal_seqs = 1;
    ubatch.n_tokens     = n_tokens;
    ubatch.n_seq_tokens = n_tokens;
    ubatch.n_seqs       = 1;
    ubatch.n_seqs_unq   = 1;
    ubatch.n_pos        = n_pos;
    ubatch.pos          = pos.data();
    ubatch.n_seq_id     = n_seq_id.data();
    ubatch.seq_id       = seq_ids.data();

    int n_warn = 0;
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        if (level == GGML_LOG_LEVEL_WARN && std::string(text).find("non-consecutive token position") != std::string::npos) {
            ++*static_cast<int *>(user_data);
        }
    }, &n_warn);
    GGML_ASSERT(mem.find_slot(ubatch));
    llama_log_set(nullptr, nullptr);
    return n_warn;
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

        for (llm_arch arch : { LLM_ARCH_DEEPSEEK2, LLM_ARCH_DEEPSEEK32, LLM_ARCH_DEEPSEEK4,
                LLM_ARCH_DEEPSEEK41, LLM_ARCH_GLM_DSA, LLM_ARCH_GLM5_NEXT, LLM_ARCH_DOTS3NOTE,
                LLM_ARCH_MISTRAL_LARGE4, LLM_ARCH_HY_V4, LLM_ARCH_MINIMAX_M3, LLM_ARCH_QWEN4EXP,
                LLM_ARCH_DFLASH }) {
            std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
            init_model(*model, 1);
            if (arch == LLM_ARCH_DEEPSEEK2) {
                model->hparams.n_embd_head_k_mla_impl = 4;
                model->hparams.n_embd_head_v_mla_impl = 4;
            }

            for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                    if (type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16) {
                        continue;
                    }
                    auto params = llama_context_default_params();
                    params.type_k = type_k;
                    params.type_v = type_v;
                    errors.clear();
                    llama_context * ctx = llama_init_from_model(model.get(), params);
                    GGML_ASSERT(ctx == nullptr);
                    if (type_k == type_v || (!model->hparams.is_mla() && arch != LLM_ARCH_DEEPSEEK4 && arch != LLM_ARCH_DEEPSEEK41)) {
                        GGML_ASSERT(errors.find("FP8 cache is not supported for architecture") != std::string::npos);
                    } else {
                        GGML_ASSERT(errors.find("model does not support different K") != std::string::npos);
                    }
                }
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
        target->layers[2].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        target->layers[2].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[1].k_cache_s = assistant->layers[0].k_cache_s;
        assistant->layers[1].v_cache_s = assistant->layers[0].v_cache_s;
        *static_cast<float *>(target->layers[2].k_cache_s->data) = 0.5f;
        *static_cast<float *>(target->layers[2].v_cache_s->data) = 0.25f;
        *static_cast<float *>(assistant->layers[0].k_cache_s->data) = 2.0f;
        *static_cast<float *>(assistant->layers[0].v_cache_s->data) = 4.0f;

        for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
            for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                llama_kv_cache source(*target, false, target->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr,
                        [](uint32_t il) { return il == 0 ? 2 : -1; }, nullptr);
                llama_kv_cache shared(*assistant, false, assistant->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, &source, nullptr, nullptr,
                        [](int32_t il) { return il == 0 ? 2 : 1; });

                GGML_ASSERT(source.get_k_scale(2) == (type_k == GGML_TYPE_F8_E4M3 ? target->layers[2].k_cache_s : nullptr));
                GGML_ASSERT(source.get_v_scale(2) == (type_v == GGML_TYPE_F8_E4M3 ? target->layers[2].v_cache_s : nullptr));
                GGML_ASSERT(source.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(source.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_storage(0) == source.get_k_storage(2));
                GGML_ASSERT(shared.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(shared.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_scale(1) == nullptr);
                GGML_ASSERT(shared.get_v_scale(1) == nullptr);
            }
        }
    }

    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35, llama_model_default_params()));
        init_model(*model, 1);
        auto make_mem = [&]() {
            return std::make_unique<llama_memory_recurrent>(*model, true, GGML_TYPE_F32, GGML_TYPE_F32, false, 1, 1, 0, [](int32_t) { return false; });
        };

        // M-RoPE: image rows share one position, the text after the image jumps forward
        auto mem = make_mem();
        GGML_ASSERT(find_slot_pos_warns(*mem, 4, { 0, 1, 2, 3 }) == 0);
        GGML_ASSERT(find_slot_pos_warns(*mem, 4, { 4, 4, 4, 4 }) == 0);
        GGML_ASSERT(find_slot_pos_warns(*mem, 4, { 4, 4 }) == 0);
        GGML_ASSERT(find_slot_pos_warns(*mem, 4, { 7, 8 }) == 0);
        GGML_ASSERT(find_slot_pos_warns(*mem, 4, { 2, 3 }) == 1);

        // single position: a skip is still reported
        mem = make_mem();
        GGML_ASSERT(find_slot_pos_warns(*mem, 1, { 0, 1, 2, 3 }) == 0);
        GGML_ASSERT(find_slot_pos_warns(*mem, 1, { 4, 5 }) == 0);
        GGML_ASSERT(find_slot_pos_warns(*mem, 1, { 7, 8 }) == 1);
    }

    llama_backend_free();
    return 0;
}
