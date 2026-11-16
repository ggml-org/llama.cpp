#include "llama-kv-cache.h"
#include "llama-model.h"

#include <algorithm>
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
    for (int bit = 0; bit < 64; ++bit) {
        GGML_ASSERT(ctz64(uint64_t{1} << bit) == bit);
        GGML_ASSERT(ctz64(UINT64_MAX << bit) == bit);
    }

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

    // SWA cells shared by several sequences (e.g. after seq_cp in a unified cache) must be reused once they
    // are masked for all of their sequences, otherwise they stay occupied forever
    // ref: https://github.com/ggml-org/llama.cpp/issues/32258
    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
        init_model(*model, 1);

        const uint32_t kv_size = 8;
        const uint32_t n_swa   = 4;

        llama_kv_cache kv(*model, false, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
                false, false, true, kv_size, 2, n_swa, LLAMA_SWA_TYPE_STANDARD, nullptr, nullptr, nullptr, nullptr);
        GGML_ASSERT(kv.get_size() == kv_size);

        auto add = [&](llama_seq_id seq_id, llama_pos pos) {
            llama_token  token    = 0;
            int32_t      n_seq_id = 1;
            llama_seq_id seq_ids[1] = { seq_id };
            llama_seq_id * seq_id_ptr = seq_ids;
            int32_t      seq_idx[LLAMA_MAX_SEQ];
            int8_t       output   = 1;
            std::fill(seq_idx, seq_idx + LLAMA_MAX_SEQ, -1);
            seq_idx[seq_id] = 0;

            llama_ubatch ubatch = {};
            ubatch.b_equal_seqs = 1;
            ubatch.n_tokens     = 1;
            ubatch.n_seq_tokens = 1;
            ubatch.n_seqs       = 1;
            ubatch.n_seqs_unq   = 1;
            ubatch.n_pos        = 1;
            ubatch.token        = &token;
            ubatch.pos          = &pos;
            ubatch.n_seq_id     = &n_seq_id;
            ubatch.seq_id       = &seq_id_ptr;
            ubatch.seq_id_unq   = seq_ids;
            ubatch.seq_idx      = seq_idx;
            ubatch.output       = &output;

            const auto sinfo = kv.find_slot(ubatch, false);
            if (sinfo.empty()) {
                return false;
            }
            kv.apply_ubatch(sinfo, ubatch);
            return true;
        };

        // shared prefix [0, 1] of seq 0 and seq 1
        GGML_ASSERT(add(0, 0));
        GGML_ASSERT(add(0, 1));
        kv.seq_cp(0, 1, -1, -1);

        // fill the rest of the cache with tokens of each sequence that are still inside the window
        for (llama_pos pos = 2; pos < 5; ++pos) {
            GGML_ASSERT(add(0, pos));
            GGML_ASSERT(add(1, pos));
        }

        // the cache is full and only the shared cells are masked for both sequences
        for (llama_pos pos = 5; pos < 16; ++pos) {
            GGML_ASSERT(add(0, pos));
            GGML_ASSERT(add(1, pos));
            GGML_ASSERT(kv.seq_pos_max(0) == pos && kv.seq_pos_max(1) == pos);
            GGML_ASSERT(kv.seq_pos_min(0) == kv.seq_pos_min(1));
            GGML_ASSERT(kv.seq_pos_max(0) - kv.seq_pos_min(0) + 1 >= (llama_pos) n_swa);
        }
    }

    llama_backend_free();
    return 0;
}
