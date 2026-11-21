#include "llama-kv-cache.h"
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

            for (bool swa : { false, true }) {
                model->hparams.is_swa_impl[0] = swa;
                for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                    for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                        if (type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16) {
                            continue;
                        }
                        auto params = llama_context_default_params();
                        params.type_k     = swa ? GGML_TYPE_F16 : type_k;
                        params.type_v     = swa ? GGML_TYPE_F16 : type_v;
                        params.type_k_swa = swa ? type_k : GGML_TYPE_COUNT;
                        params.type_v_swa = swa ? type_v : GGML_TYPE_COUNT;
                        errors.clear();
                        llama_context * ctx = llama_init_from_model(model.get(), params);
                        GGML_ASSERT(ctx == nullptr);
                        if (type_k == type_v || (!model->hparams.is_mla() && arch != LLM_ARCH_DEEPSEEK4 && arch != LLM_ARCH_DEEPSEEK41)) {
                            GGML_ASSERT(errors.find("FP8 cache is not supported for architecture") != std::string::npos);
                        } else {
                            GGML_ASSERT(errors.find(swa ? "model does not support different SWA K" : "model does not support different K") != std::string::npos);
                        }
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

    // seq_rm removes [p0, p1) of one sequence, a negative bound is open
    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
        init_model(*model, 1);

        llama_kv_cache kv(*model, false, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
                false, false, true, 32, 2, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);

        // returns the first cell that takes the tokens
        auto add = [&](llama_seq_id seq_id, llama_pos p0, uint32_t n) {
            std::vector<llama_token>    token(n, 1);
            std::vector<llama_pos>      pos(n);
            std::vector<int32_t>        n_seq_id(n, 1);
            std::vector<llama_seq_id *> seq_ids(n, &seq_id);
            std::iota(pos.begin(), pos.end(), p0);

            llama_ubatch ubatch = {};
            ubatch.n_tokens     = n;
            ubatch.n_seq_tokens = n;
            ubatch.n_seqs       = 1;
            ubatch.n_seqs_unq   = 1;
            ubatch.n_pos        = 1;
            ubatch.token        = token.data();
            ubatch.pos          = pos.data();
            ubatch.n_seq_id     = n_seq_id.data();
            ubatch.seq_id       = seq_ids.data();
            ubatch.seq_id_unq   = &seq_id;

            const auto sinfo = kv.find_slot(ubatch, false);
            GGML_ASSERT(!sinfo.empty());
            kv.apply_ubatch(sinfo, ubatch);

            return sinfo.head();
        };

        GGML_ASSERT(add(0, 0, 8) == 0);
        GGML_ASSERT(add(1, 0, 4) == 8);

        // the tail of sequence 0, the other sequence keeps its cells
        GGML_ASSERT(kv.seq_rm(0, 5, -1));
        GGML_ASSERT(kv.seq_pos_min(0) == 0 && kv.seq_pos_max(0) == 4);
        GGML_ASSERT(kv.seq_pos_min(1) == 0 && kv.seq_pos_max(1) == 3);

        // the next tokens go to the lowest freed cell
        GGML_ASSERT(add(0, 5, 2) == 5);

        // p1 is not part of the range
        GGML_ASSERT(kv.seq_rm(0, -1, 2));
        GGML_ASSERT(kv.seq_pos_min(0) == 2 && kv.seq_pos_max(0) == 6);
        GGML_ASSERT(kv.seq_rm(0, 6, 6));
        GGML_ASSERT(kv.seq_pos_max(0) == 6);

        // freed cells are taken in cell order, so a higher position can sit in a lower cell
        GGML_ASSERT(add(0, 7, 3) == 0);
        GGML_ASSERT(kv.seq_rm(0, 6, -1));
        GGML_ASSERT(add(0, 6, 1) == 0);

        // a cell shared with another sequence stays for that sequence
        kv.seq_cp(0, 1, 5, -1);
        GGML_ASSERT(kv.seq_rm(0, 5, -1));
        GGML_ASSERT(kv.seq_pos_max(0) == 4 && kv.seq_pos_max(1) == 6);

        // every cell of a repeated position is removed
        add(0, 5, 1);
        add(0, 5, 1);
        GGML_ASSERT(kv.seq_rm(0, 5, 6));
        GGML_ASSERT(kv.seq_pos_max(0) == 4);
    }

    llama_backend_free();
    return 0;
}
