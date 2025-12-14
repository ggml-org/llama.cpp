#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "chat.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // EAGLE3 specific contexts
    llama_context * ctx_encoder = NULL;
    llama_context * ctx_decoder = NULL;

    // For EAGLE3: load both draft model and target model
    if (params.speculative.eagle3) {
        llama_model_params dft_mp = llama_model_default_params();
        dft_mp.n_gpu_layers = params.speculative.n_gpu_layers;
        model_dft = llama_model_load_from_file(params.speculative.model.path.c_str(), dft_mp);
        if (!model_dft) {
            LOG_ERR("failed to load EAGLE3 draft model\n");
            return 1;
        }

        llama_model_params tgt_mp = llama_model_default_params();
        tgt_mp.n_gpu_layers = params.n_gpu_layers;
        model_tgt = llama_model_load_from_file(params.model.path.c_str(), tgt_mp);
        if (!model_tgt) {
            LOG_ERR("failed to load target model\n");
            return 1;
        }

        llama_context_params tcp = common_context_params_to_llama(params);
        tcp.eagle3_model = model_dft;  // Enable feature extraction
        ctx_tgt = llama_init_from_model(model_tgt, tcp);
    } else {
        // Standard load the target model
        auto llama_init_tgt = common_init_from_params(params);
        model_tgt = llama_init_tgt->model();
        ctx_tgt   = llama_init_tgt->context();
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    params.devices      = params.speculative.devices;
    params.model        = params.speculative.model;
    params.n_ctx        = params.speculative.n_ctx;
    params.n_batch      = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
    params.n_gpu_layers = params.speculative.n_gpu_layers;

    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    params.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;

    if (params.speculative.eagle3) {
        // EAGLE3: create encoder and decoder contexts
        llama_context_params enc_params = common_context_params_to_llama(params);
        enc_params.embeddings = true;
        ctx_encoder = llama_init_from_model(model_dft, enc_params);
        if (!ctx_encoder) {
            LOG_ERR("failed to create EAGLE3 encoder context\n");
            return 1;
        }

        llama_context_params dec_params = common_context_params_to_llama(params);
        dec_params.target_model = model_tgt;
        dec_params.embeddings = true;
        ctx_decoder = llama_init_from_model(model_dft, dec_params);
        if (!ctx_decoder) {
            LOG_ERR("failed to create EAGLE3 decoder context\n");
            return 1;
        }
    } else {
        // Standard: load draft model context
        auto llama_init_dft = common_init_from_params(params);
        model_dft = llama_init_dft->model();
        ctx_dft   = llama_init_dft->context();

        if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
            LOG_INF("the draft model '%s' is not compatible with the target model '%s'. tokens will be translated between the draft and target models.\n", params.speculative.model.path.c_str(), params.model.path.c_str());
        }
    }

    // Apply chat template for EAGLE3 if available which can increase the acceptance rate
    std::string prompt = params.prompt;
    if (params.speculative.eagle3) {
        auto chat_templates = common_chat_templates_init(model_tgt, params.chat_template);
        if (common_chat_templates_was_explicit(chat_templates.get())) {
            std::vector<common_chat_msg> chat_msgs;
            common_chat_msg user_msg;
            user_msg.role = "user";
            user_msg.content = params.prompt;
            chat_msgs.push_back(user_msg);

            common_chat_templates_inputs inputs;
            inputs.messages = chat_msgs;
            inputs.add_generation_prompt = true;
            prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            LOG_INF("%s: EAGLE3 chat template applied\n", __func__);
        }
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // how many tokens to draft each time
    int n_draft     = params.speculative.n_max;
    int n_draft_min = params.speculative.n_min;

    float p_min = params.speculative.p_min;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // eval the prompt
    llama_token id_last;
    llama_tokens prompt_tgt;
    int n_past;

    if (params.speculative.eagle3) {
        // Target model decodes full prompt and sample first token and intermediate features are extracted
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size()));

        id_last = common_sampler_sample(smpl, ctx_tgt, -1);
        common_sampler_accept(smpl, id_last, true);
        LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
        n_predict++;

        // all tokens currently in the target context
        prompt_tgt.assign(inp.begin(), inp.end());
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

        n_past = inp.size();
    } else {
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

        // note: keep the last token separate!
        id_last = inp.back();

        // all tokens currently in the target context
        prompt_tgt.assign(inp.begin(), inp.end() - 1);
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

        n_past = inp.size() - 1;
    }

    // init the speculator
    struct common_speculative_params params_spec;
    params_spec.n_draft = n_draft;
    params_spec.p_min   = p_min;

    struct common_speculative * spec = NULL;

    if (params.speculative.eagle3) {
        spec = common_speculative_init_eagle3(ctx_tgt, ctx_encoder, ctx_decoder);
    } else {
        params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft;
        spec = common_speculative_init(ctx_tgt, ctx_dft);
        for (auto &pair : params.speculative.replacements) {
            common_speculative_add_replacement_tgt_dft(spec, pair.first.c_str(), pair.second.c_str());
        }
    }

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // optionally, generate draft tokens that can be appended to the target batch
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);

        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) n_draft_min) {
                draft.clear();
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt);
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        n_past    += ids.size() - 1;
        n_drafted += draft.size(); // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    if (ctx_dft) {
        llama_perf_context_print(ctx_dft);
    } else if (ctx_encoder && ctx_decoder) {
        LOG_INF(" Eagle3 Draft encoder:\n");
        llama_perf_context_print(ctx_encoder);
        LOG_INF("\nEagle3 Draft decoder:\n");
        llama_perf_context_print(ctx_decoder);
    }

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    llama_batch_free(batch_tgt);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
