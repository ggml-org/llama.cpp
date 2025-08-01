#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "ngram-cache.h"
#include "chat.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

typedef std::unordered_map<common_ngram, std::vector<int64_t>, common_ngram_hash_function> draft_index_t;

std::string get_chat_prompt(const common_params & params, const struct common_chat_templates * chat_templates);

draft_index_t make_draft_index(
    std::vector<llama_token> & draft_tokens,
    int64_t match_length
);

int64_t find_draft_match(
    common_ngram & ngram_key,
    draft_index_t & draft_index,
    int64_t draft_pos
);

// chat template support from main.cpp
std::string get_chat_prompt(const common_params & params, const struct common_chat_templates * chat_templates) {
    std::vector<common_chat_msg> chat_msgs;

    if (params.enable_chat_template) {
        // format the system prompt (will use template default if empty)
        if (!params.system_prompt.empty()) {
            common_chat_msg new_msg;
            new_msg.role = "system";
            new_msg.content = params.system_prompt;
            common_chat_format_single(chat_templates, chat_msgs, new_msg, false, params.use_jinja);
            chat_msgs.push_back(new_msg);
        }

        // format and append the user prompt
        if (!params.prompt.empty()) {
            common_chat_msg new_msg;
            new_msg.role = "user";
            new_msg.content = params.prompt;
            common_chat_format_single(chat_templates, chat_msgs, new_msg, true, params.use_jinja);
            chat_msgs.push_back(new_msg);
        }

        // apply the chat template
        if (!params.system_prompt.empty() || !params.prompt.empty()) {
            common_chat_templates_inputs inputs;
            inputs.use_jinja = params.use_jinja;
            inputs.messages = chat_msgs;
            inputs.add_generation_prompt = !params.prompt.empty();
            return common_chat_templates_apply(chat_templates, inputs).prompt;
        }
    }

    // otherwise use the prompt as is
    return params.prompt;
}

// index the draft into a {hash: [pos]} map
draft_index_t make_draft_index(
    std::vector<llama_token> & draft_tokens,
    int64_t match_length
) {
    draft_index_t index;
    int64_t draft_size = (int64_t) draft_tokens.size();
    for (int64_t i = 0; i < draft_size - match_length + 1; i++) {
        const common_ngram ngram_key(draft_tokens.data() + i, match_length);
        auto it = index.find(ngram_key);
        if (it == index.end()) {
            index[ngram_key] = {i};
        } else {
            it->second.push_back(i);
        }
    }
    return index;
}

int64_t find_draft_match(
    common_ngram & ngram_key,
    draft_index_t & draft_index,
    int64_t draft_pos
) {
    auto it = draft_index.find(ngram_key);
    if (it != draft_index.end()) {
        for (int64_t match_idx : it->second) {
            if (match_idx >= draft_pos) {
                return match_idx;
            }
        }
    }
    return -1;
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PREDICTED)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the target model
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model.get();
    ctx   = llama_init.context.get();

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // tokenize the prompt
    auto chat_templates = common_chat_templates_init(model, params.chat_template);
    std::string prompt = get_chat_prompt(params, chat_templates.get());
    std::vector<llama_token> inp = common_tokenize(ctx, prompt, true, true);

    LOG_DBG("prompt: %s\n", prompt.c_str());

    if (inp.empty()) {
        LOG_ERR("%s: the prompt is empty\n", __func__);
        return 1;
    }

    if (llama_n_ctx(ctx) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx));

        return 1;
    }

    if (llama_n_batch(ctx) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    const int draft_min = params.speculative.n_min;
    const int draft_max = params.speculative.n_max;

    // draft text to use for prediction
    std::string draft_text = params.speculative.text;
    std::vector<llama_token> draft = common_tokenize(ctx, draft_text, false, true);
    draft_index_t draft_index = make_draft_index(draft, draft_min);
    const int n_draft = draft.size();

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    // eval the prompt
    const int n_prompt = inp.size();
    llama_decode(ctx, llama_batch_get_one(inp.data(), n_prompt - 1));

    const auto t_enc_end = ggml_time_us();
    const auto t_dec_start = ggml_time_us();

    // track stats
    int n_logits = 0;
    int n_accept = 0;

    // generation state
    int n_past = n_prompt - 1;
    int draft_pos = 0;
    bool use_draft = !draft.empty();
    llama_token id_last = inp.back();
    int batch_idx = -1;

    // tokens: all prompt + generated tokens
    std::vector<llama_token> tokens(inp);
    tokens.reserve(llama_n_ctx(ctx));

    // prefix: tokens since last match
    std::vector<llama_token> ngram;

    // batch for evaluating the model
    const int draft_size0 = std::min(draft_max, (int) llama_n_batch(ctx) - 1);
    llama_batch batch = llama_batch_init(draft_size0 + 1, 0, 1);

    // loop through (successfully) generated tokens
    for (;; n_past++) {
        // we've hit the generation limit
        if (params.n_predict >= 0 && n_past >= n_prompt + params.n_predict) {
            break;
        }

        LOG_DBG("n_past: %d, use_draft: %d, batch_idx: %d\n", n_past, use_draft, batch_idx);
        LOG_DBG("id_last: %s\n", common_token_to_piece(ctx, id_last).c_str());
        LOG_DBG("Current draft: %s\n", string_from(ctx, draft).c_str());
        LOG_DBG("Current ngram: %s\n", string_from(ctx, ngram).c_str());

        // look for earliest match in the current draft
        if (!use_draft && (int) ngram.size() >= draft_min) {
            common_ngram ngram_key(ngram.data(), draft_min);
            const int64_t match_idx = find_draft_match(ngram_key, draft_index, draft_pos);
            if (match_idx >= 0) {
                draft_pos = match_idx + draft_min;
                ngram.clear();
                use_draft = true;
                batch_idx = -1;
                n_accept += draft_min;
                LOG_DBG("draft_pos: %d\n", draft_pos);
            }
        }

        // generate new logits if needed
        if (use_draft) {
            if (batch_idx == -1 || batch_idx >= batch.n_tokens - 1) {
                const int draft_size = std::min(draft_size0, n_draft - draft_pos);

                // decode from draft
                common_batch_clear(batch);
                common_batch_add(batch, id_last, n_past, { 0 }, true);
                for (int i = 0; i < draft_size; ++i) {
                    common_batch_add(batch, draft[draft_pos + i], n_past + 1 + i, { 0 }, true);
                }
                llama_decode(ctx, batch);

                // update generation state
                batch_idx = 0;
                n_logits += draft_size + 1;

                LOG_DBG("decoded batch [idx: %d]: %s\n", batch_idx, string_from(ctx, batch).c_str());
            } else {
                batch_idx++;
            }
        } else {
            // decode from sampled token
            common_batch_clear(batch);
            common_batch_add(batch, id_last, n_past, { 0 }, true);
            llama_decode(ctx, batch);

            // update generation state
            batch_idx = 0;
            n_logits++;
        }

        // sample from the target model
        id_last = common_sampler_sample(smpl, ctx, batch_idx);
        common_sampler_accept(smpl, id_last, true);
        tokens.push_back(id_last);

        LOG_DBG("sampled token: %s\n", common_token_to_piece(ctx, id_last).c_str());

        // check for EOS
        if (llama_vocab_is_eog(vocab, id_last)) {
            LOG_DBG("EOS\n");
            break;
        }

        // did the sampled token match the draft?
        if (use_draft) {
            if (id_last == draft[draft_pos]) {
                draft_pos++;
                n_accept += 1;
            } else {
                use_draft = false;
                llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + 1, -1);
            }
        }

        // process the accepted tokens and update contexts
        const std::string token_str = common_token_to_piece(ctx, id_last);
        if (params.use_color && use_draft) {
            LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
        } else {
            LOG("%s", token_str.c_str());
        }

        // if we're not in draft mode, update ngram tokens
        if (!use_draft) {
            ngram.push_back(id_last);
            if ((int) ngram.size() > draft_min) {
                ngram.erase(ngram.begin());
            }
        }

        // are we out of draft tokens?
        if (draft_pos >= n_draft) {
            use_draft = false;
            llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + 1, -1);
        }

        LOG_DBG("\n\n");
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_prompt,   (t_enc_end - t_enc_start) / 1e6f, n_prompt / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_past - n_prompt, (t_dec_end - t_dec_start) / 1e6f, (n_past - n_prompt) / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_logits  = %d\n", n_logits);
    LOG_INF("n_accept  = %d\n", n_accept);

    LOG_INF("\n");
    LOG_INF("performance:\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
