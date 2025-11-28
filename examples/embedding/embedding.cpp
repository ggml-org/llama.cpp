#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <ctime>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static std::vector<std::string> split_lines(const std::string & s, const std::string & separator = "\n") {
    std::vector<std::string> lines;
    size_t start = 0;
    size_t end = s.find(separator);

    while (end != std::string::npos) {
        lines.push_back(s.substr(start, end - start));
        start = end + separator.length();
        end = s.find(separator, start);
    }

    lines.push_back(s.substr(start)); // Add the last part

    return lines;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(ctx), true);

    // run model
    LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s : failed to process\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float * out = output + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, embd_norm);
    }
}

// plain, pipe-friendly output: one embedding per line
static void print_raw_embeddings(const float * emb,
                                 int n_embd_count,
                                 int n_embd,
                                 const llama_model * model,
                                 enum llama_pooling_type pooling_type,
                                 int embd_normalize) {
    const uint32_t n_cls_out = llama_model_n_cls_out(model);
    const bool is_rank = (pooling_type == LLAMA_POOLING_TYPE_RANK);
    const int cols = is_rank ? std::min<int>(n_embd, (int) n_cls_out) : n_embd;

    for (int j = 0; j < n_embd_count; ++j) {
        for (int i = 0; i < cols; ++i) {
            if (embd_normalize == 0) {
                LOG("%1.0f%s", emb[j * n_embd + i], (i + 1 < cols ? " " : ""));
            } else {
                LOG("%1.7f%s", emb[j * n_embd + i], (i + 1 < cols ? " " : ""));
            }
        }
        LOG("\n");
    }
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }

    common_init();

    params.embedding = true;

    // get max number of sequences per batch
    const int n_seq_max = llama_max_parallel_sequences();

    // if the number of prompts that would be encoded is known in advance, it's more efficient to specify the
    //   --parallel argument accordingly. for convenience, if not specified, we fallback to unified KV cache
    //   in order to support any number of prompts
    if (params.n_parallel == 1) {
        LOG_INF("%s: n_parallel == 1 -> unified KV cache is enabled\n", __func__);
        params.kv_unified = true;
        params.n_parallel = n_seq_max;
    }

    // utilize the full context
    if (params.n_batch < params.n_ctx) {
        LOG_WRN("%s: setting batch size to %d\n", __func__, params.n_ctx);
        params.n_batch = params.n_ctx;
    }

    // for non-causal models, batch size must be equal to ubatch size
    if (params.attention_type != LLAMA_ATTENTION_TYPE_CAUSAL) {
        params.n_ubatch = params.n_batch;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx       = llama_n_ctx(ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        LOG_ERR("%s: computing embeddings in encoder-decoder models is not supported\n", __func__);
        return 1;
    }

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    // split the prompt into lines
    std::vector<std::string> prompts = split_lines(params.prompt, params.embd_sep);

    // max batch size
    const uint64_t n_batch = params.n_batch;

    // get added sep and eos token, if any
    const std::string added_sep_token = llama_vocab_get_add_sep(vocab) ? llama_vocab_get_text(vocab, llama_vocab_sep(vocab)) : "";
    const std::string added_eos_token = llama_vocab_get_add_eos(vocab) ? llama_vocab_get_text(vocab, llama_vocab_eos(vocab)) : "";
    const char * rerank_prompt = llama_model_chat_template(model, "rerank");

    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> inputs;
    for (const auto & prompt : prompts) {
        std::vector<llama_token> inp;

        // split classification pairs and insert expected separator tokens
        if (pooling_type == LLAMA_POOLING_TYPE_RANK && prompt.find(params.cls_sep) != std::string::npos) {
            std::vector<std::string> pairs = split_lines(prompt, params.cls_sep);
            if (rerank_prompt != nullptr) {
                const std::string query = pairs[0];
                const std::string doc = pairs[1];
                std::string final_prompt = rerank_prompt;
                string_replace_all(final_prompt, "{query}"   , query);
                string_replace_all(final_prompt, "{document}", doc  );
                inp = common_tokenize(vocab, final_prompt, true, true);
            } else {
                std::string final_prompt;
                for (size_t i = 0; i < pairs.size(); i++) {
                    final_prompt += pairs[i];
                    if (i != pairs.size() - 1) {
                        if (!added_eos_token.empty()) {
                            final_prompt += added_eos_token;
                        }
                        if (!added_sep_token.empty()) {
                            final_prompt += added_sep_token;
                        }
                    }
                }
                inp = common_tokenize(ctx, final_prompt, true, true);
            }
        } else {
            inp = common_tokenize(ctx, prompt, true, true);
        }
        if (inp.size() > n_batch) {
            LOG_ERR("%s: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        inputs.push_back(inp);
    }

    // check if the last token is SEP/EOS
    // it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
    for (auto & inp : inputs) {
        if (inp.empty() || (inp.back() != llama_vocab_sep(vocab) && inp.back() != llama_vocab_eos(vocab))) {
            LOG_WRN("%s: last token in the prompt is not SEP or EOS\n", __func__);
            LOG_WRN("%s: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n", __func__);
        }
    }

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) inputs.size(); i++) {
            LOG_INF("%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
            LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
            for (int j = 0; j < (int) inputs[i].size(); j++) {
                LOG("%6d -> '%s'\n", inputs[i][j], common_token_to_piece(ctx, inputs[i][j]).c_str());
            }
            LOG("\n\n");
        }
    }

    // initialize batch
    const int n_prompts = prompts.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // count number of embeddings
    int n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (int k = 0; k < n_prompts; k++) {
            n_embd_count += inputs[k].size();
        }
    } else {
        n_embd_count = n_prompts;
    }

    // allocate output
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> embeddings(n_embd_count * n_embd, 0);
    float * emb = embeddings.data();

    // break into batches
    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto & inp = inputs[k];

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch || s >= n_seq_max) {
            float * out = emb + e * n_embd;
            batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
            e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
            s = 0;
            common_batch_clear(batch);
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + e * n_embd;
    batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);

    if (params.embd_out.empty()) {
        LOG("\n");

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            for (int j = 0; j < n_embd_count; j++) {
                LOG("embedding %d: ", j);
                for (int i = 0; i < std::min(3, n_embd); i++) {
                    if (params.embd_normalize == 0) {
                        LOG("%6.0f ", emb[j * n_embd + i]);
                    } else {
                        LOG("%9.6f ", emb[j * n_embd + i]);
                    }
                }
                LOG(" ... ");
                for (int i = n_embd - 3; i < n_embd; i++) {
                    if (params.embd_normalize == 0) {
                        LOG("%6.0f ", emb[j * n_embd + i]);
                    } else {
                        LOG("%9.6f ", emb[j * n_embd + i]);
                    }
                }
                LOG("\n");
            }
        } else if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
            const uint32_t n_cls_out = llama_model_n_cls_out(model);
            std::vector<std::string> cls_out_labels;

            for (uint32_t i = 0; i < n_cls_out; i++) {
                const char * label = llama_model_cls_label(model, i);
                const std::string label_i(label == nullptr ? "" : label);
                cls_out_labels.emplace_back(label_i.empty() ? std::to_string(i) : label_i);
            }

            for (int j = 0; j < n_embd_count; j++) {
                for (uint32_t i = 0; i < n_cls_out; i++) {
                    // NOTE: if you change this log - update the tests in ci/run.sh
                    if (n_cls_out == 1) {
                        LOG("rerank score %d: %8.3f\n", j, emb[j * n_embd]);
                    } else {
                        LOG("rerank score %d: %8.3f [%s]\n", j, emb[j * n_embd + i], cls_out_labels[i].c_str());
                    }
                }
            }
        } else {
            // print the first part of the embeddings or for a single prompt, the full embedding
            for (int j = 0; j < n_prompts; j++) {
                LOG("embedding %d: ", j);
                for (int i = 0; i < (n_prompts > 1 ? std::min(16, n_embd) : n_embd); i++) {
                    if (params.embd_normalize == 0) {
                        LOG("%6.0f ", emb[j * n_embd + i]);
                    } else {
                        LOG("%9.6f ", emb[j * n_embd + i]);
                    }
                }
                LOG("\n");
            }

            // print cosine similarity matrix
            if (n_prompts > 1) {
                LOG("\n");
                LOG("cosine similarity matrix:\n\n");
                for (int i = 0; i < n_prompts; i++) {
                    LOG("%6.6s ", prompts[i].c_str());
                }
                LOG("\n");
                for (int i = 0; i < n_prompts; i++) {
                    for (int j = 0; j < n_prompts; j++) {
                        float sim = common_embd_similarity_cos(emb + i * n_embd, emb + j * n_embd, n_embd);
                        LOG("%6.2f ", sim);
                    }
                    LOG("%1.10s", prompts[i].c_str());
                    LOG("\n");
                }
            }
        }
    }

    if (params.embd_out == "json" || params.embd_out == "json+" || params.embd_out == "array") {
        const bool notArray = params.embd_out != "array";

        LOG(notArray ? "{\n  \"object\": \"list\",\n  \"data\": [\n" : "[");
        for (int j = 0;;) { // at least one iteration (one prompt)
            if (notArray) LOG("    {\n      \"object\": \"embedding\",\n      \"index\": %d,\n      \"embedding\": ",j);
            LOG("[");
            for (int i = 0;;) { // at least one iteration (n_embd > 0)
                LOG(params.embd_normalize == 0 ? "%1.0f" : "%1.7f", emb[j * n_embd + i]);
                i++;
                if (i < n_embd) LOG(","); else break;
            }
            LOG(notArray ? "]\n    }" : "]");
            j++;
            if (j < n_embd_count) LOG(notArray ? ",\n" : ","); else break;
        }
        LOG(notArray ? "\n  ]" : "]\n");

        if (params.embd_out == "json+" && n_prompts > 1) {
            LOG(",\n  \"cosineSimilarity\": [\n");
            for (int i = 0;;) { // at least two iteration (n_embd_count > 1)
                LOG("    [");
                for (int j = 0;;) { // at least two iteration (n_embd_count > 1)
                    float sim = common_embd_similarity_cos(emb + i * n_embd, emb + j * n_embd, n_embd);
                    LOG("%6.2f", sim);
                    j++;
                    if (j < n_embd_count) LOG(", "); else break;
                }
                LOG(" ]");
                i++;
                if (i < n_embd_count) LOG(",\n"); else break;
            }
            LOG("\n  ]");
        }

        if (notArray) LOG("\n}\n");
    } else if (params.embd_out == "raw") {
        print_raw_embeddings(emb, n_embd_count, n_embd, model, pooling_type, params.embd_normalize);
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    // clean up
    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
