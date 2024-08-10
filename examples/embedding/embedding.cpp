#include "common.h"
#include "llama.h"

#include <ctime>

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
        llama_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model * model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        // encoder-only model
        if (llama_encode(ctx, batch) < 0) {
            fprintf(stderr, "%s : failed to encode\n", __func__);
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        // decoder-only model
        if (llama_decode(ctx, batch) < 0) {
            fprintf(stderr, "%s : failed to decode\n", __func__);
        }
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
        llama_embd_normalize(embd, out, n_embd, embd_norm);
    }
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    params.embedding = true;
    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;

    print_build_info();

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        fprintf(stderr, "%s: error: computing embeddings in encoder-decoder models is not supported\n", __func__);
        return 1;
    }

    if (n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", gpt_params_get_system_info(params).c_str());
    }

    // split the prompt into lines
    std::vector<std::string> prompts = split_lines(params.prompt, params.embd_sep);

    // max batch size
    const uint64_t n_batch = params.n_batch;
    GGML_ASSERT(params.n_batch >= params.n_ctx);

    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> inputs;
    for (const auto & prompt : prompts) {
        auto inp = ::llama_tokenize(ctx, prompt, true, false);
        if (inp.size() > n_batch) {
            fprintf(stderr, "%s: error: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        inputs.push_back(inp);
    }

    // check if the last token is SEP
    // it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
    for (auto & inp : inputs) {
        if (inp.empty() || inp.back() != llama_token_sep(model)) {
            fprintf(stderr, "%s: warning: last token in the prompt is not SEP\n", __func__);
            fprintf(stderr, "%s:          'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n", __func__);
        }
    }

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) inputs.size(); i++) {
            fprintf(stderr, "%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
            fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
            for (int j = 0; j < (int) inputs[i].size(); j++) {
                fprintf(stderr, "%6d -> '%s'\n", inputs[i][j], llama_token_to_piece(ctx, inputs[i][j]).c_str());
            }
            fprintf(stderr, "\n\n");
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
    const int n_embd = llama_n_embd(model);
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
        if (batch.n_tokens + n_toks > n_batch) {
            float * out = emb + e * n_embd;
            batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
            e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
            s = 0;
            llama_batch_clear(batch);
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + e * n_embd;
    batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);

    if (params.embd_out.empty()) {
        fprintf(stdout, "\n");

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            for (int j = 0; j < n_embd_count; j++) {
                fprintf(stdout, "embedding %d: ", j);
                for (int i = 0; i < std::min(3, n_embd); i++) {
                    if (params.embd_normalize == 0) {
                        fprintf(stdout, "%6.0f ", emb[j * n_embd + i]);
                    } else {
                        fprintf(stdout, "%9.6f ", emb[j * n_embd + i]);
                    }
                }
                fprintf(stdout, " ... ");
                for (int i = n_embd - 3; i < n_embd; i++) {
                    if (params.embd_normalize == 0) {
                        fprintf(stdout, "%6.0f ", emb[j * n_embd + i]);
                    } else {
                        fprintf(stdout, "%9.6f ", emb[j * n_embd + i]);
                    }
                }
                fprintf(stdout, "\n");
            }
        } else {
            // print the first part of the embeddings or for a single prompt, the full embedding
            for (int j = 0; j < n_prompts; j++) {
                fprintf(stdout, "embedding %d: ", j);
                for (int i = 0; i < (n_prompts > 1 ? std::min(16, n_embd) : n_embd); i++) {
                    if (params.embd_normalize == 0) {
                        fprintf(stdout, "%6.0f ", emb[j * n_embd + i]);
                    } else {
                        fprintf(stdout, "%9.6f ", emb[j * n_embd + i]);
                    }
                }
                fprintf(stdout, "\n");
            }

            // print cosine similarity matrix
            if (n_prompts > 1) {
                fprintf(stdout, "\n");
                printf("cosine similarity matrix:\n\n");
                for (int i = 0; i < n_prompts; i++) {
                    fprintf(stdout, "%6.6s ", prompts[i].c_str());
                }
                fprintf(stdout, "\n");
                for (int i = 0; i < n_prompts; i++) {
                    for (int j = 0; j < n_prompts; j++) {
                        float sim = llama_embd_similarity_cos(emb + i * n_embd, emb + j * n_embd, n_embd);
                        fprintf(stdout, "%6.2f ", sim);
                    }
                    fprintf(stdout, "%1.10s", prompts[i].c_str());
                    fprintf(stdout, "\n");
                }
            }
        }
    }

    if (params.embd_out == "json" || params.embd_out == "json+" || params.embd_out == "array") {
        const bool notArray = params.embd_out != "array";

        fprintf(stdout, notArray ? "{\n  \"object\": \"list\",\n  \"data\": [\n" : "[");
        for (int j = 0;;) { // at least one iteration (one prompt)
            if (notArray) fprintf(stdout, "    {\n      \"object\": \"embedding\",\n      \"index\": %d,\n      \"embedding\": ",j);
            fprintf(stdout, "[");
            for (int i = 0;;) { // at least one iteration (n_embd > 0)
                fprintf(stdout, params.embd_normalize == 0 ? "%1.0f" : "%1.7f", emb[j * n_embd + i]);
                i++;
                if (i < n_embd) fprintf(stdout, ","); else break;
            }
            fprintf(stdout, notArray ? "]\n    }" : "]");
            j++;
            if (j < n_embd_count) fprintf(stdout, notArray ? ",\n" : ","); else break;
        }
        fprintf(stdout, notArray ? "\n  ]" : "]\n");

        if (params.embd_out == "json+" && n_prompts > 1) {
            fprintf(stdout, ",\n  \"cosineSimilarity\": [\n");
            for (int i = 0;;) { // at least two iteration (n_embd_count > 1)
                fprintf(stdout, "    [");
                for (int j = 0;;) { // at least two iteration (n_embd_count > 1)
                    float sim = llama_embd_similarity_cos(emb + i * n_embd, emb + j * n_embd, n_embd);
                    fprintf(stdout, "%6.2f", sim);
                    j++;
                    if (j < n_embd_count) fprintf(stdout, ", "); else break;
                }
                fprintf(stdout, " ]");
                i++;
                if (i < n_embd_count) fprintf(stdout, ",\n"); else break;
            }
            fprintf(stdout, "\n  ]");
        }

        if (notArray) fprintf(stdout, "\n}\n");
    }

    // clean up
    llama_print_timings(ctx);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
