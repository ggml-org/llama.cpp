#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -p \"Hello my name is\" -n 32 -np 4\n", argv[0]);
    LOG_TEE("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;

    params.prompt = "Hello my name is";
    params.n_predict = 32;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }


    // number of parallel batches
    int n_parallel = params.n_parallel;

    // total length of the sequences including the prompt
    int n_predict = params.n_predict;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = llama_model_params_from_gpt_params(params);

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(model, params.prompt, true);

    const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size())*n_parallel;

    // initialize the context

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);

    ctx_params.n_ctx   = n_kv_req;
    ctx_params.n_batch = std::max(n_predict, n_parallel);

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx);

    LOG_TEE("\n%s: n_predict = %d, n_ctx = %d, n_batch = %u, n_parallel = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, ctx_params.n_batch, n_parallel, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req (%d) > n_ctx, the required KV cache size is not big enough\n", __func__,  n_kv_req);
        LOG_TEE("%s:        either reduce n_parallel or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    // create a llama_batch
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(std::max(tokens_list.size(), (size_t) n_parallel), 0, n_parallel);

    std::vector<llama_seq_id> seq_ids(n_parallel, 0);
    for (int32_t i = 0; i < n_parallel; ++i) {
        seq_ids[i] = i;
    }

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); ++i) {
        llama_batch_add(batch, tokens_list[i], i, seq_ids, false);
    }
    GGML_ASSERT(batch.n_tokens == (int) tokens_list.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            LOG_TEE("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = llama_token_bos(model);
        }

        llama_batch_clear(batch);
        llama_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    //// assign the system KV cache to all parallel sequences
    //// this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    //for (int32_t i = 1; i < n_parallel; ++i) {
    //    llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
    //}

    if (n_parallel > 1) {
        LOG_TEE("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
    }

    // main loop

    // we will store the parallel decoded sequences in this vector
    std::vector<std::string> streams(n_parallel);

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_predict) {
        // prepare the next batch
        llama_batch_clear(batch);

        // sample the next token for each parallel sequence / stream
        for (int32_t i = 0; i < n_parallel; ++i) {
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, i_batch[i]);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            const int   top_k = 40;
            const float top_p = 0.9f;
            const float temp  = 0.4f;

            llama_sample_top_k(ctx, &candidates_p, top_k, 1);
            llama_sample_top_p(ctx, &candidates_p, top_p, 1);
            llama_sample_temp (ctx, &candidates_p, temp);

            const llama_token new_token_id = llama_sample_token(ctx, &candidates_p);

            //const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation? -> mark the stream as finished
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict) {
                i_batch[i] = -1;
                LOG_TEE("\n");
                if (n_parallel > 1) {
                    LOG_TEE("%s: stream %d finished at n_cur = %d", __func__, i, n_cur);
                }

                continue;
            }

            // if there is only one stream, we print immediately to stdout
            if (n_parallel == 1) {
                LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str());
                fflush(stdout);
            }

            streams[i] += llama_token_to_piece(ctx, new_token_id);

            i_batch[i] = batch.n_tokens;

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { i }, true);

            n_decode += 1;
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    LOG_TEE("\n");

    if (n_parallel > 1) {
        LOG_TEE("\n");

        for (int32_t i = 0; i < n_parallel; ++i) {
            LOG_TEE("sequence %d:\n\n%s%s\n\n", i, params.prompt.c_str(), streams[i].c_str());
        }
    }

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
