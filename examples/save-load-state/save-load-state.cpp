#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>
#include <chrono>

int main(int argc, char ** argv) {
    gpt_params params;

    params.prompt = "The quick brown fox";

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    print_build_info();

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    auto n_past = 0;

    std::string result0;
    std::string result1;
    std::string result2;

    // init
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // tokenize prompt
    auto tokens = llama_tokenize(ctx, params.prompt, true);

    // evaluate prompt
    llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), n_past, 0));
    n_past += tokens.size();

    // save state (rng, logits, embedding and kv_cache) to file
    {
        std::vector<uint8_t> state_mem(llama_state_get_size(ctx));
        const size_t written = llama_state_get_data(ctx, state_mem.data(), state_mem.size());

        FILE *fp_write = fopen("dump_state.bin", "wb");
        fwrite(state_mem.data(), 1, written, fp_write);
        fclose(fp_write);

        fprintf(stderr, "%s : serialized state into %zd out of a maximum of %zd bytes\n", __func__, written, state_mem.size());
    }

    // save state (last tokens)
    const auto n_past_saved = n_past;

    // first run
    printf("\nfirst run: %s", params.prompt.c_str());

    for (auto i = 0; i < params.n_predict; i++) {
        auto * logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(model);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        auto next_token = llama_sample_token(ctx, &candidates_p);
        auto next_token_str = llama_token_to_piece(ctx, next_token);

        printf("%s", next_token_str.c_str());
        result0 += next_token_str;

        if (llama_decode(ctx, llama_batch_get_one(&next_token, 1, n_past, 0))) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_free(ctx);
            llama_free_model(model);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");

    // free old context
    llama_free(ctx);

    // make new context
    auto * ctx2 = llama_new_context_with_model(model, llama_context_params_from_gpt_params(params));

    printf("\nsecond run: %s", params.prompt.c_str());

    // load state (rng, logits, embedding and kv_cache) from file
    {
        std::vector<uint8_t> state_mem;

        FILE * fp_read = fopen("dump_state.bin", "rb");
        fseek(fp_read, 0, SEEK_END);
        state_mem.resize(ftell(fp_read));
        fseek(fp_read, 0, SEEK_SET);
        const size_t read = fread(state_mem.data(), 1, state_mem.size(), fp_read);
        fclose(fp_read);

        if (read != llama_state_set_data(ctx2, state_mem.data(), state_mem.size())) {
            fprintf(stderr, "\n%s : failed to read state\n", __func__);
            llama_free(ctx2);
            llama_free_model(model);
            return 1;
        }

        fprintf(stderr, "%s : deserialized state from %zd out of a maximum of %zd bytes\n", __func__, read, state_mem.size());
    }

    // restore state (last tokens)
    n_past = n_past_saved;

    // second run
    for (auto i = 0; i < params.n_predict; i++) {
        auto * logits = llama_get_logits(ctx2);
        auto n_vocab = llama_n_vocab(model);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        auto next_token = llama_sample_token(ctx2, &candidates_p);
        auto next_token_str = llama_token_to_piece(ctx2, next_token);

        printf("%s", next_token_str.c_str());
        result1 += next_token_str;

        if (llama_decode(ctx2, llama_batch_get_one(&next_token, 1, n_past, 0))) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_free(ctx2);
            llama_free_model(model);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");

    llama_free(ctx2);

    if (result0 != result1) {
        fprintf(stderr, "\n%s : error : the 2 generations are different\n", __func__);
        return 1;
    }

    // make new context
    auto* ctx3 = llama_new_context_with_model(model, llama_context_params_from_gpt_params(params));

    printf("\nsingle seq run: %s", params.prompt.c_str());

    // load state (rng, logits, embedding and kv_cache) from file
    {
        std::vector<uint8_t> state_mem;

        FILE * fp_read = fopen("dump_state.bin", "rb");
        fseek(fp_read, 0, SEEK_END);
        state_mem.resize(ftell(fp_read));
        fseek(fp_read, 0, SEEK_SET);
        const size_t read = fread(state_mem.data(), 1, state_mem.size(), fp_read);
        fclose(fp_read);

        if (read != llama_state_set_data(ctx3, state_mem.data(), state_mem.size())) {
            fprintf(stderr, "\n%s : failed to read state\n", __func__);
            llama_free(ctx3);
            llama_free_model(model);
            return 1;
        }

        fprintf(stderr, "%s : deserialized state from %zd out of a maximum of %zd bytes\n", __func__, read, state_mem.size());
    }

    // restore state (last tokens)
    n_past = n_past_saved;

    // save seq 0 and load into seq 1
    {
        // save kv of seq 0
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx3, 0));
        const size_t ncopy = llama_state_seq_get_data(ctx3, seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            fprintf(stderr, "\n%s : seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            llama_free(ctx3);
            llama_free_model(model);
            return 1;
        }
        fprintf(stderr, "%s : seq 0 copied, %zd bytes\n", __func__, ncopy);

        // erase whole kv
        llama_kv_cache_clear(ctx3);
        fprintf(stderr, "%s : kv cache cleared\n", __func__);

        // restore kv into seq 1
        const size_t nset = llama_state_seq_set_data(ctx3, seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            fprintf(stderr, "\n%s : seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            llama_free(ctx3);
            llama_free_model(model);
            return 1;
        }
        fprintf(stderr, "%s : seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // third run with seq 1 instead of 0
    for (auto i = 0; i < params.n_predict; i++) {
        auto * logits = llama_get_logits(ctx3);
        auto n_vocab = llama_n_vocab(model);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        auto next_token = llama_sample_token(ctx3, &candidates_p);
        auto next_token_str = llama_token_to_piece(ctx3, next_token);

        printf("%s", next_token_str.c_str());
        result2 += next_token_str;

        if (llama_decode(ctx3, llama_batch_get_one(&next_token, 1, n_past, 1))) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_free(ctx3);
            llama_free_model(model);
            return 1;
        }
        n_past += 1;
    }

    printf("\n");

    llama_free(ctx3);
    llama_free_model(model);

    if (result0 != result2) {
        fprintf(stderr, "\n%s : error : the seq restore generation is different\n", __func__);
        return 1;
    }

    fprintf(stderr, "\n%s : success\n", __func__);

    return 0;
}
