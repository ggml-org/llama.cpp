#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "The quick brown fox";
    params.sampling.seed = 1234;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    auto n_past = 0;

    std::string result0;
    std::string result1;
    std::string result2;

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();

    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    // tokenize prompt
    auto tokens = common_tokenize(ctx, params.prompt, true);

    // prepare the batch
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true; // generate next token

    // evaluate prompt
    llama_decode(ctx, batch);
    n_past += batch.n_tokens;

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
        auto next_token     = llama_sampler_sample(smpl, ctx, -1);
        auto next_token_str = common_token_to_piece(ctx, next_token);

        printf("%s", next_token_str.c_str());
        result0 += next_token_str;

        common_batch_clear(batch);
        common_batch_add(batch, next_token, n_past, {0}, true);

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");

    // make new context
    llama_context * ctx2 = llama_init_from_model(model, common_context_params_to_llama(params));

    llama_sampler * smpl2 = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl2, llama_sampler_init_dist(params.sampling.seed));

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
            return 1;
        }

        fprintf(stderr, "%s : deserialized state from %zd out of a maximum of %zd bytes\n", __func__, read, state_mem.size());
    }

    // restore state (last tokens)
    n_past = n_past_saved;

    // second run
    for (auto i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl2, ctx2, -1);
        auto next_token_str = common_token_to_piece(ctx2, next_token);

        printf("%s", next_token_str.c_str());
        result1 += next_token_str;

        common_batch_clear(batch);
        common_batch_add(batch, next_token, n_past, {0}, true);

        if (llama_decode(ctx2, batch)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");

    if (result0 != result1) {
        fprintf(stderr, "\n%s : error : the 2 generations are different\n", __func__);
        return 1;
    }

    // make new context
    llama_context * ctx3 = llama_init_from_model(model, common_context_params_to_llama(params));

    llama_sampler * smpl3 = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl3, llama_sampler_init_dist(params.sampling.seed));

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
            return 1;
        }
        fprintf(stderr, "%s : seq 0 copied, %zd bytes\n", __func__, ncopy);

        // erase whole kv
        llama_kv_self_clear(ctx3);
        fprintf(stderr, "%s : kv cache cleared\n", __func__);

        // restore kv into seq 1
        const size_t nset = llama_state_seq_set_data(ctx3, seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            fprintf(stderr, "\n%s : seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return 1;
        }
        fprintf(stderr, "%s : seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // third run with seq 1 instead of 0
    for (auto i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl3, ctx3, -1);
        auto next_token_str = common_token_to_piece(ctx3, next_token);

        printf("%s", next_token_str.c_str());
        result2 += next_token_str;

        common_batch_clear(batch);
        common_batch_add(batch, next_token, n_past, {1}, true);

        if (llama_decode(ctx3, batch)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        n_past += 1;
    }

    printf("\n");

    llama_sampler_free(smpl);
    llama_sampler_free(smpl2);
    llama_sampler_free(smpl3);

    llama_batch_free(batch);

    if (result0 != result2) {
        fprintf(stderr, "\n%s : error : the seq restore generation is different\n", __func__);
        return 1;
    }

    fprintf(stderr, "\n%s : success\n", __func__);

    return 0;
}
