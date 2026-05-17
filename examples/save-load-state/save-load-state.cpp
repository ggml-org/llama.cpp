#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <vector>
#include <cstdio>


// Phase 1: tokenize the prompt, decode all but the last token, save state to disk,
// decode the last token, then generate n_predict tokens.
static std::string run_baseline_generation(struct llama_model * model, const struct common_params & params, std::string_view state_file) {
    auto * ctx = llama_init_from_model(model, common_context_params_to_llama(params));

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto tokens = common_tokenize(ctx, params.prompt, true);

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx, tokens, n_past, params.n_batch, state_file, true)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        llama_sampler_free(smpl);
        llama_free(ctx);
        return {};
    }

    printf("\nfirst run: %s", params.prompt.c_str());

    std::string result;
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl, ctx, -1);
        auto next_token_str = common_token_to_piece(ctx, next_token);

        printf("%s", next_token_str.c_str());
        result += next_token_str;

        common_batch_clear(batch);
        common_batch_add(batch, next_token, n_past, {0}, true);

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_sampler_free(smpl);
            llama_batch_free(batch);
            llama_free(ctx);
            return {};
        }
        n_past++;
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);

    return result;
}


// Phase 2: create a new context, load state from file, replay the last prompt token,
// then generate n_predict tokens.
static std::string run_state_restore_generation(struct llama_model * model, const struct common_params & params, std::string_view state_file) {
    auto * ctx = llama_init_from_model(model, common_context_params_to_llama(params));

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto tokens = common_tokenize(ctx, params.prompt, true);

    printf("\nsecond run: %s", params.prompt.c_str());

    // Load state from file
    std::vector<llama_token> unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx, state_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        fprintf(stderr, "\n%s : failed to load state\n", __func__);
        llama_sampler_free(smpl);
        llama_free(ctx);
        return {};
    }

    fprintf(stderr, "%s : loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out;
    if (!common_replay_last_token(ctx, tokens.back(), n_past)) {
        llama_sampler_free(smpl);
        llama_free(ctx);
        return {};
    }
    n_past++;

    // Generate tokens
    std::string result;
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl, ctx, -1);
        auto next_token_str = common_token_to_piece(ctx, next_token);

        printf("%s", next_token_str.c_str());
        result += next_token_str;

        common_batch_clear(batch);
        common_batch_add(batch, next_token, n_past, {0}, true);

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_sampler_free(smpl);
            llama_batch_free(batch);
            llama_free(ctx);
            return {};
        }
        n_past++;
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);

    return result;
}


// Phase 3: create a multi-seq context, load state, replay last token, migrate KV cache
// from seq 0 to seq 1 via the CPU path, then generate n_predict tokens on seq 1.
static std::string run_seq_migration_cpu_generation(struct llama_model * model, const struct common_params & params, std::string_view state_file) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto * ctx = llama_init_from_model(model, params_ctx);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto tokens = common_tokenize(ctx, params.prompt, true);

    printf("\nsingle seq run: %s", params.prompt.c_str());

    // Load state from file
    std::vector<llama_token> unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx, state_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        fprintf(stderr, "\n%s : failed to load state\n", __func__);
        llama_sampler_free(smpl);
        llama_free(ctx);
        return {};
    }

    fprintf(stderr, "%s : loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out;
    if (!common_replay_last_token(ctx, tokens.back(), n_past)) {
        llama_sampler_free(smpl);
        llama_free(ctx);
        return {};
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx, 0));
        const size_t ncopy = llama_state_seq_get_data(ctx, seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            fprintf(stderr, "\n%s : seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            llama_sampler_free(smpl);
            llama_free(ctx);
            return {};
        }
        fprintf(stderr, "%s : seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx), true);
        fprintf(stderr, "%s : kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx, seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            fprintf(stderr, "\n%s : seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            llama_sampler_free(smpl);
            llama_free(ctx);
            return {};
        }
        fprintf(stderr, "%s : seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    std::string result;
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl, ctx, -1);
        auto next_token_str = common_token_to_piece(ctx, next_token);

        printf("%s", next_token_str.c_str());
        result += next_token_str;

        common_batch_clear(batch);
        common_batch_add(batch, next_token, n_past, {1}, true);

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_sampler_free(smpl);
            llama_batch_free(batch);
            llama_free(ctx);
            return {};
        }
        n_past++;
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);

    return result;
}


// Phase 4: create a multi-seq context, load state, replay last token, migrate KV cache
// from seq 0 to seq 1 via the on-device path, then generate n_predict tokens on seq 1.
static std::string run_seq_migration_ondevice_generation(struct llama_model * model, const struct common_params & params, std::string_view state_file) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto * ctx = llama_init_from_model(model, params_ctx);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto tokens = common_tokenize(ctx, params.prompt, true);

    printf("\nsingle seq run: %s", params.prompt.c_str());

    // Load state from file
    std::vector<llama_token> unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx, state_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        fprintf(stderr, "\n%s : failed to load state\n", __func__);
        llama_sampler_free(smpl);
        llama_free(ctx);
        return {};
    }

    fprintf(stderr, "%s : loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out;
    if (!common_replay_last_token(ctx, tokens.back(), n_past)) {
        llama_sampler_free(smpl);
        llama_free(ctx);
        return {};
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx, seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            fprintf(stderr, "\n%s : seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            llama_sampler_free(smpl);
            llama_free(ctx);
            return {};
        }
        fprintf(stderr, "%s : seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx), true);
        fprintf(stderr, "%s : kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx, seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            fprintf(stderr, "\n%s : seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            llama_sampler_free(smpl);
            llama_free(ctx);
            return {};
        }
        fprintf(stderr, "%s : seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    std::string result;
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl, ctx, -1);
        auto next_token_str = common_token_to_piece(ctx, next_token);

        printf("%s", next_token_str.c_str());
        result += next_token_str;

        common_batch_clear(batch);
        common_batch_add(batch, next_token, n_past, {1}, true);

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_sampler_free(smpl);
            llama_batch_free(batch);
            llama_free(ctx);
            return {};
        }
        n_past++;
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);

    return result;
}


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "The quick brown fox";
    params.sampling.seed = 1234;

    const std::string_view state_file = "dump_state.bin";

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        printf("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();

    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // Phase 1: baseline generation (saves state to disk)
    auto result_baseline = run_baseline_generation(model, params, state_file);
    if (result_baseline.empty()) {
        return 1;
    }

    printf("\n\n");

    // Phase 2: full state restore from file
    auto result_restore = run_state_restore_generation(model, params, state_file);
    if (result_restore.empty()) {
        return 1;
    }

    printf("\n\n");

    // Phase 3: per-sequence KV migration (CPU path)
    auto result_seq_migration_cpu = run_seq_migration_cpu_generation(model, params, state_file);
    if (result_seq_migration_cpu.empty()) {
        return 1;
    }

    // Phase 4: per-sequence KV migration (on-device path)
    auto result_seq_migration_ondevice = run_seq_migration_ondevice_generation(model, params, state_file);
    if (result_seq_migration_ondevice.empty()) {
        return 1;
    }

    printf("\n");

    // Assertions
    if (result_baseline != result_restore) {
        fprintf(stderr, "\n%s : error : the 2 generations are different\n", __func__);
        return 1;
    }

    if (result_baseline != result_seq_migration_cpu) {
        fprintf(stderr, "\n%s : error : the seq restore generation is different\n", __func__);
        return 1;
    }

    if (result_baseline != result_seq_migration_ondevice) {
        fprintf(stderr, "\n%s : error : the seq restore generation is different\n", __func__);
        return 1;
    }

    fprintf(stderr, "\n%s : success\n", __func__);

    return 0;
}
