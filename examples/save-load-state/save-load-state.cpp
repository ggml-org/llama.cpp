#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <clocale>
#include <vector>

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch * get() { return &batch; }
    const llama_batch * get() const { return &batch; }
    llama_batch & operator*() { return batch; }
    const llama_batch & operator*() const { return batch; }
};

// Phase 1: tokenize the prompt, decode all but the last token, save state to disk,
// decode the last token, then generate n_predict tokens.
static std::string run_baseline_generation(struct llama_model * model, const struct common_params & params) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto tokens = common_tokenize(ctx.get(), params.prompt, true);

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOG("\nfirst run: %s", params.prompt.c_str());

    std::string result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl.get(), ctx.get(), -1);
        auto next_token_str = common_token_to_piece(ctx.get(), next_token);

        LOG("%s", next_token_str.c_str());
        result += next_token_str;

        common_batch_clear(*batch);
        common_batch_add(*batch, next_token, n_past, {0}, true);

        if (llama_decode(ctx.get(), *batch)) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}


// Phase 2: create a new context, load state from file, replay the last prompt token,
// then generate n_predict tokens and compare against expected result.
static bool run_state_restore_generation(struct llama_model * model, const struct common_params & params, const std::string & expected_result) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto tokens = common_tokenize(ctx.get(), params.prompt, true);

    LOG("\nsecond run: %s", params.prompt.c_str());

    // Load state from file
    std::vector<llama_token> unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens
    std::string result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl.get(), ctx.get(), -1);
        auto next_token_str = common_token_to_piece(ctx.get(), next_token);

        LOG("%s", next_token_str.c_str());
        result += next_token_str;

        common_batch_clear(*batch);
        common_batch_add(*batch, next_token, n_past, {0}, true);

        if (llama_decode(ctx.get(), *batch)) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG_TRC("\n%s: success\n", __func__);
    return true;
}


// Phase 3: create a multi-seq context, load state, replay last token, migrate KV cache
// from seq 0 to seq 1 via the CPU path, then generate n_predict tokens on seq 1.
static bool run_seq_migration_cpu_generation(struct llama_model * model, const struct common_params & params, const std::string & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto tokens = common_tokenize(ctx.get(), params.prompt, true);

    LOG("\nsingle seq run: %s", params.prompt.c_str());

    // Load state from file
    std::vector<llama_token> unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    std::string result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl.get(), ctx.get(), -1);
        auto next_token_str = common_token_to_piece(ctx.get(), next_token);

        LOG("%s", next_token_str.c_str());
        result += next_token_str;

        common_batch_clear(*batch);
        common_batch_add(*batch, next_token, n_past, {1}, true);

        if (llama_decode(ctx.get(), *batch)) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG_TRC("\n%s: success\n", __func__);
    return true;
}


// Phase 4: create a multi-seq context, load state, replay last token, migrate KV cache
// from seq 0 to seq 1 via the on-device path, then generate n_predict tokens on seq 1.
static bool run_seq_migration_ondevice_generation(struct llama_model * model, const struct common_params & params, const std::string & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto tokens = common_tokenize(ctx.get(), params.prompt, true);

    LOG("\nsingle seq run: %s", params.prompt.c_str());

    // Load state from file
    std::vector<llama_token> unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    std::string result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < params.n_predict; i++) {
        auto next_token     = llama_sampler_sample(smpl.get(), ctx.get(), -1);
        auto next_token_str = common_token_to_piece(ctx.get(), next_token);

        LOG("%s", next_token_str.c_str());
        result += next_token_str;

        common_batch_clear(*batch);
        common_batch_add(*batch, next_token, n_past, {1}, true);

        if (llama_decode(ctx.get(), *batch)) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG_TRC("\n%s: success\n", __func__);
    return true;
}


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "The quick brown fox";
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    // Phase 1: baseline generation (saves state to disk)
    auto result_baseline = run_baseline_generation(model, params);
    if (result_baseline.empty()) {
        return 1;
    }

    // Phase 2: full state restore from file
    if (!run_state_restore_generation(model, params, result_baseline)) {
        return 1;
    }

    // Phase 3: per-sequence KV migration (CPU path)
    if (!run_seq_migration_cpu_generation(model, params, result_baseline)) {
        return 1;
    }

    // Phase 4: per-sequence KV migration (on-device path)
    if (!run_seq_migration_ondevice_generation(model, params, result_baseline)) {
        return 1;
    }

    LOG_TRC("\n%s: all tests passed\n", __func__);

    return 0;
}
