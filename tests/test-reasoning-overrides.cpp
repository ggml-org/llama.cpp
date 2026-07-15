#include "ggml.h"
#include "llama.h"

#include "common.h"
#include "get-model.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdlib>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool cond, const std::string & msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

static llama_context_ptr make_context(llama_model * model) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 256;
    cparams.n_seq_max = 1;

    llama_context_ptr ctx { llama_init_from_model(model, cparams) };
    require(static_cast<bool>(ctx), "failed to create llama context");

    return ctx;
}

static std::vector<llama_token> decode_prompt(llama_context * ctx, const std::string & prompt) {
    auto tokens = common_tokenize(ctx, prompt, true);
    require(!tokens.empty(), "prompt tokenization produced no tokens");

    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    require(llama_decode(ctx, batch) == 0, "failed to decode prompt");

    return tokens;
}

struct sample_stats {
    size_t candidate_count = 0;
    llama_token top_token = LLAMA_TOKEN_NULL;
    float top_prob = 0.0f;
};

struct candidate_observation {
    size_t candidate_count = 0;
    size_t target_rank = SIZE_MAX;
    float target_prob = 0.0f;
};

static sample_stats sample_reasoning_stats(
        llama_model * model,
        llama_context * ctx,
        common_params_sampling params,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & generated_tokens) {
    common_sampler_ptr sampler { common_sampler_init(model, params) };
    require(static_cast<bool>(sampler), "failed to initialize common sampler");

    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    for (const auto token : generated_tokens) {
        common_sampler_accept(sampler.get(), token, true);
    }

    const int idx = static_cast<int>(prompt_tokens.size()) - 1;
    require(idx >= 0, "prompt must contain at least one token");

    (void) common_sampler_sample(sampler.get(), ctx, idx);

    auto * candidates = common_sampler_get_candidates(sampler.get(), true);
    require(candidates != nullptr, "sampler did not expose candidates");
    require(candidates->size > 0, "sampler returned no candidates");

    return sample_stats {
        candidates->size,
        candidates->data[0].id,
        candidates->data[0].p,
    };
}

static candidate_observation sample_candidate_observation(
        llama_model * model,
        llama_context * ctx,
        common_params_sampling params,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & generated_tokens,
        llama_token target_token) {
    common_sampler_ptr sampler { common_sampler_init(model, params) };
    require(static_cast<bool>(sampler), "failed to initialize common sampler");

    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    for (const auto token : generated_tokens) {
        common_sampler_accept(sampler.get(), token, true);
    }

    const int idx = static_cast<int>(prompt_tokens.size()) - 1;
    require(idx >= 0, "prompt must contain at least one token");

    (void) common_sampler_sample(sampler.get(), ctx, idx);

    auto * candidates = common_sampler_get_candidates(sampler.get(), true);
    require(candidates != nullptr, "sampler did not expose candidates");
    require(candidates->size > 0, "sampler returned no candidates");

    candidate_observation obs;
    obs.candidate_count = candidates->size;
    for (size_t i = 0; i < candidates->size; ++i) {
        if (candidates->data[i].id == target_token) {
            obs.target_rank = i;
            obs.target_prob = candidates->data[i].p;
            break;
        }
    }

    require(obs.target_rank != SIZE_MAX, "target token not found in candidate list");
    return obs;
}

static size_t sample_candidate_count(
        const llama_model * model,
        llama_context * ctx,
        common_params_sampling params,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & generated_tokens) {
    common_sampler_ptr sampler { common_sampler_init(model, params) };
    require(static_cast<bool>(sampler), "failed to initialize common sampler");

    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    for (const auto token : generated_tokens) {
        common_sampler_accept(sampler.get(), token, true);
    }

    const int idx = static_cast<int>(prompt_tokens.size()) - 1;
    require(idx >= 0, "prompt must contain at least one token");

    (void) common_sampler_sample(sampler.get(), ctx, idx);

    auto * candidates = common_sampler_get_candidates(sampler.get(), false);
    require(candidates != nullptr, "sampler did not expose candidates");

    return candidates->size;
}

static void test_reasoning_override_chain_switch(llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    require(vocab != nullptr, "failed to read model vocab");

    const int n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 2, "model vocab too small for reasoning sentinels");

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    params.top_k = 40;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
    params.reasoning_top_k = 1;
    params.reasoning_budget_tokens = 8;

    const llama_token start_token = n_vocab - 2;
    const llama_token end_token = n_vocab - 1;
    params.reasoning_budget_start = { start_token };
    params.reasoning_budget_end = { end_token };
    params.reasoning_budget_forced = { end_token };

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const size_t base_candidates = sample_candidate_count(model, ctx.get(), params, prompt_tokens, {});
    const size_t think_candidates = sample_candidate_count(model, ctx.get(), params, prompt_tokens, { start_token });
    const size_t after_end_candidates = sample_candidate_count(model, ctx.get(), params, prompt_tokens, { start_token, end_token });

    require(base_candidates == 40, "base chain did not keep the configured top_k candidate count");
    require(think_candidates == 1, "reasoning chain did not switch to the override top_k");
    require(after_end_candidates == 40, "sampler did not return to the base chain after the reasoning end tag");
}

static void test_reasoning_override_top_p(llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    require(vocab != nullptr, "failed to read model vocab");

    const int n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 2, "model vocab too small for reasoning sentinels");

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_P };
    params.top_p = 1.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_P;
    params.reasoning_top_p = 0.01f;
    params.reasoning_budget_tokens = 8;

    const llama_token start_token = n_vocab - 2;
    const llama_token end_token = n_vocab - 1;
    params.reasoning_budget_start = { start_token };
    params.reasoning_budget_end = { end_token };
    params.reasoning_budget_forced = { end_token };

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(think_stats.candidate_count < base_stats.candidate_count, "reasoning top_p did not shrink the candidate set");
}

static void test_reasoning_override_min_p(llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    require(vocab != nullptr, "failed to read model vocab");

    const int n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 2, "model vocab too small for reasoning sentinels");

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_MIN_P };
    params.min_p = 0.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_MIN_P;
    params.reasoning_min_p = 0.50f;
    params.reasoning_budget_tokens = 8;

    const llama_token start_token = n_vocab - 2;
    const llama_token end_token = n_vocab - 1;
    params.reasoning_budget_start = { start_token };
    params.reasoning_budget_end = { end_token };
    params.reasoning_budget_forced = { end_token };

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(think_stats.candidate_count < base_stats.candidate_count, "reasoning min_p did not shrink the candidate set");
}

static void test_reasoning_override_temp(llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    require(vocab != nullptr, "failed to read model vocab");

    const int n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 2, "model vocab too small for reasoning sentinels");

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    params.top_k = 40;
    params.temp = 0.80f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    params.reasoning_temp = 1.80f;
    params.reasoning_budget_tokens = 8;

    const llama_token start_token = n_vocab - 2;
    const llama_token end_token = n_vocab - 1;
    params.reasoning_budget_start = { start_token };
    params.reasoning_budget_end = { end_token };
    params.reasoning_budget_forced = { end_token };

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(base_stats.candidate_count == 40, "base chain did not keep the configured top_k candidate count");
    require(std::fabs(base_stats.top_prob - think_stats.top_prob) > 1e-6f, "reasoning temperature did not change candidate probabilities");
}

static void test_reasoning_override_penalties(llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    require(vocab != nullptr, "failed to read model vocab");

    const int n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 2, "model vocab too small for reasoning sentinels");

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_PENALTIES };
    params.penalty_last_n = 64;
    params.penalty_repeat = 1.0f;
    params.penalty_freq = 0.0f;
    params.penalty_present = 0.0f;
    params.reasoning_sampling =
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_FREQ |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_PRESENT;
    params.reasoning_penalty_repeat = 1.50f;
    params.reasoning_penalty_freq = 0.50f;
    params.reasoning_penalty_present = 0.50f;
    params.reasoning_budget_tokens = 8;

    const llama_token start_token = n_vocab - 2;
    const llama_token end_token = n_vocab - 1;
    params.reasoning_budget_start = { start_token };
    params.reasoning_budget_end = { end_token };
    params.reasoning_budget_forced = { end_token };

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning penalties did not reduce repeated-token probability");
}

int main(int argc, char ** argv) {
    llama_backend_init();

    const char * model_path = get_model_or_exit(argc, argv);

    llama_model_ptr model { llama_model_load_from_file(model_path, llama_model_default_params()) };
    require(static_cast<bool>(model), std::string("failed to load model: ") + model_path);

    test_reasoning_override_chain_switch(model.get());
    test_reasoning_override_top_p(model.get());
    test_reasoning_override_min_p(model.get());
    test_reasoning_override_temp(model.get());
    test_reasoning_override_penalties(model.get());

    return 0;
}
