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
#include <utility>
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

static std::pair<llama_token, llama_token> reasoning_sentinels(llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    require(vocab != nullptr, "failed to read model vocab");

    const int n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 2, "model vocab too small for reasoning sentinels");

    return { n_vocab - 2, n_vocab - 1 };
}

static void set_reasoning_budget(
        common_params_sampling & params,
        llama_token start_token,
        llama_token end_token,
        int32_t budget_tokens = 8) {
    params.reasoning_budget_tokens = budget_tokens;
    params.reasoning_budget_start = { start_token };
    params.reasoning_budget_end = { end_token };
    params.reasoning_budget_forced = { end_token };
}

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

static std::vector<llama_token> generate_reasoning_sequence(
        llama_model * model,
        const std::string & prompt,
        common_params_sampling params,
        const std::vector<llama_token> & accepted_tokens,
        size_t n_steps) {
    auto ctx = make_context(model);

    auto prompt_tokens = common_tokenize(ctx.get(), prompt, true);
    require(!prompt_tokens.empty(), "prompt tokenization produced no tokens");

    auto prompt_batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    require(llama_decode(ctx.get(), prompt_batch) == 0, "failed to decode prompt");

    common_sampler_ptr sampler { common_sampler_init(model, params) };
    require(static_cast<bool>(sampler), "failed to initialize common sampler");

    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    for (const auto token : accepted_tokens) {
        common_sampler_accept(sampler.get(), token, true);
        llama_token accepted = token;
        auto batch = llama_batch_get_one(&accepted, 1);
        require(llama_decode(ctx.get(), batch) == 0, "failed to decode accepted token");
    }

    std::vector<llama_token> result;
    result.reserve(n_steps);

    int idx = accepted_tokens.empty() ? static_cast<int>(prompt_tokens.size()) - 1 : 0;
    for (size_t i = 0; i < n_steps; ++i) {
        require(idx >= 0, "generation context is empty");

        const llama_token token = common_sampler_sample(sampler.get(), ctx.get(), idx);
        common_sampler_accept(sampler.get(), token, true);

        llama_token generated = token;
        auto batch = llama_batch_get_one(&generated, 1);
        require(llama_decode(ctx.get(), batch) == 0, "failed to decode generated token");

        result.push_back(token);
        idx = 0;
    }

    return result;
}

static void require_sequence_equal(
        const std::vector<llama_token> & a,
        const std::vector<llama_token> & b,
        const std::string & msg) {
    require(a == b, msg);
}

static void require_sequence_different(
        const std::vector<llama_token> & a,
        const std::vector<llama_token> & b,
        const std::string & msg) {
    require(a != b, msg);
}

static void test_reasoning_override_chain_switch(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    params.top_k = 40;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
    params.reasoning_top_k = 1;
    set_reasoning_budget(params, start_token, end_token);

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
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_P };
    params.top_p = 1.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_P;
    params.reasoning_top_p = 0.01f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(think_stats.candidate_count < base_stats.candidate_count, "reasoning top_p did not shrink the candidate set");
}

static void test_reasoning_override_min_p(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_MIN_P };
    params.min_p = 0.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_MIN_P;
    params.reasoning_min_p = 0.50f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(think_stats.candidate_count < base_stats.candidate_count, "reasoning min_p did not shrink the candidate set");
}

static void test_reasoning_override_temp(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    params.top_k = 40;
    params.temp = 0.80f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    params.reasoning_temp = 1.80f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(base_stats.candidate_count == 40, "base chain did not keep the configured top_k candidate count");
    require(std::fabs(base_stats.top_prob - think_stats.top_prob) > 1e-6f, "reasoning temperature did not change candidate probabilities");
}

static void test_reasoning_override_penalty_repeat(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_PENALTIES };
    params.penalty_last_n = 64;
    params.penalty_repeat = 1.0f;
    params.penalty_freq = 0.0f;
    params.penalty_present = 0.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
    params.reasoning_penalty_repeat = 1.50f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning repeat penalty did not reduce repeated-token probability");
}

static void test_reasoning_override_penalty_frequency(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_PENALTIES };
    params.penalty_last_n = 64;
    params.penalty_repeat = 1.0f;
    params.penalty_freq = 0.0f;
    params.penalty_present = 0.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_FREQ;
    params.reasoning_penalty_freq = 0.50f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning frequency penalty did not reduce repeated-token probability");
}

static void test_reasoning_override_penalty_presence(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_PENALTIES };
    params.penalty_last_n = 64;
    params.penalty_repeat = 1.0f;
    params.penalty_freq = 0.0f;
    params.penalty_present = 0.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_PRESENT;
    params.reasoning_penalty_present = 0.50f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning presence penalty did not reduce repeated-token probability");
}

static void test_reasoning_override_penalty_last_n(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_PENALTIES };
    params.penalty_last_n = 0;
    params.penalty_repeat = 1.0f;
    params.penalty_freq = 0.0f;
    params.penalty_present = 0.0f;
    params.reasoning_sampling =
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
    params.reasoning_penalty_last_n = 64;
    params.reasoning_penalty_repeat = 1.50f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning penalty_last_n did not reduce repeated-token probability");
}

static void test_reasoning_override_min_keep(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_P };
    params.top_p = 0.01f;
    params.min_keep = 1;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_P | COMMON_PARAMS_SAMPLING_CONFIG_MIN_KEEP;
    params.reasoning_top_p = 0.01f;
    params.reasoning_min_keep = 5;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(base_stats.candidate_count == 1, "base top_p did not keep the expected single candidate");
    require(think_stats.candidate_count >= 5, "reasoning min_keep did not keep enough candidates");
}

static void test_reasoning_override_typical(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TYPICAL_P };
    params.typ_p = 1.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TYPICAL_P;
    params.reasoning_typ_p = 0.10f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(think_stats.candidate_count < base_stats.candidate_count, "reasoning typical_p did not shrink the candidate set");
}

static void test_reasoning_override_top_n_sigma(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_N_SIGMA };
    params.top_n_sigma = -1.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_N_SIGMA;
    params.reasoning_top_n_sigma = 0.001f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, {}, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning top_n_sigma did not reduce tail-token probability");
}

static void test_reasoning_override_xtc_probability(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_XTC };
    params.min_keep = 1;
    params.xtc_probability = 0.0f;
    params.xtc_threshold = 0.0f;
    params.reasoning_sampling =
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY |
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
    params.reasoning_xtc_probability = 1.0f;
    params.reasoning_xtc_threshold = 0.0f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(think_stats.candidate_count < base_stats.candidate_count, "reasoning xtc_probability did not shrink the candidate set");
}

static void test_reasoning_override_xtc_threshold(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_XTC };
    params.min_keep = 1;
    params.xtc_probability = 1.0f;
    params.xtc_threshold = 0.99f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
    params.reasoning_xtc_threshold = 0.0f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const auto base_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, {});
    const auto think_stats = sample_reasoning_stats(model, ctx.get(), params, prompt_tokens, { start_token });

    require(think_stats.candidate_count < base_stats.candidate_count, "reasoning xtc_threshold did not shrink the candidate set");
}

static void test_reasoning_override_seed(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params_a;
    params_a.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    params_a.top_k = 40;
    params_a.temp = 0.80f;
    params_a.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_SEED;
    params_a.reasoning_seed = 123456u;
    set_reasoning_budget(params_a, start_token, end_token);

    auto params_b = params_a;
    params_b.reasoning_seed = 654321u;

    const auto seq_a_1 = generate_reasoning_sequence(model, "Hello", params_a, { start_token }, 8);
    const auto seq_a_2 = generate_reasoning_sequence(model, "Hello", params_a, { start_token }, 8);
    const auto seq_b = generate_reasoning_sequence(model, "Hello", params_b, { start_token }, 8);

    require_sequence_equal(seq_a_1, seq_a_2, "same reasoning seed did not reproduce the same sequence");
    require_sequence_different(seq_a_1, seq_b, "reasoning seed override did not change the sampled sequence");
}

static void test_reasoning_override_dynatemp_range(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params_base;
    params_base.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    params_base.top_k = 40;
    params_base.temp = 0.80f;
    params_base.dynatemp_range = 0.0f;
    params_base.dynatemp_exponent = 1.0f;
    params_base.reasoning_sampling =
        COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_RANGE |
        COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_EXPONENT;
    params_base.reasoning_dynatemp_range = 0.0f;
    params_base.reasoning_dynatemp_exponent = 1.0f;
    params_base.reasoning_seed = 777u;
    set_reasoning_budget(params_base, start_token, end_token);

    auto params_think = params_base;
    params_think.reasoning_dynatemp_range = 0.75f;

    const auto seq_base = generate_reasoning_sequence(model, "Hello", params_base, { start_token }, 8);
    const auto seq_think = generate_reasoning_sequence(model, "Hello", params_think, { start_token }, 8);

    require_sequence_different(seq_base, seq_think, "reasoning dynatemp_range did not change the sampled sequence");
}

static void test_reasoning_override_dynatemp_exponent(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params_base;
    params_base.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    params_base.top_k = 40;
    params_base.temp = 0.80f;
    params_base.dynatemp_range = 0.75f;
    params_base.dynatemp_exponent = 1.0f;
    params_base.reasoning_sampling =
        COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_RANGE |
        COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_EXPONENT;
    params_base.reasoning_dynatemp_range = 0.75f;
    params_base.reasoning_dynatemp_exponent = 1.0f;
    params_base.reasoning_seed = 888u;
    set_reasoning_budget(params_base, start_token, end_token);

    auto params_think = params_base;
    params_think.reasoning_dynatemp_exponent = 2.5f;

    const auto seq_base = generate_reasoning_sequence(model, "Hello", params_base, { start_token }, 8);
    const auto seq_think = generate_reasoning_sequence(model, "Hello", params_think, { start_token }, 8);

    require_sequence_different(seq_base, seq_think, "reasoning dynatemp_exponent did not change the sampled sequence");
}

static void test_reasoning_override_dry_multiplier(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_DRY };
    params.dry_multiplier = 0.0f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 64;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_DRY_MULTIPLIER;
    params.reasoning_dry_multiplier = 0.8f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning dry_multiplier did not reduce repeated-token probability");
}

static void test_reasoning_override_dry_base(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_DRY };
    params.dry_multiplier = 0.8f;
    params.dry_base = 1.0f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 64;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_DRY_BASE;
    params.reasoning_dry_base = 2.0f;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning dry_base did not reduce repeated-token probability");
}

static void test_reasoning_override_dry_allowed_length(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_DRY };
    params.dry_multiplier = 0.8f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 5;
    params.dry_penalty_last_n = 64;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_DRY_ALLOWED_LEN;
    params.reasoning_dry_allowed_length = 1;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning dry_allowed_length did not reduce repeated-token probability");
}

static void test_reasoning_override_dry_penalty_last_n(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_DRY };
    params.dry_multiplier = 0.8f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 1;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_DRY_PENALTY_LAST_N;
    params.reasoning_dry_penalty_last_n = 64;
    set_reasoning_budget(params, start_token, end_token);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto base_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { target_token, target_token, target_token, target_token }, target_token);
    const auto think_obs = sample_candidate_observation(model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, target_token, target_token }, target_token);

    require(think_obs.target_prob < base_obs.target_prob, "reasoning dry_penalty_last_n did not reduce repeated-token probability");
}

static void test_reasoning_override_adaptive_target(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params_base;
    params_base.samplers = { COMMON_SAMPLER_TYPE_ADAPTIVE_P };
    params_base.adaptive_target = -1.0f;
    params_base.adaptive_decay = 0.90f;
    params_base.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_ADAPTIVE_TARGET;
    params_base.reasoning_adaptive_target = -1.0f;
    params_base.reasoning_seed = 42u;
    set_reasoning_budget(params_base, start_token, end_token);

    auto params_think = params_base;
    params_think.reasoning_adaptive_target = 0.30f;

    const auto seq_base = generate_reasoning_sequence(model, "Hello", params_base, { start_token }, 8);
    const auto seq_think = generate_reasoning_sequence(model, "Hello", params_think, { start_token }, 8);

    require_sequence_different(seq_base, seq_think, "reasoning adaptive_target did not change the sampled sequence");
}

static void test_reasoning_override_adaptive_decay(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params_base;
    params_base.samplers = { COMMON_SAMPLER_TYPE_ADAPTIVE_P };
    params_base.adaptive_target = 0.20f;
    params_base.adaptive_decay = 0.20f;
    params_base.reasoning_sampling =
        COMMON_PARAMS_SAMPLING_CONFIG_ADAPTIVE_TARGET |
        COMMON_PARAMS_SAMPLING_CONFIG_ADAPTIVE_DECAY;
    params_base.reasoning_adaptive_target = 0.20f;
    params_base.reasoning_adaptive_decay = 0.20f;
    params_base.reasoning_seed = 314u;
    set_reasoning_budget(params_base, start_token, end_token);

    auto params_think = params_base;
    params_think.reasoning_adaptive_decay = 0.95f;

    const auto seq_base = generate_reasoning_sequence(model, "Hello", params_base, { start_token }, 8);
    const auto seq_think = generate_reasoning_sequence(model, "Hello", params_think, { start_token }, 8);

    require_sequence_different(seq_base, seq_think, "reasoning adaptive_decay did not change the sampled sequence");
}

static void test_reasoning_override_mirostat(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params_base;
    params_base.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    params_base.top_k = 40;
    params_base.temp = 0.80f;
    params_base.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT;
    params_base.reasoning_mirostat = 0;
    params_base.reasoning_mirostat_tau = 5.0f;
    params_base.reasoning_mirostat_eta = 0.10f;
    params_base.reasoning_seed = 11u;
    set_reasoning_budget(params_base, start_token, end_token);

    auto params_think = params_base;
    params_think.reasoning_mirostat = 2;

    const auto seq_base = generate_reasoning_sequence(model, "Hello", params_base, { start_token }, 8);
    const auto seq_think = generate_reasoning_sequence(model, "Hello", params_think, { start_token }, 8);

    require_sequence_different(seq_base, seq_think, "reasoning mirostat did not change the sampled sequence");
}

static void test_reasoning_override_mirostat_tau(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params_base;
    params_base.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    params_base.top_k = 40;
    params_base.temp = 0.80f;
    params_base.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU;
    params_base.reasoning_mirostat = 2;
    params_base.reasoning_mirostat_tau = 3.0f;
    params_base.reasoning_mirostat_eta = 0.10f;
    params_base.reasoning_seed = 22u;
    set_reasoning_budget(params_base, start_token, end_token);

    auto params_think = params_base;
    params_think.reasoning_mirostat_tau = 7.0f;

    const auto seq_base = generate_reasoning_sequence(model, "Hello", params_base, { start_token }, 8);
    const auto seq_think = generate_reasoning_sequence(model, "Hello", params_think, { start_token }, 8);

    require_sequence_different(seq_base, seq_think, "reasoning mirostat_tau did not change the sampled sequence");
}

static void test_reasoning_override_mirostat_eta(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params_base;
    params_base.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    params_base.top_k = 40;
    params_base.temp = 0.80f;
    params_base.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA;
    params_base.reasoning_mirostat = 2;
    params_base.reasoning_mirostat_tau = 5.0f;
    params_base.reasoning_mirostat_eta = 0.05f;
    params_base.reasoning_seed = 33u;
    set_reasoning_budget(params_base, start_token, end_token);

    auto params_think = params_base;
    params_think.reasoning_mirostat_eta = 0.30f;

    const auto seq_base = generate_reasoning_sequence(model, "Hello", params_base, { start_token }, 8);
    const auto seq_think = generate_reasoning_sequence(model, "Hello", params_think, { start_token }, 8);

    require_sequence_different(seq_base, seq_think, "reasoning mirostat_eta did not change the sampled sequence");
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
    test_reasoning_override_penalty_repeat(model.get());
    test_reasoning_override_penalty_frequency(model.get());
    test_reasoning_override_penalty_presence(model.get());
    test_reasoning_override_penalty_last_n(model.get());
    test_reasoning_override_min_keep(model.get());
    test_reasoning_override_typical(model.get());
    test_reasoning_override_top_n_sigma(model.get());
    test_reasoning_override_xtc_probability(model.get());
    test_reasoning_override_xtc_threshold(model.get());
    test_reasoning_override_seed(model.get());
    test_reasoning_override_dynatemp_range(model.get());
    test_reasoning_override_dynatemp_exponent(model.get());
    test_reasoning_override_dry_multiplier(model.get());
    test_reasoning_override_dry_base(model.get());
    test_reasoning_override_dry_allowed_length(model.get());
    test_reasoning_override_dry_penalty_last_n(model.get());
    test_reasoning_override_adaptive_target(model.get());
    test_reasoning_override_adaptive_decay(model.get());
    test_reasoning_override_mirostat(model.get());
    test_reasoning_override_mirostat_tau(model.get());
    test_reasoning_override_mirostat_eta(model.get());

    return 0;
}
