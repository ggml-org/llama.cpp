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

static void accept_and_decode(
        common_sampler * sampler,
        llama_context * ctx,
        llama_token token) {
    common_sampler_accept(sampler, token, true);

    auto batch = llama_batch_get_one(&token, 1);
    require(llama_decode(ctx, batch) == 0, "failed to decode generated token");
}

static llama_token sample_accept_and_decode(
        common_sampler * sampler,
        llama_context * ctx,
        int idx) {
    const llama_token token = common_sampler_sample(sampler, ctx, idx);
    accept_and_decode(sampler, ctx, token);
    return token;
}

static std::vector<llama_token> generate_boundary_sequence(
        llama_model * model,
        common_params_sampling params,
        llama_token start_token,
        llama_token end_token) {
    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    common_sampler_ptr sampler { common_sampler_init(model, params) };
    require(static_cast<bool>(sampler), "failed to initialize common sampler");

    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    std::vector<llama_token> result;
    auto sample_n = [&](size_t n, int & idx) {
        for (size_t i = 0; i < n; ++i) {
            result.push_back(sample_accept_and_decode(sampler.get(), ctx.get(), idx));
            idx = 0;
        }
    };

    int idx = static_cast<int>(prompt_tokens.size()) - 1;
    sample_n(1, idx);

    accept_and_decode(sampler.get(), ctx.get(), start_token);
    idx = 0;
    sample_n(3, idx);

    accept_and_decode(sampler.get(), ctx.get(), end_token);
    idx = 0;
    sample_n(2, idx);

    accept_and_decode(sampler.get(), ctx.get(), start_token);
    idx = 0;
    sample_n(2, idx);

    accept_and_decode(sampler.get(), ctx.get(), end_token);
    idx = 0;
    sample_n(2, idx);

    return result;
}

static void require_candidates_equal(
        common_sampler * a,
        common_sampler * b,
        const std::string & msg) {
    auto * candidates_a = common_sampler_get_candidates(a, true);
    auto * candidates_b = common_sampler_get_candidates(b, true);

    require(candidates_a != nullptr && candidates_b != nullptr, msg + ": missing candidates");
    require(candidates_a->size == candidates_b->size, msg + ": candidate count differs");

    for (size_t i = 0; i < candidates_a->size; ++i) {
        const auto & lhs = candidates_a->data[i];
        const auto & rhs = candidates_b->data[i];
        require(lhs.id == rhs.id, msg + ": candidate token differs");
        require(lhs.logit == rhs.logit || std::fabs(lhs.logit - rhs.logit) < 1e-6f,
                msg + ": candidate logit differs");
        require(std::fabs(lhs.p - rhs.p) < 1e-6f, msg + ": candidate probability differs");
    }
}

static void test_reasoning_equal_value_rng_continuity(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = {
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
    params.seed = 12345;
    params.top_k = 40;
    params.temp = 0.80f;
    params.logit_bias = {
        { start_token, -INFINITY },
        { end_token,   -INFINITY },
    };
    set_reasoning_budget(params, start_token, end_token, 64);

    auto params_equal_temp = params;
    params_equal_temp.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    params_equal_temp.reasoning_temp = params.temp;

    const auto base_temp = generate_boundary_sequence(model, params, start_token, end_token);
    const auto equal_temp = generate_boundary_sequence(model, params_equal_temp, start_token, end_token);
    require_sequence_equal(base_temp, equal_temp, "equal temperature override changed the global dist sequence");

    params.samplers = {
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
    params.seed = 54321;
    params.min_keep = 1;
    params.xtc_probability = 0.50f;
    params.xtc_threshold = 0.05f;

    auto params_equal_xtc = params;
    params_equal_xtc.reasoning_sampling =
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY |
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
    params_equal_xtc.reasoning_xtc_probability = params.xtc_probability;
    params_equal_xtc.reasoning_xtc_threshold = params.xtc_threshold;

    const auto base_xtc = generate_boundary_sequence(model, params, start_token, end_token);
    const auto equal_xtc = generate_boundary_sequence(model, params_equal_xtc, start_token, end_token);
    require_sequence_equal(base_xtc, equal_xtc, "equal XTC override changed the shared RNG sequence");

    params.samplers = {
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
        COMMON_SAMPLER_TYPE_ADAPTIVE_P,
    };
    params.seed = 13579;
    params.adaptive_target = 0.20f;
    params.adaptive_decay = 0.80f;

    auto params_equal_adaptive = params;
    params_equal_adaptive.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    params_equal_adaptive.reasoning_temp = params.temp;

    const auto base_adaptive = generate_boundary_sequence(model, params, start_token, end_token);
    const auto equal_adaptive = generate_boundary_sequence(model, params_equal_adaptive, start_token, end_token);
    require_sequence_equal(base_adaptive, equal_adaptive, "equal override changed the shared adaptive-p sequence");

    params.samplers.clear();
    params.seed = 97531;
    params.mirostat = 2;
    params.mirostat_tau = 5.0f;
    params.mirostat_eta = 0.1f;

    auto params_equal_mirostat = params;
    params_equal_mirostat.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    params_equal_mirostat.reasoning_temp = params.temp;

    const auto base_mirostat = generate_boundary_sequence(model, params, start_token, end_token);
    const auto equal_mirostat = generate_boundary_sequence(model, params_equal_mirostat, start_token, end_token);
    require_sequence_equal(base_mirostat, equal_mirostat, "equal override changed the shared Mirostat sequence");
}

static void test_reasoning_initial_state(llama_model * model) {
    const llama_token end_token = reasoning_sentinels(model).second;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    params.top_k = 40;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
    params.reasoning_top_k = 1;
    params.generation_prompt = " Hello";
    params.reasoning_budget_start = common_tokenize(vocab, params.generation_prompt, false, true);
    require(!params.reasoning_budget_start.empty(), "generation prompt tokenization produced no tokens");
    params.reasoning_budget_end = { end_token };
    params.reasoning_budget_forced = { end_token };
    params.reasoning_budget_tokens = 64;

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    const size_t initial_candidates = sample_candidate_count(model, ctx.get(), params, prompt_tokens, {});
    require(initial_candidates == 1, "generation prompt did not initialize the reasoning sampling state");

    params.generation_prompt.clear();
    const size_t idle_candidates = sample_candidate_count(model, ctx.get(), params, prompt_tokens, {});
    require(idle_candidates == 40, "reasoning override activated without a generated or prefilled start marker");
}

static void test_reasoning_forced_ending(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    params.top_k = 40;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
    params.reasoning_top_k = 1;
    set_reasoning_budget(params, start_token, end_token, 1);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token forced_prefix = prompt_tokens.back();
    params.reasoning_budget_forced = { forced_prefix, end_token };

    common_sampler_ptr sampler { common_sampler_init(model, params) };
    require(static_cast<bool>(sampler), "failed to initialize forced-ending sampler");
    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    common_sampler_accept(sampler.get(), start_token, true);
    common_sampler_accept(sampler.get(), forced_prefix, true);

    const int idx = static_cast<int>(prompt_tokens.size()) - 1;
    const llama_token first_forced = common_sampler_sample(sampler.get(), ctx.get(), idx);
    require(first_forced == forced_prefix, "reasoning budget did not force the budget message token");
    common_sampler_accept(sampler.get(), first_forced, true);

    const llama_token final_forced = common_sampler_sample(sampler.get(), ctx.get(), idx);
    require(final_forced == end_token, "reasoning budget did not force the reasoning end token");
    common_sampler_accept(sampler.get(), final_forced, true);

    (void) common_sampler_sample(sampler.get(), ctx.get(), idx);
    auto * candidates = common_sampler_get_candidates(sampler.get(), false);
    require(candidates != nullptr && candidates->size == 40,
            "base sampling parameters were not restored after the forced ending");
}

static void test_reasoning_inert_override(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    params.seed = 86420;
    params.temp = 0.80f;
    params.logit_bias = {
        { start_token, -INFINITY },
        { end_token,   -INFINITY },
    };
    set_reasoning_budget(params, start_token, end_token, 64);

    auto inert = params;
    inert.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
    inert.reasoning_top_k = 1;

    const auto base_sequence = generate_boundary_sequence(model, params, start_token, end_token);
    const auto inert_sequence = generate_boundary_sequence(model, inert, start_token, end_token);
    require_sequence_equal(base_sequence, inert_sequence, "absent top-k override changed the sampler topology");
}

static void test_reasoning_mirostat_validation(llama_model * model) {
    common_params_sampling params;
    params.mirostat = 2;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
    params.reasoning_top_k = 1;

    bool rejected = false;
    try {
        common_sampler_ptr sampler { common_sampler_init(model, params) };
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "Mirostat accepted an override for a sampler outside its topology");

    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    params.reasoning_temp = 1.0f;
    common_sampler_ptr sampler { common_sampler_init(model, params) };
    require(static_cast<bool>(sampler), "Mirostat rejected the supported reasoning temperature override");
}

static void test_reasoning_penalty_history_continuity(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_PENALTIES };
    params.penalty_last_n = 64;
    params.penalty_repeat = 1.0f;
    params.penalty_freq = 0.50f;
    params.penalty_present = 0.0f;
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_FREQ;
    params.reasoning_penalty_freq = 0.75f;
    set_reasoning_budget(params, start_token, end_token, 64);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto reasoning_without_pre = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens, { start_token }, target_token);
    const auto reasoning_with_pre = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens, { target_token, target_token, start_token }, target_token);
    require(reasoning_with_pre.target_prob < reasoning_without_pre.target_prob,
            "reasoning penalties did not retain pre-reasoning history");

    const auto base_without_reasoning = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens, { start_token, end_token }, target_token);
    const auto base_with_reasoning = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens, { start_token, target_token, target_token, end_token }, target_token);
    require(base_with_reasoning.target_prob < base_without_reasoning.target_prob,
            "base penalties did not retain reasoning history");

    const auto second_without_history = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens, { start_token, end_token, start_token }, target_token);
    const auto second_with_history = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens,
            { start_token, target_token, end_token, target_token, start_token }, target_token);
    require(second_with_history.target_prob < second_without_history.target_prob,
            "second reasoning block did not retain the complete token history");
}

static void test_reasoning_dry_history_continuity(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_DRY };
    params.dry_multiplier = 0.80f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 64;
    params.dry_sequence_breakers.clear();
    params.reasoning_sampling = COMMON_PARAMS_SAMPLING_CONFIG_DRY_MULTIPLIER;
    params.reasoning_dry_multiplier = params.dry_multiplier;
    set_reasoning_budget(params, start_token, end_token, 64);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");
    const llama_token target_token = prompt_tokens.back();

    const auto reasoning_without_pre = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens,
            { start_token, target_token, target_token }, target_token);
    const auto reasoning_with_pre = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens,
            { target_token, target_token, target_token, start_token, target_token, target_token }, target_token);
    require(reasoning_with_pre.target_prob < reasoning_without_pre.target_prob,
            "reasoning DRY sampler did not retain pre-reasoning history");

    const auto base_without_reasoning = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens,
            { start_token, end_token, target_token, target_token }, target_token);
    const auto base_with_reasoning = sample_candidate_observation(
            model, ctx.get(), params, prompt_tokens,
            { start_token, target_token, target_token, target_token, end_token, target_token, target_token }, target_token);
    require(base_with_reasoning.target_prob < base_without_reasoning.target_prob,
            "base DRY sampler did not retain reasoning history");
}

static void test_reasoning_clone_and_reset(llama_model * model) {
    const auto [start_token, end_token] = reasoning_sentinels(model);

    common_params_sampling params;
    params.samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
    params.seed = 24680;
    params.top_k = 40;
    params.min_keep = 1;
    params.xtc_probability = 0.65f;
    params.xtc_threshold = 0.05f;
    params.temp = 0.80f;
    params.penalty_last_n = 64;
    params.penalty_repeat = 1.10f;
    params.dry_multiplier = 0.50f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 64;
    params.dry_sequence_breakers.clear();
    params.reasoning_sampling =
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY |
        COMMON_PARAMS_SAMPLING_CONFIG_TEMP |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT |
        COMMON_PARAMS_SAMPLING_CONFIG_DRY_MULTIPLIER;
    params.reasoning_xtc_probability = 0.35f;
    params.reasoning_temp = 1.10f;
    params.reasoning_penalty_repeat = 1.30f;
    params.reasoning_dry_multiplier = 0.80f;
    params.logit_bias = {
        { start_token, -INFINITY },
        { end_token,   -INFINITY },
    };
    set_reasoning_budget(params, start_token, end_token, 64);

    auto ctx = make_context(model);
    const auto prompt_tokens = decode_prompt(ctx.get(), "Hello");

    auto init_params = params;
    common_sampler_ptr sampler { common_sampler_init(model, init_params) };
    require(static_cast<bool>(sampler), "failed to initialize common sampler");
    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    (void) sample_accept_and_decode(
            sampler.get(), ctx.get(), static_cast<int>(prompt_tokens.size()) - 1);
    accept_and_decode(sampler.get(), ctx.get(), start_token);
    (void) sample_accept_and_decode(sampler.get(), ctx.get(), 0);

    common_sampler_ptr clone { common_sampler_clone(sampler.get()) };
    require(static_cast<bool>(clone), "failed to clone common sampler");

    for (int i = 0; i < 4; ++i) {
        const llama_token source_token = common_sampler_sample(sampler.get(), ctx.get(), 0);
        const llama_token clone_token = common_sampler_sample(clone.get(), ctx.get(), 0);
        require(source_token == clone_token, "reasoning clone changed the sampled token");
        require_candidates_equal(sampler.get(), clone.get(), "reasoning clone");

        common_sampler_accept(sampler.get(), source_token, true);
        common_sampler_accept(clone.get(), clone_token, true);
        llama_token decoded = source_token;
        auto batch = llama_batch_get_one(&decoded, 1);
        require(llama_decode(ctx.get(), batch) == 0, "failed to decode cloned sample");
    }

    clone.reset();
    common_sampler_reset(sampler.get());

    auto fresh_params = params;
    common_sampler_ptr fresh { common_sampler_init(model, fresh_params) };
    require(static_cast<bool>(fresh), "failed to initialize reset reference sampler");
    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
        common_sampler_accept(fresh.get(), token, false);
    }

    const llama_token reset_base = common_sampler_sample(sampler.get(), ctx.get(), 0);
    const llama_token fresh_base = common_sampler_sample(fresh.get(), ctx.get(), 0);
    require(reset_base == fresh_base, "reset did not restore the base sampling state");
    require_candidates_equal(sampler.get(), fresh.get(), "base state after reset");

    common_sampler_accept(sampler.get(), start_token, true);
    common_sampler_accept(fresh.get(), start_token, true);
    const llama_token reset_reasoning = common_sampler_sample(sampler.get(), ctx.get(), 0);
    const llama_token fresh_reasoning = common_sampler_sample(fresh.get(), ctx.get(), 0);
    require(reset_reasoning == fresh_reasoning, "reset did not restore the reasoning sampling state");
    require_candidates_equal(sampler.get(), fresh.get(), "reasoning state after reset");

    common_sampler_ptr checkpoint { common_sampler_clone(sampler.get()) };
    common_sampler_ptr rollback_reference { common_sampler_clone(sampler.get()) };
    require(static_cast<bool>(checkpoint) && static_cast<bool>(rollback_reference),
            "failed to clone reasoning sampler for rollback");

    for (int i = 0; i < 3; ++i) {
        const llama_token token = common_sampler_sample(sampler.get(), ctx.get(), 0);
        common_sampler_accept(sampler.get(), token, true);
    }
    sampler = std::move(checkpoint);

    for (int i = 0; i < 4; ++i) {
        const llama_token restored_token = common_sampler_sample(sampler.get(), ctx.get(), 0);
        const llama_token reference_token = common_sampler_sample(rollback_reference.get(), ctx.get(), 0);
        require(restored_token == reference_token, "reasoning sampler rollback changed the sampled token");
        require_candidates_equal(sampler.get(), rollback_reference.get(), "reasoning sampler rollback");
        common_sampler_accept(sampler.get(), restored_token, true);
        common_sampler_accept(rollback_reference.get(), reference_token, true);
    }
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
    const size_t second_think_candidates = sample_candidate_count(model, ctx.get(), params, prompt_tokens, { start_token, end_token, start_token });

    require(base_candidates == 40, "base chain did not keep the configured top_k candidate count");
    require(think_candidates == 1, "reasoning chain did not switch to the override top_k");
    require(after_end_candidates == 40, "sampler did not return to the base chain after the reasoning end tag");
    require(second_think_candidates == 1, "sampler did not reactivate the override for a second reasoning block");
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
    params_base.seed = 777u;
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
    params_base.seed = 888u;
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

int main(int argc, char ** argv) {
    llama_backend_init();

    const char * model_path = get_model_or_exit(argc, argv);

    llama_model_ptr model { llama_model_load_from_file(model_path, llama_model_default_params()) };
    require(static_cast<bool>(model), std::string("failed to load model: ") + model_path);

    test_reasoning_equal_value_rng_continuity(model.get());
    test_reasoning_initial_state(model.get());
    test_reasoning_forced_ending(model.get());
    test_reasoning_inert_override(model.get());
    test_reasoning_mirostat_validation(model.get());
    test_reasoning_penalty_history_continuity(model.get());
    test_reasoning_dry_history_continuity(model.get());
    test_reasoning_clone_and_reset(model.get());

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
    test_reasoning_override_dynatemp_range(model.get());
    test_reasoning_override_dynatemp_exponent(model.get());
    test_reasoning_override_dry_multiplier(model.get());
    test_reasoning_override_dry_base(model.get());
    test_reasoning_override_dry_allowed_length(model.get());
    test_reasoning_override_dry_penalty_last_n(model.get());

    return 0;
}
