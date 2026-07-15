#include "ggml.h"
#include "llama.h"

#include "common.h"
#include "get-model.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdlib>
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

int main(int argc, char ** argv) {
    llama_backend_init();

    const char * model_path = get_model_or_exit(argc, argv);

    llama_model_ptr model { llama_model_load_from_file(model_path, llama_model_default_params()) };
    require(static_cast<bool>(model), std::string("failed to load model: ") + model_path);

    test_reasoning_override_chain_switch(model.get());

    return 0;
}
