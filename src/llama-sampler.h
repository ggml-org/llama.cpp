#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

// set trigger_suppressed on a grammar sampler (e.g. to suppress triggers during reasoning)
// the sampler must have been created by llama_sampler_init_grammar* or this is a no-op
void llama_sampler_grammar_set_trigger_suppressed(struct llama_sampler * smpl, bool suppressed);

struct llama_sampler * llama_sampler_init_dry_testing(
        int32_t context_size,
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
