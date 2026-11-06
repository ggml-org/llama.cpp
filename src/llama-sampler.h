#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_output_sizes {
    size_t logits = 0;
    size_t probs = 0;
    size_t candidates = 0;
};

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    llama_sampler_output_sizes output_sizes;
    bool output_sizes_valid = false;
    uint64_t revision = 0;

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

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
uint64_t llama_sampler_backend_revision(const llama_sampler * sampler);
void llama_sampler_backend_init_output_sizes(llama_sampler * sampler, int64_t n_vocab);
llama_sampler_output_sizes llama_sampler_backend_output_sizes(const llama_sampler * sampler, int64_t n_vocab);
void llama_sampler_backend_begin(llama_sampler * sampler);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
