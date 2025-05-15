#pragma once

// TODO: rename llama-sampling.h/.cpp to llama-sampler.h/.cpp ?

#include "llama-grammar.h"

#include <unordered_map>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    std::vector<struct llama_sampler *> samplers;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct llama_sampler * llama_sampler_init_grammar_impl(
        const struct llama_vocab & vocab,
                      const char * grammar_str,
                      const char * grammar_root);
