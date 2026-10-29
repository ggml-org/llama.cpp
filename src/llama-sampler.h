#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// Borrowed sampling outputs for one row.
struct llama_sampling_output_view {
    llama_token token = LLAMA_TOKEN_NULL;
    const float * probs = nullptr;
    const float * sampled_logits = nullptr;
    const llama_token * candidates = nullptr;
    const float * logits = nullptr;
    uint32_t n_probs = 0;
    uint32_t n_logits = 0;
};

// Borrowed outputs from the last successful encode/decode. Do not use after the context outputs change or the context is freed.
struct llama_outputs {
    const llama_vocab * vocab = nullptr;
    uint32_t n_vocab = 0;
    const float * embeddings = nullptr;
    const float * embeddings_nextn = nullptr;

    // Positive indices are batch token positions; negative indices count from the last output row.
    // An invalid index returns an empty row.
    llama_sampling_output_view get_sampling_ith(int32_t i) const {
        const int64_t row = output_resolve_row(i);
        return row >= 0 ? sampling[row] : llama_sampling_output_view{};
    }

    // Unmasked NextN embeddings use non-negative batch token positions, including tokens without logits.
    const float * get_embeddings_nextn_ith(int32_t i) const {
        const int64_t row = embeddings_nextn_masked ? output_resolve_row(i) : i;
        return embeddings_nextn && row >= 0 && (size_t) row < n_embeddings_nextn
            ? embeddings_nextn + (size_t) row * n_embd : nullptr;
    }

    const float * get_embeddings_layer_inp(uint32_t lid) const {
        GGML_ASSERT(lid < embeddings_layer_inp.size() && embeddings_layer_inp[lid]);
        return embeddings_layer_inp[lid];
    }

private:
    friend struct llama_context;
    int64_t output_resolve_row(int32_t i) const {
        int64_t row = i;
        if (i < 0) {
            row += (int64_t) sampling.size();
        } else {
            if (!output_ids || (size_t) i >= output_ids->size()) {
                return -1;
            }
            row = (*output_ids)[i];
        }
        return row >= 0 && (size_t) row < sampling.size() ? row : -1;
    }

    std::vector<llama_sampling_output_view> sampling;
    std::vector<const float *> embeddings_layer_inp;
    const std::vector<int32_t> * output_ids = nullptr;
    uint32_t n_embd = 0;
    size_t n_embeddings_nextn = 0;
    bool embeddings_nextn_masked = false;
};

// Wait for the current batch and reorder its requested outputs once. Does not copy output tensors.
LLAMA_API const llama_outputs * llama_get_outputs(llama_context * ctx);

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

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
void llama_sampler_backend_begin(llama_sampler * sampler);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
