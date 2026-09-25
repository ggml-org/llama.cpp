#pragma once

#include "llama.h"
#include "common.h"

#include <cstdint>
#include <vector>


struct common_speculative_prefill_result {
    std::vector<int32_t> kept_indices;      // sorted original token positions kept for target model
    std::vector<float>   importance_scores; // per-token aggregated importance scores (size = n_prompt)
    int64_t              t_draft_eval_us = 0;
    int64_t              t_estimate_us   = 0;
    int32_t              n_prompt_orig   = 0;
    int32_t              n_prompt_kept   = 0;
};

// calculate 1D average-pooled smoothed token importance from raw attention matrices
std::vector<float> common_speculative_prefill_compute_importance(
    const std::vector<std::vector<float>> & attn_layers,
    int32_t n_prompt,
    int32_t n_heads,
    int32_t pool_kernel_size);

// select kept token indices using chunk-based top-k or token-level ranking
std::vector<int32_t> common_speculative_prefill_select_indices(
    const std::vector<float> & importance_scores,
    const common_params_speculative_prefill & params);

// execute draft prefill, lookahead decoding, attention collection, and index selection
common_speculative_prefill_result common_speculative_prefill_execute(
    llama_context * ctx_dft,
    common_sampler * smpl_dft,
    const std::vector<llama_token> & prompt,
    llama_seq_id seq_id,
    const common_params_speculative_prefill & params);
