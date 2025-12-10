#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;  // max drafted tokens
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft
};

struct mtp_kv_update_data {
    llama_token id;
    int32_t n_past;
    int32_t tok_idx;
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft
);

void common_speculative_free(struct common_speculative * spec);

bool common_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_gen_draft(
    struct common_speculative * spec,
    struct common_speculative_params   params,
    const llama_tokens & prompt,
    llama_token   id_last);

/**
 * @brief Generates speculative draft tokens using the Multi-Token Prediction (MTP) architecture.
 * 
 * This function performs a recursive generation loop using the MTP head (e.g., Eagle/NextN).
 * It uses the fixed hidden state from the main model's last step and updates the MTP layer's 
 * internal KV cache autoregressively.
 * 
 * @param smpl      The sampler instance.
 * @param ctx       The llama context (shared between Main and MTP).
 * @param params    Speculative parameters (n_draft, p_min).
 * @param id_last   The last confirmed token ID from the main model.
 * @param n_past    The number of tokens in the validated past (start position for drafting).
 * @param seq_id    The sequence ID to use for drafting.
 * 
 * @return std::vector<llama_token> The generated draft tokens.
 */
llama_tokens mtp_speculative_gen_draft(
    struct common_sampler* smpl,
    struct llama_context* ctx,
    struct common_speculative_params params,
    llama_token id_last,
    int32_t n_past,
    llama_seq_id seq_id);

void mtp_update_kv_cache(struct llama_context * ctx, const llama_batch& batch, bool is_prompt_warmup);

void mtp_accept_tokens(
    struct llama_context * ctx,
    const std::vector<llama_token> & ids,
    int32_t n_past_base,
    llama_seq_id seq_id
);
