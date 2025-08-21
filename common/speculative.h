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
llama_token mtp_speculative_gen_draft(
    struct common_sampler* smpl,
    struct llama_context* ctx,
    llama_token id_last,
    int32_t n_past,
    int32_t last_tok_idx);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_gen_draft(
               struct common_speculative * spec,
        struct common_speculative_params   params,
                      const llama_tokens & prompt,
                             llama_token   id_last);

void mtp_update_kv_cache(struct llama_context * ctx, std::vector<mtp_kv_update_data>& tokens);
