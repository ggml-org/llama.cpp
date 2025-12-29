#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;  // max drafted tokens
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft
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
 * Perform speculative generation using the model's own token history.
 * Searches for a matching pattern in the token history and returns draft tokens.
 *
 * @param tokens    Token history to search in
 * @param sampled   Last sampled token
 * @param n_draft_min Minimum number of draft tokens required
 * @param n_draft_max Maximum number of draft tokens to generate
 * @return Vector of draft tokens, empty if no matching pattern is found
 */
llama_tokens common_speculative_gen_self_draft(
                    const llama_tokens & tokens,
                    llama_token          sampled,
                    size_t         n_draft_min,
                    size_t         n_draft_max);
