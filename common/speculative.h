#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);
// starts a new generation while preserving at most the retained common prefix that is
// still valid in both the target and draft contexts
void common_speculative_begin(
        common_speculative * spec,
        const llama_tokens & prompt,
        llama_pos            retained_prefix_len);

llama_pos common_speculative_get_committed_prefix_len(
        const common_speculative * spec);

void common_speculative_invalidate_retained_state(
        common_speculative * spec);

// supplies the token/hidden-state source used by the next MTP first pass; start_pos
// is the target-context position of source_tokens[0]
void common_speculative_set_first_pass_source(
        common_speculative * spec,
        const llama_tokens & source_tokens,
        const float *        hidden_states,
        int32_t              n_tokens,
        int32_t              n_embd,
        llama_pos            start_pos);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model;
// batch_idxs maps the frontier token and accepted draft tokens back to verifier output rows
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted, const std::vector<int32_t> & batch_idxs);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
