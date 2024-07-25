#pragma once

#include "llama-impl.h"

struct llama_vocab;
struct llama_sampling;

struct llama_grammar {
    const llama_grammar_rules  rules;
          llama_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8 partial_utf8;
};

//
// internal API
//

struct llama_grammar * llama_grammar_init_impl(
            const llama_grammar_element ** rules,
                                 size_t    n_rules,
                                 size_t    start_rule_index);

void llama_grammar_free_impl(struct llama_grammar * grammar);

struct llama_grammar * llama_grammar_copy_impl(const struct llama_grammar * grammar);

void llama_grammar_sample_impl(
        const struct llama_grammar * grammar,
          const struct llama_vocab * vocab,
       const struct llama_sampling * smpl,
            llama_token_data_array * candidates);

void llama_grammar_accept_token_impl(
              struct llama_grammar * grammar,
          const struct llama_vocab * vocab,
       const struct llama_sampling * smpl,
                       llama_token   token);
