#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// token_matcher: watches for a specific sequence of tokens.
// advance() returns true when the full sequence has been matched.
struct token_matcher {
    std::vector<llama_token> tokens;
    size_t pos = 0;

    // Feed a token. Returns true if the full sequence was just completed.
    bool advance(llama_token token);

    // Number of tokens matched so far (partial match length).
    size_t matched() const { return pos; }

    void reset() { pos = 0; }
};

// --- Reasoning Budget SSM ---------------------------------------------------

enum matcher_ssm_rb_state {
    MATCHER_SSM_RB_IDLE,          // waiting for start sequence
    MATCHER_SSM_RB_COUNTING,      // counting down tokens
    MATCHER_SSM_RB_WAITING_UTF8,  // budget exhausted, waiting for UTF-8 completion
    MATCHER_SSM_RB_FORCING,       // forcing end-of-reasoning tokens
    MATCHER_SSM_RB_DONE,          // passthrough forever
};

// --- Tool Call Grammar SSM ---------------------------------------------------

enum matcher_ssm_tcg_state {
    MATCHER_SSM_TCG_OUT_OF_THINKING,  // passthrough; watching for think-start or tool-call-start
    MATCHER_SSM_TCG_IN_THINKING,      // passthrough; watching for think-end
    MATCHER_SSM_TCG_GRAMMAR_SAMPLING, // delegating apply to attached grammar sampler
};

// --- Matcher sampler construction --------------------------------------------

// Create a matcher sampler containing a reasoning budget SSM.
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for start_tokens sequence
//   COUNTING:     counting down remaining tokens, watching for natural end_tokens
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_tokens   - token sequence that activates counting
//   end_tokens     - token sequence for natural deactivation
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   prefill_tokens - tokens already present in the prompt; used to determine the
//                    initial state: COUNTING if they begin with start_tokens (but
//                    don't also end with end_tokens), IDLE otherwise.
//                    COUNTING with budget <= 0 is promoted to FORCING.
//
struct llama_sampler * common_matcher_sampler_init_reasoning_budget(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        const std::vector<llama_token> & prefill_tokens = {});

// Variant that takes an explicit initial state (used by tests and clone).
// COUNTING with budget <= 0 is promoted to FORCING.
struct llama_sampler * common_matcher_sampler_init_reasoning_budget(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        matcher_ssm_rb_state             initial_state);

// Add a tool call grammar SSM to an existing matcher sampler.
// The grammar sampler must be non-lazy (trigger detection is handled by the matcher).
// Ownership of grammar_sampler is transferred to the matcher.
//
// thinking_start/end_tokens: the reasoning block delimiters (same as reasoning budget).
//   When inside a thinking block, the tool call trigger is suppressed.
// tool_call_start_seqs: one or more token sequences that activate grammar sampling.
//   The first sequence to fully match fires; its tokens are replayed into the grammar.
//
void common_matcher_sampler_add_tool_call_grammar(
        struct llama_sampler                          * matcher_sampler,
        const std::vector<llama_token>                & thinking_start_tokens,
        const std::vector<llama_token>                & thinking_end_tokens,
        const std::vector<std::vector<llama_token>>   & tool_call_start_seqs,
        struct llama_sampler                          * grammar_sampler);

// Create a matcher sampler containing only a tool call grammar SSM (no reasoning budget).
struct llama_sampler * common_matcher_sampler_init_tool_call_grammar(
        const std::vector<llama_token>                & thinking_start_tokens,
        const std::vector<llama_token>                & thinking_end_tokens,
        const std::vector<std::vector<llama_token>>   & tool_call_start_seqs,
        struct llama_sampler                          * grammar_sampler);
