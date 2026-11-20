#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,          // waiting for start sequence
    REASONING_BUDGET_COUNTING,      // counting down tokens
    REASONING_BUDGET_FORCING,       // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8,  // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,          // passthrough forever
    // local soft-budget extension (only reachable when soft_ratio/grace are configured)
    REASONING_BUDGET_SOFT_PENDING,  // soft threshold reached, waiting for a line boundary
    REASONING_BUDGET_SOFT_FORCING,  // forcing the soft wrap-up message, then back to COUNTING
    REASONING_BUDGET_HARD_PENDING,  // budget exhausted, bounded grace region before forced end
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for a start sequence
//   COUNTING:     counting down remaining tokens, watching for a natural end sequence
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_seqs     - token sequences, any of which activates counting
//   end_seqs       - token sequences, any of which naturally deactivates
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state = REASONING_BUDGET_IDLE,
        // local soft-budget extension: defaults keep exact upstream behavior
        float                             soft_ratio   = -1.0f, // fraction of budget for the soft warning; <= 0 or > 1 disables
        const llama_tokens              & soft_tokens  = {},    // tokenized soft wrap-up message (injected once, near a line boundary)
        int32_t                           grace_tokens = 0);    // extra tokens allowed past the budget before forced termination

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

// The end sequence that transitioned the sampler to DONE, or nullptr if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl);

// Manually transition the reasoning budget sampler into the FORCING state.
// Returns true if the transition occurred.
bool common_reasoning_budget_force(struct llama_sampler * smpl);
