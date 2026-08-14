#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,          // waiting for start sequence
    REASONING_BUDGET_INTRO_FORCING, // forcing the intro/announcement message
    REASONING_BUDGET_COUNTING,      // counting down tokens
    REASONING_BUDGET_SOFT_PENDING,  // soft threshold crossed, waiting for a newline boundary
    REASONING_BUDGET_SOFT_FORCING,  // forcing the soft warning message
    REASONING_BUDGET_HARD_PENDING,  // budget exhausted, waiting (bounded) for a paragraph boundary
    REASONING_BUDGET_FORCING,       // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8,  // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,          // passthrough forever
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> INTRO_FORCING -> COUNTING -> SOFT_PENDING -> SOFT_FORCING -> COUNTING -> HARD_PENDING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:          passthrough, watching for a start sequence
//   INTRO_FORCING: forces intro_forced_tokens token-by-token right as the block starts, then proceeds to COUNTING (or straight to FORCING if budget <= 0)
//   COUNTING:      counting down remaining tokens, watching for a natural end sequence
//   SOFT_PENDING:  soft threshold crossed, waiting for a newline token before warning
//   SOFT_FORCING:  forces soft_forced_tokens token-by-token, then returns to COUNTING
//   HARD_PENDING:  budget exhausted, waiting (up to grace_tokens) for a paragraph boundary before forcing
//   WAITING_UTF8:  waiting to force, allowing tokens to complete a UTF-8 sequence
//   FORCING:       forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:          passthrough forever
//
// The hard cutoff always takes priority over the soft warning: if the budget is
// exhausted before a newline boundary is found in SOFT_PENDING, the soft warning
// is abandoned and the hard-cutoff path proceeds as normal (including any grace
// period below).
//
// Intro tokens (like soft/forced tokens) do not count against the budget: the
// countdown only starts once INTRO_FORCING completes.
//
// When the budget is exhausted (from COUNTING or SOFT_PENDING), if grace_tokens > 0
// the sampler enters HARD_PENDING instead of forcing immediately: it waits for a
// paragraph boundary (two adjacent newlines) so the cutoff lands at a clean break,
// but forces anyway once grace_tokens more tokens have passed without one - the
// hard guarantee on total length is bounded by budget + grace_tokens, never open-ended.
//
// Parameters:
//   vocab               - vocabulary (used for UTF-8/paragraph boundary detection; can be nullptr)
//   start_seqs          - token sequences, any of which activates counting
//   end_seqs            - token sequences, any of which naturally deactivates
//   forced_tokens       - token sequence forced when budget expires
//   soft_forced_tokens  - token sequence forced once at the soft threshold (empty = disabled)
//   intro_forced_tokens - token sequence forced once right as the block starts (empty = disabled)
//   budget              - max tokens allowed in the reasoning block
//   soft_ratio          - fraction of budget consumed at which to trigger the soft warning (<= 0 disables it)
//   grace_tokens        - max tokens to wait for a paragraph boundary after the budget expires (<= 0 = force immediately, no wait)
//   initial_state       - initial state
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        const llama_tokens              & soft_forced_tokens,
        const llama_tokens              & intro_forced_tokens,
        int32_t                           budget,
        float                              soft_ratio = -1.0f,
        int32_t                            grace_tokens = 0,
        common_reasoning_budget_state     initial_state = REASONING_BUDGET_IDLE);

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

// The end sequence that transitioned the sampler to DONE, or nullptr if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl);

// Manually transition the reasoning budget sampler into the FORCING state.
// Returns true if the transition occurred.
bool common_reasoning_budget_force(struct llama_sampler * smpl);
