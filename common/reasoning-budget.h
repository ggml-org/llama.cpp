#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> FORCING -> DONE
//   IDLE:     passthrough, watching for start_tokens sequence
//   COUNTING: counting down remaining tokens, watching for natural end_tokens
//   FORCING:  forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:     passthrough forever
//
// Parameters:
//   vocab                - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_tokens         - token sequence that activates counting
//   end_tokens           - token sequence for natural deactivation
//   forced_tokens        - token sequence forced when budget expires
//   budget               - max tokens allowed in the reasoning block
//   activate_immediately - if true, skip IDLE and start in COUNTING
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        bool                             activate_immediately);
