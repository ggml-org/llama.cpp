#pragma once

// CPUOPTI: Schema-aware token skipping — skip forward passes for grammar-forced tokens
// When a grammar constraint forces exactly one valid next token, that token can be
// emitted without running a forward pass (the grammar would force it regardless).
// Tier 1 — Exact: mathematically identical outputs.

#include "llama.h"
#include "llama-grammar.h"

#include <cstdint>
#include <vector>

//
// Single-token skip query result
//

struct llama_opt_skip_result {
    bool        can_skip;      // True if the grammar forces exactly one token
    llama_token forced_token;  // The forced token (valid only if can_skip is true)
};

//
// Multi-token skip query result (greedy lookahead)
//

struct llama_opt_skip_sequence {
    std::vector<llama_token> tokens;   // Sequence of forced tokens
    uint32_t                 n_skip;   // Number of tokens that can be skipped
};

//
// Query the grammar for the next forced token
// Returns can_skip=true if the grammar's current state has exactly one valid continuation
//
llama_opt_skip_result llama_opt_schema_query_next(
    const llama_grammar * grammar,
    const llama_vocab   * vocab);

//
// Query for a sequence of consecutive forced tokens (greedy lookahead)
// Walks the grammar forward up to max_lookahead steps, collecting forced tokens
// Stops as soon as a state has more than one valid continuation
//
llama_opt_skip_sequence llama_opt_schema_query_sequence(
    const llama_grammar * grammar,
    const llama_vocab   * vocab,
    uint32_t              max_lookahead = 32);

//
// Accumulator for deferred batch fill of skipped tokens
// Collects skipped tokens so their KV projections can be computed in a single batch
//

class llama_opt_skip_accumulator {
public:
    // Add a skipped token at the given position
    void push(llama_token token, llama_pos pos);

    // Get accumulated tokens for batch processing
    const std::vector<llama_token> & tokens() const;
    const std::vector<llama_pos>   & positions() const;

    // Check if there are pending tokens
    bool has_pending() const;
    uint32_t n_pending() const;

    // Clear after batch processing
    void clear();

private:
    std::vector<llama_token> tokens_;
    std::vector<llama_pos>   positions_;
};
