#pragma once

// CPUOPTI: Structural KV cache diffing — incremental prefill via token-level diffs
// Avoids recomputing KV projections for unchanged tokens between turns.
// Handles mid-context insertions, deletions, and replacements.
// Tier 1 — Exact: bitwise identical outputs.

#include "llama.h"

#include <cstdint>
#include <vector>

//
// Diff operation types
//

enum llama_opt_diff_op {
    DIFF_KEEP,     // Token unchanged at same/new position — reuse KV (with possible RoPE correction)
    DIFF_INSERT,   // New token — compute KV
    DIFF_DELETE,   // Token removed — free KV slot
    DIFF_REPLACE,  // Token changed — recompute KV
};

//
// A contiguous span of tokens with the same diff operation
//

struct llama_opt_diff_span {
    llama_opt_diff_op op;
    uint32_t          prev_start;  // Start index in previous context (-1 for INSERT)
    uint32_t          curr_start;  // Start index in current context (-1 for DELETE)
    uint32_t          length;      // Number of tokens in this span
};

using llama_opt_diff_result = std::vector<llama_opt_diff_span>;

//
// Diff engine
//

class llama_opt_diff_engine {
public:
    explicit llama_opt_diff_engine(uint32_t block_size);

    // Compute diff between previous and current context
    // Returns a list of KEEP/INSERT/DELETE/REPLACE spans
    llama_opt_diff_result compute(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr) const;

    // Summarize a diff result for statistics
    struct diff_summary {
        uint32_t n_keep    = 0;
        uint32_t n_insert  = 0;
        uint32_t n_delete  = 0;
        uint32_t n_replace = 0;
    };

    static diff_summary summarize(const llama_opt_diff_result & diff);

private:
    uint32_t block_size_;

    // Fast path: detect simple append (prev is a prefix of curr)
    bool try_append_diff(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr,
        llama_opt_diff_result & result) const;

    // Fast path: detect simple truncation (curr is a prefix of prev)
    bool try_truncate_diff(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr,
        llama_opt_diff_result & result) const;

    // General diff: find longest common prefix, longest common suffix,
    // then handle the middle section
    llama_opt_diff_result general_diff(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr) const;
};

//
// RoPE position correction
// When tokens shift position but content is unchanged, apply exact rotational correction
//

void llama_opt_rope_correction(
    float     * k_data,         // K tensor data to correct in place
    uint32_t    n_dims,         // Number of dimensions per head
    llama_pos   old_pos,        // Original position
    llama_pos   new_pos,        // New position
    float       rope_freq_base, // From model hparams
    float       rope_freq_scale // From model hparams
);

//
// Context history — tracks previous turn's tokens for diffing
//

class llama_opt_context_history {
public:
    // Record the current turn's context (swaps prev ↔ curr)
    void record(const llama_token * tokens, uint32_t n_tokens);

    // Access previous turn's data
    const llama_token * prev_tokens() const;
    uint32_t            prev_n_tokens() const;

    // Check if we have a previous turn to diff against
    bool has_prev() const;

    // Clear all history
    void clear();

private:
    std::vector<llama_token> prev_;
    std::vector<llama_token> curr_;
    bool has_prev_ = false;
};
