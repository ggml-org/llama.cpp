// CPUOPTI: Structural KV cache diffing — incremental prefill

#include "llama-kv-cache-diff.h"
#include "llama-context-hash.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

//
// llama_opt_diff_engine
//

llama_opt_diff_engine::llama_opt_diff_engine(uint32_t block_size)
    : block_size_(block_size) {}

llama_opt_diff_result llama_opt_diff_engine::compute(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr) const {

    // Empty cases
    if (n_prev == 0 && n_curr == 0) {
        return {};
    }

    if (n_prev == 0) {
        // Everything is new
        return {{ DIFF_INSERT, 0, 0, n_curr }};
    }

    if (n_curr == 0) {
        // Everything deleted
        return {{ DIFF_DELETE, 0, 0, n_prev }};
    }

    // Fast path: append-only
    llama_opt_diff_result result;
    if (try_append_diff(prev, n_prev, curr, n_curr, result)) {
        return result;
    }

    // Fast path: truncation
    if (try_truncate_diff(prev, n_prev, curr, n_curr, result)) {
        return result;
    }

    // General diff
    return general_diff(prev, n_prev, curr, n_curr);
}

llama_opt_diff_engine::diff_summary llama_opt_diff_engine::summarize(
        const llama_opt_diff_result & diff) {

    diff_summary s;
    for (const auto & span : diff) {
        switch (span.op) {
            case DIFF_KEEP:    s.n_keep    += span.length; break;
            case DIFF_INSERT:  s.n_insert  += span.length; break;
            case DIFF_DELETE:  s.n_delete  += span.length; break;
            case DIFF_REPLACE: s.n_replace += span.length; break;
        }
    }
    return s;
}

bool llama_opt_diff_engine::try_append_diff(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr,
        llama_opt_diff_result & result) const {

    if (n_curr < n_prev) {
        return false;
    }

    // Check if prev is a prefix of curr
    for (uint32_t i = 0; i < n_prev; i++) {
        if (prev[i] != curr[i]) {
            return false;
        }
    }

    result.clear();

    if (n_prev > 0) {
        result.push_back({ DIFF_KEEP, 0, 0, n_prev });
    }

    if (n_curr > n_prev) {
        result.push_back({ DIFF_INSERT, n_prev, n_prev, n_curr - n_prev });
    }

    return true;
}

bool llama_opt_diff_engine::try_truncate_diff(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr,
        llama_opt_diff_result & result) const {

    if (n_curr > n_prev) {
        return false;
    }

    // Check if curr is a prefix of prev
    for (uint32_t i = 0; i < n_curr; i++) {
        if (prev[i] != curr[i]) {
            return false;
        }
    }

    result.clear();

    if (n_curr > 0) {
        result.push_back({ DIFF_KEEP, 0, 0, n_curr });
    }

    if (n_prev > n_curr) {
        result.push_back({ DIFF_DELETE, n_curr, n_curr, n_prev - n_curr });
    }

    return true;
}

llama_opt_diff_result llama_opt_diff_engine::general_diff(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr) const {

    llama_opt_diff_result result;

    // Find longest common prefix
    uint32_t prefix_len = 0;
    while (prefix_len < n_prev && prefix_len < n_curr &&
           prev[prefix_len] == curr[prefix_len]) {
        prefix_len++;
    }

    // Find longest common suffix (not overlapping with prefix)
    uint32_t suffix_len = 0;
    while (suffix_len < (n_prev - prefix_len) &&
           suffix_len < (n_curr - prefix_len) &&
           prev[n_prev - 1 - suffix_len] == curr[n_curr - 1 - suffix_len]) {
        suffix_len++;
    }

    // Emit common prefix as KEEP
    if (prefix_len > 0) {
        result.push_back({ DIFF_KEEP, 0, 0, prefix_len });
    }

    // Handle the middle section
    const uint32_t prev_mid_len = n_prev - prefix_len - suffix_len;
    const uint32_t curr_mid_len = n_curr - prefix_len - suffix_len;

    if (prev_mid_len > 0 && curr_mid_len > 0) {
        // Both have middle content — attempt block-level matching within the middle
        // For simplicity in Phase 1, treat the entire middle as REPLACE if lengths differ,
        // or do a token-by-token comparison if lengths match

        if (prev_mid_len == curr_mid_len) {
            // Same length — find changed regions within the middle
            uint32_t i = 0;
            while (i < prev_mid_len) {
                const uint32_t pi = prefix_len + i;
                const uint32_t ci = prefix_len + i;

                if (prev[pi] == curr[ci]) {
                    // Find extent of matching region
                    uint32_t match_len = 0;
                    while (i + match_len < prev_mid_len &&
                           prev[pi + match_len] == curr[ci + match_len]) {
                        match_len++;
                    }
                    result.push_back({ DIFF_KEEP, pi, ci, match_len });
                    i += match_len;
                } else {
                    // Find extent of changed region
                    uint32_t change_len = 0;
                    while (i + change_len < prev_mid_len &&
                           prev[pi + change_len] != curr[ci + change_len]) {
                        change_len++;
                    }
                    result.push_back({ DIFF_REPLACE, pi, ci, change_len });
                    i += change_len;
                }
            }
        } else {
            // Different lengths — emit DELETE + INSERT
            result.push_back({ DIFF_DELETE, prefix_len, prefix_len, prev_mid_len });
            result.push_back({ DIFF_INSERT, prefix_len, prefix_len, curr_mid_len });
        }
    } else if (prev_mid_len > 0) {
        // Only prev has middle content — pure deletion
        result.push_back({ DIFF_DELETE, prefix_len, prefix_len, prev_mid_len });
    } else if (curr_mid_len > 0) {
        // Only curr has middle content — pure insertion
        result.push_back({ DIFF_INSERT, prefix_len, prefix_len, curr_mid_len });
    }

    // Emit common suffix as KEEP
    if (suffix_len > 0) {
        result.push_back({ DIFF_KEEP, n_prev - suffix_len, n_curr - suffix_len, suffix_len });
    }

    return result;
}

//
// RoPE position correction
//

void llama_opt_rope_correction(
        float     * k_data,
        uint32_t    n_dims,
        llama_pos   old_pos,
        llama_pos   new_pos,
        float       rope_freq_base,
        float       rope_freq_scale) {

    if (old_pos == new_pos) {
        return; // No correction needed
    }

    const llama_pos delta = new_pos - old_pos;

    // Apply rotational correction for each dimension pair
    // RoPE operates on pairs of dimensions: (d, d+1) for d = 0, 2, 4, ...
    for (uint32_t d = 0; d < n_dims; d += 2) {
        const float freq = 1.0f / powf(rope_freq_base, (float)d / (float)n_dims);
        const float theta_delta = (float)delta * freq * rope_freq_scale;

        const float cos_delta = cosf(theta_delta);
        const float sin_delta = sinf(theta_delta);

        // Rotate the pair (k_data[d], k_data[d+1])
        const float k0 = k_data[d];
        const float k1 = k_data[d + 1];

        k_data[d]     = k0 * cos_delta - k1 * sin_delta;
        k_data[d + 1] = k0 * sin_delta + k1 * cos_delta;
    }
}

//
// Context history
//

void llama_opt_context_history::record(const llama_token * tokens, uint32_t n_tokens) {
    // Swap curr → prev, then store new tokens as curr
    prev_.swap(curr_);
    curr_.assign(tokens, tokens + n_tokens);
    has_prev_ = true;
}

const llama_token * llama_opt_context_history::prev_tokens() const {
    return prev_.data();
}

uint32_t llama_opt_context_history::prev_n_tokens() const {
    return (uint32_t) prev_.size();
}

bool llama_opt_context_history::has_prev() const {
    return has_prev_ && !prev_.empty();
}

void llama_opt_context_history::clear() {
    prev_.clear();
    curr_.clear();
    has_prev_ = false;
}
