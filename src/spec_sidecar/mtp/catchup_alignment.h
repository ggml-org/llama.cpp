#pragma once

#include <cstdint>

namespace spec_sidecar_mtp {

// Native MTP shifts target hidden rows right by one: token row p consumes
// target hidden row p - 1. A negative source row denotes the previously
// committed hidden state, or the all-zero initial state at BOS.
constexpr int32_t hidden_source_row(int32_t token_batch_row) {
    return token_batch_row - 1;
}

// A non-BOS catch-up requires the hidden row retained at the committed tip.
// If a restore or truncate invalidated it, fail closed instead of constructing
// the first MTP row with an incorrect all-zero hidden state.
constexpr bool can_begin_catchup(int32_t state_pos_max, bool committed_hidden_valid) {
    return state_pos_max == 0 || committed_hidden_valid;
}

// The retained hidden row remains valid only while the committed tip stays at
// the same position. Moving the tip requires a matching historical hidden row.
constexpr bool committed_hidden_matches_tip(int32_t current_pos_max, int32_t restored_pos_max) {
    return current_pos_max == restored_pos_max;
}

// Draft kernels write KV ahead of the committed cursor. Capacity checks must
// include that entire lookahead, not just the latest catch-up row.
constexpr int64_t draft_storage_required(int32_t n_past, int32_t n_draft) {
    return (int64_t) n_past + (int64_t) n_draft;
}

} // namespace spec_sidecar_mtp
