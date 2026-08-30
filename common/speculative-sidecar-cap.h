#pragma once

#include "speculative.h"

#include <algorithm>
#include <cstdint>

// A sidecar n-gram proposal is capped at the configured MTP width by default.
// A per-request speculative.n_max override remains authoritative and bypasses
// this internal sidecar safety cap.
struct common_speculative_sidecar_cap_config {
    int width = 0;
};

inline bool common_speculative_sidecar_cap_request_enabled(
        const common_speculative_sidecar_cap_config & config,
        const common_speculative_draft_params & dp) {
    return config.width > 0 && !dp.n_max_user_override;
}

inline int common_speculative_sidecar_cap_limit(
        const common_speculative_sidecar_cap_config & config,
        const common_speculative_draft_params & dp) {
    int limit = config.width;
    if (dp.n_max > 0) {
        limit = std::min(limit, dp.n_max);
    }
    return std::max(0, limit);
}

inline void common_speculative_sidecar_cap_trim(
        const common_speculative_sidecar_cap_config & config,
        const common_speculative_draft_params & dp,
        llama_tokens & result) {
    if (result.empty() || !common_speculative_sidecar_cap_request_enabled(config, dp)) {
        return;
    }

    const int limit = common_speculative_sidecar_cap_limit(config, dp);
    if (limit > 0 && (int) result.size() > limit) {
        result.resize((size_t) limit);
    }
}
