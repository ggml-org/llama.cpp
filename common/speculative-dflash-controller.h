#pragma once

#include <algorithm>
#include <cstdint>

enum class common_speculative_dflash_controller_mode {
    OFF,
    TRACE,
    BATCH,
};

struct common_speculative_dflash_controller_config {
    common_speculative_dflash_controller_mode mode = common_speculative_dflash_controller_mode::OFF;
    int32_t max_depth = 4;
};

struct common_speculative_dflash_controller_decision {
    int32_t input_depth = 0;
    int32_t depth = 0;
    bool limited_by_batch = false;
};

inline int32_t common_speculative_dflash_batch_depth_cap(int32_t active_batch, int32_t max_depth) {
    if (max_depth <= 1 || active_batch <= 1) {
        return std::max(1, max_depth);
    }
    // Measured gfx1030 TP4 sidecar schedule. K4V-hit sequences have already
    // left the neural drafting set before active_batch is counted. Only the
    // qualified active-two workload is shortened.
    if (active_batch == 2) {
        return std::min(2, max_depth);
    }
    return max_depth;
}

inline common_speculative_dflash_controller_decision common_speculative_dflash_controller_select(
        const common_speculative_dflash_controller_config & config,
        int32_t active_batch,
        int32_t requested_depth) {
    common_speculative_dflash_controller_decision result;
    result.input_depth = std::max(0, std::min(config.max_depth, requested_depth));
    result.depth = result.input_depth;
    if (result.input_depth > 0) {
        result.depth = common_speculative_dflash_batch_depth_cap(active_batch, result.input_depth);
        result.limited_by_batch = result.depth < result.input_depth;
    }
    return result;
}

inline int32_t common_speculative_dflash_controller_pre_draft_cap(
        const common_speculative_dflash_controller_config & config,
        int32_t active_batch,
        int32_t requested_depth,
        bool request_override) {
    if (config.mode != common_speculative_dflash_controller_mode::BATCH ||
            request_override || requested_depth <= 0) {
        return requested_depth;
    }
    return common_speculative_dflash_controller_select(config, active_batch, requested_depth).depth;
}
