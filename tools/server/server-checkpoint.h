#pragma once

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <list>

namespace server_checkpoint {

static inline int32_t ladder_min_step(int64_t n_tokens, int32_t min_step) {
    if (min_step <= 0) {
        return 0;
    }

    int64_t step = min_step;
    int64_t span = 4096;

    while (n_tokens >= 2*span && step < 32768) {
        step *= 2;
        span *= 2;
    }

    return (int32_t) std::min<int64_t>(step, 32768);
}

static inline bool should_create_mid_prompt_checkpoint(
        int64_t n_tokens_start,
        int64_t n_tokens_total,
        bool    near_prompt_end) {
    return n_tokens_start > 0 && n_tokens_start < n_tokens_total && !near_prompt_end;
}

static inline std::list<common_prompt_checkpoint>::iterator find_redundant_checkpoint(
        std::list<common_prompt_checkpoint> & checkpoints) {
    GGML_ASSERT(!checkpoints.empty());

    if (checkpoints.size() <= 2) {
        return checkpoints.begin();
    }

    auto best_it  = std::next(checkpoints.begin());
    auto best_gap = std::numeric_limits<int64_t>::max();

    for (auto it = std::next(checkpoints.begin()); std::next(it) != checkpoints.end(); ++it) {
        const auto prev = std::prev(it);
        const auto next = std::next(it);

        const int64_t gap = next->n_tokens - prev->n_tokens;
        if (gap < best_gap) {
            best_it  = it;
            best_gap = gap;
        }
    }

    return best_it;
}

struct prune_result {
    size_t n_pruned = 0;
    size_t size_pruned = 0;
};

static inline size_t checkpoint_size(const std::list<common_prompt_checkpoint> & checkpoints) {
    size_t res = 0;

    for (const auto & ckpt : checkpoints) {
        res += ckpt.size();
    }

    return res;
}

static inline prune_result prune_checkpoints_to_limit(
        std::list<common_prompt_checkpoint> & checkpoints,
        size_t max_size) {
    prune_result res;

    size_t cur_size = checkpoint_size(checkpoints);

    while (!checkpoints.empty() && cur_size > max_size) {
        auto it = find_redundant_checkpoint(checkpoints);
        const size_t size_pruned = it->size();

        checkpoints.erase(it);
        cur_size -= size_pruned;

        ++res.n_pruned;
        res.size_pruned += size_pruned;
    }

    return res;
}

} // namespace server_checkpoint
