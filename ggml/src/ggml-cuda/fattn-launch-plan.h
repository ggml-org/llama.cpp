#pragma once

#include <cstdint>
#include <limits>

struct ggml_cuda_fattn_stream_k_plan {
    int32_t blocks = 0;
    bool use_stream_k = false;
    bool needs_fixup = false;
};

// Plan stock Stream-K partitioning without allocating device storage. Failure preserves output.
static inline bool ggml_cuda_fattn_stream_k_plan_make(
        int64_t ntiles_dst, int64_t ntiles_kv, int64_t max_blocks,
        bool force_stream_k, bool prefer_whole_tiles,
        ggml_cuda_fattn_stream_k_plan & output) {
    const int64_t limit = std::numeric_limits<int32_t>::max();
    if (ntiles_dst <= 0 || ntiles_dst > limit || ntiles_kv <= 0 || ntiles_kv > limit ||
            max_blocks <= 0 || max_blocks > limit || ntiles_kv > limit/ntiles_dst) {
        return false;
    }

    const int64_t waves = (ntiles_dst + max_blocks - 1)/max_blocks;
    const int64_t efficiency = 100*ntiles_dst/(max_blocks*waves);
    const bool use_stream_k = !(prefer_whole_tiles && efficiency >= 75) &&
        (force_stream_k || efficiency < 75);
    int64_t blocks = ntiles_dst;
    if (use_stream_k) {
        const int64_t total_tiles = ntiles_kv*ntiles_dst;
        const int64_t raw = max_blocks < total_tiles ? max_blocks : total_tiles;
        const int64_t rounded = (raw/ntiles_dst)*ntiles_dst;
        const int64_t loss = rounded > 0 ? 100*(raw-rounded)/raw : 100;
        blocks = loss <= 5 ? rounded : raw;
    }

    output = {int32_t(blocks), use_stream_k, ntiles_dst%blocks != 0};
    return true;
}
