#pragma once

#include <cstddef>
#include <cstdint>
#include "kv-stream-span.h"

struct ggml_cuda_fattn_mma_resume_control {
    char * state = nullptr;
    int32_t first = 0, end = 0;
};

struct ggml_cuda_fattn_mma_resume_layout {
    size_t slots = 0, thread_bytes = 0, bytes = 0;
};

// A native CTA/output-tile intersection owns slot (output_tile + CTA), including seam fragments.
static inline bool ggml_cuda_fattn_mma_resume_layout_make(
        size_t destinations, size_t blocks, uint32_t threads, uint32_t columns, uint32_t groups,
        ggml_cuda_fattn_mma_resume_layout & output, size_t kv_tiles = 0) {
    if (!destinations || !blocks || destinations > SIZE_MAX-(blocks-1) ||
            !threads || threads%32 || threads > 1024 ||
            (columns != 1 && columns != 2 && columns != 4 && columns != 8) ||
            (groups != 2 && groups != 8) || (columns < 4 && groups != 8) ||
            (kv_tiles && destinations > SIZE_MAX/kv_tiles)) return false;
    ggml_cuda_fattn_mma_resume_layout next;
    next.slots=destinations+blocks-1;
    // Head-256 Turing MMA fragments hold 128 or 256 bytes, plus two FP32 max/sum pairs.
    next.thread_bytes=columns*groups <= 16 ? 144 : 272;
    // One native K tile per CTA cannot cross a legal complete-tile wave boundary.
    if (kv_tiles && blocks == destinations*kv_tiles) {
        next.slots=0;
        output=next;
        return true;
    }
    if (next.slots > SIZE_MAX/threads || next.slots*threads > SIZE_MAX/next.thread_bytes) return false;
    next.bytes=next.slots*threads*next.thread_bytes;
    output=next;
    return true;
}

// Require complete native tiles and exact physical coverage, excluding the masked final padding.
static inline bool ggml_cuda_fattn_mma_resume_wave_validate(
        const ggml_cuda_kv_span * spans, size_t count, size_t active, size_t capacity, size_t tile,
        size_t first, size_t end, size_t heads, size_t key_row, size_t value_row) {
    if (!spans || !count || count > 3 || !active || active > INT32_MAX || capacity > INT32_MAX ||
            (active+255)/256*256 != capacity || !tile || tile > 256 || (tile&(tile-1)) ||
            first >= end || end > capacity || first >= active || first%tile || end%tile ||
            (end > active && end != capacity) || heads > INT32_MAX || key_row > INT32_MAX || value_row > INT32_MAX) return false;
    ggml_cuda_kv_span relative[3];
    for (size_t i=0;i<count;++i) {
        if (spans[i].first < int64_t(first)) return false;
        relative[i]=spans[i]; relative[i].first-=int32_t(first);
    }
    return ggml_cuda_kv_spans_validate(relative,count,(end < active ? end : active)-first,
        int64_t(heads),int64_t(key_row),int64_t(value_row));
}
