#pragma once

#include <cstddef>
#include <cstdint>
#include "kv-stream-span.h"

struct ggml_cuda_fattn_mma_resume_control {
    char * state = nullptr;
    int32_t first = 0, end = 0;
    const uint32_t * block_ids = nullptr;
    uint32_t native_blocks = 0;
    const uint32_t * state_ids = nullptr;
    bool state_per_block = false;
    const int32_t * mask_bounds = nullptr;
};

struct ggml_cuda_fattn_mma_resume_layout {
    size_t slots = 0, thread_bytes = 0, bytes = 0, map_entries = 0;
    bool block_indexed = false;
};

// Validate the bounded native linear schedule before enumerating its checkpoint intersections.
static inline bool ggml_cuda_fattn_mma_resume_schedule_valid(size_t destinations, size_t blocks, size_t tiles) {
    return destinations && tiles && destinations <= size_t(INT32_MAX)/tiles &&
        blocks && blocks <= destinations*tiles;
}

// Count only intersections that can span two legal waves; singleton tiles never need a checkpoint.
static inline size_t ggml_cuda_fattn_mma_resume_state_slots(size_t destinations, size_t blocks, size_t tiles) {
    size_t slots=0;
    const uint64_t total=destinations*tiles;
    for (size_t block=0;block<blocks;++block) {
        const size_t begin=block*total/blocks, stop=(block+1)*total/blocks;
        for (size_t d=begin/tiles;d<=(stop-1)/tiles;++d) {
            const size_t first=begin > d*tiles ? begin : d*tiles;
            const size_t end=stop < (d+1)*tiles ? stop : (d+1)*tiles;
            slots+=end-first > 1;
        }
    }
    return slots;
}

// Build a dense checkpoint-ID table without changing the native split or reduction IDs.
// A short table or invalid schedule is rejected before any caller storage is modified.
static inline bool ggml_cuda_fattn_mma_resume_state_map(
        size_t destinations, size_t blocks, size_t tiles, uint32_t * ids, size_t entries) {
    if (!ggml_cuda_fattn_mma_resume_schedule_valid(destinations,blocks,tiles) || !ids ||
            entries != destinations+blocks-1) return false;
    for (size_t i=0;i<entries;++i) ids[i]=UINT32_MAX;
    uint32_t slot=0;
    const uint64_t total=destinations*tiles;
    for (size_t block=0;block<blocks;++block) {
        const size_t begin=block*total/blocks, stop=(block+1)*total/blocks;
        for (size_t d=begin/tiles;d<=(stop-1)/tiles;++d) {
            const size_t first=begin > d*tiles ? begin : d*tiles;
            const size_t end=stop < (d+1)*tiles ? stop : (d+1)*tiles;
            if (end-first > 1) ids[d+block]=slot++;
        }
    }
    return true;
}

// Intersect a wave with the original linear CTA interval, including both seams and any complete middle tile.
static inline bool ggml_cuda_fattn_mma_resume_block_active(
        int64_t destinations, int64_t tiles, int64_t blocks, int64_t block, int64_t first, int64_t end) {
    if (destinations <= 0 || tiles <= 0 || destinations > INT32_MAX/tiles || blocks <= 0 ||
            blocks > destinations*tiles || block < 0 || block >= blocks || first < 0 || first >= end || end > tiles) return false;
    const int64_t total=destinations*tiles;
    const int64_t begin=block*total/blocks, stop=(block+1)*total/blocks;
    const int64_t d0=begin/tiles, d1=(stop-1)/tiles;
    if (d1 > d0+1) return true;
    if (d0 == d1) return begin%tiles < end && first < (stop-1)%tiles+1;
    return begin%tiles < end || first < (stop-1)%tiles+1;
}

// A native CTA/output-tile intersection owns slot (output_tile + CTA), including seam fragments.
static inline bool ggml_cuda_fattn_mma_resume_layout_make(
        size_t destinations, size_t blocks, uint32_t threads, uint32_t columns, uint32_t groups,
        ggml_cuda_fattn_mma_resume_layout & output, size_t kv_tiles = 0, bool compact = false) {
    if (!destinations || !blocks || destinations > SIZE_MAX-(blocks-1) ||
            !threads || threads%32 || threads > 1024 ||
            (columns != 1 && columns != 2 && columns != 4 && columns != 8) ||
            (groups != 2 && groups != 8) || (columns < 4 && groups != 8) ||
            (kv_tiles && destinations > SIZE_MAX/kv_tiles) ||
            (compact && !ggml_cuda_fattn_mma_resume_schedule_valid(destinations,blocks,kv_tiles))) return false;
    ggml_cuda_fattn_mma_resume_layout next;
    next.slots=destinations+blocks-1;
    // Head-256 Turing MMA fragments hold 128 or 256 bytes, plus two FP32 max/sum pairs.
    next.thread_bytes=columns*groups <= 16 ? 144 : 272;
    if (compact) {
        next.slots=ggml_cuda_fattn_mma_resume_state_slots(destinations,blocks,kv_tiles);
        // Uniform native splits stay inside one output tile and all have at least two K tiles.
        // Their original CTA ID is already a dense checkpoint ID, with no device lookup needed.
        next.block_indexed=blocks%destinations == 0 && destinations*kv_tiles/blocks >= 2;
        if (next.slots && next.slots != destinations+blocks-1 && !next.block_indexed)
            next.map_entries=destinations+blocks-1;
    }
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
