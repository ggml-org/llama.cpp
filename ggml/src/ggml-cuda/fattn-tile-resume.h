#pragma once

#include "kv-stream-span.h"
#include <cstddef>
#include <initializer_list>

struct ggml_cuda_fattn_tile_span_table {
    ggml_cuda_kv_span spans[3];
    int32_t count = 0, tokens = 0;
    bool contiguous_k = false, contiguous_v = false;
};

struct ggml_cuda_fattn_tile_resume_geometry {
    uint32_t width = 0, query_heads = 0, kv_heads = 0, queries = 0;
    uint32_t ncols1 = 0, ncols2 = 0, nthreads = 0, nbatch_fa = 0, splits = 0, acc_bytes = 0;
};

struct ggml_cuda_fattn_tile_resume_control {
    char * state = nullptr;
    int32_t first = 0, end = 0;
    bool reset = false, finish = false;
};

// The leading table preserves the K-argument ABI; the caller retains scratch and source grants through completion.
struct ggml_cuda_fattn_tile_resume_descriptor {
    ggml_cuda_fattn_tile_span_table table;
    ggml_cuda_fattn_tile_resume_control control;
};
static_assert(offsetof(ggml_cuda_fattn_tile_resume_descriptor,table) == 0,"table must be first");

struct ggml_cuda_fattn_tile_resume_layout {
    ggml_cuda_fattn_tile_resume_geometry geometry;
    size_t tokens = 0, thread_bytes = 0, state_offset = 0, state_bytes = 0;
    size_t partial_offset = 0, meta_offset = 0, output_offset = 0, output_bytes = 0, bytes = 0;
};

// Size native per-thread bytes before reductions, stock split outputs, and a private final publication plane.
static inline bool ggml_cuda_fattn_tile_resume_layout_make(
        const ggml_cuda_fattn_tile_resume_geometry & g, size_t tokens, ggml_cuda_fattn_tile_resume_layout & output) {
    auto power_two = [](uint32_t v){return v && !(v&(v-1));};
    if ((g.width != 40 && g.width != 64 && g.width != 72 && g.width != 256 && g.width != 512) ||
            !g.query_heads || g.query_heads > 65535 || !g.kv_heads || g.query_heads%g.kv_heads ||
            !g.queries || g.queries > 8 || (g.ncols1 != 1 && g.ncols1 != 2 && g.ncols1 != 4 && g.ncols1 != 8) ||
            !power_two(g.ncols2) || g.ncols2 > 32 || (g.query_heads/g.kv_heads)%g.ncols2 ||
            (g.nthreads != 64 && g.nthreads != 128 && g.nthreads != 256) ||
            !power_two(g.nbatch_fa) || g.nbatch_fa > 256 || !g.splits || g.splits > 65535 ||
            (g.acc_bytes != 4 && g.acc_bytes != 8) || tokens < g.queries || tokens > INT32_MAX) return false;
    const size_t columns = size_t(g.ncols1)*g.ncols2, warps = g.nthreads/32;
    if (columns < 2 || columns > 32 || (columns >= warps ? columns%warps : warps%columns)) return false;
    const size_t cpw = columns > warps ? columns/warps : 1, np = warps > columns ? warps/columns : 1;
    if (g.nbatch_fa%(np*32) || (g.ncols2 > 1 && tokens%g.nbatch_fa)) return false;
    ggml_cuda_fattn_tile_resume_layout next;
    next.geometry = g; next.tokens = tokens;
    next.thread_bytes = cpw*(8+((g.width+63)/64)*g.acc_bytes);
    size_t threads = (size_t(g.queries)+g.ncols1-1)/g.ncols1;
    for (size_t extent : {size_t(g.query_heads/g.ncols2),size_t(g.splits),size_t(g.nthreads),next.thread_bytes}) {
        if (extent > SIZE_MAX/threads) return false;
        threads *= extent;
    }
    next.state_bytes = threads;
    next.state_offset = (sizeof(ggml_cuda_fattn_tile_resume_descriptor)+127)/128*128;
    auto append = [](size_t & cursor, size_t bytes, size_t & offset) {
        if (cursor > SIZE_MAX-127) return false;
        offset = (cursor+127)/128*128;
        if (bytes > SIZE_MAX-offset) return false;
        cursor = offset+bytes; return true;
    };
    size_t cursor = next.state_offset;
    if (!append(cursor,next.state_bytes,next.state_offset)) return false;
    size_t rows = size_t(g.query_heads)*g.queries;
    if (rows > SIZE_MAX/(size_t(g.width)*sizeof(float))) return false;
    next.output_bytes = rows*g.width*sizeof(float);
    if (next.output_bytes > SIZE_MAX/g.splits || rows > SIZE_MAX/g.splits/8) return false;
    if (!append(cursor,next.output_bytes*g.splits,next.partial_offset) ||
            !append(cursor,rows*g.splits*8,next.meta_offset) ||
            !append(cursor,next.output_bytes,next.output_offset)) return false;
    next.bytes = cursor;
    output = next;
    return true;
}

// Physical allocation and grant checks belong to the owner; this validates the low-level wave and scratch metadata.
static inline bool ggml_cuda_fattn_tile_resume_wave_make(
        const ggml_cuda_fattn_tile_span_table & table, const ggml_cuda_fattn_tile_resume_layout & layout,
        void * workspace, size_t capacity, int64_t k_row_bytes, int64_t v_row_bytes, size_t first, size_t end,
        ggml_cuda_fattn_tile_resume_descriptor & output, bool allow_masked_padding = false) {
    ggml_cuda_fattn_tile_resume_layout checked;
    if (!ggml_cuda_fattn_tile_resume_layout_make(layout.geometry,layout.tokens,checked) ||
            checked.thread_bytes != layout.thread_bytes || checked.state_bytes != layout.state_bytes ||
            checked.state_offset != layout.state_offset || checked.partial_offset != layout.partial_offset ||
            checked.meta_offset != layout.meta_offset || checked.output_offset != layout.output_offset ||
            checked.output_bytes != layout.output_bytes || checked.bytes != layout.bytes ||
            !workspace || capacity < layout.bytes || table.count < 1 || table.count > 3 ||
            table.tokens < int32_t(layout.geometry.queries) || size_t(table.tokens) > layout.tokens ||
            (size_t(table.tokens) != layout.tokens && (!allow_masked_padding ||
                layout.tokens != (size_t(table.tokens)+255)/256*256 || first >= size_t(table.tokens) ||
                (end > size_t(table.tokens) && end != layout.tokens))) ||
            first >= end || end > layout.tokens || first%layout.geometry.nbatch_fa ||
            (end != layout.tokens && end%layout.geometry.nbatch_fa)) return false;
    const auto base = reinterpret_cast<uintptr_t>(workspace);
    if (base%16 || layout.bytes > UINTPTR_MAX-base) return false;
    auto next = ggml_cuda_fattn_tile_resume_descriptor{};
    next.table = table;
    const size_t physical_end = end < size_t(table.tokens) ? end : size_t(table.tokens);
    for (int i = 0; i < table.count; ++i) {
        if (next.table.spans[i].first < int32_t(first)) return false;
        next.table.spans[i].first -= int32_t(first);
    }
    if (!ggml_cuda_kv_spans_validate(next.table.spans,size_t(table.count),physical_end-first,
            layout.geometry.kv_heads,k_row_bytes,v_row_bytes)) return false;
    for (int i = 0; i < table.count; ++i) {
        const auto & span = table.spans[i];
        const uintptr_t addresses[2] = {uintptr_t(span.k),uintptr_t(span.v)};
        const int64_t strides[2] = {span.k_token_stride,span.v_token_stride};
        const int64_t heads[2] = {span.k_head_stride,span.v_head_stride};
        const int64_t rows[2] = {k_row_bytes,v_row_bytes};
        for (int plane = 0; plane < 2; ++plane) {
            const size_t bytes = size_t((span.tokens-1)*strides[plane]+(layout.geometry.kv_heads-1)*heads[plane]+rows[plane]);
            if (addresses[plane] < base+layout.bytes && base < addresses[plane]+bytes) return false;
        }
    }
    next.table = table;
    next.table.tokens = int32_t(layout.tokens);
    next.control = {reinterpret_cast<char *>(workspace)+layout.state_offset,int32_t(first),int32_t(end),first == 0,end == layout.tokens};
    output = next;
    return true;
}

// Host submission bookkeeping only; complete() is called after the caller's real read-completion fence.
class ggml_cuda_fattn_tile_resume_cursor {
    size_t tokens_, tile_, next_ = 0, end_ = 0;
    uint64_t generation_;
    bool pending_ = false, cancelled_ = false, published_ = false;
public:
    ggml_cuda_fattn_tile_resume_cursor(size_t tokens,size_t tile, uint64_t generation = 0) : tokens_(tokens),tile_(tile),generation_(generation) {
        cancelled_ = !tokens || tokens > INT32_MAX || !tile || (tile&(tile-1));
    }
    size_t next() const {return next_;}
    bool can_reuse() const {return !pending_;}
    bool ready_to_publish() const {return !cancelled_ && !pending_ && !published_ && next_ == tokens_;}
    // Reserve one ordered wave without advancing the committed read cursor.
    bool begin(size_t end) {return begin(end,generation_);}
    bool begin(size_t end, uint64_t generation) {
        if (generation != generation_ || !tile_ || pending_ || cancelled_ || published_ || next_ == tokens_ || end <= next_ || end > tokens_ ||
                next_%tile_ || (end != tokens_ && end%tile_)) return false;
        end_ = end; pending_ = true; return true;
    }
    // Cancellation stops future work but does not release an in-flight source slot.
    void cancel() {cancelled_ = true;}
    bool complete(bool success) {return complete(success,generation_);}
    bool complete(bool success, uint64_t generation) {
        if (generation != generation_ || !pending_) return false;
        pending_ = false;
        if (!success) cancelled_ = true;
        if (cancelled_) return false;
        next_ = end_; return true;
    }
    // This gates the caller's public copy; failed publication cannot be retried against possibly changed state.
    bool publish(bool success) {return publish(success,generation_);}
    bool publish(bool success, uint64_t generation) {
        if (generation != generation_ || !ready_to_publish()) return false;
        if (!success) {cancelled_ = true; return false;}
        published_ = true; return true;
    }
};
