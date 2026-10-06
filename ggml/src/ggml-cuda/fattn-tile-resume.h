#pragma once

#include "fattn-tile-spans.h"
#include <cstddef>
#include <algorithm>
#include <utility>

struct ggml_cuda_fattn_tile_resume_geometry {
    uint32_t width = 0, query_heads = 0, kv_heads = 0, queries = 0;
    uint32_t ncols1 = 0, ncols2 = 0, nthreads = 0, nbatch_fa = 0, splits = 0, acc_bytes = 0;
};

struct ggml_cuda_fattn_tile_resume_control {
    char * state = nullptr;
    int32_t first = 0, end = 0;
    bool reset = false, finish = false;
};

// The leading table keeps C4b's K-argument ABI; the caller retains scratch and source grants through completion.
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
    if ((g.width != 40 && g.width != 64 && g.width != 72 && g.width != 256) ||
            !g.query_heads || g.query_heads > 65535 || !g.kv_heads || g.query_heads%g.kv_heads ||
            !g.queries || g.queries > 4 || (g.ncols1 != 1 && g.ncols1 != 2 && g.ncols1 != 4) ||
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

// Reject stale sizes, incomplete windows, overlapping scratch and non-tile wave boundaries before submitting work.
static inline bool ggml_cuda_fattn_tile_resume_wave_make(
        const ggml_kv_stream_span_plan_view & view, const ggml_cuda_fattn_tile_resume_layout & layout,
        ggml_backend_buffer_t workspace, size_t first, size_t end, ggml_cuda_fattn_tile_resume_descriptor & output,
        bool allow_masked_padding = false) {
    ggml_cuda_fattn_tile_resume_layout checked;
    if (!ggml_cuda_fattn_tile_resume_layout_make(layout.geometry,layout.tokens,checked) ||
            checked.thread_bytes != layout.thread_bytes || checked.state_bytes != layout.state_bytes ||
            checked.state_offset != layout.state_offset || checked.partial_offset != layout.partial_offset ||
            checked.meta_offset != layout.meta_offset || checked.output_offset != layout.output_offset ||
            checked.output_bytes != layout.output_bytes || checked.bytes != layout.bytes ||
            !workspace || ggml_backend_buffer_get_size(workspace) < layout.bytes ||
            (view.active_tokens != layout.tokens && (!allow_masked_padding ||
                view.active_tokens > layout.tokens || layout.tokens != (view.active_tokens+255)/256*256 ||
                end != layout.tokens || first >= view.active_tokens)) || view.query_tokens != layout.geometry.queries ||
            view.shape.heads != layout.geometry.kv_heads || view.shape.head_dim_k != layout.geometry.width ||
            view.shape.head_dim_v != layout.geometry.width || first >= end || end > layout.tokens ||
            first%layout.geometry.nbatch_fa || (end != layout.tokens && end%layout.geometry.nbatch_fa)) return false;
    const auto base = uintptr_t(ggml_backend_buffer_get_base(workspace));
    if (!base || base%16 || layout.bytes > UINTPTR_MAX-base) return false;
    ggml_cuda_fattn_tile_resume_descriptor next;
    if (!ggml_cuda_fattn_tile_span_window_make(view,first,std::min(end,view.active_tokens),next.table)) return false;
    // The caller's causal mask hides the absent padded tail; no address is formed outside a retained span.
    next.table.tokens = int32_t(layout.tokens);
    for (int i = 0; i < next.table.count; ++i) {
        const auto & span = next.table.spans[i];
        for (auto range : {std::pair{uintptr_t(span.k),size_t(span.tokens)*size_t(span.k_token_stride)},
                std::pair{uintptr_t(span.v),size_t(span.tokens)*size_t(span.v_token_stride)}})
            if (range.first < base+layout.bytes && base < range.first+range.second) return false;
    }
    next.control = {reinterpret_cast<char *>(base+layout.state_offset),int32_t(first),int32_t(end),first == 0,end == layout.tokens};
    output = next;
    return true;
}

// Host submission bookkeeping only; complete() is called after the caller's real read-completion fence.
class ggml_cuda_fattn_tile_resume_cursor {
    size_t tokens_, tile_, next_ = 0, end_ = 0;
    bool pending_ = false, cancelled_ = false, published_ = false;
public:
    ggml_cuda_fattn_tile_resume_cursor(size_t tokens,size_t tile) : tokens_(tokens),tile_(tile) {
        cancelled_ = !tokens || tokens > INT32_MAX || !tile || (tile&(tile-1));
    }
    size_t next() const {return next_;}
    bool can_reuse() const {return !pending_;}
    bool ready_to_publish() const {return !cancelled_ && !pending_ && !published_ && next_ == tokens_;}
    // Reserve one ordered wave without advancing the committed read cursor.
    bool begin(size_t end) {
        if (!tile_ || pending_ || cancelled_ || published_ || next_ == tokens_ || end <= next_ || end > tokens_ ||
                next_%tile_ || (end != tokens_ && end%tile_)) return false;
        end_ = end; pending_ = true; return true;
    }
    // Cancellation stops future work but does not release an in-flight source slot.
    void cancel() {cancelled_ = true;}
    bool complete(bool success) {
        if (!pending_) return false;
        pending_ = false;
        if (!success) cancelled_ = true;
        if (cancelled_) return false;
        next_ = end_; return true;
    }
    // This gates the caller's public copy; failed publication cannot be retried against possibly changed state.
    bool publish(bool success) {
        if (!ready_to_publish()) return false;
        if (!success) {cancelled_ = true; return false;}
        published_ = true; return true;
    }
};
