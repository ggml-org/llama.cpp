#pragma once

#include "kv-stream-span.h"
#include "../ggml-kv-stream.h"

#include <limits>
#include <initializer_list>

// One complete layer: resident prefix, ring suffix, and optional ring wrap. The caller retains the original grants.
struct ggml_cuda_fattn_tile_span_table {
    ggml_cuda_kv_span spans[3];
    int32_t count = 0, tokens = 0;
    bool contiguous_k = false, contiguous_v = false;
};

struct ggml_cuda_fattn_tile_span_workspace {
    size_t partial_offset = 0, meta_offset = 0, bytes = 0;
};

// Only the span table and stock split outputs use global scratch; conversion stays inside the shared tile.
static inline bool ggml_cuda_fattn_tile_span_workspace_make(
        size_t heads, size_t queries, size_t width, size_t splits, ggml_cuda_fattn_tile_span_workspace & output) {
    if (!heads || !queries || queries > 4 || !width || !splits) return false;
    size_t rows = heads;
    for (size_t extent : {queries,splits}) {
        if (extent > SIZE_MAX/rows) return false;
        rows *= extent;
    }
    if (width > SIZE_MAX/sizeof(float) || rows > SIZE_MAX/(width*sizeof(float)) || rows > SIZE_MAX/(2*sizeof(float))) return false;
    const size_t partial_bytes = splits == 1 ? 0 : rows*width*sizeof(float);
    const size_t meta_bytes = splits == 1 ? 0 : rows*2*sizeof(float);
    ggml_cuda_fattn_tile_span_workspace next;
    next.partial_offset = (sizeof(ggml_cuda_fattn_tile_span_table)+127)/128*128;
    if (partial_bytes > SIZE_MAX-next.partial_offset-127) return false;
    next.meta_offset = (next.partial_offset+partial_bytes+127)/128*128;
    if (meta_bytes > SIZE_MAX-next.meta_offset) return false;
    next.bytes = next.meta_offset+meta_bytes;
    output = next;
    return true;
}

static inline bool ggml_cuda_fattn_tile_type_supported(int32_t type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32 ||
        type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
        type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1;
}

// Translate retained grants only after checking exact coverage and both buffer bounds. Failure preserves output.
static inline bool ggml_cuda_fattn_tile_span_window_make(
        const ggml_kv_stream_span_plan_view & view, size_t first, size_t end, ggml_cuda_fattn_tile_span_table & output) {
    if (!view.spans || !view.count || view.count > 3 || !view.query_tokens || view.query_tokens > view.active_tokens ||
            view.active_tokens > INT32_MAX || !ggml_cuda_fattn_tile_type_supported(view.shape.type_k) ||
            !ggml_cuda_fattn_tile_type_supported(view.shape.type_v) || first >= end || end > view.active_tokens) return false;
    ggml_cuda_fattn_tile_span_table next{};
    next.count = int32_t(view.count); next.tokens = int32_t(view.active_tokens);
    // Canonical token-major views with multiple heads use stock's strided converter.
    next.contiguous_k = next.contiguous_v = view.shape.heads == 1;
    size_t next_token = first;
    for (size_t i = 0; i < view.count; ++i) {
        const auto & span = view.spans[i];
        if (span.token_begin != next_token || !span.tokens || span.tokens > end-next_token) return false;
        ggml_kv_stream_layout layout;
        if (ggml_kv_stream_layout_make(view.shape,span.tokens,layout).status != ggml_kv_stream_status::success ||
                layout.k_token_bytes > INT64_MAX || layout.v_token_bytes > INT64_MAX) return false;
        const char * pointers[2];
        size_t plane = 0;
        for (auto buffer : {span.k_buffer,span.v_buffer}) {
            const size_t offset = plane ? span.v_offset : span.k_offset;
            const size_t bytes = plane ? layout.v_bytes : layout.k_bytes;
            if (!buffer) return false;
            const size_t capacity = ggml_backend_buffer_get_size(buffer);
            const auto base = uintptr_t(ggml_backend_buffer_get_base(buffer));
            if (!base || offset > capacity || bytes > capacity-offset ||
                    offset > UINTPTR_MAX-base || bytes > UINTPTR_MAX-base-offset ||
                    offset%view.shape.alignment || (base+offset)%16) return false;
            pointers[plane++] = reinterpret_cast<const char *>(base+offset);
        }
        next.spans[i] = {pointers[0],pointers[1],int32_t(next_token),int32_t(span.tokens),
            int64_t(layout.k_token_bytes),int64_t(layout.v_token_bytes),int64_t(layout.k_row_bytes),int64_t(layout.v_row_bytes)};
        next_token += span.tokens;
    }
    if (next_token != end) return false;
    output = next;
    return true;
}

// The ordinary complete-layer adapter is the full logical window.
static inline bool ggml_cuda_fattn_tile_spans_make(
        const ggml_kv_stream_span_plan_view & view, ggml_cuda_fattn_tile_span_table & output) {
    return ggml_cuda_fattn_tile_span_window_make(view,0,view.active_tokens,output);
}
