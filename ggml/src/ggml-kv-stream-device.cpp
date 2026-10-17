#include "ggml-kv-stream-device.h"

bool ggml_kv_stream_resume_layout_make(
        uint32_t heads, uint32_t queries, uint32_t splits, uint32_t values,
        ggml_kv_stream_resume_plan & output, uint32_t width) {
    if (!heads || !queries || queries > 2 || !splits || (values != 8 && values != 16 && values != 32) ||
            !width || width > INT32_MAX ||
            size_t(heads) > SIZE_MAX/queries ||
            size_t(heads)*queries > SIZE_MAX/splits) return false;
    const size_t rows = size_t(heads)*queries*splits;
    const size_t stride = 128*(2+size_t(values))*sizeof(float);
    if (rows > (SIZE_MAX-127)/stride) return false;
    ggml_kv_stream_resume_plan next;
    next.heads = heads; next.queries = queries; next.splits = splits; next.values_per_thread = values;
    next.head_dim_k = next.head_dim_v = width;
    next.state_bytes = rows*stride;
    next.partial_offset = (next.state_bytes+127)/128*128;
    if (size_t(width) > SIZE_MAX/sizeof(float) ||
            rows > (SIZE_MAX-next.partial_offset-127)/(size_t(width)*sizeof(float))) return false;
    next.meta_offset = (next.partial_offset+rows*size_t(width)*sizeof(float)+127)/128*128;
    if (rows > (SIZE_MAX-next.meta_offset)/(2*sizeof(float))) return false;
    next.bytes = next.meta_offset+rows*2*sizeof(float);
    output = next; return true;
}

// Quoting must not narrow an invalid dimension or infer support from the byte size alone.
static bool request_valid(const ggml_kv_stream_attention_request & request) {
    const auto & shape=request.shape;
    ggml_kv_stream_layout layout;
    return request.query_heads && request.query_heads <= INT32_MAX && request.query_tokens &&
        request.query_tokens <= INT32_MAX && request.tokens >= request.query_tokens && request.tokens <= INT64_MAX &&
        shape.head_dim_k > 0 && shape.head_dim_k <= INT32_MAX &&
        shape.head_dim_v > 0 && shape.head_dim_v <= INT32_MAX && shape.heads > 0 && shape.heads <= INT32_MAX &&
        request.query_heads%shape.heads == 0 && request.spans &&
        ggml_kv_stream_layout_make(shape,request.tokens,layout).status == ggml_kv_stream_status::success;
}

static bool legacy_geometry(const ggml_kv_stream_attention_request & request) {
    return request.shape.head_dim_k == 256 && request.shape.head_dim_v == 256 &&
        request.shape.page_tokens == 256 && request.shape.alignment == 128;
}

bool ggml_kv_stream_resume_plan_query(const ggml_kv_stream_partial_ops * ops, ggml_backend_t backend,
        const ggml_kv_stream_attention_request & request, ggml_kv_stream_resume_plan & output) {
    if (!ops || !request_valid(request)) return false;
    ggml_kv_stream_resume_plan next;
    if (ops->version >= 15 && ops->geometry && ops->geometry->resume_plan) {
        if (!ops->geometry->resume_plan(backend,request,next) ||
                next.head_dim_k != request.shape.head_dim_k || next.head_dim_v != request.shape.head_dim_v) return false;
    } else {
        if (ops->version < 5 || !ops->resume_plan || !legacy_geometry(request) ||
                !ops->resume_plan(backend,request.shape.type_k,request.shape.type_v,
                    request.query_heads,uint32_t(request.shape.heads),request.query_tokens,request.tokens,next)) return false;
    }
    output=next;
    return true;
}

bool ggml_kv_stream_workspace_query(const ggml_kv_stream_partial_ops * ops, ggml_backend_t backend,
        const ggml_kv_stream_attention_request & request, ggml_kv_stream_workspace_kind kind, size_t & output) {
    if (!ops || !request_valid(request)) return false;
    size_t bytes=0;
    bool valid=false;
    switch (kind) {
        case ggml_kv_stream_workspace_kind::decode:
        case ggml_kv_stream_workspace_kind::prefill:
        case ggml_kv_stream_workspace_kind::mma:
        case ggml_kv_stream_workspace_kind::prefill_mask_bounds: break;
        default: return false;
    }
    if (ops->version >= 15 && ops->geometry && ops->geometry->workspace) {
        valid=ops->geometry->workspace(backend,request,kind,bytes);
    } else {
        if (!legacy_geometry(request)) return false;
        const auto & shape=request.shape;
        switch (kind) {
            case ggml_kv_stream_workspace_kind::decode:
                valid=ops->version >= 10 && ops->decode_workspace && ops->decode_workspace(backend,
                    shape.type_k,shape.type_v,request.query_heads,uint32_t(shape.heads),request.query_tokens,request.tokens,bytes);
                break;
            case ggml_kv_stream_workspace_kind::prefill:
                valid=ops->version >= 12 && ops->prefill_workspace && ops->prefill_workspace(backend,
                    shape.type_k,shape.type_v,request.query_heads,uint32_t(shape.heads),request.query_tokens,request.tokens,bytes);
                break;
            case ggml_kv_stream_workspace_kind::mma:
                valid=ops->version >= 9 && ops->mma_workspace && ops->mma_workspace(backend,
                    shape.type_k,shape.type_v,request.query_heads,uint32_t(shape.heads),request.tokens,request.spans,bytes);
                break;
            case ggml_kv_stream_workspace_kind::prefill_mask_bounds:
                valid=ops->version >= 14 && ops->prefill_mask_bounds && ops->prefill_mask_bounds(backend,
                    shape.type_k,shape.type_v,request.query_heads,uint32_t(shape.heads),request.query_tokens,request.tokens);
                break;
            default: return false;
        }
    }
    if (!valid) return false;
    output=bytes;
    return true;
}

// Reject the entire shape before publishing offsets, including overflow beyond the selected tile.
bool ggml_kv_stream_query_tile_make(size_t queries, size_t heads, size_t first, ggml_kv_stream_query_tile & output) {
    if (!heads || first >= queries || queries > SIZE_MAX/heads) return false;
    const size_t count = queries-first < 256 ? queries-first : 256;
    output = {count,first*heads,count*heads};
    return true;
}

// Keep both exports and the normalized staging plane aligned; publish only a complete layout.
ggml_kv_stream_partial_result ggml_kv_stream_block_layout_make(size_t rows, size_t width, ggml_kv_stream_block_layout & output) {
    ggml_kv_stream_block_layout next;
    auto result = ggml_kv_stream_partial_layout_make(rows, 2, width, 128, next.partial);
    if (result.status != ggml_kv_stream_partial_status::success) return result;
    const auto overflow = ggml_kv_stream_partial_result{ggml_kv_stream_partial_status::overflow};
    if (next.partial.bytes > SIZE_MAX - 127) return overflow;
    next.second_offset = (next.partial.bytes + 127)/128*128;
    if (next.second_offset > SIZE_MAX - next.partial.bytes || next.second_offset + next.partial.bytes > SIZE_MAX - 127) return overflow;
    next.value_offset = (next.second_offset + next.partial.bytes + 127)/128*128;
    next.value_bytes = next.partial.numerator_bytes/2;
    if (next.value_offset > SIZE_MAX - next.value_bytes) return overflow;
    next.status_offset = next.value_offset + next.value_bytes;
    if (next.status_offset > SIZE_MAX - sizeof(uint32_t)) return overflow;
    next.bytes = next.status_offset + sizeof(uint32_t);
    output = next;
    return {};
}
