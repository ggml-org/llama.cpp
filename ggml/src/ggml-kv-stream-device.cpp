#include "ggml-kv-stream-device.h"

bool ggml_kv_stream_resume_layout_make(
        uint32_t heads, uint32_t queries, uint32_t splits, uint32_t values,
        ggml_kv_stream_resume_plan & output) {
    if (!heads || !queries || queries > 2 || !splits || (values != 8 && values != 32) ||
            size_t(heads) > SIZE_MAX/queries ||
            size_t(heads)*queries > SIZE_MAX/splits) return false;
    const size_t rows = size_t(heads)*queries*splits;
    const size_t stride = 128*(2+size_t(values))*sizeof(float);
    if (rows > (SIZE_MAX-127)/stride) return false;
    ggml_kv_stream_resume_plan next;
    next.heads = heads; next.queries = queries; next.splits = splits; next.values_per_thread = values;
    next.state_bytes = rows*stride;
    next.partial_offset = (next.state_bytes+127)/128*128;
    if (rows > (SIZE_MAX-next.partial_offset-127)/(256*sizeof(float))) return false;
    next.meta_offset = (next.partial_offset+rows*256*sizeof(float)+127)/128*128;
    if (rows > (SIZE_MAX-next.meta_offset)/(2*sizeof(float))) return false;
    next.bytes = next.meta_offset+rows*2*sizeof(float);
    output = next; return true;
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
