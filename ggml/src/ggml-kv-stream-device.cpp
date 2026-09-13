#include "ggml-kv-stream-device.h"

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
