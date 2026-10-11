#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// Device-visible physical extent for one ordered interval of a logical K/V sequence.
struct ggml_cuda_kv_span {
    const char * k = nullptr;
    const char * v = nullptr;
    int32_t first = 0;
    int32_t tokens = 0;
    int64_t k_token_stride = 0;
    int64_t v_token_stride = 0;
    int64_t k_head_stride = 0;
    int64_t v_head_stride = 0;
};

// Validate encoded plane strides independently; the owner must also validate allocation extents.
static inline bool ggml_cuda_kv_spans_validate(
        const ggml_cuda_kv_span * spans, size_t count, int64_t tokens, int64_t heads,
        int64_t k_row_bytes, int64_t v_row_bytes) {
    const int64_t limit = std::numeric_limits<int64_t>::max();
    if (!spans || count == 0 || count > 3 || tokens <= 0 || tokens > INT32_MAX ||
            heads <= 0 || heads > INT32_MAX || k_row_bytes <= 0 || v_row_bytes <= 0 ||
            k_row_bytes > INT32_MAX || v_row_bytes > INT32_MAX || k_row_bytes%16 || v_row_bytes%16) return false;
    int64_t first = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & span = spans[i];
        if (span.first != first || span.tokens <= 0 || span.tokens > tokens-first) return false;
        const char * bases[2] = {span.k, span.v};
        const int64_t rows[2] = {span.k_token_stride, span.v_token_stride};
        const int64_t head_strides[2] = {span.k_head_stride, span.v_head_stride};
        const int64_t row_bytes[2] = {k_row_bytes, v_row_bytes};
        for (int plane = 0; plane < 2; ++plane) {
            const auto address = reinterpret_cast<uintptr_t>(bases[plane]);
            const int64_t row_stride = rows[plane], head_stride = head_strides[plane];
            if (!address || address%16 || row_stride < row_bytes[plane] || row_stride%16 ||
                    head_stride < row_bytes[plane] || head_stride%16 || (span.tokens-1) > limit/row_stride) return false;
            const int64_t row_offset = (span.tokens-1)*row_stride;
            if (heads-1 > (limit-row_offset)/head_stride) return false;
            const int64_t last = row_offset+(heads-1)*head_stride;
            if (last > limit-row_bytes[plane] || uint64_t(last+row_bytes[plane]) > std::numeric_limits<uintptr_t>::max()-address) return false;
        }
        first += span.tokens;
    }
    return first == tokens;
}

// Preserve FP16 admission while deriving its encoded row sizes without overflow.
static inline bool ggml_cuda_kv_spans_f16_validate(
        const ggml_cuda_kv_span * spans, size_t count, int64_t tokens, int64_t heads, int64_t dimension) {
    if (dimension <= 0 || dimension > INT32_MAX/2 || dimension%8) return false;
    return ggml_cuda_kv_spans_validate(spans,count,tokens,heads,dimension*2,dimension*2);
}
