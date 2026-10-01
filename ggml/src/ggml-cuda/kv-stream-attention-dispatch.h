#pragma once

#include "ggml.h"

#include <cstdint>

enum class ggml_cuda_kv_stream_attention_path { none, vector, mma };

static inline ggml_cuda_kv_stream_attention_path ggml_cuda_kv_stream_attention_select(
        int cc, uint32_t queries, ggml_type key, ggml_type value) {
    if (queries < 1 || queries > 2) return ggml_cuda_kv_stream_attention_path::none;
    if (cc >= 890) return ggml_cuda_kv_stream_attention_path::vector;
    if (cc >= 800 && queries == 1 && (ggml_is_quantized(key) || ggml_is_quantized(value))) {
        return ggml_cuda_kv_stream_attention_path::vector;
    }
    if (cc >= 800 && queries == 2 && key == GGML_TYPE_Q8_0 && value == GGML_TYPE_Q4_0) {
        return ggml_cuda_kv_stream_attention_path::mma;
    }
    return ggml_cuda_kv_stream_attention_path::none;
}

static inline int ggml_cuda_kv_stream_mma_ncols1(uint32_t queries, int gqa_ratio) {
    return queries == 2 && gqa_ratio > 4 ? 2 : 4;
}
