#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstddef>

// Keep native 128-token ownership; only a final incomplete tile needs guarded loads.
static inline size_t ggml_cuda_kv_stream_vector_bulk_tokens(size_t first, size_t tokens) {
    return first%128 == 0 ? tokens/128*128 : 0;
}

enum class ggml_cuda_kv_stream_attention_path { none, vector, mma };

// Other conversion modes can round differently; compact masks require the general loader.
static inline bool ggml_cuda_kv_stream_mma_dense_strided(bool compact_mask, uint32_t conversion_flags) {
    return !compact_mask && conversion_flags == 0;
}

static inline ggml_cuda_kv_stream_attention_path ggml_cuda_kv_stream_attention_select(
        int cc, uint32_t queries, ggml_type key, ggml_type value) {
    if (queries < 1 || queries > 2) return ggml_cuda_kv_stream_attention_path::none;
    if (cc >= 890) return ggml_cuda_kv_stream_attention_path::vector;
    if (cc >= 610 && (queries == 1 || cc < 700) && (ggml_is_quantized(key) || ggml_is_quantized(value))) {
        return ggml_cuda_kv_stream_attention_path::vector;
    }
    if (cc >= 750 && queries == 2 && key == GGML_TYPE_Q8_0 && value == GGML_TYPE_Q4_0) {
        return ggml_cuda_kv_stream_attention_path::mma;
    }
    return ggml_cuda_kv_stream_attention_path::none;
}

// Match stock columns for admitted head-256 groups; Turing-only builds cap tiles at 32 columns.
static inline int ggml_cuda_kv_stream_mma_ncols1(uint32_t queries, int gqa_ratio, bool turing_only = false) {
    if (!queries || queries > uint32_t(INT32_MAX) || (gqa_ratio != 2 && gqa_ratio != 6 && gqa_ratio != 8)) return 0;
    const int ncols2 = gqa_ratio > 4 ? 8 : 2;
    if (queries <= uint32_t(8/ncols2)) return 8/ncols2;
    if (queries <= uint32_t(16/ncols2)) return 16/ncols2;
    return queries <= uint32_t(32/ncols2) || turing_only ? 32/ncols2 : 64/ncols2;
}

static inline int ggml_cuda_kv_stream_mma_query_tiles(uint32_t queries, int gqa_ratio, bool turing_only = false) {
    const int columns = ggml_cuda_kv_stream_mma_ncols1(queries, gqa_ratio, turing_only);
    return columns ? int((queries+uint32_t(columns)-1)/uint32_t(columns)) : 0;
}

// Match the head-256 vector body's compiled copy width, not a newer GPU's physical capability.
static inline uint32_t ggml_cuda_kv_stream_vector_values_per_thread(ggml_type value, int compiled_cc) {
    return value == GGML_TYPE_F16 || value == GGML_TYPE_BF16 ? (compiled_cc < 700 ? 16 : 32) : 8;
}
