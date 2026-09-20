#pragma once

#include "ggml-kv-stream.h"

#include <vector>

enum class ggml_kv_stream_reference_status {
    success,
    invalid_plan,
    invalid_shape,
    invalid_buffer,
    invalid_value,
    unsupported_type,
    overflow,
    allocation_failed,
};

struct ggml_kv_stream_reference_result {
    ggml_kv_stream_reference_status status = ggml_kv_stream_reference_status::success;
    size_t row = SIZE_MAX;
    size_t token = SIZE_MAX;
};

// Q is packed as [query][query_head][head_dim_k]. An optional mask is [query][active_token].
struct ggml_kv_stream_attention_reference_input {
    const float * q = nullptr;
    size_t q_count = 0;
    const float * mask = nullptr;
    size_t mask_count = 0;
    size_t query_heads = 0;
    float scale = 0;
    float logit_softcap = 0;
};

struct ggml_kv_stream_attention_reference_output {
    size_t queries = 0;
    size_t heads = 0;
    size_t width = 0;
    // Packed as [query][query_head][head_dim_v].
    std::vector<float> value;
    // One entry per query/head row. All-masked rows contain zero values.
    std::vector<uint8_t> empty;
};

// Scalar host oracle for a retained span plan. It is not an inference or performance path.
GGML_API ggml_kv_stream_reference_result ggml_kv_stream_attention_reference(
        ggml_kv_stream_span_plan_t plan,
        const ggml_kv_stream_attention_reference_input & input,
        ggml_kv_stream_attention_reference_output & output);
