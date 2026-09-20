#pragma once

#include "ggml-backend-memory.h"

enum class ggml_kv_stream_status {
    success, invalid_type, unsupported_storage, invalid_shape, invalid_alignment,
    overflow, page_out_of_range, capability_mismatch, unsupported_write, unsupported_attention,
    invalid_tensor, unsupported_stride, unsupported_mask, unsupported_sinks,
    invalid_span, invalid_buffer, allocation_failed,
};
enum class ggml_kv_stream_operand { none, k, v };
struct ggml_kv_stream_result {
    ggml_kv_stream_status status = ggml_kv_stream_status::success;
    ggml_kv_stream_operand operand = ggml_kv_stream_operand::none;
};

struct ggml_kv_stream_shape {
    // Integer GGML type codes permit validation of unknown values without invalid-enum loads.
    int32_t type_k = GGML_TYPE_COUNT;
    int32_t type_v = GGML_TYPE_COUNT;
    int64_t head_dim_k = 0;
    int64_t head_dim_v = 0;
    int64_t heads = 0;
    int64_t page_tokens = 0;
    size_t alignment = 128;
};

// One contiguous K plane followed by an aligned V plane, not interleaved page records.
struct ggml_kv_stream_layout {
    size_t k_row_bytes = 0, v_row_bytes = 0;
    size_t k_token_bytes = 0, v_token_bytes = 0;
    size_t k_bytes = 0, v_offset = 0, v_bytes = 0, bytes = 0;
    size_t tokens = 0, pages = 0, tail_tokens = 0;
};

// Offsets are relative to the separate source planes, not to a packed K+V allocation.
struct ggml_kv_stream_page {
    size_t token_begin = 0, tokens = 0;
    size_t k_offset = 0, k_bytes = 0, v_offset = 0, v_bytes = 0;
};

struct ggml_kv_stream_type_capabilities {
    int32_t type = GGML_TYPE_COUNT;
    bool storage = false;
    // Includes validation of the producer dtype/layout, not only the destination encoding.
    bool online_write = false;
    bool direct_attention = false;
    bool convert_f16 = false;
};

// Advertise actual backend implementations. Per-type support alone does not establish pair support.
struct ggml_kv_stream_capabilities {
    ggml_kv_stream_type_capabilities k, v;
    bool direct_pair = false;
    bool f16_attention = false;
};
enum class ggml_kv_stream_attention { direct, f16 };
struct ggml_kv_stream_execution {
    ggml_kv_stream_attention attention = ggml_kv_stream_attention::direct;
    ggml_kv_stream_layout storage;
    // Only conversion planes; excludes attention partials, accumulators, write staging, and backend scratch.
    ggml_kv_stream_layout conversion;
};

struct ggml_kv_stream_attention_limits {
    int64_t head_dim_k = 0, head_dim_v = 0;
    int64_t key_token_multiple = 0;
    int64_t page_tokens = 0;
    size_t alignment = 128;
};

// Creation input retains arena leases so their address ranges cannot be repartitioned.
struct ggml_kv_stream_span_source {
    ggml_backend_memory_lease_t k_lease = nullptr;
    ggml_backend_memory_lease_t v_lease = nullptr;
    size_t token_begin = 0;
    size_t tokens = 0;
    size_t k_offset = 0;
    size_t v_offset = 0;
};

// One physical range in an ordered logical KV sequence. Offsets are relative to each buffer.
struct ggml_kv_stream_span {
    ggml_backend_buffer_t k_buffer = nullptr;
    ggml_backend_buffer_t v_buffer = nullptr;
    size_t token_begin = 0;
    size_t tokens = 0;
    size_t k_offset = 0;
    size_t v_offset = 0;
};

struct ggml_kv_stream_span_plan;
using ggml_kv_stream_span_plan_t = ggml_kv_stream_span_plan *;

// Borrowed immutable metadata valid until the owning plan is freed.
struct ggml_kv_stream_span_plan_view {
    ggml_kv_stream_shape shape;
    const ggml_kv_stream_span * spans = nullptr;
    size_t count = 0;
    size_t active_tokens = 0;
    size_t query_tokens = 0;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
};

// Geometry and tensor-validation functions below are metadata-only and leave output unchanged on failure.
GGML_API ggml_kv_stream_result ggml_kv_stream_layout_make(
        const ggml_kv_stream_shape & shape, size_t tokens, ggml_kv_stream_layout & output);
GGML_API ggml_kv_stream_result ggml_kv_stream_page_make(
        const ggml_kv_stream_shape & shape, size_t live_tokens, size_t page, ggml_kv_stream_page & output);
// Resolve both write paths and either a native pair or bounded F16 conversion for the supplied span size.
GGML_API ggml_kv_stream_result ggml_kv_stream_resolve(
        const ggml_kv_stream_shape & shape, const ggml_kv_stream_capabilities & capabilities,
        size_t span_tokens, ggml_kv_stream_execution & output);
// Validate tensor metadata before allocating or inspecting payloads. Q supports either dense head/token ordering.
GGML_API ggml_kv_stream_result ggml_kv_stream_attention_validate(
        const ggml_tensor * dst, const ggml_kv_stream_attention_limits & limits,
        const ggml_kv_stream_capabilities & capabilities, size_t span_tokens, ggml_kv_stream_execution & output);
// Validate exact logical coverage and retain every backing lease until plan_free.
GGML_API ggml_kv_stream_result ggml_kv_stream_span_plan_make(
        const ggml_kv_stream_shape & shape, const ggml_kv_stream_span_source * spans, size_t count,
        size_t active_tokens, size_t query_tokens, ggml_kv_stream_span_plan_t & output);
// Share one immutable plan among asynchronous consumers.
GGML_API ggml_kv_stream_span_plan_t ggml_kv_stream_span_plan_retain(ggml_kv_stream_span_plan_t plan);
GGML_API void ggml_kv_stream_span_plan_free(ggml_kv_stream_span_plan_t plan);
// Return a borrowed immutable view without transferring buffer ownership.
GGML_API bool ggml_kv_stream_span_plan_get_view(
        ggml_kv_stream_span_plan_t plan, ggml_kv_stream_span_plan_view & output);
