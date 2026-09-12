#include "ggml-kv-stream.h"

#include <algorithm>
#include <cstring>
#include <limits>

using status = ggml_kv_stream_status;
using operand = ggml_kv_stream_operand;

// Check arithmetic before evaluation; no GGML tensor or backend allocation is involved.
static bool multiply(size_t a, size_t b, size_t & out) {
    if (b && a > SIZE_MAX/b) return false;
    out = a*b;
    return true;
}
static bool add(size_t a, size_t b, size_t & out) {
    if (a > SIZE_MAX-b) return false;
    out = a+b;
    return true;
}

// Validate metadata before calling GGML type helpers, including removed enum entries and auxiliary formats.
static ggml_kv_stream_result row_bytes(int32_t code, int64_t dim, operand side, size_t & out) {
    if (code < 0 || code >= GGML_TYPE_COUNT) return {status::invalid_type, side};
    const auto type = static_cast<ggml_type>(code);
    const int64_t block = ggml_blck_size(type);
    const size_t bytes = ggml_type_size(type);
    if (block <= 0 || bytes == 0) return {status::invalid_type, side};
    if ((type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 && !ggml_is_quantized(type)) ||
            type == GGML_TYPE_Q8_1 || type == GGML_TYPE_Q8_K) return {status::unsupported_storage, side};
    if (dim <= 0 || dim % block != 0) return {status::invalid_shape, side};
    const uint64_t blocks = static_cast<uint64_t>(dim/block);
    if (blocks > SIZE_MAX || !multiply(static_cast<size_t>(blocks), bytes, out)) return {status::overflow, side};
    return {};
}

// Build complete planes for a span; multiplying a padded single-page record is not equivalent.
ggml_kv_stream_result ggml_kv_stream_layout_make(
        const ggml_kv_stream_shape & shape, size_t tokens, ggml_kv_stream_layout & output) {
    ggml_kv_stream_layout next;
    auto result = row_bytes(shape.type_k, shape.head_dim_k, operand::k, next.k_row_bytes);
    if (result.status != status::success) return result;
    result = row_bytes(shape.type_v, shape.head_dim_v, operand::v, next.v_row_bytes);
    if (result.status != status::success) return result;
    if (shape.heads <= 0 || shape.page_tokens <= 0) return {status::invalid_shape};
    if (static_cast<uint64_t>(shape.heads) > SIZE_MAX || static_cast<uint64_t>(shape.page_tokens) > SIZE_MAX) return {status::overflow};
    if (shape.alignment == 0 || (shape.alignment & (shape.alignment-1)) != 0) return {status::invalid_alignment};
    const size_t heads = static_cast<size_t>(shape.heads), page = static_cast<size_t>(shape.page_tokens);
    if (!multiply(next.k_row_bytes, heads, next.k_token_bytes) ||
            !multiply(next.k_token_bytes, tokens, next.k_bytes)) return {status::overflow, operand::k};
    if (!multiply(next.v_row_bytes, heads, next.v_token_bytes) ||
            !multiply(next.v_token_bytes, tokens, next.v_bytes)) return {status::overflow, operand::v};
    size_t padded;
    if (!add(next.k_bytes, shape.alignment-1, padded)) return {status::overflow, operand::k};
    next.v_offset = padded & ~(shape.alignment-1);
    if (!add(next.v_offset, next.v_bytes, next.bytes)) return {status::overflow, operand::v};
    next.tokens = tokens;
    next.pages = tokens/page + (tokens % page != 0);
    next.tail_tokens = tokens ? (tokens % page ? tokens % page : page) : 0;
    output = next;
    return {};
}

// Slice live rows without rounding a partial last page into nonexistent source data.
ggml_kv_stream_result ggml_kv_stream_page_make(
        const ggml_kv_stream_shape & shape, size_t live_tokens, size_t page, ggml_kv_stream_page & output) {
    ggml_kv_stream_layout whole;
    const auto result = ggml_kv_stream_layout_make(shape, live_tokens, whole);
    if (result.status != status::success) return result;
    if (page >= whole.pages) return {status::page_out_of_range};
    ggml_kv_stream_page next;
    if (!multiply(page, static_cast<size_t>(shape.page_tokens), next.token_begin)) return {status::overflow};
    next.tokens = std::min(static_cast<size_t>(shape.page_tokens), live_tokens-next.token_begin);
    if (!multiply(next.token_begin, whole.k_token_bytes, next.k_offset) ||
            !multiply(next.tokens, whole.k_token_bytes, next.k_bytes)) return {status::overflow, operand::k};
    if (!multiply(next.token_begin, whole.v_token_bytes, next.v_offset) ||
            !multiply(next.tokens, whole.v_token_bytes, next.v_bytes)) return {status::overflow, operand::v};
    output = next;
    return {};
}

// Storage, new-row writes, direct pairs, and conversion are independent backend promises.
ggml_kv_stream_result ggml_kv_stream_resolve(
        const ggml_kv_stream_shape & shape, const ggml_kv_stream_capabilities & caps,
        size_t span_tokens, ggml_kv_stream_execution & output) {
    if (span_tokens == 0) return {status::invalid_shape};
    ggml_kv_stream_execution next;
    const auto geometry = ggml_kv_stream_layout_make(shape, span_tokens, next.storage);
    if (geometry.status != status::success) return geometry;
    if (caps.k.type != shape.type_k) return {status::capability_mismatch, operand::k};
    if (caps.v.type != shape.type_v) return {status::capability_mismatch, operand::v};
    if (!caps.k.storage) return {status::unsupported_storage, operand::k};
    if (!caps.v.storage) return {status::unsupported_storage, operand::v};
    if (!caps.k.online_write) return {status::unsupported_write, operand::k};
    if (!caps.v.online_write) return {status::unsupported_write, operand::v};
    if (caps.direct_pair && caps.k.direct_attention && caps.v.direct_attention) {
        next.attention = ggml_kv_stream_attention::direct;
    } else if (caps.f16_attention && caps.k.convert_f16 && caps.v.convert_f16) {
        next.attention = ggml_kv_stream_attention::f16;
        auto converted = shape;
        converted.type_k = converted.type_v = GGML_TYPE_F16;
        const auto result = ggml_kv_stream_layout_make(converted, span_tokens, next.conversion);
        if (result.status != status::success) return result;
    } else {
        return {status::unsupported_attention};
    }
    output = next;
    return {};
}

// Dense scalar tensors may have arbitrary unused strides on unit axes, as in GGML views.
static bool dense_scalar(const ggml_tensor & tensor, size_t element_size) {
    size_t stride = element_size;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor.ne[i] <= 0 || static_cast<uint64_t>(tensor.ne[i]) > SIZE_MAX ||
                (tensor.ne[i] > 1 && tensor.nb[i] != stride) ||
                !multiply(stride, static_cast<size_t>(tensor.ne[i]), stride)) return false;
    }
    return true;
}

// Validate public attention metadata, not device pointers or private per-query kernel descriptors.
ggml_kv_stream_result ggml_kv_stream_attention_validate(
        const ggml_tensor * dst, const ggml_kv_stream_attention_limits & limits,
        const ggml_kv_stream_capabilities & caps, size_t span_tokens, ggml_kv_stream_execution & output) {
    if (!dst || dst->op != GGML_OP_FLASH_ATTN_EXT || dst->type != GGML_TYPE_F32) return {status::invalid_tensor};
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    if (!q || !k || !v || q->type != GGML_TYPE_F32) return {status::invalid_tensor};
    if (dst->src[4]) return {status::unsupported_sinks};
    if (limits.head_dim_k <= 0 || limits.head_dim_v <= 0 || limits.key_token_multiple <= 0 ||
            limits.page_tokens <= 0 || limits.page_tokens % limits.key_token_multiple != 0 ||
            static_cast<uint64_t>(limits.key_token_multiple) > SIZE_MAX ||
            span_tokens == 0 || span_tokens % static_cast<size_t>(limits.key_token_multiple) != 0) return {status::invalid_shape};
    for (const auto * tensor : {q, k, v, dst}) {
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (tensor->ne[i] <= 0 || tensor->ne[i] > INT32_MAX) return {status::invalid_tensor};
        }
        if (tensor->ne[3] != 1) return {status::invalid_tensor};
    }
    if (q->ne[0] != limits.head_dim_k || k->ne[0] != q->ne[0] || v->ne[0] != limits.head_dim_v ||
            k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0 ||
            k->ne[1] % limits.key_token_multiple != 0 ||
            dst->ne[0] != v->ne[0] || dst->ne[1] != q->ne[2] || dst->ne[2] != q->ne[1]) return {status::invalid_shape};
    const ggml_kv_stream_shape shape{k->type, v->type, k->ne[0], v->ne[0], k->ne[2], limits.page_tokens, limits.alignment};
    ggml_kv_stream_layout storage;
    auto result = ggml_kv_stream_layout_make(shape, static_cast<size_t>(k->ne[1]), storage);
    if (result.status != status::success) return result;
    if (k->nb[0] != ggml_type_size(k->type) || k->nb[1] != storage.k_token_bytes || k->nb[2] != storage.k_row_bytes)
        return {status::unsupported_stride, operand::k};
    if (v->nb[0] != ggml_type_size(v->type) || v->nb[1] != storage.v_token_bytes || v->nb[2] != storage.v_row_bytes)
        return {status::unsupported_stride, operand::v};

    size_t q_row, q_token, q_head, rows, extent, end;
    if (!multiply(static_cast<size_t>(q->ne[0]), sizeof(float), q_row) ||
            !multiply(q_row, static_cast<size_t>(q->ne[2]), q_token) ||
            !multiply(q_row, static_cast<size_t>(q->ne[1]), q_head) ||
            !multiply(static_cast<size_t>(q->ne[1]), static_cast<size_t>(q->ne[2]), rows) || rows > INT32_MAX ||
            !multiply(static_cast<size_t>(q->ne[1]-1), q->nb[1], extent) ||
            !multiply(static_cast<size_t>(q->ne[2]-1), q->nb[2], end) ||
            !add(extent, end, extent) || !add(extent, q_row, extent)) return {status::overflow};
    if (q->nb[0] != sizeof(float) || q->nb[1] % sizeof(float) || q->nb[2] % sizeof(float) ||
            !((q->nb[2] == q_row && q->nb[1] >= q_token) || (q->nb[1] == q_row && q->nb[2] >= q_head)) ||
            !dense_scalar(*dst, sizeof(float))) return {status::unsupported_stride};
    float max_bias;
    std::memcpy(&max_bias, &dst->op_params[1], sizeof(max_bias));
    if ((!mask && max_bias > 0) || (mask && (mask->type != GGML_TYPE_F16 ||
            mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1] || mask->ne[2] <= 0 ||
            q->ne[2] % mask->ne[2] != 0 || mask->ne[3] != 1 || !dense_scalar(*mask, sizeof(ggml_fp16_t))))) return {status::unsupported_mask};
    ggml_kv_stream_execution next;
    result = ggml_kv_stream_resolve(shape, caps, span_tokens, next);
    if (result.status != status::success) return result;
    output = next;
    return {};
}
