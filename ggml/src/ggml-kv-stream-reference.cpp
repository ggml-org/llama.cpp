#include "ggml-kv-stream-reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

using status = ggml_kv_stream_reference_status;

// Check every flattened size before allocation or pointer arithmetic.
static bool multiply(size_t a, size_t b, size_t & output) {
    if (b && a > SIZE_MAX/b) return false;
    output = a*b;
    return true;
}

// Validate a host-readable scalar prefix without dereferencing it.
static bool readable(const void * data, size_t count, size_t element) {
    if (!data || reinterpret_cast<uintptr_t>(data) % element) return false;
    size_t bytes;
    return multiply(count, element, bytes) && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(data);
}

// Decode one physical K/V row into FP32 without depending on a backend implementation.
static bool decode_row(ggml_type type, const void * source, size_t count, float * output) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(output, source, count*sizeof(float));
        return true;
    }
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) return false;
    traits->to_float(source, output, count);
    return true;
}

// Resolve a logical token through the ordered spans while preserving global token order.
static const ggml_kv_stream_span * find_span(
        const ggml_kv_stream_span_plan_view & plan, size_t token, size_t & local) {
    for (size_t i = 0; i < plan.count; ++i) {
        const auto & span = plan.spans[i];
        if (token >= span.token_begin && token - span.token_begin < span.tokens) {
            local = token - span.token_begin;
            return &span;
        }
    }
    return nullptr;
}

// Evaluate the logical sequence continuously; physical span boundaries never reset softmax state.
ggml_kv_stream_reference_result ggml_kv_stream_attention_reference(
        ggml_kv_stream_span_plan_t plan,
        const ggml_kv_stream_attention_reference_input & input,
        ggml_kv_stream_attention_reference_output & output) {
    ggml_kv_stream_span_plan_view spans;
    if (!ggml_kv_stream_span_plan_get_view(plan, spans)) return {status::invalid_plan};
    if (input.query_heads == 0 || spans.shape.heads <= 0 ||
            input.query_heads % size_t(spans.shape.heads) != 0 ||
            spans.shape.head_dim_k <= 0 || spans.shape.head_dim_v <= 0) return {status::invalid_shape};
    if (!std::isfinite(input.scale) || !std::isfinite(input.logit_softcap) || input.logit_softcap < 0)
        return {status::invalid_value};

    const size_t queries = spans.query_tokens;
    const size_t q_heads = input.query_heads;
    const size_t kv_heads = size_t(spans.shape.heads);
    const size_t dk = size_t(spans.shape.head_dim_k);
    const size_t dv = size_t(spans.shape.head_dim_v);
    const size_t tokens = spans.active_tokens;
    size_t rows, q_required, output_count, mask_required;
    if (!multiply(queries, q_heads, rows) ||
            !multiply(rows, dk, q_required) ||
            !multiply(rows, dv, output_count) ||
            !multiply(queries, tokens, mask_required)) return {status::overflow};
    if (input.q_count < q_required || !readable(input.q, q_required, sizeof(float)))
        return {status::invalid_buffer};
    if ((!input.mask && input.mask_count != 0) ||
            (input.mask && (input.mask_count < mask_required ||
                !readable(input.mask, mask_required, sizeof(float))))) return {status::invalid_buffer};
    for (size_t i = 0; i < q_required; ++i)
        if (!std::isfinite(input.q[i])) return {status::invalid_value, i/dk};

    const auto type_k = static_cast<ggml_type>(spans.shape.type_k);
    const auto type_v = static_cast<ggml_type>(spans.shape.type_v);
    if ((type_k != GGML_TYPE_F32 && !ggml_get_type_traits(type_k)->to_float) ||
            (type_v != GGML_TYPE_F32 && !ggml_get_type_traits(type_v)->to_float))
        return {status::unsupported_type};
    for (size_t i = 0; i < spans.count; ++i)
        if (!ggml_backend_buffer_is_host(spans.spans[i].k_buffer) ||
                !ggml_backend_buffer_is_host(spans.spans[i].v_buffer)) return {status::invalid_buffer};

    ggml_kv_stream_layout layout;
    const auto geometry = ggml_kv_stream_layout_make(spans.shape, tokens, layout);
    if (geometry.status != ggml_kv_stream_status::success) return {status::invalid_plan};

    try {
        ggml_kv_stream_attention_reference_output next;
        next.queries = queries;
        next.heads = q_heads;
        next.width = dv;
        if (output_count > next.value.max_size() || rows > next.empty.max_size())
            return {status::overflow};
        next.value.assign(output_count, 0);
        next.empty.assign(rows, 0);
        std::vector<double> scores(tokens);
        std::vector<double> weighted(dv);
        std::vector<float> key(dk), value(dv);
        const size_t group = q_heads/kv_heads;

        for (size_t row = 0; row < rows; ++row) {
            const size_t query = row/q_heads;
            const size_t head = row%q_heads;
            const size_t kv_head = head/group;
            const float * q = input.q + row*dk;
            double maximum = -std::numeric_limits<double>::infinity();

            for (size_t token = 0; token < tokens; ++token) {
                const float mask = input.mask ? input.mask[query*tokens + token] : 0;
                if (std::isinf(mask) && mask < 0) {
                    scores[token] = -std::numeric_limits<double>::infinity();
                    continue;
                }
                if (!std::isfinite(mask)) return {status::invalid_value, row, token};

                size_t local;
                const auto * span = find_span(spans, token, local);
                if (!span) return {status::invalid_plan, row, token};
                const auto * source = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(span->k_buffer)) +
                    span->k_offset + local*layout.k_token_bytes + kv_head*layout.k_row_bytes;
                if (!decode_row(type_k, source, dk, key.data())) return {status::unsupported_type, row, token};
                double score = 0;
                for (size_t d = 0; d < dk; ++d) {
                    if (!std::isfinite(key[d])) return {status::invalid_value, row, token};
                    score += double(q[d])*double(key[d]);
                }
                score *= double(input.scale);
                if (input.logit_softcap != 0)
                    score = double(input.logit_softcap)*std::tanh(score/double(input.logit_softcap));
                score += double(mask);
                if (!std::isfinite(score)) return {status::invalid_value, row, token};
                scores[token] = score;
                maximum = std::max(maximum, score);
            }
            if (!std::isfinite(maximum)) {
                next.empty[row] = 1;
                continue;
            }

            double denominator = 0;
            for (double score : scores)
                if (std::isfinite(score)) denominator += std::exp(score - maximum);
            if (!std::isfinite(denominator) || denominator <= 0) return {status::overflow, row};
            std::fill(weighted.begin(), weighted.end(), 0);
            for (size_t token = 0; token < tokens; ++token) {
                if (!std::isfinite(scores[token])) continue;
                size_t local;
                const auto * span = find_span(spans, token, local);
                if (!span) return {status::invalid_plan, row, token};
                const auto * source = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(span->v_buffer)) +
                    span->v_offset + local*layout.v_token_bytes + kv_head*layout.v_row_bytes;
                if (!decode_row(type_v, source, dv, value.data())) return {status::unsupported_type, row, token};
                const double weight = std::exp(scores[token] - maximum);
                for (size_t d = 0; d < dv; ++d) {
                    if (!std::isfinite(value[d])) return {status::invalid_value, row, token};
                    weighted[d] += weight*double(value[d]);
                }
            }
            for (size_t d = 0; d < dv; ++d) {
                const double value = weighted[d]/denominator;
                if (!std::isfinite(value) ||
                        std::abs(value) > double(std::numeric_limits<float>::max()))
                    return {status::overflow, row};
                next.value[row*dv + d] = float(value);
            }
        }
        output = std::move(next);
        return {};
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed};
    } catch (const std::length_error &) {
        return {status::overflow};
    }
}
