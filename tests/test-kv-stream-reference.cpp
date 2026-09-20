#include "../ggml/src/ggml-kv-stream-reference.h"
#include "testing.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using status = ggml_kv_stream_reference_status;

struct stored_plan {
    ggml_backend_memory_arena_t arena = nullptr;
    ggml_backend_memory_lease_t lease = nullptr;
    ggml_kv_stream_span_plan_t plan = nullptr;

    ~stored_plan() {
        ggml_kv_stream_span_plan_free(plan);
        ggml_backend_memory_lease_free(lease);
        ggml_backend_memory_arena_free(arena);
    }
};

static ggml_kv_stream_shape shape(ggml_type k, ggml_type v) {
    return {k, v, 32, 32, 2, 16, 128};
}

static void encode_row(ggml_type type, const float * input, size_t count, void * output) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(output, input, count*sizeof(float));
    } else {
        ggml_get_type_traits(type)->from_float_ref(input, output, count);
    }
}

static std::unique_ptr<stored_plan> make_plan(
        ggml_type type_k, ggml_type type_v, size_t tokens, size_t queries,
        const std::vector<size_t> & cuts, const std::vector<float> & keys,
        const std::vector<float> & values) {
    if (cuts.size() < 2 || cuts.front() != 0 || cuts.back() != tokens) return {};
    auto result = std::make_unique<stored_plan>();
    result->arena = ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(), 262144);
    ggml_backend_memory_region region;
    if (!result->arena ||
            !ggml_backend_memory_arena_begin(result->arena, GGML_BACKEND_MEMORY_PLAN_NONE) ||
            !ggml_backend_memory_arena_reserve(
                result->arena, 1, 262144, 128, GGML_BACKEND_MEMORY_REGION_NONE, &region) ||
            !ggml_backend_memory_arena_commit(result->arena)) return {};
    result->lease = ggml_backend_memory_arena_acquire(result->arena, 1);
    if (!result->lease) return {};

    const auto spec = shape(type_k, type_v);
    ggml_kv_stream_layout layout;
    if (ggml_kv_stream_layout_make(spec, tokens, layout).status != ggml_kv_stream_status::success) return {};
    auto * base = static_cast<uint8_t *>(
        ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(result->lease)));
    const size_t spans = cuts.size() - 1;
    std::vector<ggml_kv_stream_span_source> sources;
    sources.reserve(spans);
    for (size_t i = 0; i < spans; ++i) {
        const size_t count = cuts[i + 1] - cuts[i];
        const size_t physical = spans - i - 1;
        const size_t k_offset = physical*4096;
        const size_t v_offset = 131072 + physical*4096;
        sources.push_back({result->lease, result->lease, cuts[i], count, k_offset, v_offset});
        for (size_t local = 0; local < count; ++local) {
            const size_t token = cuts[i] + local;
            for (size_t head = 0; head < size_t(spec.heads); ++head) {
                const size_t logical = (token*size_t(spec.heads) + head)*size_t(spec.head_dim_k);
                encode_row(type_k, keys.data() + logical, size_t(spec.head_dim_k),
                    base + k_offset + local*layout.k_token_bytes + head*layout.k_row_bytes);
                encode_row(type_v, values.data() + logical, size_t(spec.head_dim_v),
                    base + v_offset + local*layout.v_token_bytes + head*layout.v_row_bytes);
            }
        }
    }
    if (ggml_kv_stream_span_plan_make(
            spec, sources.data(), sources.size(), tokens, queries, result->plan).status !=
            ggml_kv_stream_status::success) return {};
    return result;
}

static ggml_kv_stream_attention_reference_input input(
        const std::vector<float> & q, const std::vector<float> & mask,
        size_t query_heads, float scale, float softcap = 0) {
    return {q.data(), q.size(), mask.empty() ? nullptr : mask.data(), mask.size(), query_heads, scale, softcap};
}

static std::vector<float> scalar(
        const ggml_kv_stream_shape & spec, size_t tokens, size_t queries, size_t query_heads,
        const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
        const std::vector<float> & mask, float scale, float softcap) {
    const size_t dk = size_t(spec.head_dim_k), dv = size_t(spec.head_dim_v);
    const size_t kv_heads = size_t(spec.heads), group = query_heads/kv_heads;
    std::vector<float> output(queries*query_heads*dv);
    for (size_t query = 0; query < queries; ++query) for (size_t head = 0; head < query_heads; ++head) {
        const size_t kv_head = head/group;
        std::vector<double> scores(tokens);
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t token = 0; token < tokens; ++token) {
            const float mv = mask.empty() ? 0 : mask[query*tokens + token];
            if (std::isinf(mv) && mv < 0) {
                scores[token] = -std::numeric_limits<double>::infinity();
                continue;
            }
            double score = 0;
            for (size_t d = 0; d < dk; ++d)
                score += double(q[(query*query_heads + head)*dk + d])*
                    double(k[(token*kv_heads + kv_head)*dk + d]);
            score *= scale;
            if (softcap != 0) score = double(softcap)*std::tanh(score/double(softcap));
            score += mv;
            scores[token] = score;
            maximum = std::max(maximum, score);
        }
        if (!std::isfinite(maximum)) continue;
        double denominator = 0;
        for (double score : scores) if (std::isfinite(score)) denominator += std::exp(score - maximum);
        for (size_t d = 0; d < dv; ++d) {
            double sum = 0;
            for (size_t token = 0; token < tokens; ++token) if (std::isfinite(scores[token]))
                sum += std::exp(scores[token] - maximum)*
                    double(v[(token*kv_heads + kv_head)*dv + d]);
            output[(query*query_heads + head)*dv + d] = float(sum/denominator);
        }
    }
    return output;
}

static void close(testing & t, const std::vector<float> & expected,
        const std::vector<float> & actual, float tolerance = 1e-6f) {
    if (!t.assert_equal(expected.size(), actual.size())) return;
    for (size_t i = 0; i < expected.size(); ++i)
        t.assert_true(std::isfinite(actual[i]) && std::abs(expected[i] - actual[i]) <= tolerance);
}

int main() {
    testing t;

    t.test("tg1_to_tg4_arbitrary_partitions_match_scalar_attention", [](testing & t) {
        constexpr size_t tokens = 17, kv_heads = 2, query_heads = 4, dim = 32;
        std::mt19937 rng(7201);
        std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
        std::vector<float> keys(tokens*kv_heads*dim), values(keys.size());
        for (auto & value : keys) value = dist(rng);
        for (auto & value : values) value = dist(rng);
        for (size_t queries = 1; queries <= 4; ++queries) {
            std::vector<float> q(queries*query_heads*dim), mask(queries*tokens);
            for (auto & value : q) value = dist(rng);
            for (size_t query = 0; query < queries; ++query) for (size_t token = 0; token < tokens; ++token)
                mask[query*tokens + token] = token <= tokens - queries + query ? 0.03f*float(token%3) : -INFINITY;
            const auto expected = scalar(
                shape(GGML_TYPE_F32, GGML_TYPE_F32), tokens, queries, query_heads,
                q, keys, values, mask, 0.25f, 0.7f);
            std::vector<float> first;
            for (const auto & cuts : std::vector<std::vector<size_t>>{
                    {0, tokens}, {0, 1, 5, 6, 16, tokens},
                    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, tokens}}) {
                auto plan = make_plan(GGML_TYPE_F32, GGML_TYPE_F32, tokens, queries, cuts, keys, values);
                if (!t.assert_true(plan && plan->plan)) continue;
                ggml_kv_stream_attention_reference_output output;
                if (!t.assert_true(ggml_kv_stream_attention_reference(
                        plan->plan, input(q, mask, query_heads, 0.25f, 0.7f), output).status == status::success)) continue;
                t.assert_equal(queries, output.queries);
                t.assert_equal(query_heads, output.heads);
                t.assert_equal(dim, output.width);
                close(t, expected, output.value);
                if (first.empty()) first = output.value;
                else t.assert_true(first == output.value);
            }
        }
    });

    t.test("quantized_partition_boundaries_are_deterministic", [](testing & t) {
        constexpr size_t tokens = 17, kv_heads = 2, query_heads = 4, queries = 4, dim = 32;
        std::vector<float> q(queries*query_heads*dim), keys(tokens*kv_heads*dim), values(keys.size());
        for (size_t i = 0; i < q.size(); ++i) q[i] = 0.2f*std::sin(float(i)*0.11f);
        for (size_t i = 0; i < keys.size(); ++i) {
            keys[i] = 0.3f*std::cos(float(i)*0.07f);
            values[i] = 0.25f*std::sin(float(i)*0.13f);
        }
        std::vector<float> mask(queries*tokens);
        for (size_t query = 0; query < queries; ++query) for (size_t token = 0; token < tokens; ++token)
            mask[query*tokens + token] = token <= 13 + query ? 0 : -INFINITY;
        for (const auto & pair : {
                std::pair{GGML_TYPE_F16, GGML_TYPE_F16},
                std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}}) {
            auto contiguous = make_plan(pair.first, pair.second, tokens, queries, {0, tokens}, keys, values);
            auto segmented = make_plan(pair.first, pair.second, tokens, queries, {0, 1, 4, 9, 16, tokens}, keys, values);
            if (!t.assert_true(contiguous && segmented && contiguous->plan && segmented->plan)) continue;
            ggml_kv_stream_attention_reference_output a, b;
            if (!t.assert_true(ggml_kv_stream_attention_reference(
                    contiguous->plan, input(q, mask, query_heads, 0.2f), a).status == status::success)) continue;
            if (!t.assert_true(ggml_kv_stream_attention_reference(
                    segmented->plan, input(q, mask, query_heads, 0.2f), b).status == status::success)) continue;
            t.assert_true(a.value == b.value);
            t.assert_true(a.empty == b.empty);
        }
    });

    t.test("gqa_and_all_masked_rows_have_explicit_zero_output", [](testing & t) {
        constexpr size_t tokens = 7, queries = 2, query_heads = 4, kv_heads = 2, dim = 32;
        std::vector<float> q(queries*query_heads*dim, 0.25f);
        std::vector<float> keys(tokens*kv_heads*dim), values(keys.size());
        for (size_t token = 0; token < tokens; ++token) for (size_t head = 0; head < kv_heads; ++head)
            for (size_t d = 0; d < dim; ++d) {
                keys[(token*kv_heads + head)*dim + d] = 0.01f*float(token + head + d);
                values[(token*kv_heads + head)*dim + d] = float(10*head + token) + 0.001f*float(d);
            }
        std::vector<float> mask(queries*tokens, 0);
        std::fill_n(mask.begin(), tokens, -INFINITY);
        auto plan = make_plan(GGML_TYPE_F32, GGML_TYPE_F32, tokens, queries, {0, 3, tokens}, keys, values);
        if (!t.assert_true(plan && plan->plan)) return;
        ggml_kv_stream_attention_reference_output output;
        if (!t.assert_true(ggml_kv_stream_attention_reference(
                plan->plan, input(q, mask, query_heads, 0.1f), output).status == status::success)) return;
        t.assert_true(output.empty == std::vector<uint8_t>({1, 1, 1, 1, 0, 0, 0, 0}));
        for (size_t i = 0; i < query_heads*dim; ++i) t.assert_equal(0.0f, output.value[i]);
        t.assert_true(output.value[(query_heads + 0)*dim] < output.value[(query_heads + 2)*dim]);
        ggml_kv_stream_attention_reference_output unmasked;
        const std::vector<float> no_mask;
        t.assert_true(ggml_kv_stream_attention_reference(
            plan->plan, input(q, no_mask, query_heads, 0.1f), unmasked).status == status::success);
        t.assert_true(std::none_of(unmasked.empty.begin(), unmasked.empty.end(), [](uint8_t value) { return value != 0; }));
    });

    t.test("host_oracle_matches_stock_contiguous_cpu_attention", [](testing & t) {
        constexpr size_t tokens = 17, queries = 4, query_heads = 4, kv_heads = 2, dim = 32;
        std::vector<float> q(queries*query_heads*dim), keys(tokens*kv_heads*dim), values(keys.size());
        for (size_t i = 0; i < q.size(); ++i) q[i] = 0.1f*std::sin(float(i)*0.17f);
        for (size_t i = 0; i < keys.size(); ++i) {
            keys[i] = 0.15f*std::cos(float(i)*0.09f);
            values[i] = 0.2f*std::sin(float(i)*0.05f);
        }
        std::vector<float> mask(queries*tokens);
        for (size_t query = 0; query < queries; ++query) for (size_t token = 0; token < tokens; ++token)
            mask[query*tokens + token] = token <= 13 + query ? 0.01f*float(token%3) : -INFINITY;

        auto plan = make_plan(GGML_TYPE_F32, GGML_TYPE_F32, tokens, queries, {0, 3, 11, tokens}, keys, values);
        if (!t.assert_true(plan && plan->plan)) return;
        ggml_kv_stream_attention_reference_output reference;
        if (!t.assert_true(ggml_kv_stream_attention_reference(
                plan->plan, input(q, mask, query_heads, 0.2f), reference).status == status::success)) return;

        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_context_ptr context(ggml_init({65536, nullptr, true}));
        if (!t.assert_true(backend && context)) return;
        auto * qt = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dim, queries, query_heads);
        auto * kt = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dim, tokens, kv_heads);
        auto * vt = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dim, tokens, kv_heads);
        auto * mt = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, tokens, queries);
        auto * result = ggml_flash_attn_ext(context.get(), qt, kt, vt, mt, 0.2f, 0, 0);
        ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
        auto * graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, result);
        ggml_backend_buffer_ptr tensors(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
        if (!t.assert_true(tensors != nullptr)) return;

        std::vector<float> q_stock(q.size()), k_stock(keys.size()), v_stock(values.size());
        std::vector<ggml_fp16_t> mask_stock(mask.size());
        for (size_t query = 0; query < queries; ++query) for (size_t head = 0; head < query_heads; ++head)
            std::copy_n(q.data() + (query*query_heads + head)*dim, dim,
                q_stock.data() + (head*queries + query)*dim);
        for (size_t token = 0; token < tokens; ++token) for (size_t head = 0; head < kv_heads; ++head) {
            std::copy_n(keys.data() + (token*kv_heads + head)*dim, dim,
                k_stock.data() + (head*tokens + token)*dim);
            std::copy_n(values.data() + (token*kv_heads + head)*dim, dim,
                v_stock.data() + (head*tokens + token)*dim);
        }
        for (size_t i = 0; i < mask.size(); ++i) mask_stock[i] = ggml_fp32_to_fp16(mask[i]);
        ggml_backend_tensor_set(qt, q_stock.data(), 0, q_stock.size()*sizeof(float));
        ggml_backend_tensor_set(kt, k_stock.data(), 0, k_stock.size()*sizeof(float));
        ggml_backend_tensor_set(vt, v_stock.data(), 0, v_stock.size()*sizeof(float));
        ggml_backend_tensor_set(mt, mask_stock.data(), 0, mask_stock.size()*sizeof(ggml_fp16_t));
        if (!t.assert_true(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS)) return;
        std::vector<float> stock(reference.value.size());
        ggml_backend_tensor_get(result, stock.data(), 0, stock.size()*sizeof(float));
        close(t, stock, reference.value, 2e-5f);
    });

    t.test("invalid_inputs_fail_without_publishing_output", [](testing & t) {
        constexpr size_t tokens = 7, queries = 2, query_heads = 4, dim = 32;
        std::vector<float> q(queries*query_heads*dim, 0.1f);
        std::vector<float> keys(tokens*2*dim, 0.2f), values(keys.size(), 0.3f), mask(queries*tokens, 0);
        auto plan = make_plan(GGML_TYPE_F32, GGML_TYPE_F32, tokens, queries, {0, tokens}, keys, values);
        if (!t.assert_true(plan && plan->plan)) return;
        ggml_kv_stream_attention_reference_output output;
        output.queries = 77;
        auto request = input(q, mask, query_heads, 0.1f);
        t.assert_true(ggml_kv_stream_attention_reference(nullptr, request, output).status == status::invalid_plan);
        request.q_count--;
        t.assert_true(ggml_kv_stream_attention_reference(plan->plan, request, output).status == status::invalid_buffer);
        request = input(q, mask, 3, 0.1f);
        t.assert_true(ggml_kv_stream_attention_reference(plan->plan, request, output).status == status::invalid_shape);
        request = input(q, mask, query_heads, 0.1f);
        request.mask_count--;
        t.assert_true(ggml_kv_stream_attention_reference(plan->plan, request, output).status == status::invalid_buffer);
        request = input(q, mask, query_heads, INFINITY);
        t.assert_true(ggml_kv_stream_attention_reference(plan->plan, request, output).status == status::invalid_value);
        q[0] = NAN;
        request = input(q, mask, query_heads, 0.1f);
        t.assert_true(ggml_kv_stream_attention_reference(plan->plan, request, output).status == status::invalid_value);
        q[0] = 0.1f;
        mask[0] = INFINITY;
        request = input(q, mask, query_heads, 0.1f);
        t.assert_true(ggml_kv_stream_attention_reference(plan->plan, request, output).status == status::invalid_value);
        t.assert_equal(size_t(77), output.queries);
    });

    return t.summary();
}
