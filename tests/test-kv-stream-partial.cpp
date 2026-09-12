#include "../ggml/src/ggml-kv-stream-partial.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"

#include <cmath>
#include <random>

using status = ggml_kv_stream_partial_status;

// Independent scalar producer: masked scores are -infinity and empty chunks export zero mass/numerator.
static ggml_kv_stream_partial_batch parts(size_t rows, size_t tokens, size_t width,
        const std::vector<float> & scores, const std::vector<float> & values, const std::vector<size_t> & cuts) {
    ggml_kv_stream_partial_batch b;
    b.rows = rows; b.parts = cuts.size()-1; b.width = width;
    b.numerator.assign(rows*b.parts*width, 0);
    b.meta.resize(rows*b.parts);
    for (size_t r = 0; r < rows; ++r) for (size_t p = 0; p < b.parts; ++p) {
        double peak = -INFINITY;
        for (size_t k = cuts[p]; k < cuts[p+1]; ++k) peak = std::max(peak, double(scores[r*tokens+k]));
        if (!std::isfinite(peak)) continue;
        double sum = 0; std::vector<double> weighted(width, 0);
        for (size_t k = cuts[p]; k < cuts[p+1]; ++k) {
            const double w = std::exp(double(scores[r*tokens+k])-peak);
            sum += w;
            for (size_t d = 0; d < width; ++d) weighted[d] += w*values[(r*tokens+k)*width+d];
        }
        b.meta[r*b.parts+p] = {float(peak), float(sum)};
        for (size_t d = 0; d < width; ++d) b.numerator[(r*b.parts+p)*width+d] = float(weighted[d]);
    }
    return b;
}

// Unsplit normalization does not call either the partial producer or merge implementation.
static std::vector<float> oracle(size_t rows, size_t tokens, size_t width,
        const std::vector<float> & scores, const std::vector<float> & values) {
    std::vector<float> out(rows*width, 0);
    for (size_t r = 0; r < rows; ++r) {
        double peak = -INFINITY, sum = 0;
        for (size_t k = 0; k < tokens; ++k) peak = std::max(peak, double(scores[r*tokens+k]));
        if (!std::isfinite(peak)) continue;
        for (size_t k = 0; k < tokens; ++k) sum += std::exp(double(scores[r*tokens+k])-peak);
        for (size_t d = 0; d < width; ++d) {
            double result = 0;
            for (size_t k = 0; k < tokens; ++k) result += std::exp(double(scores[r*tokens+k])-peak)/sum*values[(r*tokens+k)*width+d];
            out[r*width+d] = float(result);
        }
    }
    return out;
}

static void near(testing & t, const std::vector<float> & a, const std::vector<float> & b, float tolerance = 2e-5f) {
    if (!t.assert_equal(a.size(), b.size())) return;
    for (size_t i = 0; i < a.size(); ++i) t.assert_true(std::isfinite(b[i]) && std::abs(a[i]-b[i]) <= tolerance);
}

int main() {
    testing t;
    t.test("checked_packed_layout_and_overflow", [](testing & t) {
        ggml_kv_stream_partial_layout layout;
        t.assert_true(ggml_kv_stream_partial_layout_make(2, 3, 7, 128, layout).status == status::success);
        t.assert_equal(size_t(42), layout.elements);
        t.assert_equal(size_t(6), layout.entries);
        t.assert_equal(size_t(168), layout.numerator_bytes);
        t.assert_equal(size_t(256), layout.meta_offset);
        t.assert_equal(size_t(48), layout.meta_bytes);
        t.assert_equal(size_t(304), layout.bytes);
        for (size_t axis = 0; axis < 3; ++axis) {
            size_t dims[] = {2, 3, 7}; dims[axis] = 0;
            t.assert_true(ggml_kv_stream_partial_layout_make(dims[0], dims[1], dims[2], 128, layout).status == status::invalid_shape);
            dims[axis] = SIZE_MAX;
            t.assert_true(ggml_kv_stream_partial_layout_make(dims[0], dims[1], dims[2], 128, layout).status == status::overflow);
            t.assert_equal(size_t(304), layout.bytes);
        }
        t.assert_true(ggml_kv_stream_partial_layout_make(1, 1, SIZE_MAX/4, 8, layout).status == status::overflow);
        t.assert_true(ggml_kv_stream_partial_layout_make(1, 1, SIZE_MAX/4 - 1, 8, layout).status == status::overflow);
        t.assert_equal(size_t(304), layout.bytes);
        for (size_t alignment : {size_t(0), size_t(4), size_t(12)})
            t.assert_true(ggml_kv_stream_partial_layout_make(2, 3, 7, alignment, layout).status == status::invalid_shape);
    });
    t.test("unequal_blocks_must_not_average_normalized_values", [](testing & t) {
        ggml_kv_stream_partial_batch input{1, 2, 1, {0, 90}, {{0, 1}, {0, 9}}};
        ggml_kv_stream_partial_batch merged;
        if (!t.assert_true(ggml_kv_stream_partial_merge({input.view()}, merged).status == status::success)) return;
        ggml_kv_stream_partial_value value;
        t.assert_true(ggml_kv_stream_partial_normalize(merged.view(), value).status == status::success);
        near(t, {9}, value.value);
        t.assert_equal(uint8_t(0), value.empty[0]);
    });
    t.test("empty_blocks_are_identity_and_empty_rows_are_explicit", [](testing & t) {
        ggml_kv_stream_partial_batch input{2, 3, 2, std::vector<float>(12, 0), std::vector<ggml_kv_stream_partial_meta>(6)};
        input.meta[0] = {std::numeric_limits<float>::max(), 0};
        input.meta[1] = {-1000, 2}; input.numerator[2] = 6; input.numerator[3] = -4;
        input.meta[2] = {-std::numeric_limits<float>::max()/2, 0};
        ggml_kv_stream_partial_batch merged;
        if (!t.assert_true(ggml_kv_stream_partial_merge({input.view()}, merged).status == status::success)) return;
        ggml_kv_stream_partial_value result;
        t.assert_true(ggml_kv_stream_partial_normalize(merged.view(), result).status == status::success);
        near(t, {3, -2, 0, 0}, result.value);
        t.assert_true(result.empty == std::vector<uint8_t>({0, 1}));
        t.assert_true(std::isinf(merged.meta[1].max_logit) && merged.meta[1].max_logit < 0);
    });
    t.test("partitions_and_causal_tails_match_unsplit_softmax", [](testing & t) {
        std::mt19937 rng(5304);
        std::uniform_real_distribution<float> dist(-1, 1);
        for (size_t rows : {size_t(1), size_t(6)}) for (size_t tokens : {size_t(1), size_t(7), size_t(33), size_t(257)})
            for (size_t width : {size_t(1), size_t(7), size_t(64), size_t(256)}) {
                std::vector<float> scores(rows*tokens), values(rows*tokens*width);
                for (auto & x : values) x = dist(rng);
                for (size_t r = 0; r < rows; ++r) for (size_t k = 0; k < tokens; ++k)
                    scores[r*tokens+k] = (rows > 1 && r == 0) || k >= (r+1)*tokens/rows || k%7 == 3 ? -INFINITY : 30*dist(rng);
                const auto expected = oracle(rows, tokens, width, scores, values);
                for (size_t step : {size_t(1), size_t(2), size_t(7), size_t(16), size_t(64)}) {
                    std::vector<size_t> cuts{0, 0};
                    for (size_t k = step; k < tokens; k += step) cuts.push_back(k);
                    cuts.push_back(tokens); cuts.push_back(tokens);
                    auto input = parts(rows, tokens, width, scores, values, cuts);
                    ggml_kv_stream_partial_batch merged;
                    if (!t.assert_true(ggml_kv_stream_partial_merge({input.view()}, merged).status == status::success)) continue;
                    ggml_kv_stream_partial_value normalized;
                    if (!t.assert_true(ggml_kv_stream_partial_normalize(merged.view(), normalized).status == status::success)) continue;
                    near(t, expected, normalized.value);
                }
            }
    });
    t.test("hierarchical_merge_is_alias_safe_and_order_independent", [](testing & t) {
        constexpr size_t rows = 3, tokens = 19, width = 7;
        std::vector<float> scores(rows*tokens), values(rows*tokens*width);
        for (size_t i = 0; i < scores.size(); ++i) scores[i] = float(int(i%17)-8)*3;
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i)*.13f);
        auto input = parts(rows, tokens, width, scores, values, {0, 1, 4, 7, 18, 19});
        ggml_kv_stream_partial_batch accumulator;
        for (size_t j = input.parts; j > 0; --j) {
            ggml_kv_stream_partial_batch part{rows, 1, width, std::vector<float>(rows*width), std::vector<ggml_kv_stream_partial_meta>(rows)};
            for (size_t r = 0; r < rows; ++r) {
                part.meta[r] = input.meta[r*input.parts+j-1];
                std::copy_n(input.numerator.data() + (r*input.parts+j-1)*width, width, part.numerator.data()+r*width);
            }
            std::vector<ggml_kv_stream_partial_view> views{part.view()};
            if (accumulator.rows) views.push_back(accumulator.view());
            if (!t.assert_true(ggml_kv_stream_partial_merge(views, accumulator).status == status::success)) return;
        }
        ggml_kv_stream_partial_value result;
        t.assert_true(ggml_kv_stream_partial_normalize(accumulator.view(), result).status == status::success);
        near(t, oracle(rows, tokens, width, scores, values), result.value);
    });
    t.test("extreme_logits_and_positive_subnormal_mass", [](testing & t) {
        const float maximum = std::numeric_limits<float>::max(), tiny = std::numeric_limits<float>::denorm_min();
        ggml_kv_stream_partial_batch input{1, 2, 1, {3, 99}, {{maximum, 1}, {-maximum, 1}}}, merged;
        if (!t.assert_true(ggml_kv_stream_partial_merge({input.view()}, merged).status == status::success)) return;
        ggml_kv_stream_partial_value value;
        t.assert_true(ggml_kv_stream_partial_normalize(merged.view(), value).status == status::success);
        near(t, {3}, value.value);
        input = {1, 1, 1, {tiny}, {{0, tiny}}};
        t.assert_true(ggml_kv_stream_partial_normalize(input.view(), value).status == status::success);
        near(t, {1}, value.value);
        t.assert_equal(uint8_t(0), value.empty[0]);
    });
    t.test("malformed_partials_reject_without_changing_output", [](testing & t) {
        for (int bad = 0; bad < 8; ++bad) {
            ggml_kv_stream_partial_batch input{1, 1, 1, {1}, {{0, 1}}};
            if (bad == 0) input.meta[0].normalizer = -1;
            if (bad == 1) input.meta[0].normalizer = INFINITY;
            if (bad == 2) input.meta[0].normalizer = NAN;
            if (bad == 3) input.meta[0].max_logit = INFINITY;
            if (bad == 4) input.meta[0].max_logit = -INFINITY;
            if (bad == 5) input.meta[0].max_logit = NAN;
            if (bad == 6) input.numerator[0] = NAN;
            if (bad == 7) input.meta[0].normalizer = 0;
            ggml_kv_stream_partial_batch output; output.rows = 77;
            output.numerator = {123}; output.meta = {{4, 5}};
            const auto result = ggml_kv_stream_partial_merge({input.view()}, output);
            t.assert_true(result.status == status::invalid_partial);
            t.assert_true(result.input == 0 && result.row == 0 && result.part == 0);
            t.assert_equal(size_t(77), output.rows);
            t.assert_true(output.numerator == std::vector<float>({123}));
            t.assert_true(output.meta.size() == 1 && output.meta[0].max_logit == 4 && output.meta[0].normalizer == 5);
        }
    });
    t.test("buffer_shape_and_prefix_capacity_validation", [](testing & t) {
        ggml_kv_stream_partial_batch input{1, 1, 1, {2, NAN}, {{0, 1}, {NAN, NAN}}}, output;
        t.assert_true(ggml_kv_stream_partial_merge({input.view()}, output).status == status::success);
        output.rows = 99;
        auto view = input.view(); view.numerator_count = 0;
        t.assert_true(ggml_kv_stream_partial_merge({view}, output).status == status::invalid_buffer);
        view = input.view(); view.meta = nullptr;
        t.assert_true(ggml_kv_stream_partial_merge({view}, output).status == status::invalid_buffer);
        view = input.view(); view.width = 2;
        t.assert_true(ggml_kv_stream_partial_merge({input.view(), view}, output).status == status::invalid_shape);
        t.assert_true(ggml_kv_stream_partial_merge({}, output).status == status::invalid_shape);
        t.assert_equal(size_t(99), output.rows);
    });
    t.test("fp32_publication_overflow_is_explicit", [](testing & t) {
        const float maximum = std::numeric_limits<float>::max();
        ggml_kv_stream_partial_batch input{1, 2, 1, {maximum, maximum}, {{0, 1}, {0, 1}}}, output;
        output.rows = 7;
        t.assert_true(ggml_kv_stream_partial_merge({input.view()}, output).status == status::overflow);
        input.numerator = {0, 0}; input.meta = {{0, maximum}, {0, maximum}};
        t.assert_true(ggml_kv_stream_partial_merge({input.view()}, output).status == status::overflow);
        t.assert_equal(size_t(7), output.rows);
        input = {1, 1, 1, {maximum}, {{0, std::numeric_limits<float>::denorm_min()}}};
        ggml_kv_stream_partial_value normalized; normalized.rows = 9; normalized.value = {55}; normalized.empty = {1};
        t.assert_true(ggml_kv_stream_partial_normalize(input.view(), normalized).status == status::overflow);
        t.assert_equal(size_t(9), normalized.rows);
        t.assert_true(normalized.value == std::vector<float>({55}) && normalized.empty == std::vector<uint8_t>({1}));
    });
    t.test("misaligned_and_wrapping_views_reject_before_reading", [](testing & t) {
        ggml_kv_stream_partial_batch input{1, 1, 1, {1}, {{0, 1}}}, output;
        output.rows = 73;
        for (int bad = 0; bad < 4; ++bad) {
            auto view = input.view();
            if (bad == 0) view.numerator = reinterpret_cast<const float *>(reinterpret_cast<uintptr_t>(view.numerator) + 1);
            if (bad == 1) view.meta = reinterpret_cast<const ggml_kv_stream_partial_meta *>(reinterpret_cast<uintptr_t>(view.meta) + 4);
            if (bad == 2) view.numerator = reinterpret_cast<const float *>(UINTPTR_MAX - 3);
            if (bad == 3) view.meta = reinterpret_cast<const ggml_kv_stream_partial_meta *>(UINTPTR_MAX - 7);
            t.assert_true(ggml_kv_stream_partial_merge({view}, output).status == status::invalid_buffer);
            t.assert_equal(size_t(73), output.rows);
        }
    });
    t.test("normalize_requires_merged_parts_and_handles_aliases", [](testing & t) {
        ggml_kv_stream_partial_batch input{1, 2, 1, {0, 0}, {{0, 1}, {0, 1}}};
        ggml_kv_stream_partial_value out; out.rows = 51; out.value = {-3, 9};
        t.assert_true(ggml_kv_stream_partial_normalize(input.view(), out).status == status::invalid_shape);
        t.assert_equal(size_t(51), out.rows);
        ggml_kv_stream_partial_meta meta{0, 3};
        ggml_kv_stream_partial_view alias{1, 1, 2, out.value.data(), out.value.size(), &meta, 1};
        t.assert_true(ggml_kv_stream_partial_normalize(alias, out).status == status::success);
        near(t, {-1, 3}, out.value);
        input = {2, 1, 1, {0, -0.0f}, {{0, 1}, {0, -0.0f}}};
        t.assert_true(ggml_kv_stream_partial_normalize(input.view(), out).status == status::success);
        t.assert_true(out.empty == std::vector<uint8_t>({0, 1}));
        near(t, {0, 0}, out.value);
    });
    t.test("packed_row_order_matches_ordinary_ggml_attention", [](testing & t) {
        constexpr size_t width = 64, tokens = 32, heads = 2, queries = 4, rows = heads*queries;
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
        if (!t.assert_true(backend && ctx)) return;
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, queries, heads);
        auto * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, width, tokens, heads);
        auto * v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, width, tokens, heads);
        auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, tokens, queries);
        auto * output = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f/8, 0, 0);
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
        auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!t.assert_true(buffer != nullptr)) return;
        std::vector<float> qdata(width*rows);
        std::vector<ggml_fp16_t> kdata(width*tokens*heads), vdata(kdata.size()), masks(tokens*queries);
        for (size_t i = 0; i < qdata.size(); ++i) qdata[i] = .2f*std::sin(float(i)*.3f);
        for (size_t i = 0; i < kdata.size(); ++i) {
            kdata[i] = ggml_fp32_to_fp16(.2f*std::cos(float(i)*.07f));
            vdata[i] = ggml_fp32_to_fp16(.2f*std::sin(float(i)*.13f));
        }
        std::vector<float> scores(rows*tokens), values(rows*tokens*width);
        for (size_t query = 0; query < queries; ++query) for (size_t h = 0; h < heads; ++h) for (size_t token = 0; token < tokens; ++token) {
            double dot = 0;
            for (size_t d = 0; d < width; ++d) {
                const size_t source = (h*tokens+token)*width+d;
                dot += double(qdata[(h*queries+query)*width+d])*ggml_fp16_to_fp32(kdata[source]);
                values[((query*heads+h)*tokens+token)*width+d] = ggml_fp16_to_fp32(vdata[source]);
            }
            const bool visible = token < (query+1)*8;
            scores[(query*heads+h)*tokens+token] = visible ? float(dot/8) : -INFINITY;
            masks[query*tokens+token] = ggml_fp32_to_fp16(visible ? 0 : -INFINITY);
        }
        ggml_backend_tensor_set(q, qdata.data(), 0, qdata.size()*sizeof(float));
        ggml_backend_tensor_set(k, kdata.data(), 0, kdata.size()*sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, vdata.data(), 0, vdata.size()*sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(mask, masks.data(), 0, masks.size()*sizeof(ggml_fp16_t));
        if (!t.assert_true(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS)) return;
        std::vector<float> ordinary(width*rows);
        ggml_backend_tensor_get(output, ordinary.data(), 0, ordinary.size()*sizeof(float));
        auto input = parts(rows, tokens, width, scores, values, {0, 1, 7, 16, 32});
        ggml_kv_stream_partial_batch merged;
        if (!t.assert_true(ggml_kv_stream_partial_merge({input.view()}, merged).status == status::success)) return;
        ggml_kv_stream_partial_value result;
        if (!t.assert_true(ggml_kv_stream_partial_normalize(merged.view(), result).status == status::success)) return;
        near(t, ordinary, result.value, 1e-3f);
        near(t, oracle(rows, tokens, width, scores, values), result.value);
    });
    t.test("long_incremental_merge_preserves_fp32_accuracy", [](testing & t) {
        constexpr size_t rows = 2, tokens = 4096, width = 7;
        std::vector<float> scores(rows*tokens), values(rows*tokens*width);
        for (size_t r = 0; r < rows; ++r) for (size_t k = 0; k < tokens; ++k) {
            scores[r*tokens+k] = r ? 1000.0f : float(k)/40.0f - 100.0f;
            for (size_t d = 0; d < width; ++d) values[(r*tokens+k)*width+d] = std::sin(float(k*width+d)*.07f);
        }
        std::vector<size_t> cuts;
        for (size_t k = 0; k <= tokens; k += 4) cuts.push_back(k);
        auto input = parts(rows, tokens, width, scores, values, cuts);
        ggml_kv_stream_partial_batch merged;
        for (size_t p = 0; p < input.parts; ++p) {
            ggml_kv_stream_partial_batch part{rows, 1, width, std::vector<float>(rows*width), std::vector<ggml_kv_stream_partial_meta>(rows)};
            for (size_t r = 0; r < rows; ++r) {
                part.meta[r] = input.meta[r*input.parts+p];
                std::copy_n(input.numerator.data() + (r*input.parts+p)*width, width, part.numerator.data()+r*width);
            }
            std::vector<ggml_kv_stream_partial_view> inputs{part.view()};
            if (merged.rows) inputs.push_back(merged.view());
            if (!t.assert_true(ggml_kv_stream_partial_merge(inputs, merged).status == status::success)) return;
        }
        ggml_kv_stream_partial_value normalized;
        t.assert_true(ggml_kv_stream_partial_normalize(merged.view(), normalized).status == status::success);
        near(t, oracle(rows, tokens, width, scores, values), normalized.value, 1e-4f);
    });
    return t.summary();
}
