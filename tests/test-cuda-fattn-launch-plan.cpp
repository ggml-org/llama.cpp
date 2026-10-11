#include "fattn-launch-plan.h"
#include "kv-stream-span.h"

#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>

// Keep the pinned upstream formula independent of the extracted planner.
static ggml_cuda_fattn_stream_k_plan reference_plan(int dst, int kv, int occupancy, bool force, bool whole) {
    const int waves = (dst + occupancy - 1)/occupancy;
    const int efficiency = 100*dst/(occupancy*waves);
    const bool stream = !(whole && efficiency >= 75) && (force || efficiency < 75);
    int blocks = dst;
    if (stream) {
        const int raw = occupancy < kv*dst ? occupancy : kv*dst;
        const int rounded = (raw/dst)*dst;
        const int loss = rounded > 0 ? 100*(raw-rounded)/raw : 100;
        blocks = loss <= 5 ? rounded : raw;
    }
    return {blocks, stream, dst%blocks != 0};
}

static void check_plan(int dst, int kv, int occupancy, bool force, bool whole,
        int blocks, bool stream, bool fixup) {
    ggml_cuda_fattn_stream_k_plan actual;
    assert(ggml_cuda_fattn_stream_k_plan_make(dst, kv, occupancy, force, whole, actual));
    assert(actual.blocks == blocks);
    assert(actual.use_stream_k == stream);
    assert(actual.needs_fixup == fixup);
}

static void test_boundaries() {
    check_plan(75, 16, 100, false, false, 75, false, false);
    check_plan(74, 16, 100, false, false, 100, true, true);
    check_plan(75, 16, 100, true, false, 100, true, true);
    check_plan(75, 16, 100, true, true, 75, false, false);
    check_plan(74, 16, 100, true, true, 100, true, true);
    check_plan(19, 32, 100, true, false, 95, true, true);
    check_plan(19, 32, 101, true, false, 95, true, true);
    check_plan(19, 32, 102, true, false, 102, true, true);
    check_plan(1, 1, 128, true, false, 1, true, false);
    check_plan(8, 2, 128, true, false, 16, true, true);
    check_plan(8, 16, 32, true, false, 32, true, true);
}

static void test_stock_formula() {
    for (int dst = 1; dst <= 64; ++dst) {
        for (int kv = 1; kv <= 32; ++kv) {
            for (int occupancy = 1; occupancy <= 48; ++occupancy) {
                for (bool force : {false, true}) {
                    for (bool whole : {false, true}) {
                        const auto expected = reference_plan(dst, kv, occupancy, force, whole);
                        ggml_cuda_fattn_stream_k_plan actual;
                        assert(ggml_cuda_fattn_stream_k_plan_make(dst, kv, occupancy, force, whole, actual));
                        assert(actual.blocks == expected.blocks);
                        assert(actual.use_stream_k == expected.use_stream_k);
                        assert(actual.needs_fixup == expected.needs_fixup);
                    }
                }
            }
        }
    }
}

static void test_invalid_inputs() {
    const int64_t limit = std::numeric_limits<int32_t>::max();
    const int64_t invalid[][3] = {{0, 1, 1}, {1, 0, 1}, {1, 1, 0},
        {-1, 1, 1}, {1, -1, 1}, {1, 1, -1},
        {limit+1, 1, 1}, {1, limit+1, 1}, {1, 1, limit+1},
        {limit, 2, 1}, {2, limit, 1}, {limit, limit, limit},
        {std::numeric_limits<int64_t>::max(), 1, 1}};
    for (const auto & input : invalid) {
        ggml_cuda_fattn_stream_k_plan output{7, true, true};
        assert(!ggml_cuda_fattn_stream_k_plan_make(input[0], input[1], input[2], false, false, output));
        assert(output.blocks == 7 && output.use_stream_k && output.needs_fixup);
    }
}

static void test_large_valid_inputs() {
    const int64_t limit = std::numeric_limits<int32_t>::max();
    ggml_cuda_fattn_stream_k_plan output;
    assert(ggml_cuda_fattn_stream_k_plan_make(limit, 1, limit, true, false, output));
    assert(output.blocks == limit && output.use_stream_k && !output.needs_fixup);
    assert(ggml_cuda_fattn_stream_k_plan_make(limit, 1, limit-1, true, false, output));
    assert(output.blocks == limit-1 && output.use_stream_k && output.needs_fixup);
    assert(ggml_cuda_fattn_stream_k_plan_make(1, limit, limit, false, false, output));
    assert(output.blocks == limit && output.use_stream_k && output.needs_fixup);
}

static void test_f16_span_admission() {
    ggml_cuda_kv_span spans[3] = {
        {reinterpret_cast<const char *>(0x10000), reinterpret_cast<const char *>(0x20000), 0, 17, 512, 512, 8704, 8704},
        {reinterpret_cast<const char *>(0x30000), reinterpret_cast<const char *>(0x40000), 17, 30, 2048, 2048, 512, 512},
        {reinterpret_cast<const char *>(0x50000), reinterpret_cast<const char *>(0x60000), 47, 18, 512, 512, 9216, 9216},
    };
    assert(ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256));
    assert(ggml_cuda_kv_spans_f16_validate(spans, 2, 47, 4, 256));
    assert(ggml_cuda_kv_spans_f16_validate(spans, 1, 17, 4, 256));
    assert(!ggml_cuda_kv_spans_f16_validate(nullptr, 3, 65, 4, 256));
    assert(!ggml_cuda_kv_spans_f16_validate(spans, 0, 65, 4, 256));
    assert(!ggml_cuda_kv_spans_f16_validate(spans, 4, 65, 4, 256));
    for (int64_t dimension : {0, -1, 255}) assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, dimension));
    for (int64_t heads : {0, -1}) assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, heads, 256));
    for (int64_t tokens : {0, -1, 64, 66}) assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, tokens, 4, 256));
    for (int index = 0; index < 3; ++index) {
        const auto original = spans[index];
        spans[index].first++; assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256)); spans[index] = original;
        spans[index].tokens = 0; assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256)); spans[index] = original;
        spans[index].k = nullptr; assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256)); spans[index] = original;
        spans[index].v = nullptr; assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256)); spans[index] = original;
        spans[index].k = reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(original.k)+1);
        assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256)); spans[index] = original;
        spans[index].v = reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(original.v)+1);
        assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256)); spans[index] = original;
        spans[index].k = reinterpret_cast<const char *>(std::numeric_limits<uintptr_t>::max()-15);
        assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256)); spans[index] = original;
        for (auto field : {&ggml_cuda_kv_span::k_token_stride, &ggml_cuda_kv_span::v_token_stride,
                &ggml_cuda_kv_span::k_head_stride, &ggml_cuda_kv_span::v_head_stride}) {
            for (int64_t stride : {0, -16, 496, 513}) {
                spans[index].*field = stride;
                assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256));
                spans[index] = original;
            }
            spans[index].*field = std::numeric_limits<int64_t>::max()-15;
            assert(!ggml_cuda_kv_spans_f16_validate(spans, 3, 65, 4, 256));
            spans[index] = original;
        }
    }
}

static void test_encoded_span_admission() {
    ggml_cuda_kv_span span{reinterpret_cast<const char *>(0x10000),reinterpret_cast<const char *>(0x20000),
        0,32,272,144,8704,4608};
    assert(ggml_cuda_kv_spans_validate(&span,1,32,4,272,144));
    assert(!ggml_cuda_kv_spans_validate(&span,1,32,4,512,512));
    assert(!ggml_cuda_kv_spans_validate(&span,1,32,4,0,144));
    assert(!ggml_cuda_kv_spans_validate(&span,1,32,4,272,-16));
    assert(!ggml_cuda_kv_spans_validate(&span,1,32,4,273,144));
    assert(!ggml_cuda_kv_spans_validate(&span,1,32,4,272,145));
    const auto original = span;
    span.v_token_stride = 128;
    assert(!ggml_cuda_kv_spans_validate(&span,1,32,4,272,144));
    span = original; span.k_token_stride = 256;
    assert(!ggml_cuda_kv_spans_validate(&span,1,32,4,272,144));
    span = original; span.k_head_stride = 272; span.k_token_stride = 1088;
    span.v_head_stride = 144; span.v_token_stride = 576;
    assert(ggml_cuda_kv_spans_validate(&span,1,32,4,272,144));
}

int main() {
    test_boundaries();
    test_stock_formula();
    test_invalid_inputs();
    test_large_valid_inputs();
    test_f16_span_admission();
    test_encoded_span_admission();
    std::puts("FlashAttention launch and span planning: 6 test groups passed, 393216 stock-formula cases");
}
