#include "../ggml/src/ggml-kv-stream.h"
#include "testing.h"

#include <limits>

using status = ggml_kv_stream_status;
using operand = ggml_kv_stream_operand;

static ggml_kv_stream_shape shape() {
    return {GGML_TYPE_F16, GGML_TYPE_F16, 32, 32, 1, 16, 16};
}

struct allocation {
    ggml_backend_memory_arena_t arena = nullptr;
    ggml_backend_memory_lease_t lease = nullptr;

    explicit allocation(size_t bytes = 4096) {
        arena = ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(), bytes);
        ggml_backend_memory_region region;
        if (!arena ||
                !ggml_backend_memory_arena_begin(arena, GGML_BACKEND_MEMORY_PLAN_NONE) ||
                !ggml_backend_memory_arena_reserve(arena, 1, bytes, 16, GGML_BACKEND_MEMORY_REGION_NONE, &region) ||
                !ggml_backend_memory_arena_commit(arena)) return;
        lease = ggml_backend_memory_arena_acquire(arena, 1);
    }

    ~allocation() {
        ggml_backend_memory_lease_free(lease);
        ggml_backend_memory_arena_free(arena);
    }

    bool valid() const {
        return arena && lease;
    }
};

static ggml_kv_stream_span_source span(
        ggml_backend_memory_lease_t k, ggml_backend_memory_lease_t v,
        size_t first, size_t tokens, size_t k_offset, size_t v_offset) {
    return {k, v, first, tokens, k_offset, v_offset};
}

int main() {
    testing t;

    t.test("one_and_multiple_spans_cover_one_logical_sequence", [](testing & t) {
        allocation storage;
        if (!t.assert_true(storage.valid())) return;

        ggml_kv_stream_span_plan_t one = nullptr;
        const auto single = span(storage.lease, storage.lease, 0, 8, 0, 1024);
        if (!t.assert_true(ggml_kv_stream_span_plan_make(shape(), &single, 1, 8, 4, one).status == status::success)) {
            return;
        }
        ggml_kv_stream_span_plan_view view;
        t.assert_true(ggml_kv_stream_span_plan_get_view(one, view));
        t.assert_equal(size_t(1), view.count);
        t.assert_equal(size_t(8), view.active_tokens);
        t.assert_equal(size_t(4), view.query_tokens);
        t.assert_equal(size_t(512), view.k_bytes);
        t.assert_equal(size_t(512), view.v_bytes);
        t.assert_equal(size_t(8), view.spans[0].tokens);
        ggml_kv_stream_span_plan_free(one);

        const ggml_kv_stream_span_source split[] = {
            span(storage.lease, storage.lease, 0, 3, 0, 1024),
            span(storage.lease, storage.lease, 3, 5, 256, 1280),
        };
        ggml_kv_stream_span_plan_t multiple = nullptr;
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), split, 2, 8, 1, multiple).status == status::success);
        t.assert_true(ggml_kv_stream_span_plan_get_view(multiple, view));
        t.assert_equal(size_t(2), view.count);
        t.assert_equal(size_t(3), view.spans[1].token_begin);
        ggml_kv_stream_span_plan_free(multiple);
    });

    t.test("rejects_gaps_overlaps_and_reordered_spans_transactionally", [](testing & t) {
        allocation storage;
        if (!t.assert_true(storage.valid())) return;
        const size_t starts[][2] = {{0, 5}, {0, 3}, {4, 0}};
        const size_t counts[][2] = {{4, 3}, {4, 5}, {4, 4}};
        for (size_t i = 0; i < 3; ++i) {
            const ggml_kv_stream_span_source spans[] = {
                span(storage.lease, storage.lease, starts[i][0], counts[i][0], 0, 1024),
                span(storage.lease, storage.lease, starts[i][1], counts[i][1], 256, 1280),
            };
            auto * sentinel = reinterpret_cast<ggml_kv_stream_span_plan_t>(uintptr_t(1));
            const auto result = ggml_kv_stream_span_plan_make(shape(), spans, 2, 8, 1, sentinel);
            t.assert_true(result.status == status::invalid_span);
            t.assert_true(sentinel == reinterpret_cast<ggml_kv_stream_span_plan_t>(uintptr_t(1)));
        }
    });

    t.test("rejects_empty_spans_and_invalid_query_width", [](testing & t) {
        allocation storage;
        if (!t.assert_true(storage.valid())) return;
        ggml_kv_stream_span_plan_t plan = nullptr;
        auto value = span(storage.lease, storage.lease, 0, 8, 0, 1024);
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), nullptr, 0, 8, 1, plan).status == status::invalid_span);
        value.tokens = 0;
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 8, 1, plan).status == status::invalid_span);
        value.tokens = 8;
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 8, 0, plan).status == status::invalid_shape);
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 8, 9, plan).status == status::invalid_shape);
        t.assert_true(plan == nullptr);
    });

    t.test("validates_offsets_alignment_and_each_buffer_extent", [](testing & t) {
        allocation k(512);
        allocation v(512);
        if (!t.assert_true(k.valid() && v.valid())) return;

        ggml_kv_stream_span_plan_t plan = nullptr;
        auto value = span(k.lease, v.lease, 0, 4, 256, 256);
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan).status == status::success);
        ggml_kv_stream_span_plan_free(plan);
        plan = nullptr;

        value.k_offset = 257;
        auto result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan);
        t.assert_true(result.status == status::invalid_alignment && result.operand == operand::k);
        value.k_offset = 272;
        result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan);
        t.assert_true(result.status == status::invalid_buffer && result.operand == operand::k);
        value.k_offset = 256;
        value.v_offset = 272;
        result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan);
        t.assert_true(result.status == status::invalid_buffer && result.operand == operand::v);
        value.v_offset = 256;
        value.k_lease = nullptr;
        result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan);
        t.assert_true(result.status == status::invalid_buffer && result.operand == operand::k);
        t.assert_true(plan == nullptr);
    });

    t.test("checked_arithmetic_rejects_logical_and_byte_overflow", [](testing & t) {
        allocation storage;
        if (!t.assert_true(storage.valid())) return;
        ggml_kv_stream_span_plan_t plan = nullptr;
        auto value = span(storage.lease, storage.lease, std::numeric_limits<size_t>::max() - 1, 4, 0, 0);
        t.assert_true(ggml_kv_stream_span_plan_make(
            shape(), &value, 1, std::numeric_limits<size_t>::max(), 1, plan).status == status::overflow);
        value = span(storage.lease, storage.lease, 0, 2, std::numeric_limits<size_t>::max() - 63, 0);
        const auto result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 2, 1, plan);
        t.assert_true(result.status == status::overflow && result.operand == operand::k);
        t.assert_true(plan == nullptr);
    });

    t.test("plan_retains_arena_lease_until_final_release", [](testing & t) {
        allocation k(1024);
        allocation v(1024);
        if (!t.assert_true(k.valid() && v.valid())) return;
        void * const k_base = ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(k.lease));
        void * const v_base = ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(v.lease));
        const auto value = span(k.lease, v.lease, 0, 4, 0, 0);
        ggml_kv_stream_span_plan_t plan = nullptr;
        if (!t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan).status == status::success)) {
            return;
        }
        auto * retained = ggml_kv_stream_span_plan_retain(plan);
        t.assert_true(retained == plan);
        ggml_backend_memory_lease_free(k.lease);
        ggml_backend_memory_lease_free(v.lease);
        k.lease = nullptr;
        v.lease = nullptr;

        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(k.arena));
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(v.arena));
        t.assert_true(ggml_backend_memory_arena_quiesce(k.arena));
        t.assert_true(ggml_backend_memory_arena_quiesce(v.arena));
        t.assert_true(ggml_backend_memory_arena_get_state(k.arena) == GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING);
        t.assert_true(ggml_backend_memory_arena_get_state(v.arena) == GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING);

        ggml_kv_stream_span_plan_view view;
        t.assert_true(ggml_kv_stream_span_plan_get_view(plan, view));
        t.assert_true(ggml_backend_buffer_get_base(view.spans[0].k_buffer) == k_base);
        t.assert_true(ggml_backend_buffer_get_base(view.spans[0].v_buffer) == v_base);
        ggml_backend_buffer_clear(view.spans[0].k_buffer, 0xa5);
        ggml_backend_buffer_clear(view.spans[0].v_buffer, 0x5a);
        t.assert_equal(uint8_t(0xa5), static_cast<const uint8_t *>(k_base)[0]);
        t.assert_equal(uint8_t(0x5a), static_cast<const uint8_t *>(v_base)[0]);

        ggml_kv_stream_span_plan_free(plan);
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(k.arena));
        t.assert_true(ggml_kv_stream_span_plan_get_view(retained, view));
        ggml_kv_stream_span_plan_free(retained);
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(k.arena));
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(v.arena));
        t.assert_true(ggml_backend_memory_arena_get_state(k.arena) == GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
        t.assert_true(ggml_backend_memory_arena_get_state(v.arena) == GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
    });

    t.test("invalid_plan_view_leaves_output_unchanged", [](testing & t) {
        ggml_kv_stream_span_plan_view view;
        view.count = 77;
        t.assert_true(!ggml_kv_stream_span_plan_get_view(nullptr, view));
        t.assert_equal(size_t(77), view.count);
        t.assert_true(ggml_kv_stream_span_plan_retain(nullptr) == nullptr);
        ggml_kv_stream_span_plan_free(nullptr);
    });

    return t.summary();
}
