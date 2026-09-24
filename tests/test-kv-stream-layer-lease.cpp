#include "../src/llama-kv-stream-layer-lease.h"
#include "../ggml/src/ggml-kv-stream-reference.h"
#include "testing.h"

#include <algorithm>
#include <memory>
#include <vector>

struct fixture {
    llama_kv_stream_policy_config policy;
    llama_kv_stream_policy_state state;
    ggml_backend_memory_arena_t arena = nullptr;
    ggml_backend_memory_lease_t pool = nullptr;
    size_t active_tokens = 4*256 - 7;

    fixture(uint32_t pages = 6, uint32_t ring = 4) {
        policy.shape = {GGML_TYPE_F16, GGML_TYPE_F16, 32, 32, 2, 256, 128};
        policy.capabilities = {
            {GGML_TYPE_F16, true, true, true, true},
            {GGML_TYPE_F16, true, true, true, true},
            true, true,
        };
        ggml_kv_stream_layout page;
        GGML_ASSERT(ggml_kv_stream_layout_make(policy.shape, 256, page).status == ggml_kv_stream_status::success);
        policy.pool_bytes = size_t(pages)*page.bytes;
        policy.layers = 2;
        policy.initial_ring_slots = ring;
        policy.fixed_ring = true;
        GGML_ASSERT(llama_kv_stream_policy_initialize(policy, state).status == llama_kv_stream_policy_status::success);
        GGML_ASSERT(state.ring_slots == ring);

        arena = ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(), policy.pool_bytes);
        ggml_backend_memory_region region;
        GGML_ASSERT(arena);
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena, GGML_BACKEND_MEMORY_PLAN_NONE));
        GGML_ASSERT(ggml_backend_memory_arena_reserve(
            arena, 1, policy.pool_bytes, 128, GGML_BACKEND_MEMORY_REGION_NONE, &region));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena));
        pool = ggml_backend_memory_arena_acquire(arena, 1);
        GGML_ASSERT(pool);
    }

    ~fixture() {
        ggml_backend_memory_lease_free(pool);
        ggml_backend_memory_arena_free(arena);
    }

    llama_kv_stream_layer_lease_layout layout(
            uint64_t revision = 11, uint64_t generation = 29) const {
        return {policy, state, active_tokens, revision, generation};
    }
};

static llama_kv_stream_complete_layer_request request(
        uint32_t layer, uint32_t queries, uint64_t revision = 11, uint64_t generation = 29) {
    return {layer, queries, revision, generation};
}

int main() {
    testing t;

    t.test("complete_plan_covers_resident_prefix_and_retained_ring_suffix", [](testing & t) {
        fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto * lease = owner->acquire(request(0, 4));
        if (!t.assert_true(lease != nullptr)) return;

        t.assert_equal(uint32_t(0), llama_kv_stream_complete_layer_lease_layer(lease));
        t.assert_equal(size_t(256), llama_kv_stream_complete_layer_lease_resident_tokens(lease));
        t.assert_equal(size_t(3), llama_kv_stream_complete_layer_lease_ring_slots(lease));
        t.assert_equal(size_t(0), llama_kv_stream_complete_layer_lease_ring_first(lease));
        t.assert_equal(uint64_t(11), llama_kv_stream_complete_layer_lease_layout_revision(lease));
        t.assert_equal(uint64_t(29), llama_kv_stream_complete_layer_lease_content_generation(lease));

        auto * plan = llama_kv_stream_complete_layer_lease_plan(lease);
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(plan && ggml_kv_stream_span_plan_get_view(plan, view))) return;
        t.assert_equal(size_t(2), view.count);
        t.assert_equal(f.active_tokens, view.active_tokens);
        t.assert_equal(size_t(4), view.query_tokens);
        t.assert_equal(size_t(0), view.spans[0].token_begin);
        t.assert_equal(size_t(256), view.spans[0].tokens);
        t.assert_equal(size_t(256), view.spans[1].token_begin);
        t.assert_equal(f.active_tokens - 256, view.spans[1].tokens);
        llama_kv_stream_policy_layout physical;
        GGML_ASSERT(llama_kv_stream_policy_layout_make(
            f.policy, f.state, f.active_tokens, physical).status == llama_kv_stream_policy_status::success);
        t.assert_equal(physical.layers[0].offset, view.spans[0].k_offset);
        t.assert_equal(physical.layers[0].offset + physical.layers[0].planes.v_offset, view.spans[0].v_offset);
        t.assert_equal(size_t(0), view.spans[1].k_offset);
        t.assert_equal(physical.ring.v_offset, view.spans[1].v_offset);

        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(
            ggml_backend_memory_lease_buffer(f.pool)));
        const size_t resident = llama_kv_stream_complete_layer_lease_resident_tokens(lease);
        for (size_t token = 0; token < f.active_tokens; ++token) {
            const bool in_resident = token < resident;
            const size_t local = in_resident ? token : token - resident;
            const size_t k_offset = in_resident ? physical.layers[0].offset : 0;
            const size_t v_offset = in_resident ?
                physical.layers[0].offset + physical.layers[0].planes.v_offset : physical.ring.v_offset;
            const auto & planes = in_resident ? physical.layers[0].planes : physical.ring;
            for (size_t head = 0; head < 2; ++head) {
                auto * k = reinterpret_cast<ggml_fp16_t *>(
                    base + k_offset + local*planes.k_token_bytes + head*planes.k_row_bytes);
                auto * v = reinterpret_cast<ggml_fp16_t *>(
                    base + v_offset + local*planes.v_token_bytes + head*planes.v_row_bytes);
                std::fill_n(k, 32, ggml_fp32_to_fp16(0));
                std::fill_n(v, 32, ggml_fp32_to_fp16(float(token)));
            }
        }
        std::vector<float> q(4*2*32, 0);
        ggml_kv_stream_attention_reference_output output;
        const ggml_kv_stream_attention_reference_input input{
            q.data(), q.size(), nullptr, 0, 2, 1.0f, 0};
        if (!t.assert_true(ggml_kv_stream_attention_reference(
                plan, input, output).status == ggml_kv_stream_reference_status::success)) return;
        for (float value : output.value) t.assert_equal(508.0f, value);
        t.assert_equal(size_t(1), owner->active_reservations());
        t.assert_equal(size_t(1), owner->active_leases());
        t.assert_equal(size_t(3), owner->ring_slots_used());
        t.assert_true(!owner->can_repartition());

        llama_kv_stream_complete_layer_lease_free(lease);
        t.assert_equal(size_t(0), owner->active_reservations());
        t.assert_equal(size_t(0), owner->active_leases());
        t.assert_equal(size_t(0), owner->ring_slots_used());
        t.assert_true(owner->can_repartition());
    });

    t.test("different_query_widths_share_one_physical_reservation", [](testing & t) {
        fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto * catchup = owner->acquire(request(0, 4));
        auto * draft = owner->acquire(request(0, 1));
        if (!t.assert_true(catchup && draft)) return;
        t.assert_equal(
            llama_kv_stream_complete_layer_lease_ring_first(catchup),
            llama_kv_stream_complete_layer_lease_ring_first(draft));
        t.assert_equal(size_t(1), owner->active_reservations());
        t.assert_equal(size_t(2), owner->active_leases());
        t.assert_equal(size_t(3), owner->ring_slots_used());
        t.assert_true(owner->acquire(request(1, 1)) == nullptr);

        auto * retained = llama_kv_stream_complete_layer_lease_retain(draft);
        t.assert_true(retained == draft);
        llama_kv_stream_complete_layer_lease_free(catchup);
        llama_kv_stream_complete_layer_lease_free(draft);
        t.assert_equal(size_t(1), owner->active_leases());
        t.assert_equal(size_t(3), owner->ring_slots_used());
        ggml_kv_stream_span_plan_view view;
        t.assert_true(ggml_kv_stream_span_plan_get_view(
            llama_kv_stream_complete_layer_lease_plan(retained), view));
        t.assert_equal(size_t(1), view.query_tokens);
        llama_kv_stream_complete_layer_lease_free(retained);
        t.assert_equal(size_t(0), owner->active_leases());
        t.assert_equal(size_t(0), owner->ring_slots_used());
    });

    t.test("live_reservation_rejects_repartition_until_final_retirement", [](testing & t) {
        fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto * lease = owner->acquire(request(0, 1));
        if (!t.assert_true(lease != nullptr)) return;

        auto changed = f.layout(12, 29);
        changed.policy.initial_ring_slots = 2;
        GGML_ASSERT(llama_kv_stream_policy_initialize(
            changed.policy, changed.state).status == llama_kv_stream_policy_status::success);
        t.assert_true(!owner->rebind(changed));
        t.assert_equal(uint64_t(11), owner->layout_revision());
        llama_kv_stream_complete_layer_lease_free(lease);
        t.assert_true(owner->rebind(changed));
        t.assert_equal(uint64_t(12), owner->layout_revision());
        t.assert_equal(uint64_t(29), owner->content_generation());

        t.assert_true(owner->acquire(request(0, 1, 11, 29)) == nullptr);
        auto * current = owner->acquire(request(0, 1, 12, 29));
        if (!t.assert_true(current != nullptr)) return;
        t.assert_equal(size_t(512), llama_kv_stream_complete_layer_lease_resident_tokens(current));
        t.assert_equal(size_t(2), llama_kv_stream_complete_layer_lease_ring_slots(current));
        llama_kv_stream_complete_layer_lease_free(current);
    });

    t.test("generation_layer_and_capacity_failures_roll_back_ring_ownership", [](testing & t) {
        fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        for (const auto & invalid : {
                request(2, 1), request(0, 0), request(0, uint32_t(f.active_tokens + 1)),
                request(0, 1, 10, 29), request(0, 1, 11, 28)}) {
            t.assert_true(owner->acquire(invalid) == nullptr);
            t.assert_equal(size_t(0), owner->active_reservations());
            t.assert_equal(size_t(0), owner->ring_slots_used());
        }

        fixture insufficient(5, 1);
        auto small = llama_kv_stream_layer_lease_owner::create(insufficient.pool, insufficient.layout());
        if (!t.assert_true(bool(small))) return;
        t.assert_true(small->acquire(request(0, 1)) == nullptr);
        t.assert_equal(size_t(0), small->active_reservations());
        t.assert_equal(size_t(0), small->ring_slots_used());

        ggml_backend_memory_arena_t tiny_arena =
            ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(), 128);
        ggml_backend_memory_region region;
        GGML_ASSERT(tiny_arena);
        GGML_ASSERT(ggml_backend_memory_arena_begin(tiny_arena, GGML_BACKEND_MEMORY_PLAN_NONE));
        GGML_ASSERT(ggml_backend_memory_arena_reserve(
            tiny_arena, 7, 128, 128, GGML_BACKEND_MEMORY_REGION_NONE, &region));
        GGML_ASSERT(ggml_backend_memory_arena_commit(tiny_arena));
        auto * tiny = ggml_backend_memory_arena_acquire(tiny_arena, 7);
        t.assert_true(!llama_kv_stream_layer_lease_owner::create(tiny, f.layout()));
        ggml_backend_memory_lease_free(tiny);
        ggml_backend_memory_arena_free(tiny_arena);
    });

    t.test("close_and_owner_destruction_preserve_delayed_consumer", [](testing & t) {
        fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto * lease = owner->acquire(request(0, 4));
        if (!t.assert_true(lease != nullptr)) return;
        auto * delayed = llama_kv_stream_complete_layer_lease_retain(lease);
        owner->close();
        t.assert_true(owner->closed());
        t.assert_true(owner->acquire(request(0, 1)) == nullptr);
        t.assert_true(!owner->rebind(f.layout(12, 30)));
        owner.reset();

        llama_kv_stream_complete_layer_lease_free(lease);
        ggml_kv_stream_span_plan_view view;
        t.assert_true(ggml_kv_stream_span_plan_get_view(
            llama_kv_stream_complete_layer_lease_plan(delayed), view));
        t.assert_equal(f.active_tokens, view.active_tokens);
        llama_kv_stream_complete_layer_lease_free(delayed);
    });

    t.test("ring_guard_freezes_mtp_slots_until_target_sequence_retires", [](testing & t) {
        fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto * mtp = owner->acquire(request(0, 1));
        if (!t.assert_true(mtp != nullptr)) return;
        auto guard = owner->hold_ring();
        if (!t.assert_true(bool(guard))) return;
        t.assert_true(guard->pool_buffer() == ggml_backend_memory_lease_buffer(f.pool));
        t.assert_equal(size_t(4), guard->blocked_slots().size());
        for (size_t i = 0; i < 3; ++i) t.assert_equal(uint8_t(1), guard->blocked_slots()[i]);
        t.assert_equal(uint8_t(0), guard->blocked_slots()[3]);
        llama_kv_stream_complete_layer_lease_free(mtp);
        t.assert_equal(size_t(0), owner->ring_slots_used());
        t.assert_true(!owner->can_repartition());
        t.assert_true(owner->acquire(request(1, 1)) == nullptr);
        t.assert_true(!owner->rebind(f.layout(12, 29)));
        guard.reset();
        t.assert_true(owner->can_repartition());
        t.assert_true(owner->rebind(f.layout(12, 29)));
        auto * next = owner->acquire(request(1, 1, 12, 29));
        t.assert_true(next != nullptr);
        llama_kv_stream_complete_layer_lease_free(next);
    });
    return t.summary();
}
