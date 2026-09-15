#include "../src/llama-kv-stream-publication.h"
#include "../ggml/src/ggml-backend-memory.h"
#include "testing.h"

#include <memory>
#include <vector>

using domain = llama_kv_stream_publication_domain;
using plane = llama_kv_stream_publication_plane;
using completion = llama_kv_stream_publication_completion;
using ticket = llama_kv_stream_publication_ticket;

static bool finish_pair(ticket & value, uint32_t pair, domain where, bool k = true, bool v = true) {
    completion ck, cv;
    if (!value.submit(pair, plane::k, where, ck) || !value.submit(pair, plane::v, where, cv)) return false;
    return ck.finish(k) && cv.finish(v);
}

static bool finish_batch(ticket & value, domain where) {
    for (uint32_t pair = 0; pair < value.pairs(); ++pair) if (!finish_pair(value, pair, where)) return false;
    return true;
}

struct lease_fixture {
    using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
    using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};

    lease_fixture() {
        arena.reset(ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(), 4096));
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena.get(), 0));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 9, 0, 4096, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 9));
        GGML_ASSERT(lease);
    }
};

static std::unique_ptr<llama_kv_stream_publications> publications(size_t tokens = 8, size_t capacity = 32,
        uint64_t generation = 7, uint64_t sequence = 1) {
    return llama_kv_stream_publications::create({tokens, capacity, generation, sequence});
}

int main() {
    testing t;

    t.test("initial_and_invalid_reservations_are_transactional", [](testing & t) {
        t.assert_true(!llama_kv_stream_publications::create({9, 8, 1, 1}));
        t.assert_true(!llama_kv_stream_publications::create({0, 8, 0, 1}));
        t.assert_true(!llama_kv_stream_publications::create({0, 8, 1, 0}));
        auto state = publications();
        if (!t.assert_true(state != nullptr)) return;
        const auto initial = state->frontiers();
        t.assert_equal(size_t(8), initial.reserved);
        t.assert_equal(size_t(8), initial.device);
        t.assert_equal(size_t(8), initial.host);
        t.assert_equal(size_t(8), initial.committed);
        t.assert_equal(uint64_t(7), state->generation());
        ticket valid;
        t.assert_true(state->reserve(8, 2, 2, {}, {}, valid));
        t.assert_true(valid.pending());
        t.assert_equal(uint64_t(1), valid.sequence());
        t.assert_equal(uint64_t(7), valid.generation());
        t.assert_equal(size_t(8), valid.first());
        t.assert_equal(size_t(2), valid.count());
        t.assert_equal(uint32_t(2), valid.pairs());
        ticket output;
        auto owner = std::make_shared<int>(3);
        const std::vector<std::shared_ptr<void>> null_owner{std::shared_ptr<void>{}};
        for (bool result : {
                state->reserve(8, 1, 1, {}, {}, output),
                state->reserve(10, 0, 1, {}, {}, output),
                state->reserve(10, 1, 0, {}, {}, output),
                state->reserve(10, 23, 1, {}, {}, output),
                state->reserve(10, 1, 1, null_owner, {}, output),
                state->reserve(10, 1, 1, {owner}, {nullptr}, output),
                state->reserve(10, 1, 1, {}, {}, valid)}) t.assert_true(!result);
        t.assert_true(!output.pending());
        t.assert_equal(size_t(10), state->frontiers().reserved);
        t.assert_equal(size_t(1), state->pending());
        t.assert_true(valid.cancel());
        t.assert_true(valid.retire());
    });

    t.test("kv_pairs_and_frontiers_publish_atomically", [](testing & t) {
        auto state = publications(); ticket value;
        if (!t.assert_true(state && state->reserve(8, 2, 2, {}, {}, value))) return;
        completion dk, dv;
        t.assert_true(value.submit(0, plane::k, domain::device, dk));
        t.assert_true(value.submit(0, plane::v, domain::device, dv));
        t.assert_true(dk.finish());
        t.assert_true(!value.ready(0, domain::device));
        t.assert_equal(size_t(8), state->frontiers().device);
        t.assert_true(dv.finish());
        t.assert_true(value.ready(0, domain::device));
        t.assert_equal(size_t(8), state->frontiers().device);
        t.assert_true(finish_pair(value, 1, domain::device));
        t.assert_equal(size_t(10), state->frontiers().device);
        completion hk, hv;
        t.assert_true(value.submit(0, plane::k, domain::host, hk));
        t.assert_true(value.submit(0, plane::v, domain::host, hv));
        t.assert_true(hk.finish());
        t.assert_true(!value.ready(0, domain::host));
        t.assert_equal(size_t(8), state->frontiers().host);
        t.assert_true(hv.finish());
        t.assert_true(finish_pair(value, 1, domain::host));
        const auto complete = state->frontiers();
        t.assert_equal(size_t(10), complete.host);
        t.assert_equal(size_t(10), complete.committed);
        t.assert_true(value.committed());
        t.assert_true(!value.cancel());
        t.assert_true(value.retire());
        t.assert_equal(size_t(0), state->pending());
    });

    t.test("host_and_device_frontiers_advance_independently", [](testing & t) {
        auto state = publications(); ticket value;
        if (!t.assert_true(state && state->reserve(8, 1, 1, {}, {}, value))) return;
        t.assert_true(finish_batch(value, domain::host));
        auto frontier = state->frontiers();
        t.assert_equal(size_t(8), frontier.device);
        t.assert_equal(size_t(9), frontier.host);
        t.assert_equal(size_t(8), frontier.committed);
        t.assert_true(!value.committed());
        t.assert_true(finish_batch(value, domain::device));
        frontier = state->frontiers();
        t.assert_equal(size_t(9), frontier.device);
        t.assert_equal(size_t(9), frontier.host);
        t.assert_equal(size_t(9), frontier.committed);
        t.assert_true(value.committed());
        t.assert_true(value.retire());
    });

    t.test("out_of_order_batches_do_not_skip_frontier_gaps", [](testing & t) {
        auto state = publications(); ticket first, second;
        if (!t.assert_true(state && state->reserve(8, 2, 1, {}, {}, first) && state->reserve(10, 3, 1, {}, {}, second))) return;
        t.assert_true(finish_batch(second, domain::host));
        t.assert_true(finish_batch(second, domain::device));
        t.assert_true(second.ready(0, domain::host) && second.ready(0, domain::device));
        t.assert_true(!second.committed());
        auto frontier = state->frontiers();
        t.assert_equal(size_t(8), frontier.device);
        t.assert_equal(size_t(8), frontier.host);
        t.assert_equal(size_t(8), frontier.committed);
        t.assert_true(finish_batch(first, domain::device));
        frontier = state->frontiers();
        t.assert_equal(size_t(13), frontier.device);
        t.assert_equal(size_t(8), frontier.host);
        t.assert_true(finish_batch(first, domain::host));
        frontier = state->frontiers();
        t.assert_equal(size_t(13), frontier.host);
        t.assert_equal(size_t(13), frontier.committed);
        t.assert_true(first.committed() && second.committed());
        t.assert_true(first.retire() && second.retire());
    });

    t.test("failed_or_cancelled_batches_close_without_publication", [](testing & t) {
        auto failed = publications(); ticket value; completion result;
        if (!t.assert_true(failed && failed->reserve(8, 1, 1, {}, {}, value) &&
                value.submit(0, plane::k, domain::device, result))) return;
        t.assert_true(result.finish(false));
        t.assert_true(value.failed() && failed->failed());
        t.assert_true(!value.ready(0, domain::device));
        t.assert_equal(size_t(8), failed->frontiers().committed);
        ticket rejected;
        t.assert_true(!failed->reserve(9, 1, 1, {}, {}, rejected));
        t.assert_true(value.retire());

        auto cancelled = publications(); ticket pending;
        if (!t.assert_true(cancelled && cancelled->reserve(8, 1, 1, {}, {}, pending))) return;
        t.assert_true(pending.cancel());
        t.assert_true(pending.cancel());
        t.assert_true(cancelled->failed());
        t.assert_equal(size_t(8), cancelled->frontiers().committed);
        t.assert_true(pending.retire());
    });

    t.test("tickets_retain_parent_and_lease_until_explicit_retirement", [](testing & t) {
        lease_fixture f;
        auto parent = std::make_shared<int>(7); std::weak_ptr<int> weak = parent;
        auto state = publications(); ticket value;
        if (!t.assert_true(state && state->reserve(8, 1, 1, {parent}, {f.lease.get()}, value))) return;
        parent.reset(); f.lease.reset();
        t.assert_true(!weak.expired());
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.arena.get()));
        t.assert_true(finish_batch(value, domain::device));
        t.assert_true(finish_batch(value, domain::host));
        t.assert_true(!weak.expired());
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.arena.get()));
        t.assert_true(value.retire());
        t.assert_true(weak.expired());
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.arena.get()));
    });

    t.test("abandoned_ticket_waits_for_submitted_completion", [](testing & t) {
        auto parent = std::make_shared<int>(11); std::weak_ptr<int> weak = parent;
        auto state = publications(); completion work;
        {
            ticket value;
            if (!t.assert_true(state && state->reserve(8, 1, 1, {parent}, {}, value) &&
                    value.submit(0, plane::k, domain::device, work))) return;
            parent.reset();
        }
        t.assert_true(state->failed());
        t.assert_true(!weak.expired());
        t.assert_true(work.finish());
        t.assert_true(weak.expired());
        t.assert_equal(size_t(0), state->pending());
    });

    t.test("completion_and_ticket_moves_preserve_failure_safety", [](testing & t) {
        auto state = publications(); ticket original;
        if (!t.assert_true(state && state->reserve(8, 1, 1, {}, {}, original))) return;
        completion first, second;
        t.assert_true(original.submit(0, plane::k, domain::device, first));
        t.assert_true(!original.submit(0, plane::k, domain::device, second));
        completion invalid;
        t.assert_true(!original.submit(1, plane::v, domain::host, invalid));
        t.assert_true(!original.submit(0, static_cast<plane>(9), domain::host, invalid));
        t.assert_true(!original.submit(0, plane::v, static_cast<domain>(9), invalid));
        t.assert_true(!invalid.pending());
        t.assert_true(!original.retire());
        completion moved(std::move(first));
        t.assert_true(!first.pending() && moved.pending());
        second = std::move(moved);
        t.assert_true(!moved.pending() && second.pending());
        t.assert_true(second.finish());
        t.assert_true(!second.finish());
        ticket owner(std::move(original));
        t.assert_true(!original.pending() && owner.pending());
        t.assert_true(owner.cancel() && owner.retire());
    });

    t.test("sequence_exhaustion_is_explicit_and_never_wraps", [](testing & t) {
        auto state = publications(0, 2, 1, UINT64_MAX); ticket last, rejected;
        if (!t.assert_true(state && state->reserve(0, 1, 1, {}, {}, last))) return;
        t.assert_equal(UINT64_MAX, last.sequence());
        t.assert_true(state->exhausted());
        t.assert_true(!state->reserve(1, 1, 1, {}, {}, rejected));
        t.assert_equal(size_t(1), state->frontiers().reserved);
        t.assert_true(last.cancel() && last.retire());
    });

    t.test("tracker_destruction_keeps_in_flight_resources_alive", [](testing & t) {
        auto parent = std::make_shared<int>(19); std::weak_ptr<int> weak = parent;
        ticket value; completion work;
        {
            auto state = publications();
            if (!t.assert_true(state && state->reserve(8, 1, 1, {parent}, {}, value) &&
                    value.submit(0, plane::v, domain::host, work))) return;
        }
        parent.reset(); value.cancel(); value = {};
        t.assert_true(!weak.expired());
        t.assert_true(work.finish());
        t.assert_true(weak.expired());
    });

    return t.summary();
}
