#include "memory-transition-test.h"

int main() {
    testing t;

    t.test("arena_metadata_enumeration", [](testing & t) {
        fixture f(t);
        ggml_backend_memory_region region{};
        t.assert_true(ggml_backend_memory_arena_get_region_at(f.arena.get(), 0, &region));
        t.assert_equal(uint64_t(11), region.id);
        t.assert_true(ggml_backend_memory_arena_get_region_at(f.arena.get(), 1, &region));
        t.assert_equal(uint64_t(12), region.id);
        t.assert_true(!ggml_backend_memory_arena_get_region_at(f.arena.get(), 2, &region));
        t.assert_equal(uint64_t(12), region.id);
        t.assert_true(!ggml_backend_memory_arena_get_region_at(nullptr, 0, &region));
        t.assert_true(!ggml_backend_memory_arena_get_region_at(f.arena.get(), 0, nullptr));
    });

    t.test("executor_quiesce_closes_submission_before_drain", [](testing & t) {
        fixture f(t);
        f.workspace.executor.quiesce();
        t.assert_true(!f.workspace.executor.ready());
        t.assert_true(!f.workspace.executor.acquire({f.workspace.lease.get()}, 1));
        t.assert_true(f.workspace.executor.matches({f.workspace.lease.get()}, 1));
    });

    t.test("ordered_activation_preserves_persistent_capture_and_old_lease", [](testing & t) {
        fixture f(t);
        const auto old_generation = ggml_backend_memory_lease_generation(f.preserved.lease.get());
        auto * old_view = ggml_backend_memory_lease_buffer(f.preserved.lease.get());
        f.workspace.work.push_back(f.workspace.executor.acquire({f.workspace.lease.get()}, 1));
        f.workspace.work.push_back(f.workspace.executor.acquire({f.workspace.lease.get()}, 1));
        f.preserved.work.push_back(f.preserved.executor.acquire({f.preserved.lease.get()}, 1));
        f.workspace.hook = [&](const std::string & name) {
            t.assert_equal(uint64_t(0), f.transition.admit());
            t.assert_true(f.transition.activate(f.arenas).status == status::busy);
            t.assert_true(ggml_backend_memory_arena_get_state(f.arena.get()) != GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
            if (name == "bind") {
                t.assert_equal(uint64_t(2), ggml_backend_memory_arena_generation(f.arena.get()));
            }
        };
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activated);
        const std::vector<std::string> expected = {
            "quiesce:11", "quiesce:12", "drain:11", "drain:12", "invalidate:11", "invalidate:12",
            "release:11", "release:12", "bind:11", "bind:12", "activate:11", "activate:12", "cleanup:12", "cleanup:11",
        };
        t.assert_true(f.events == expected);
        t.assert_equal(2, f.workspace.drained);
        t.assert_equal(0, f.preserved.drained);
        t.assert_equal(1, f.workspace.destroyed);
        t.assert_equal(0, f.preserved.destroyed);
        t.assert_equal(old_generation, ggml_backend_memory_lease_generation(f.preserved.lease.get()));
        t.assert_true(old_view == ggml_backend_memory_lease_buffer(f.preserved.lease.get()));
        t.assert_equal(size_t(1), f.preserved.executor.outstanding());
        t.assert_true(f.preserved.executor.ready());
        t.assert_true(f.transition.active_layout() != nullptr);
        t.assert_true(f.transition.pending_layout() == nullptr);
        t.assert_true(f.transition.state() == state::idle);
        t.assert_true(ggml_backend_memory_arena_get_state(f.arena.get()) == GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
        const auto admission = f.transition.admit();
        t.assert_true(admission != 0 && f.transition.finish(admission));
    });

    t.test("unchanged_nonpersistent_view_is_rebound_when_arena_changes", [](testing & t) {
        fixture f(t, false);
        auto * old_view = ggml_backend_memory_lease_buffer(f.preserved.lease.get());
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activated);
        t.assert_equal(1, f.preserved.destroyed);
        t.assert_true(old_view != ggml_backend_memory_lease_buffer(f.preserved.lease.get()));
        t.assert_equal(uint64_t(2), ggml_backend_memory_lease_generation(f.preserved.lease.get()));
    });

    t.test("identical_layout_avoids_commit_but_runtime_change_retires_capture", [](testing & t) {
        fixture f(t);
        if (!f.prepare(t)) return;
        if (!t.assert_true(f.transition.activate(f.arenas).status == status::activated)) return;
        const auto generation = ggml_backend_memory_arena_generation(f.arena.get());
        f.workspace.runtime_changed = true;
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activated);
        t.assert_equal(generation, ggml_backend_memory_arena_generation(f.arena.get()));
        t.assert_equal(2, f.workspace.destroyed);
        t.assert_equal(0, f.preserved.destroyed);
        f.workspace.unchanged = f.preserved.unchanged = true;
        t.assert_true(f.transition.prepare(f.target).status == status::no_change);
        t.assert_true(f.transition.active_layout() != nullptr);
    });

    t.test("invalid_arena_preflight_has_no_callbacks_or_gate_changes", [](testing & t) {
        fixture f(t);
        if (!f.prepare(t)) return;
        auto wrong = f.arenas;
        wrong[0].domain = 99;
        t.assert_true(f.transition.activate(wrong).status == status::invalid_arena);
        wrong = f.arenas;
        wrong[0].arena = nullptr;
        t.assert_true(f.transition.activate(wrong).status == status::invalid_arena);
        t.assert_true(f.transition.activate({}).status == status::invalid_arena);
        t.assert_true(f.events.empty());
        t.assert_true(f.transition.state() == state::prepared);
        t.assert_true(f.workspace.executor.ready());
        t.assert_true(ggml_backend_memory_arena_get_state(f.arena.get()) == GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
        t.assert_true(f.transition.cancel());
    });

    t.test("stale_fixed_snapshot_is_rejected_before_quiesce", [](testing & t) {
        fixture f(t);
        f.target.fixed[0].region.offset = 3*f.a;
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::invalid_arena);
        t.assert_true(f.events.empty());
        t.assert_true(f.transition.cancel());
    });

    t.test("incomplete_drain_does_not_release_or_commit", [](testing & t) {
        fixture f(t);
        f.workspace.work.push_back(f.workspace.executor.acquire({f.workspace.lease.get()}, 1));
        f.workspace.complete = false;
        if (!f.prepare(t)) return;
        const auto result = f.transition.activate(f.arenas);
        t.assert_true(result.status == status::activation_failed);
        t.assert_true(result.failed_at == state::draining);
        t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(f.arena.get()));
        t.assert_true(f.workspace.lease != nullptr);
        t.assert_equal(0, f.workspace.destroyed);
        t.assert_true(f.transition.state() == state::failed);
        t.assert_equal(uint64_t(0), f.transition.admit());
        t.assert_true(!f.transition.cancel());
        t.assert_true(f.transition.pending_layout() != nullptr);
    });

    t.test("outstanding_external_lease_blocks_commit_and_activation", [](testing & t) {
        fixture f(t);
        lease_ptr external(ggml_backend_memory_arena_acquire(f.arena.get(), 11), ggml_backend_memory_lease_free);
        if (!f.prepare(t)) return;
        const auto result = f.transition.activate(f.arenas);
        t.assert_true(result.status == status::activation_failed);
        t.assert_true(result.failed_at == state::committing);
        t.assert_equal(size_t(0), result.arena);
        t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(f.arena.get()));
        t.assert_equal(0, f.workspace.activations);
        t.assert_true(f.transition.active_layout() == nullptr);
        t.assert_equal(uint64_t(0), f.transition.admit());
    });

    t.test("callback_failures_never_reopen_execution", [](testing & t) {
        for (const auto & step : {"quiesce", "drain", "invalidate", "release", "bind", "activate"}) {
            fixture f(t);
            f.workspace.fail_at = step;
            if (!f.prepare(t)) return;
            const auto result = f.transition.activate(f.arenas);
            t.assert_true(result.status == status::activation_failed);
            t.assert_equal(size_t(0), result.consumer);
            t.assert_true(f.transition.state() == state::failed);
            t.assert_equal(uint64_t(0), f.transition.admit());
            t.assert_true(!f.transition.cancel());
            t.assert_true(ggml_backend_memory_arena_get_state(f.arena.get()) != GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
        }
    });

    t.test("callback_exception_retains_failure_diagnostics", [](testing & t) {
        fixture f(t);
        f.workspace.throw_at = "bind";
        if (!f.prepare(t)) return;
        const auto result = f.transition.activate(f.arenas);
        t.assert_true(result.status == status::consumer_exception);
        t.assert_true(result.failed_at == state::binding);
        t.assert_true(result.exception != nullptr);
        t.assert_true(f.transition.state() == state::failed);
    });

    t.test("cancellation_after_quiesce_stays_closed", [](testing & t) {
        fixture f(t);
        f.workspace.hook = [&](const std::string & step) {
            if (step == "drain") {
                t.assert_true(f.transition.cancel());
            }
        };
        if (!f.prepare(t)) return;
        const auto result = f.transition.activate(f.arenas);
        t.assert_true(result.status == status::cancelled);
        t.assert_true(result.failed_at == state::draining);
        t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(f.arena.get()));
        t.assert_true(f.transition.state() == state::failed);
        t.assert_true(f.workspace.lease != nullptr);
    });


    t.test("native_alignment_capacity_and_existing_gate_are_checked", [](testing & t) {
        fixture f(t);
        f.target.plan.stages[0].requirements[0].alignment = 2*f.a;
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::invalid_arena);
        t.assert_true(f.transition.cancel());
        f.events.clear(); // Cancellation cleanup belongs to the previous proposal.
        f.target.plan.stages[0].requirements[0].alignment = f.a;
        if (!f.prepare(t)) return;
        arena_ptr small(ggml_backend_memory_arena_new(f.buft, f.a), ggml_backend_memory_arena_free);
        t.assert_true(f.transition.activate({{1, LLAMA_MEMORY_ALLOCATION_HOST, small.get()}}).status == status::invalid_arena);
        GGML_ASSERT(ggml_backend_memory_arena_quiesce(f.arena.get()));
        t.assert_true(f.transition.activate(f.arenas).status == status::invalid_arena);
        t.assert_true(ggml_backend_memory_arena_get_state(f.arena.get()) != GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
        GGML_ASSERT(ggml_backend_memory_arena_resume(f.arena.get()));
        t.assert_true(f.events.empty());
        t.assert_true(f.transition.cancel());
    });

    t.test("successful_parent_binding_cannot_be_silently_substituted", [](testing & t) {
        fixture f(t);
        if (!f.prepare(t)) return;
        if (!t.assert_true(f.transition.activate(f.arenas).status == status::activated)) return;
        auto other = make_arena(f.buft);
        f.events.clear();
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate({{1, LLAMA_MEMORY_ALLOCATION_HOST, other.get()}}).status == status::invalid_arena);
        t.assert_true(f.events.empty());
        t.assert_true(f.transition.active_layout() != nullptr);
        t.assert_true(f.transition.cancel());
        t.assert_true(f.workspace.executor.ready());
    });

    t.test("partial_activation_is_not_published_or_admitted", [](testing & t) {
        fixture f(t);
        f.preserved.fail_at = "activate";
        if (!f.prepare(t)) return;
        const auto result = f.transition.activate(f.arenas);
        t.assert_true(result.status == status::activation_failed);
        t.assert_true(result.failed_at == state::activating);
        t.assert_equal(size_t(1), result.consumer);
        t.assert_equal(1, f.workspace.activations);
        t.assert_equal(0, f.preserved.activations);
        t.assert_true(f.transition.active_layout() == nullptr);
        t.assert_true(f.transition.pending_layout() != nullptr);
        t.assert_equal(uint64_t(0), f.transition.admit());
    });

    t.test("partial_multi_arena_commit_remains_failed_closed", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto * buft = ggml_backend_get_default_buffer_type(backend.get());
        const size_t a = ggml_backend_buft_get_alignment(buft);
        auto single = [&](uint64_t id) {
            arena_ptr arena(ggml_backend_memory_arena_new(buft, 4*a), ggml_backend_memory_arena_free);
            GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), 0));
            GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), id, 0, a, a, 0, nullptr));
            GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
            return arena;
        };
        auto one = single(11);
        auto two = single(12);
        std::vector<std::string> events;
        component first(t, one.get(), 11, events), second(t, two.get(), 12, events);
        llama_memory_transition transition({&first, &second});
        llama_memory_transition_target target{
            {
                {{1, LLAMA_MEMORY_ALLOCATION_HOST, 0}, {2, LLAMA_MEMORY_ALLOCATION_HOST, 0}},
                {{11, 1, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::discardable},
                 {12, 2, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::discardable}},
                {{10, {}, {{11, 2*a, 2*a, a, LLAMA_MEMORY_ACCESS_WRITE, 0},
                           {12, 2*a, 2*a, a, LLAMA_MEMORY_ACCESS_WRITE, 0}}}}, {}, {},
            },
            10, {{1, LLAMA_MEMORY_ALLOCATION_HOST, 4*a, a}, {2, LLAMA_MEMORY_ALLOCATION_HOST, 4*a, a}}, {},
        };
        lease_ptr external(ggml_backend_memory_arena_acquire(two.get(), 12), ggml_backend_memory_lease_free);
        t.assert_true(transition.prepare(target).status == status::prepared);
        const auto result = transition.activate({{1, LLAMA_MEMORY_ALLOCATION_HOST, one.get()},
                                                 {2, LLAMA_MEMORY_ALLOCATION_HOST, two.get()}});
        t.assert_true(result.status == status::activation_failed);
        t.assert_true(result.failed_at == state::committing);
        t.assert_equal(size_t(1), result.arena);
        t.assert_equal(uint64_t(2), ggml_backend_memory_arena_generation(one.get()));
        t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(two.get()));
        t.assert_true(ggml_backend_memory_arena_get_state(one.get()) != GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
        t.assert_true(ggml_backend_memory_arena_get_state(two.get()) != GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
        t.assert_equal(0, first.activations);
        t.assert_equal(0, second.activations);
        t.assert_equal(uint64_t(0), transition.admit());
        t.assert_true(!transition.cancel());
    });

    return t.summary();
}
