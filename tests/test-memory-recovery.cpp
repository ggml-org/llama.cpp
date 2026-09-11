#include "memory-transition-test.h"
#include "../ggml/src/ggml-backend-impl.h"

// Inject a backend view-creation failure without changing production allocation code.
struct view_fault {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_t, size_t, size_t);
    inline static view_fault * current = nullptr;
    ggml_backend_buffer_t parent;
    factory original;
    size_t remaining;
    size_t attempts = 0;
    int exception_kind = 0;

    view_fault(ggml_backend_memory_arena_t arena, size_t successes) :
        parent(ggml_backend_memory_arena_parent(arena)), original(parent->view_buffer), remaining(successes) {
        GGML_ASSERT(current == nullptr && original != nullptr);
        current = this;
        parent->view_buffer = create;
    }
    ~view_fault() {
        parent->view_buffer = original;
        current = nullptr;
    }
    view_fault(const view_fault &) = delete;
    view_fault & operator=(const view_fault &) = delete;

    static ggml_backend_buffer_t create(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
        GGML_ASSERT(current && buffer == current->parent);
        ++current->attempts;
        if (current->remaining == 0) {
            if (current->exception_kind == 1) throw std::bad_alloc();
            if (current->exception_kind == 2) throw std::runtime_error("view factory");
            return nullptr;
        }
        --current->remaining;
        return current->original(buffer, offset, size);
    }
};

// Check consumer state and real old extents, not only the coordinator's logical snapshot.
static void check_restored(testing & t, fixture & f, int value = 11) {
    t.assert_true(f.transition.state() == state::idle);
    t.assert_true(f.transition.pending_layout() == nullptr);
    t.assert_true(f.transition.last_failure() == nullptr);
    if (!t.assert_true(f.workspace.lease != nullptr)) return;
    ggml_backend_memory_region region{};
    t.assert_true(ggml_backend_memory_lease_get_region(f.workspace.lease.get(), &region));
    t.assert_equal(f.a, region.size);
    t.assert_equal(value, *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(f.workspace.lease.get()))));
    t.assert_equal(uint64_t(1), f.workspace.revision);
    t.assert_true(f.workspace.executor.ready());
    t.assert_true(f.preserved.executor.ready());
    t.assert_true(ggml_backend_memory_arena_get_state(f.arena.get()) == GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
    const auto admission = f.transition.admit();
    t.assert_true(admission != 0 && f.transition.finish(admission));
}

int main() {
    testing t;

    t.test("recover_requires_failed_activation", [](testing & t) {
        fixture f(t);
        t.assert_true(f.transition.recover().status == status::not_failed);
        t.assert_true(f.transition.last_failure() == nullptr);
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.recover().status == status::busy);
        t.assert_true(f.transition.cancel());
    });

    t.test("recover_each_forward_callback_boundary", [](testing & t) {
        for (const auto & step : {"quiesce", "drain", "invalidate", "release", "bind", "activate"}) {
            fixture f(t);
            f.workspace.fail_at = step;
            if (!f.prepare(t)) return;
            const auto failed = f.transition.activate(f.arenas);
            t.assert_true(failed.status == status::activation_failed);
            if (t.assert_true(f.transition.last_failure() != nullptr)) {
                t.assert_true(f.transition.last_failure()->failed_at == failed.failed_at);
            }
            f.workspace.fail_at.clear();
            if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
            check_restored(t, f);
        }
    });

    t.test("recovery_backup_is_taken_after_old_work_completes", [](testing & t) {
        fixture f(t);
        f.workspace.work.push_back(f.workspace.executor.acquire({f.workspace.lease.get()}, 1));
        f.workspace.complete = false;
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).failed_at == state::draining);
        f.workspace.complete = true;
        if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
        check_restored(t, f, 12);
        t.assert_equal(1, f.workspace.drained);
    });

    t.test("partial_activation_restores_data_and_runtime_revision", [](testing & t) {
        fixture f(t);
        f.preserved.fail_at = "activate";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).failed_at == state::activating);
        if (!t.assert_true(f.workspace.lease != nullptr)) return;
        *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(f.workspace.lease.get()))) = 999;
        f.preserved.fail_at.clear();
        if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
        check_restored(t, f);
        t.assert_equal(2, f.workspace.destroyed);
        t.assert_equal(0, f.preserved.destroyed);
    });

    t.test("last_successful_snapshot_survives_recovery_and_retry", [](testing & t) {
        fixture f(t);
        f.target.plan.stages[0].requirements[0].size_min = f.a;
        f.target.plan.stages[0].requirements[0].size_preferred = f.a;
        if (!f.prepare(t)) return;
        if (!t.assert_true(f.transition.activate(f.arenas).status == status::activated)) return;
        const auto * original = f.transition.active_layout();
        f.target.plan.stages[0].requirements[0].size_min = 2*f.a;
        f.target.plan.stages[0].requirements[0].size_preferred = 2*f.a;
        f.workspace.fail_at = "bind";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activation_failed);
        f.workspace.fail_at.clear();
        if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
        t.assert_true(f.transition.active_layout() == original);
        check_restored(t, f);
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activated);
    });

    t.test("persistent_capture_survives_forward_and_restore_commits", [](testing & t) {
        fixture f(t);
        const auto generation = ggml_backend_memory_lease_generation(f.preserved.lease.get());
        auto * view = ggml_backend_memory_lease_buffer(f.preserved.lease.get());
        f.preserved.work.push_back(f.preserved.executor.acquire({f.preserved.lease.get()}, 1));
        f.workspace.fail_at = "bind";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activation_failed);
        f.workspace.fail_at.clear();
        if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
        t.assert_equal(uint64_t(3), ggml_backend_memory_arena_generation(f.arena.get()));
        t.assert_equal(generation, ggml_backend_memory_lease_generation(f.preserved.lease.get()));
        t.assert_true(view == ggml_backend_memory_lease_buffer(f.preserved.lease.get()));
        t.assert_equal(size_t(1), f.preserved.executor.outstanding());
        t.assert_equal(0, f.preserved.destroyed);
    });

    t.test("consumer_without_provable_restore_invalidates_session", [](testing & t) {
        fixture f(t);
        f.workspace.fail_at = "bind";
        f.workspace.can_recover = false;
        if (!f.prepare(t)) return;
        const auto original = f.transition.activate(f.arenas);
        t.assert_true(original.status == status::activation_failed);
        t.assert_true(f.transition.recover().status == status::session_invalid);
        t.assert_true(f.transition.state() == state::invalid);
        t.assert_equal(uint64_t(0), f.transition.admit());
        t.assert_true(!f.transition.cancel());
        t.assert_true(f.transition.recover().status == status::session_invalid);
        if (t.assert_true(f.transition.last_failure() != nullptr)) {
            t.assert_true(f.transition.last_failure()->status == original.status);
            t.assert_true(f.transition.last_failure()->failed_at == original.failed_at);
        }
    });

    t.test("failed_restore_phase_is_terminal_and_closed", [](testing & t) {
        for (const auto & step : {"restore_quiesce", "restore_drain", "restore_invalidate", "restore_release", "restore_bind", "restore_activate"}) {
            fixture f(t);
            f.workspace.fail_at = "bind";
            if (!f.prepare(t)) return;
            t.assert_true(f.transition.activate(f.arenas).status == status::activation_failed);
            f.workspace.fail_at = step;
            const auto result = f.transition.recover();
            t.assert_true(result.status == status::recovery_failed);
            t.assert_true(f.transition.state() == state::invalid);
            t.assert_equal(uint64_t(0), f.transition.admit());
            t.assert_true(!f.transition.cancel());
            t.assert_true(ggml_backend_memory_arena_get_state(f.arena.get()) != GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
        }
    });

    t.test("restore_exception_does_not_replace_original_failure", [](testing & t) {
        fixture f(t);
        f.workspace.fail_at = "bind";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activation_failed);
        f.workspace.fail_at.clear();
        f.workspace.throw_at = "restore_bind";
        const auto result = f.transition.recover();
        t.assert_true(result.status == status::recovery_failed);
        t.assert_true(result.exception != nullptr);
        if (t.assert_true(f.transition.last_failure() != nullptr)) {
            t.assert_true(f.transition.last_failure()->status == status::activation_failed);
            t.assert_true(f.transition.last_failure()->exception == nullptr);
        }
    });

    t.test("foreign_committed_metadata_is_not_overwritten", [](testing & t) {
        fixture f(t);
        f.preserved.fail_at = "release";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).failed_at == state::releasing);
        GGML_ASSERT(ggml_backend_memory_arena_begin(f.arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(f.arena.get(), 11, 0, f.a, f.a, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(f.arena.get(), 99, 3*f.a, f.a, f.a, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(f.arena.get()));
        const auto generation = ggml_backend_memory_arena_generation(f.arena.get());
        t.assert_true(f.transition.recover().status == status::session_invalid);
        t.assert_true(f.transition.state() == state::invalid);
        t.assert_equal(generation, ggml_backend_memory_arena_generation(f.arena.get()));
        ggml_backend_memory_region extra{};
        t.assert_true(ggml_backend_memory_arena_get_region(f.arena.get(), 99, &extra));
    });

    t.test("partial_view_creation_failure_restores_original_views", [](testing & t) {
        fixture f(t, false);
        if (!f.prepare(t)) return;
        {
            view_fault fault(f.arena.get(), 1);
            const auto result = f.transition.activate(f.arenas);
            t.assert_true(result.failed_at == state::committing);
            t.assert_equal(size_t(2), fault.attempts);
        }
        t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(f.arena.get()));
        if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
        t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(f.arena.get()));
        check_restored(t, f);
    });

    t.test("view_creation_failure_during_restore_invalidates_session", [](testing & t) {
        fixture f(t);
        f.workspace.fail_at = "bind";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activation_failed);
        f.workspace.fail_at.clear();
        {
            view_fault fault(f.arena.get(), 0);
            const auto result = f.transition.recover();
            t.assert_true(result.status == status::recovery_failed);
            t.assert_true(result.failed_at == state::committing);
            t.assert_equal(size_t(1), fault.attempts);
            t.assert_true(f.transition.state() == state::invalid);
        }
        t.assert_equal(uint64_t(2), ggml_backend_memory_arena_generation(f.arena.get()));
        t.assert_equal(uint64_t(0), f.transition.admit());
    });

    t.test("external_candidate_lease_blocks_metadata_restoration", [](testing & t) {
        fixture f(t);
        f.workspace.fail_at = "bind";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activation_failed);
        GGML_ASSERT(ggml_backend_memory_arena_resume(f.arena.get()));
        lease_ptr external(ggml_backend_memory_arena_acquire(f.arena.get(), 11), ggml_backend_memory_lease_free);
        GGML_ASSERT(ggml_backend_memory_arena_quiesce(f.arena.get()));
        f.workspace.fail_at.clear();
        const auto result = f.transition.recover();
        t.assert_true(result.status == status::recovery_failed);
        t.assert_true(result.failed_at == state::committing);
        t.assert_true(f.transition.state() == state::invalid);
        ggml_backend_memory_region region{};
        t.assert_true(ggml_backend_memory_lease_get_region(external.get(), &region));
        t.assert_equal(2*f.a, region.size);
    });

    t.test("restore_callbacks_cannot_reenter_or_cancel_recovery", [](testing & t) {
        fixture f(t);
        f.workspace.fail_at = "bind";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activation_failed);
        f.workspace.fail_at.clear();
        f.workspace.hook = [&](const std::string & step) {
            if (step.find("restore_") == 0) {
                t.assert_equal(uint64_t(0), f.transition.admit());
                t.assert_true(f.transition.prepare(f.target).status == status::busy);
                t.assert_true(f.transition.activate(f.arenas).status == status::busy);
                t.assert_true(f.transition.recover().status == status::busy);
                t.assert_true(!f.transition.cancel());
                t.assert_true(f.transition.last_failure() != nullptr);
            }
        };
        if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
        check_restored(t, f);
        auto restored = std::find(f.events.begin(), f.events.end(), "restore_cleanup:11");
        auto original = std::find(f.events.begin(), f.events.end(), "cleanup:12");
        t.assert_true(restored < original);
    });


    t.test("partial_view_factory_exceptions_are_recoverable", [](testing & t) {
        for (int kind : {1, 2}) {
            fixture f(t, false);
            if (!f.prepare(t)) return;
            {
                view_fault fault(f.arena.get(), 1);
                fault.exception_kind = kind;
                const auto result = f.transition.activate(f.arenas);
                t.assert_true(result.failed_at == state::committing);
                t.assert_true(result.status == (kind == 1 ? status::activation_failed : status::consumer_exception));
                t.assert_true(bool(result.exception) == (kind == 2));
                t.assert_equal(size_t(2), fault.attempts);
            }
            t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(f.arena.get()));
            if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
            check_restored(t, f);
        }
    });

    t.test("cancelled_transition_can_restore_after_drain", [](testing & t) {
        fixture f(t);
        f.workspace.work.push_back(f.workspace.executor.acquire({f.workspace.lease.get()}, 1));
        f.workspace.hook = [&](const std::string & name) {
            if (name == "drain") t.assert_true(f.transition.cancel());
        };
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::cancelled);
        f.workspace.hook = {};
        if (!t.assert_true(f.transition.recover().status == status::recovered)) return;
        check_restored(t, f, 12);
    });

    t.test("recovery_preparation_exception_keeps_original_diagnostics", [](testing & t) {
        fixture f(t, false);
        f.workspace.fail_at = "bind";
        if (!f.prepare(t)) return;
        t.assert_true(f.transition.activate(f.arenas).status == status::activation_failed);
        f.workspace.fail_at.clear();
        f.preserved.throw_at = "prepare_recovery";
        const auto result = f.transition.recover();
        t.assert_true(result.status == status::recovery_failed);
        t.assert_true(result.failed_at == state::recovering);
        t.assert_true(result.exception != nullptr);
        t.assert_equal(size_t(1), result.consumer);
        t.assert_true(f.transition.state() == state::invalid);
        t.assert_true(f.transition.last_failure()->failed_at == state::binding);
        t.assert_equal(uint64_t(0), f.transition.admit());
    });

    t.test("partial_multi_arena_commit_restores_only_changed_arena", [](testing & t) {
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
        auto * old_view = ggml_backend_memory_lease_buffer(external.get());
        t.assert_true(transition.prepare(target).status == status::prepared);
        const auto result = transition.activate({{1, LLAMA_MEMORY_ALLOCATION_HOST, one.get()},
                                                 {2, LLAMA_MEMORY_ALLOCATION_HOST, two.get()}});
        t.assert_true(result.failed_at == state::committing);
        t.assert_equal(size_t(1), result.arena);
        t.assert_equal(uint64_t(2), ggml_backend_memory_arena_generation(one.get()));
        t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(two.get()));
        if (!t.assert_true(transition.recover().status == status::recovered)) return;
        t.assert_equal(uint64_t(3), ggml_backend_memory_arena_generation(one.get()));
        t.assert_equal(uint64_t(1), ggml_backend_memory_arena_generation(two.get()));
        t.assert_true(old_view == ggml_backend_memory_lease_buffer(second.lease.get()));
        for (auto * part : {&first, &second}) {
            ggml_backend_memory_region region{};
            t.assert_true(ggml_backend_memory_lease_get_region(part->lease.get(), &region));
            t.assert_equal(a, region.size);
            t.assert_equal(int(part->id), *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(part->lease.get()))));
            t.assert_true(part->executor.ready());
            t.assert_equal(uint64_t(1), part->revision);
            t.assert_true(ggml_backend_memory_arena_get_state(part->arena) == GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
        }
        const auto admission = transition.admit();
        t.assert_true(admission != 0 && transition.finish(admission));
    });

    return t.summary();
}
