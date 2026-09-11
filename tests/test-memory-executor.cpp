#include "../src/llama-memory-executor.h"
#include "testing.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <functional>
#include <stdexcept>
#include <utility>

using status = llama_memory_executor_status;
using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct fixture {
    ggml_backend_ptr backend{ggml_backend_cpu_init()};
    size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend.get()));
    arena_ptr arena{ggml_backend_memory_arena_new(ggml_backend_get_default_buffer_type(backend.get()), 4*alignment),
                    ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    uint32_t flags;

    // Use real CPU arena/lease ownership while keeping execution fake.
    explicit fixture(bool persistent = false) :
        flags(persistent ? GGML_BACKEND_MEMORY_REGION_PERSISTENT : GGML_BACKEND_MEMORY_REGION_NONE) {
        GGML_ASSERT(backend && arena);
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 11, 0, alignment, alignment, flags, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 12, 3*alignment, alignment, alignment, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 11));
        GGML_ASSERT(lease);
        *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(lease.get()))) = 7;
    }

    // Try to reuse captured storage; active capture/execution leases must reject the commit.
    bool grow() {
        GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 11, 0, 2*alignment, alignment, flags, nullptr));
        const bool result = ggml_backend_memory_arena_commit(arena.get());
        if (!result) {
            ggml_backend_memory_arena_rollback(arena.get());
        }
        GGML_ASSERT(ggml_backend_memory_arena_resume(arena.get()));
        return result;
    }
};

struct observation {
    testing & t;
    ggml_backend_memory_arena_t arena;
    int destroyed = 0;
    int pending = 0;
    int computed = 0;
    std::vector<int> events = {};
};

struct fake_capture : llama_memory_executable {
    observation & seen;
    const int * data;

    fake_capture(observation & seen, ggml_backend_memory_lease_t lease) :
        seen(seen), data(static_cast<const int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(lease)))) {}

    // Native capture destruction must happen before its final lease is released.
    ~fake_capture() override {
        seen.t.assert_equal(0, seen.pending);
        seen.t.assert_true(ggml_backend_memory_arena_lease_count(seen.arena) != 0);
        ++seen.destroyed;
        seen.events.push_back(3);
    }
};

struct fake_backend : llama_memory_executor_backend {
    observation & seen;
    std::vector<llama_memory_execution> queue;
    bool completed = false;
    bool leave_pins = false;
    bool throw_error = false;
    int drains = 0;
    std::function<void()> on_drain;

    explicit fake_backend(observation & seen) : seen(seen) {}

    // Queue a pin for either simulated compute or copy work; stale acquisition never queues work.
    bool enqueue(llama_memory_execution execution) {
        if (!execution) {
            return false;
        }
        queue.push_back(std::move(execution));
        ++seen.pending;
        return true;
    }

    // Release pins only after the fake completion signal; callbacks can inspect drain-time protection.
    bool drain() override {
        ++drains;
        seen.events.push_back(1);
        if (on_drain) {
            on_drain();
        }
        if (throw_error) {
            throw std::runtime_error("fake drain failure");
        }
        if (!completed) {
            return false;
        }
        if (!leave_pins) {
            for (auto & execution : queue) {
                seen.computed += *static_cast<fake_capture *>(execution.executable())->data;
                --seen.pending;
                seen.events.push_back(2);
                execution.reset();
            }
            queue.clear();
        }
        return true;
    }
};

// Adopt one capture while retaining a raw identity only for assertions, never for submission.
static bool capture(testing & t, fixture & f, observation & seen, llama_memory_executor & executor,
        uint64_t revision = 1) {
    std::unique_ptr<llama_memory_executable> native = std::make_unique<fake_capture>(seen, f.lease.get());
    if (!t.assert_true(executor.capture(native, {f.lease.get()}, revision))) {
        return false;
    }
    t.assert_true(native == nullptr);
    return true;
}

int main() {
    testing t;

    t.test("capture_matches_bindings_and_pins_execution", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor)) {
            return;
        }
        t.assert_true(executor.ready());
        t.assert_true(executor.matches({f.lease.get()}, 1));
        auto execution = executor.acquire({f.lease.get()}, 1);
        t.assert_true(bool(execution));
        t.assert_true(execution.executable() != nullptr);
        t.assert_equal(size_t(1), executor.outstanding());
        auto moved = std::move(execution);
        t.assert_true(!execution);
        t.assert_true(bool(moved));
        moved.reset();
        t.assert_equal(size_t(0), executor.outstanding());
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_equal(1, seen.destroyed);
        t.assert_true(!executor.ready());
    });

    t.test("runtime_revision_rejects_stale_submission", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor, 8)) {
            return;
        }
        t.assert_true(!executor.matches({f.lease.get()}, 9));
        t.assert_true(!backend.enqueue(executor.acquire({f.lease.get()}, 9)));
        t.assert_equal(0, seen.pending);
        t.assert_true(backend.enqueue(executor.acquire({f.lease.get()}, 8)));
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_equal(7, seen.computed);
    });

    t.test("same_region_id_on_different_storage_is_stale", [](testing & t) {
        fixture first, second;
        observation seen{t, first.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, first, seen, executor)) {
            return;
        }
        t.assert_true(!executor.matches({second.lease.get()}, 1));
        t.assert_true(!executor.acquire({second.lease.get()}, 1));
        t.assert_true(!executor.matches({}, 1));
        t.assert_true(!executor.matches({nullptr}, 1));
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
    });

    t.test("invalid_capture_preserves_caller_ownership", [](testing & t) {
        fixture first, second;
        observation seen{t, first.arena.get()};
        llama_memory_executor executor;
        std::unique_ptr<llama_memory_executable> native;
        t.assert_true(!executor.capture(native, {first.lease.get()}, 1));
        native = std::make_unique<fake_capture>(seen, first.lease.get());
        t.assert_true(!executor.capture(native, {nullptr}, 1));
        t.assert_true(native != nullptr);
        t.assert_true(!executor.capture(native, {first.lease.get(), second.lease.get()}, 1));
        t.assert_true(native != nullptr);
        t.assert_true(!executor.ready());
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(first.arena.get()));
    });

    t.test("aliases_are_normalized_and_matching_is_order_independent", [](testing & t) {
        fixture f;
        lease_ptr alias(ggml_backend_memory_arena_acquire(f.arena.get(), 11), ggml_backend_memory_lease_free);
        lease_ptr second(ggml_backend_memory_arena_acquire(f.arena.get(), 12), ggml_backend_memory_lease_free);
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        std::unique_ptr<llama_memory_executable> native = std::make_unique<fake_capture>(seen, f.lease.get());
        if (!t.assert_true(executor.capture(native, {f.lease.get(), alias.get(), second.get(), f.lease.get()}, 1))) {
            return;
        }
        t.assert_true(executor.matches({second.get(), alias.get()}, 1));
        t.assert_true(!executor.matches({alias.get()}, 1));
        f.lease.reset();
        alias.reset();
        second.reset();
        t.assert_equal(size_t(2), ggml_backend_memory_arena_lease_count(f.arena.get()));
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.arena.get()));
    });

    t.test("older_persistent_lease_survives_unrelated_arena_generation", [](testing & t) {
        fixture f(true);
        const uint64_t old_generation = ggml_backend_memory_lease_generation(f.lease.get());
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor)) {
            return;
        }
        GGML_ASSERT(ggml_backend_memory_arena_quiesce(f.arena.get()));
        GGML_ASSERT(ggml_backend_memory_arena_begin(f.arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(f.arena.get(), 12, 2*f.alignment, f.alignment, f.alignment, 0, nullptr));
        t.assert_true(ggml_backend_memory_arena_commit(f.arena.get()));
        GGML_ASSERT(ggml_backend_memory_arena_resume(f.arena.get()));
        lease_ptr fresh(ggml_backend_memory_arena_acquire(f.arena.get(), 11), ggml_backend_memory_lease_free);
        t.assert_true(ggml_backend_memory_lease_generation(fresh.get()) > old_generation);
        t.assert_equal(old_generation, ggml_backend_memory_lease_generation(f.lease.get()));
        t.assert_true(executor.matches({fresh.get()}, 1));
        t.assert_true(executor.retire_if_affected(backend, {12, 99}).status == status::unchanged);
        t.assert_true(executor.ready());
        t.assert_equal(0, backend.drains);
        t.assert_equal(0, seen.destroyed);
        t.assert_true(backend.enqueue(executor.acquire({fresh.get()}, 1)));
        backend.completed = true;
        t.assert_true(executor.retire_if_affected(backend, {11}).status == status::retired);
        t.assert_equal(7, seen.computed);
    });

    t.test("capture_blocks_reuse_even_without_submitted_work", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor)) {
            return;
        }
        f.lease.reset();
        t.assert_true(!f.grow());
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_true(f.grow());
        t.assert_equal(1, seen.destroyed);
    });

    t.test("compute_and_copy_pins_require_drain_before_reuse", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor)) {
            return;
        }
        t.assert_true(backend.enqueue(executor.acquire({f.lease.get()}, 1)));
        t.assert_true(backend.enqueue(executor.acquire({f.lease.get()}, 1)));
        f.lease.reset();
        t.assert_equal(size_t(2), executor.outstanding());
        t.assert_true(executor.retire(backend).status == status::drain_failed);
        t.assert_true(!executor.ready());
        t.assert_true(!f.grow());
        t.assert_equal(0, seen.destroyed);
        backend.on_drain = [&]() {
            t.assert_true(!executor.ready());
            t.assert_true(!f.grow());
            t.assert_true(executor.retire(backend).status == status::busy);
        };
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_true(seen.events == std::vector<int>({1, 1, 2, 2, 3}));
        t.assert_equal(14, seen.computed);
        t.assert_true(f.grow());
    });

    t.test("drain_cannot_invalidate_while_pins_remain", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor)) {
            return;
        }
        t.assert_true(backend.enqueue(executor.acquire({f.lease.get()}, 1)));
        backend.completed = true;
        backend.leave_pins = true;
        t.assert_true(executor.retire(backend).status == status::pending);
        t.assert_true(!executor.ready());
        t.assert_true(!executor.acquire({f.lease.get()}, 1));
        t.assert_equal(0, seen.destroyed);
        backend.leave_pins = false;
        t.assert_true(executor.retire(backend).status == status::retired);
    });

    t.test("drain_exception_keeps_capture_and_leases", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor)) {
            return;
        }
        t.assert_true(backend.enqueue(executor.acquire({f.lease.get()}, 1)));
        f.lease.reset();
        backend.throw_error = true;
        const auto result = executor.retire(backend);
        t.assert_true(result.status == status::drain_failed);
        t.assert_true(result.exception != nullptr);
        t.assert_equal(0, seen.destroyed);
        t.assert_true(!f.grow());
        backend.throw_error = false;
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_true(f.grow());
    });

    t.test("execution_pin_outlives_executor_and_arena_owner", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        {
            llama_memory_executor executor;
            if (!capture(t, f, seen, executor)) {
                return;
            }
            t.assert_true(backend.enqueue(executor.acquire({f.lease.get()}, 1)));
            f.lease.reset();
            f.arena.reset();
        }
        t.assert_equal(0, seen.destroyed);
        backend.completed = true;
        t.assert_true(backend.drain());
        t.assert_equal(7, seen.computed);
        t.assert_equal(1, seen.destroyed);
    });

    t.test("replacement_requires_retiring_old_capture", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()}, other{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor)) {
            return;
        }
        std::unique_ptr<llama_memory_executable> replacement = std::make_unique<fake_capture>(other, f.lease.get());
        t.assert_true(!executor.capture(replacement, {f.lease.get()}, 2));
        t.assert_true(replacement != nullptr);
        t.assert_true(executor.matches({f.lease.get()}, 1));
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_true(executor.capture(replacement, {f.lease.get()}, 2));
        t.assert_true(executor.matches({f.lease.get()}, 2));
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_equal(1, other.destroyed);
    });

    t.test("empty_executor_and_unrelated_changes_are_noops", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        t.assert_true(executor.retire(backend).status == status::unchanged);
        t.assert_true(!executor.ready());
        t.assert_equal(size_t(0), executor.outstanding());
        if (!capture(t, f, seen, executor)) {
            return;
        }
        t.assert_true(!executor.affected_by({}));
        t.assert_true(!executor.affected_by({12}));
        t.assert_true(executor.affected_by({12, 11}));
        t.assert_true(executor.retire_if_affected(backend, {}).status == status::unchanged);
        t.assert_true(executor.retire_if_affected(backend, {12}).status == status::unchanged);
        t.assert_equal(0, backend.drains);
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
    });


    t.test("empty_dependency_capture_still_requires_pins_to_finish", [](testing & t) {
        struct empty_capture : llama_memory_executable {
            int & destroyed;
            explicit empty_capture(int & destroyed) : destroyed(destroyed) {}
            ~empty_capture() override { ++destroyed; }
        };
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        int destroyed = 0;
        llama_memory_executor executor;
        std::unique_ptr<llama_memory_executable> native = std::make_unique<empty_capture>(destroyed);
        if (!t.assert_true(executor.capture(native, {}, 0))) {
            return;
        }
        t.assert_true(executor.matches({}, 0));
        t.assert_true(!executor.matches({f.lease.get()}, 0));
        t.assert_true(!executor.matches({}, 1));
        t.assert_true(!executor.affected_by({11}));
        auto execution = executor.acquire({}, 0);
        t.assert_true(bool(execution));
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::pending);
        t.assert_equal(0, destroyed);
        execution.reset();
        t.assert_true(executor.retire(backend).status == status::retired);
        t.assert_equal(1, destroyed);
    });

    t.test("unrelated_noop_does_not_reopen_a_failed_retirement", [](testing & t) {
        fixture f;
        observation seen{t, f.arena.get()};
        fake_backend backend(seen);
        llama_memory_executor executor;
        if (!capture(t, f, seen, executor)) {
            return;
        }
        t.assert_true(executor.retire(backend).status == status::drain_failed);
        t.assert_true(executor.retire_if_affected(backend, {12}).status == status::unchanged);
        t.assert_equal(1, backend.drains);
        t.assert_true(executor.matches({f.lease.get()}, 1));
        t.assert_true(!executor.ready());
        t.assert_true(!executor.acquire({f.lease.get()}, 1));
        backend.completed = true;
        t.assert_true(executor.retire(backend).status == status::retired);
    });

    t.test("zero_resource_identifier_is_not_a_capture_dependency", [](testing & t) {
        fixture f(true);
        GGML_ASSERT(ggml_backend_memory_arena_begin(f.arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(f.arena.get(), 0, 2*f.alignment, f.alignment, f.alignment, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(f.arena.get()));
        lease_ptr zero(ggml_backend_memory_arena_acquire(f.arena.get(), 0), ggml_backend_memory_lease_free);
        observation seen{t, f.arena.get()};
        llama_memory_executor executor;
        std::unique_ptr<llama_memory_executable> native = std::make_unique<fake_capture>(seen, f.lease.get());
        t.assert_true(!executor.capture(native, {zero.get()}, 1));
        t.assert_true(native != nullptr);
        t.assert_true(!executor.ready());
    });

    return t.summary();
}
