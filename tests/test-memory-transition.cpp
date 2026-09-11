#include "../src/llama-memory-transition.h"
#include "testing.h"

#include <functional>
#include <new>
#include <stdexcept>
#include <utility>

using status = llama_memory_transition_status;
using phase = llama_memory_transition_state;
static constexpr size_t no_consumer = std::numeric_limits<size_t>::max();

// Exercise planning without creating backend storage or running a model.
static llama_memory_transition_target make_target() {
    return {
        {
            {{1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}},
            {{11, 1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, llama_memory_content::preserve}},
            {{10, {}, {{11, 32, 64, 16, LLAMA_MEMORY_ACCESS_WRITE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}}}},
            {},
            {11},
        },
        10,
        {{1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 64, 16}},
        {},
    };
}

struct fake_consumer : llama_memory_consumer {
    testing & t;
    std::vector<int> & events;
    int id;
    int active_value = 42;
    int outstanding = 0;
    int calls = 0;
    bool changed = true;
    bool fail = false;
    bool throw_error = false;
    bool throw_allocation = false;
    std::function<void()> on_prepare;
    std::function<void()> on_discard;

    struct preparation : llama_memory_preparation {
        fake_consumer & owner;
        const llama_memory_transition_target & target;
        const llama_memory_layout & layout;

        // Keep borrowed metadata references so sanitizers verify destruction order.
        preparation(fake_consumer & owner, const llama_memory_transition_target & target,
                const llama_memory_layout & layout) : owner(owner), target(target), layout(layout) {
            ++owner.outstanding;
        }

        // Roll back temporary ownership while the target and layout still exist.
        ~preparation() override {
            owner.t.assert_equal(uint64_t(10), target.stage);
            owner.t.assert_equal(size_t(64), layout.arenas[0].budget.capacity);
            --owner.outstanding;
            owner.events.push_back(-owner.id);
            if (owner.on_discard) {
                owner.on_discard();
            }
        }
    };

    fake_consumer(testing & t, std::vector<int> & events, int id) : t(t), events(events), id(id) {}

    // Allow controlled failure, exceptions, and reentrant cancellation after producing temporary state.
    bool prepare(const llama_memory_transition_target & target, const llama_memory_layout & layout,
            std::unique_ptr<llama_memory_preparation> & output) override {
        ++calls;
        events.push_back(id);
        if (changed) {
            output = std::make_unique<preparation>(*this, target, layout);
        }
        if (on_prepare) {
            on_prepare();
        }
        if (throw_allocation) {
            throw std::bad_alloc();
        }
        if (throw_error) {
            throw std::runtime_error("fake preparation failure");
        }
        return !fail;
    }
};

// Compare structured errors without depending on log wording.
static void check(testing & t, const llama_memory_transition_result & result,
        status expected, size_t consumer = no_consumer) {
    t.assert_equal(static_cast<int>(expected), static_cast<int>(result.status));
    t.assert_equal(consumer, result.consumer);
}

// Confirm old consumer state is usable again after discarding a proposed transition.
static void check_idle(testing & t, llama_memory_transition & transition, fake_consumer & consumer) {
    t.assert_true(transition.state() == phase::idle);
    t.assert_true(transition.pending_target() == nullptr);
    t.assert_true(transition.pending_layout() == nullptr);
    t.assert_equal(42, consumer.active_value);
    t.assert_equal(0, consumer.outstanding);
    const auto admission = transition.admit();
    t.assert_true(admission != 0);
    t.assert_true(transition.finish(admission));
}

int main() {
    testing t;

    t.test("admission_rejects_overlap_and_stale_completion", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        t.assert_true(transition.state() == phase::idle);
        t.assert_true(!transition.finish(0));
        const auto first = transition.admit();
        t.assert_true(first != 0);
        t.assert_true(transition.state() == phase::executing);
        t.assert_equal(uint64_t(0), transition.admit());
        check(t, transition.prepare(make_target()), status::busy);
        t.assert_true(!transition.cancel());
        t.assert_true(!transition.finish(first + 1));
        t.assert_true(transition.finish(first));
        t.assert_true(!transition.finish(first));
        const auto second = transition.admit();
        t.assert_true(second > first);
        t.assert_true(!transition.finish(first));
        t.assert_true(transition.state() == phase::executing);
        t.assert_true(transition.finish(second));
        t.assert_equal(0, consumer.calls);
    });

    t.test("prepared_state_closes_admission_until_discarded", [](testing & t) {
        std::vector<int> events;
        fake_consumer first(t, events, 1), second(t, events, 2);
        llama_memory_transition transition({&first, &second});
        check(t, transition.prepare(make_target()), status::prepared);
        t.assert_true(transition.state() == phase::prepared);
        t.assert_equal(uint64_t(0), transition.admit());
        check(t, transition.prepare(make_target()), status::busy);
        t.assert_true(!transition.finish(1));
        t.assert_equal(1, first.outstanding);
        t.assert_equal(1, second.outstanding);
        t.assert_equal(42, first.active_value);
        if (t.assert_true(transition.pending_layout() != nullptr)) {
            t.assert_equal(size_t(64), transition.pending_layout()->arenas[0].used);
        }
        t.assert_true(transition.cancel());
        t.assert_true(events == std::vector<int>({1, 2, -2, -1}));
        t.assert_true(!transition.cancel());
        check_idle(t, transition, first);
        t.assert_equal(0, second.outstanding);
    });

    t.test("prepare_callback_cannot_reenter_or_reopen_admission", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        consumer.on_prepare = [&]() {
            t.assert_true(transition.state() == phase::preparing);
            t.assert_true(transition.pending_layout() == nullptr);
            t.assert_true(transition.pending_target() == nullptr);
            t.assert_equal(uint64_t(0), transition.admit());
            t.assert_true(!transition.finish(1));
            check(t, transition.prepare(make_target()), status::busy);
        };
        check(t, transition.prepare(make_target()), status::prepared);
        t.assert_equal(1, consumer.calls);
        t.assert_true(transition.cancel());
    });

    t.test("failed_consumer_partial_output_is_discarded", [](testing & t) {
        std::vector<int> events;
        fake_consumer first(t, events, 1), second(t, events, 2);
        llama_memory_transition transition({&first, &second});
        first.fail = true;
        check(t, transition.prepare(make_target()), status::consumer_failed, 0);
        t.assert_true(events == std::vector<int>({1, -1}));
        t.assert_equal(0, second.calls);
        check_idle(t, transition, first);
        first.changed = false;
        check(t, transition.prepare(make_target()), status::consumer_failed, 0);
        check_idle(t, transition, first);
    });

    t.test("later_failure_rolls_back_in_reverse_order", [](testing & t) {
        std::vector<int> events;
        fake_consumer first(t, events, 1), second(t, events, 2), third(t, events, 3);
        llama_memory_transition transition({&first, &second, &third});
        second.fail = true;
        check(t, transition.prepare(make_target()), status::consumer_failed, 1);
        t.assert_true(events == std::vector<int>({1, 2, -2, -1}));
        t.assert_equal(0, third.calls);
        t.assert_equal(0, second.outstanding);
        check_idle(t, transition, first);
    });

    t.test("rollback_callbacks_cannot_reopen_the_gate", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        consumer.on_discard = [&]() {
            t.assert_true(transition.state() == phase::discarding);
            t.assert_equal(uint64_t(0), transition.admit());
            t.assert_true(!transition.finish(1));
            t.assert_true(!transition.cancel());
            check(t, transition.prepare(make_target()), status::busy);
        };
        check(t, transition.prepare(make_target()), status::prepared);
        t.assert_true(transition.cancel());
        check_idle(t, transition, consumer);
    });

    t.test("consumer_exception_keeps_diagnostics_and_active_state", [](testing & t) {
        std::vector<int> events;
        fake_consumer first(t, events, 1), second(t, events, 2);
        llama_memory_transition transition({&first, &second});
        second.throw_error = true;
        const auto result = transition.prepare(make_target());
        check(t, result, status::consumer_exception, 1);
        if (t.assert_true(result.exception != nullptr)) {
            try {
                std::rethrow_exception(result.exception);
            } catch (const std::runtime_error & error) {
                t.assert_equal(std::string("fake preparation failure"), std::string(error.what()));
            }
        }
        t.assert_true(events == std::vector<int>({1, 2, -2, -1}));
        check_idle(t, transition, first);
        second.throw_error = false;
        check(t, transition.prepare(make_target()), status::prepared);
        t.assert_true(transition.cancel());
    });

    t.test("allocation_exception_discards_partial_preparations", [](testing & t) {
        std::vector<int> events;
        fake_consumer first(t, events, 1), second(t, events, 2);
        llama_memory_transition transition({&first, &second});
        second.throw_allocation = true;
        const auto result = transition.prepare(make_target());
        check(t, result, status::allocation_failed, 1);
        t.assert_true(result.exception != nullptr);
        t.assert_true(events == std::vector<int>({1, 2, -2, -1}));
        check_idle(t, transition, first);
    });

    t.test("reentrant_cancellation_waits_for_prepare_to_return", [](testing & t) {
        std::vector<int> events;
        fake_consumer first(t, events, 1), second(t, events, 2);
        llama_memory_transition transition({&first, &second});
        first.on_prepare = [&]() {
            t.assert_true(transition.cancel());
            t.assert_true(transition.cancel());
            t.assert_true(transition.state() == phase::preparing);
            t.assert_equal(1, first.outstanding);
        };
        check(t, transition.prepare(make_target()), status::cancelled, 0);
        t.assert_true(events == std::vector<int>({1, -1}));
        t.assert_equal(0, second.calls);
        check_idle(t, transition, first);
        first.on_prepare = {};
        check(t, transition.prepare(make_target()), status::prepared);
        t.assert_true(transition.cancel());
    });

    t.test("all_consumers_must_confirm_no_change", [](testing & t) {
        std::vector<int> events;
        fake_consumer first(t, events, 1), second(t, events, 2);
        first.changed = second.changed = false;
        llama_memory_transition transition({&first, &second});
        check(t, transition.prepare(make_target()), status::no_change);
        t.assert_true(events == std::vector<int>({1, 2}));
        check_idle(t, transition, first);
        second.changed = true;
        check(t, transition.prepare(make_target()), status::prepared);
        t.assert_equal(0, first.outstanding);
        t.assert_equal(1, second.outstanding);
        t.assert_true(transition.cancel());
        t.assert_equal(2, first.calls);
        t.assert_equal(2, second.calls);
    });

    t.test("identical_metadata_does_not_skip_consumer_state_checks", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        const auto target = make_target();
        consumer.changed = false;
        check(t, transition.prepare(target), status::no_change);
        consumer.changed = true;
        check(t, transition.prepare(target), status::prepared);
        t.assert_equal(2, consumer.calls);
        t.assert_true(transition.cancel());
        check(t, transition.prepare(target), status::prepared);
        t.assert_true(transition.cancel());
        t.assert_equal(3, consumer.calls);
    });

    t.test("invalid_consumer_registrations_do_not_call_consumers", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition empty({});
        check(t, empty.prepare(make_target()), status::invalid_consumer);
        llama_memory_transition missing({nullptr});
        check(t, missing.prepare(make_target()), status::invalid_consumer, 0);
        llama_memory_transition duplicate({&consumer, &consumer});
        check(t, duplicate.prepare(make_target()), status::invalid_consumer, 1);
        t.assert_equal(0, consumer.calls);
        t.assert_true(empty.state() == phase::idle);
        t.assert_true(missing.state() == phase::idle);
        t.assert_true(duplicate.state() == phase::idle);
    });

    t.test("layout_errors_prevent_preparation", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        auto target = make_target();
        target.stage = 99;
        auto result = transition.prepare(target);
        check(t, result, status::layout_error);
        t.assert_true(result.layout.status == llama_memory_layout_status::missing_stage);
        target = make_target();
        target.budgets[0].capacity = 1;
        result = transition.prepare(target);
        check(t, result, status::layout_error);
        t.assert_true(result.layout.status == llama_memory_layout_status::placement_failed);
        t.assert_equal(uint64_t(11), result.layout.resource);
        target = make_target();
        target.plan.stages[0].dependencies = {10};
        result = transition.prepare(target);
        check(t, result, status::layout_error);
        t.assert_true(result.layout.plan.status == llama_memory_plan_status::invalid_dependency_order);
        t.assert_equal(0, consumer.calls);
        check_idle(t, transition, consumer);
    });

    t.test("pending_metadata_is_an_owned_snapshot", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        auto target = make_target();
        check(t, transition.prepare(target), status::prepared);
        target.stage = 99;
        target.plan.resources.clear();
        target.plan.stages.clear();
        target.budgets.clear();
        if (t.assert_true(transition.pending_target() != nullptr)) {
            t.assert_equal(uint64_t(10), transition.pending_target()->stage);
            t.assert_equal(size_t(1), transition.pending_target()->plan.resources.size());
            t.assert_equal(size_t(64), transition.pending_target()->budgets[0].capacity);
        }
        t.assert_true(transition.cancel());
    });

    t.test("destruction_discards_pending_tokens_before_metadata", [](testing & t) {
        std::vector<int> events;
        fake_consumer first(t, events, 1), second(t, events, 2);
        {
            llama_memory_transition transition({&first, &second});
            check(t, transition.prepare(make_target()), status::prepared);
        }
        t.assert_true(events == std::vector<int>({1, 2, -2, -1}));
        t.assert_equal(0, first.outstanding);
        t.assert_equal(0, second.outstanding);
        t.assert_equal(42, first.active_value);
    });

    t.test("old_completion_cannot_release_a_pending_transition", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        const auto admission = transition.admit();
        t.assert_true(transition.finish(admission));
        check(t, transition.prepare(make_target()), status::prepared);
        t.assert_true(!transition.finish(admission));
        t.assert_true(transition.state() == phase::prepared);
        t.assert_true(transition.cancel());
    });

    t.test("repeated_failure_noop_and_cancel_do_not_leak_state", [](testing & t) {
        std::vector<int> events;
        events.reserve(256);
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        for (int i = 0; i < 16; ++i) {
            consumer.fail = true;
            check(t, transition.prepare(make_target()), status::consumer_failed, 0);
            check_idle(t, transition, consumer);
            consumer.fail = false;
            consumer.changed = false;
            check(t, transition.prepare(make_target()), status::no_change);
            check_idle(t, transition, consumer);
            consumer.changed = true;
            check(t, transition.prepare(make_target()), status::prepared);
            t.assert_true(transition.cancel());
            check_idle(t, transition, consumer);
        }
    });


    t.test("cancellation_is_honored_for_noop_and_normal_failure", [](testing & t) {
        for (bool changed : {false, true}) {
            std::vector<int> events;
            fake_consumer consumer(t, events, 1);
            llama_memory_transition transition({&consumer});
            consumer.changed = changed;
            consumer.fail = changed;
            consumer.on_prepare = [&]() { t.assert_true(transition.cancel()); };
            check(t, transition.prepare(make_target()), status::cancelled, 0);
            check_idle(t, transition, consumer);
        }
    });

    t.test("cancellation_does_not_hide_a_thrown_exception", [](testing & t) {
        std::vector<int> events;
        fake_consumer consumer(t, events, 1);
        llama_memory_transition transition({&consumer});
        consumer.on_prepare = [&]() { t.assert_true(transition.cancel()); };
        consumer.throw_error = true;
        const auto result = transition.prepare(make_target());
        check(t, result, status::consumer_exception, 0);
        t.assert_true(result.exception != nullptr);
        check_idle(t, transition, consumer);
    });

    return t.summary();
}
