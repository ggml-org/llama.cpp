#include "../src/llama-memory-workspace.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <stdexcept>

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using status = llama_memory_transition_status;

// A real CPU allocation with view support withheld for scheduler-fallback tests.
static ggml_backend_buffer_t no_views_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buffer = ggml_backend_cpu_buffer_type()->iface.alloc_buffer(buft, size);
    if (buffer) buffer->view_buffer = nullptr;
    return buffer;
}

struct late_failure : llama_memory_consumer {
    bool fail = false;
    struct token : llama_memory_preparation {
        late_failure & owner;
        explicit token(late_failure & owner) : owner(owner) {}
        bool quiesce(const std::vector<llama_memory_resource_id> &) override { return true; }
        bool drain() override { return true; }
        bool invalidate() override { return true; }
        bool release() override { return true; }
        bool bind(const std::vector<llama_memory_region_binding> &) override { return true; }
        bool activate() override { return !owner.fail; }
        bool prepare_recovery(std::unique_ptr<llama_memory_preparation> &) override { return true; }
    };
    bool prepare(const llama_memory_transition_target &, const llama_memory_layout &,
            std::unique_ptr<llama_memory_preparation> & output) override {
        output = std::make_unique<token>(*this);
        return true;
    }
};

struct fixture {
    ggml_backend_ptr first{ggml_backend_cpu_init()}, second{ggml_backend_cpu_init()};
    ggml_backend_buffer_type separate = *ggml_backend_cpu_buffer_type();
    std::vector<ggml_backend_t> backends{first.get(), second.get()};
    std::vector<ggml_backend_buffer_type_t> bufts;
    ggml_backend_sched_ptr sched;
    std::vector<arena_ptr> parents;
    std::vector<llama_memory_transition_arena> arenas;
    std::vector<llama_memory_workspace_group> groups;
    llama_memory_transition_target target;
    int quiesces = 0, invalidations = 0;
    bool fail_invalidate = false, throw_invalidate = false;
    std::unique_ptr<llama_memory_workspace> workspace;
    late_failure later;
    std::unique_ptr<llama_memory_transition> transition;

    // Use measured phase maxima and the real scheduler's shared-buffer-type grouping.
    fixture(bool alias = true, bool manage_second = true, bool second_views = true) {
        if (!second_views) separate.iface.alloc_buffer = no_views_alloc;
        bufts = {ggml_backend_cpu_buffer_type(), alias ? ggml_backend_cpu_buffer_type() : &separate};
        sched.reset(ggml_backend_sched_new(backends.data(), bufts.data(), 2, 256, false, true));
        GGML_ASSERT(sched);
        const size_t sizes[] = {4096, 8192, 2048, 4096};
        ggml_backend_memory_workspace_group measured[2];
        size_t count = 2;
        GGML_ASSERT(ggml_backend_memory_plan_workspace_groups(bufts.data(), sizes, 2, 2, measured, &count));
        target.plan.stages = {{10, {}, {}}, {20, {10}, {}}};
        target.stage = 10;
        for (size_t i = 0; i < count; ++i) {
            if (i == 1 && !manage_second) continue;
            const auto & group = measured[i];
            const uint64_t domain = i + 1, id = i + 11;
            groups.push_back({group, {id, domain, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::discardable}});
            parents.emplace_back(ggml_backend_memory_arena_new(group.buft, 2*group.size), ggml_backend_memory_arena_free);
            GGML_ASSERT(parents.back());
            arenas.push_back({domain, LLAMA_MEMORY_ALLOCATION_HOST, parents.back().get()});
            target.plan.domains.push_back({domain, LLAMA_MEMORY_ALLOCATION_HOST, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            target.budgets.push_back({domain, LLAMA_MEMORY_ALLOCATION_HOST, 2*group.size, group.alignment});
        }
        workspace = std::make_unique<llama_memory_workspace>(sched.get(), groups, llama_memory_workspace_hooks{
            [&] { ++quiesces; return true; },
            [&] {
                ++invalidations;
                if (throw_invalidate) throw std::runtime_error("workspace invalidation");
                return !fail_invalidate;
            },
        });
        transition = std::make_unique<llama_memory_transition>(std::vector<llama_memory_consumer *>{workspace.get(), &later});
    }

    // Build the consumer declarations once, then prepare and activate against supplied parent arenas.
    bool start(testing & t) {
        if (!t.assert_true(workspace->register_resources(target.plan, {10, 20}))) return false;
        if (!t.assert_true(transition->prepare(target).status == status::prepared)) return false;
        return t.assert_true(transition->activate(arenas).status == status::activated);
    }

    // Force a new offset without changing the maximum workspace grant.
    void move_workspace() {
        target.plan.resources.push_back({99, 1, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::discardable});
        auto & requirements = target.plan.stages[0].requirements;
        requirements.insert(requirements.begin(), {99, groups[0].workspace.alignment, groups[0].workspace.alignment,
            groups[0].workspace.alignment, LLAMA_MEMORY_ACCESS_WRITE, 0});
    }

    // Reserve and evaluate with the scheduler to prove attachment is usable, not just retained.
    void compute(testing & t, size_t slot = 0) {
        ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
        auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        auto * output = ggml_scale(ctx.get(), input, 2.0f);
        auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_set_tensor_backend(sched.get(), input, backends[slot]);
        ggml_backend_sched_set_tensor_backend(sched.get(), output, backends[slot]);
        if (!t.assert_true(ggml_backend_sched_reserve(sched.get(), graph))) return;
        if (!t.assert_true(ggml_backend_sched_alloc_graph(sched.get(), graph))) return;
        const float data[4] = {1, 2, 3, 4};
        float result[4] = {};
        ggml_backend_tensor_set(input, data, 0, sizeof(data));
        t.assert_true(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 4; ++i) t.assert_equal(2*data[i], result[i]);
        ggml_backend_sched_reset(sched.get());
    }
};

int main() {
    testing t;
    t.test("registers_phase_maximum_once_for_aliased_slots", [](testing & t) {
        fixture f;
        t.assert_true(f.workspace->register_resources(f.target.plan, {10, 20}));
        t.assert_equal(size_t(1), f.target.plan.resources.size());
        if (!t.assert_true(!f.target.plan.stages[0].requirements.empty())) return;
        for (const auto & stage : f.target.plan.stages) {
            t.assert_equal(size_t(1), stage.requirements.size());
            t.assert_equal(size_t(8192), stage.requirements[0].size_min);
            t.assert_equal(size_t(8192), stage.requirements[0].size_preferred);
            t.assert_true(stage.requirements[0].access == LLAMA_MEMORY_ACCESS_WRITE);
        }
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        t.assert_equal(size_t(0), ggml_backend_memory_arena_region_count(f.parents[0].get()));
    });

    t.test("registration_rejects_collisions_without_partial_output", [](testing & t) {
        fixture f;
        t.assert_true(!f.workspace->register_resources(f.target.plan, {10, 10}));
        t.assert_true(!f.workspace->register_resources(f.target.plan, {99}));
        t.assert_true(f.target.plan.resources.empty());
        if (!t.assert_true(f.workspace->register_resources(f.target.plan, {10, 20}))) return;
        t.assert_true(!f.workspace->register_resources(f.target.plan, {10}));
        t.assert_equal(size_t(1), f.target.plan.resources.size());
        t.assert_equal(size_t(1), f.target.plan.stages[0].requirements.size());
    });

    t.test("attach_reserve_compute_and_detach_shared_slots", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        t.assert_true(f.workspace->ready());
        t.assert_equal(size_t(8192), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        // Shared storage is reported once, on the first slot; both slots still borrow its lease.
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.parents[0].get()));
        f.compute(t, 1);
        t.assert_true(f.workspace->close());
        t.assert_true(!f.workspace->ready());
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.parents[0].get()));
        t.assert_true(f.workspace->close());
    });

    t.test("unchanged_maximum_avoids_invalidation_in_decode", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        const int invalidations = f.invalidations, quiesces = f.quiesces;
        const auto generation = ggml_backend_memory_arena_generation(f.parents[0].get());
        f.target.stage = 20;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activated);
        t.assert_equal(invalidations, f.invalidations);
        t.assert_equal(quiesces, f.quiesces);
        t.assert_equal(generation, ggml_backend_memory_arena_generation(f.parents[0].get()));
        f.compute(t);
    });

    t.test("cancelled_preparation_does_not_touch_active_scheduler", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        const int invalidations = f.invalidations;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(!f.workspace->close());
        t.assert_true(f.transition->cancel());
        t.assert_equal(invalidations, f.invalidations);
        t.assert_true(f.workspace->ready());
        f.compute(t);
    });

    t.test("partial_attachment_failure_leaves_foreign_range_alone", [](testing & t) {
        fixture f(false);
        ggml_backend_buffer_ptr external(ggml_backend_buft_alloc_buffer(f.bufts[1], 8192));
        GGML_ASSERT(ggml_backend_sched_set_buffer_range(f.sched.get(), f.second.get(), external.get(), 0, 8192));
        if (!t.assert_true(f.workspace->register_resources(f.target.plan, {10, 20}))) return;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        t.assert_true(!f.workspace->ready());
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        t.assert_equal(size_t(8192), ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
        for (auto & arena : f.parents) t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(arena.get()));
        t.assert_true(f.transition->recover().status == status::recovered);
        t.assert_true(!f.workspace->ready());
        GGML_ASSERT(ggml_backend_sched_clear_buffer_range(f.sched.get(), f.second.get()));
    });

    t.test("moved_workspace_recovers_after_later_consumer_failure", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        f.move_workspace();
        f.later.fail = true;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        f.later.fail = false;
        t.assert_true(f.transition->recover().status == status::recovered);
        ggml_backend_memory_region region{};
        t.assert_true(ggml_backend_memory_arena_get_region(f.parents[0].get(), 11, &region));
        t.assert_equal(size_t(0), region.offset);
        t.assert_true(f.workspace->ready());
        f.compute(t);
    });

    t.test("failed_invalidation_does_not_release_old_lease", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        f.move_workspace();
        f.fail_invalidate = true;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        t.assert_true(!f.workspace->ready());
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.parents[0].get()));
        f.fail_invalidate = false;
        t.assert_true(f.transition->recover().status == status::recovered);
        t.assert_true(f.workspace->ready());
        f.compute(t);
    });

    t.test("omitted_nonview_group_uses_scheduler_fallback", [](testing & t) {
        fixture f(false, false, false);
        if (!f.start(t)) return;
        t.assert_equal(size_t(1), f.groups.size());
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
        f.compute(t, 1);
        const auto fallback_size = ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get());
        t.assert_true(fallback_size > 0);
        t.assert_true(f.workspace->close());
        t.assert_equal(fallback_size, ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
    });

    t.test("destruction_detaches_before_parent_owner_disappears", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        f.transition.reset();
        f.workspace.reset();
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.parents[0].get()));
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
    });

    t.test("rejects_mislabeled_and_noncanonical_scheduler_groups", [](testing & t) {
        fixture f;
        for (int bad : {0, 1, 2, 3}) {
            auto groups = f.groups;
            if (bad == 0) groups[0].workspace.buft = &f.separate;
            if (bad == 1) groups[0].workspace.first_slot = 1;
            if (bad == 2) groups[0].workspace.first_slot = 9;
            if (bad == 3) groups.push_back(groups[0]);
            llama_memory_workspace consumer(f.sched.get(), groups, {[] { return true; }, [] { return true; }});
            auto plan = f.target.plan;
            t.assert_true(!consumer.register_resources(plan, {10}));
            t.assert_true(plan.resources.empty());
        }
    });

    t.test("missing_execution_hooks_are_rejected", [](testing & t) {
        fixture f;
        llama_memory_workspace consumer(f.sched.get(), f.groups, {});
        t.assert_true(!consumer.register_resources(f.target.plan, {10}));
        std::unique_ptr<llama_memory_preparation> proposal;
        t.assert_true(!consumer.prepare(f.target, {}, proposal));
        t.assert_true(!proposal);
    });

    t.test("incorrect_workspace_grant_is_rejected_before_preparation", [](testing & t) {
        fixture f;
        if (!t.assert_true(f.workspace->register_resources(f.target.plan, {10, 20}))) return;
        f.target.plan.stages[0].requirements[0].size_min /= 2;
        f.target.plan.stages[0].requirements[0].size_preferred /= 2;
        t.assert_true(f.transition->prepare(f.target).status == status::consumer_failed);
        t.assert_equal(0, f.quiesces);
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
    });

    t.test("invalidation_exception_remains_closed_and_recoverable", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        f.move_workspace();
        f.throw_invalidate = true;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        const auto result = f.transition->activate(f.arenas);
        t.assert_true(result.status == status::consumer_exception && result.exception != nullptr);
        t.assert_true(!f.workspace->ready());
        f.throw_invalidate = false;
        t.assert_true(f.transition->recover().status == status::recovered);
        f.compute(t);
    });

    t.test("repeated_create_reserve_detach_releases_all_leases", [](testing & t) {
        for (int i = 0; i < 16; ++i) {
            fixture f(false);
            if (!f.start(t)) return;
            f.compute(t, size_t(i % 2));
            t.assert_true(f.workspace->close());
            for (auto & parent : f.parents) t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(parent.get()));
        }
    });


    t.test("recovery_remaps_workspace_indices_when_budget_order_changes", [](testing & t) {
        fixture f(false);
        if (!f.start(t)) return;
        f.move_workspace();
        std::swap(f.target.budgets[0], f.target.budgets[1]);
        std::swap(f.arenas[0], f.arenas[1]);
        f.later.fail = true;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        f.later.fail = false;
        if (!t.assert_true(f.transition->recover().status == status::recovered)) return;
        t.assert_true(f.workspace->ready());
        f.compute(t);
        f.compute(t, 1);
    });

    return t.summary();
}
