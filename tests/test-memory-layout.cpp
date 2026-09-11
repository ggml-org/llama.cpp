#include "../src/llama-memory-layout.h"
#include "testing.h"

#include <limits>

using status = llama_memory_layout_status;
static constexpr size_t no_arena = std::numeric_limits<size_t>::max();

// Compare every region attribute; persistent identity includes alignment and flags.
static bool same_region(const ggml_backend_memory_region & a, const ggml_backend_memory_region & b) {
    return a.id == b.id && a.offset == b.offset && a.size == b.size &&
           a.alignment == b.alignment && a.flags == b.flags;
}

// Compare the full output to detect partial publication on any failure path.
static bool same_layout(const llama_memory_layout & a, const llama_memory_layout & b) {
    if (a.arenas.size() != b.arenas.size()) {
        return false;
    }
    for (size_t i = 0; i < a.arenas.size(); ++i) {
        const auto & x = a.arenas[i];
        const auto & y = b.arenas[i];
        if (x.budget.domain != y.budget.domain || x.budget.allocation_class != y.budget.allocation_class ||
                x.budget.capacity != y.budget.capacity || x.budget.alignment != y.budget.alignment ||
                x.used != y.used || x.high_water != y.high_water || x.unused != y.unused ||
                x.regions.size() != y.regions.size()) {
            return false;
        }
        for (size_t j = 0; j < x.regions.size(); ++j) {
            if (!same_region(x.regions[j], y.regions[j])) {
                return false;
            }
        }
    }
    return true;
}

struct fixture {
    llama_memory_execution_plan plan = {
        {{1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL | LLAMA_MEMORY_ALLOCATION_MANAGED, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}},
        {
            {11, 1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, llama_memory_content::preserve},
            {12, 1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, llama_memory_content::discardable},
            {13, 1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, llama_memory_content::reconstructible},
        },
        {{10, {}, {
            {11, 64, 128, 16, LLAMA_MEMORY_ACCESS_WRITE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS},
            {12, 32, 96, 32, LLAMA_MEMORY_ACCESS_WRITE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS},
        }}},
        {},
        {},
    };
    llama_memory_stage_id stage = 10;
    std::vector<llama_memory_arena_budget> budgets = {{1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 128, 16}};
    std::vector<llama_memory_fixed_region> fixed;
    llama_memory_layout output = {{{{77, LLAMA_MEMORY_ALLOCATION_HOST, 99, 1}, {{777, 0, 1, 1, 0}}, 1, 1, 98}}};

    // Every rejected input must leave an existing usable output byte-for-byte equivalent by field.
    bool check(testing & t, status expected, uint64_t resource = 0, size_t arena = no_arena) {
        const auto before = output;
        const auto result = llama_memory_layout_minimum(plan, stage, budgets, fixed, output);
        const bool matched = t.assert_equal(static_cast<int>(expected), static_cast<int>(result.status));
        t.assert_equal(resource, result.resource);
        t.assert_equal(arena, result.arena);
        if (expected != status::success) {
            t.assert_true(same_layout(before, output));
        }
        return matched;
    }
};

int main() {
    testing t;

    t.test("minimum_not_preferred_and_exact_fit", [](testing & t) {
        fixture f;
        f.budgets[0].capacity = 96;
        if (!f.check(t, status::success) || !t.assert_equal(size_t(1), f.output.arenas.size())) {
            return;
        }
        const auto & arena = f.output.arenas[0];
        if (!t.assert_equal(size_t(2), arena.regions.size())) {
            return;
        }
        t.assert_true(same_region(arena.regions[0], {11, 0, 64, 16, 0}));
        t.assert_true(same_region(arena.regions[1], {12, 64, 32, 32, 0}));
        t.assert_equal(size_t(96), arena.used);
        t.assert_equal(size_t(96), arena.high_water);
        t.assert_equal(size_t(0), arena.unused);
        f.budgets[0].capacity = 95;
        f.check(t, status::placement_failed, 12, 0);
    });

    t.test("offset_alignment_and_exact_byte_sizes", [](testing & t) {
        fixture f;
        f.plan.stages[0].requirements[0].size_min = 17;
        f.plan.stages[0].requirements[1].size_min = 1;
        f.budgets[0].capacity = 33;
        if (!f.check(t, status::success) || !t.assert_equal(size_t(1), f.output.arenas.size())) {
            return;
        }
        const auto & arena = f.output.arenas[0];
        if (!t.assert_equal(size_t(2), arena.regions.size())) {
            return;
        }
        t.assert_true(same_region(arena.regions[0], {11, 0, 17, 16, 0}));
        t.assert_true(same_region(arena.regions[1], {12, 32, 1, 32, 0}));
        t.assert_equal(size_t(18), arena.used);
        t.assert_equal(size_t(33), arena.high_water);
        t.assert_equal(size_t(15), arena.unused);
    });

    t.test("parent_alignment_is_a_lower_bound", [](testing & t) {
        fixture f;
        f.budgets[0].alignment = 64;
        if (!f.check(t, status::success) || !t.assert_equal(size_t(1), f.output.arenas.size())) {
            return;
        }
        const auto & regions = f.output.arenas[0].regions;
        if (!t.assert_equal(size_t(2), regions.size())) {
            return;
        }
        t.assert_equal(size_t(64), regions[0].alignment);
        t.assert_equal(size_t(64), regions[1].alignment);
    });

    t.test("persistent_regions_reserved_before_new_minima", [](testing & t) {
        fixture f;
        f.fixed = {{0, {12, 0, 32, 32, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        if (!f.check(t, status::success) || !t.assert_equal(size_t(1), f.output.arenas.size())) {
            return;
        }
        const auto & regions = f.output.arenas[0].regions;
        if (!t.assert_equal(size_t(2), regions.size())) {
            return;
        }
        t.assert_true(same_region(regions[0], f.fixed[0].region));
        t.assert_true(same_region(regions[1], {11, 32, 64, 16, 0}));
    });

    t.test("persistent_identity_survives_a_smaller_preference", [](testing & t) {
        fixture f;
        f.plan.stages[0].requirements[0].size_min = 1;
        f.plan.stages[0].requirements[0].size_preferred = 2;
        f.fixed = {{0, {11, 0, 64, 64, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        if (!f.check(t, status::success) || !t.assert_equal(size_t(1), f.output.arenas.size())) {
            return;
        }
        const auto & regions = f.output.arenas[0].regions;
        if (t.assert_equal(size_t(2), regions.size())) {
            t.assert_true(same_region(regions[0], f.fixed[0].region));
        }
    });

    t.test("persistent_overlap_and_duplicate_identity", [](testing & t) {
        fixture f;
        f.fixed = {
            {0, {11, 0, 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT}},
            {0, {12, 32, 32, 32, GGML_BACKEND_MEMORY_REGION_PERSISTENT}},
        };
        f.check(t, status::invalid_fixed_region, 12, 0);
        f.fixed[1].region = {11, 64, 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT};
        f.check(t, status::invalid_fixed_region, 11, 0);
        f.fixed[1].region = {12, 64, 32, 32, GGML_BACKEND_MEMORY_REGION_PERSISTENT};
        f.check(t, status::success);
    });

    t.test("invalid_persistent_extent_and_flags", [](testing & t) {
        const ggml_backend_memory_region invalid[] = {
            {11, 0, 64, 16, 0},
            {11, 0, 64, 16, 3},
            {11, 0, 0, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
            {11, 0, 63, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
            {11, 1, 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
            {11, 0, 64, 8, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
            {11, 0, 64, 0, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
            {11, 0, 64, 3, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
            {11, 128, 1, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
            {11, std::numeric_limits<size_t>::max(), 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
            {11, 64, std::numeric_limits<size_t>::max(), 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT},
        };
        for (const auto & region : invalid) {
            fixture f;
            f.fixed = {{0, region}};
            f.check(t, status::invalid_fixed_region, 11, 0);
        }
    });

    t.test("fixed_binding_requires_matching_resource_and_budget", [](testing & t) {
        fixture f;
        f.fixed = {{99, {11, 0, 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        f.check(t, status::invalid_fixed_region, 11, 99);
        f.fixed[0].arena = 0;
        f.fixed[0].region.id = 99;
        f.check(t, status::invalid_fixed_region, 99, 0);
        f.fixed[0].region.id = 13;
        f.check(t, status::invalid_fixed_region, 13, 0);
        f.fixed[0].region.id = 11;
        f.budgets[0].allocation_class = LLAMA_MEMORY_ALLOCATION_MANAGED;
        f.check(t, status::invalid_fixed_region, 11, 0);
    });

    t.test("zero_minima_need_no_storage_but_fixed_regions_stay", [](testing & t) {
        fixture f;
        for (auto & req : f.plan.stages[0].requirements) {
            req.size_min = 0;
        }
        f.budgets.clear();
        if (f.check(t, status::success)) {
            t.assert_true(f.output.arenas.empty());
        }
        f.budgets = {{1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 0, 16}};
        if (f.check(t, status::success) && t.assert_equal(size_t(1), f.output.arenas.size())) {
            t.assert_true(f.output.arenas[0].regions.empty());
            t.assert_equal(size_t(0), f.output.arenas[0].unused);
        }
        f.budgets[0].capacity = 128;
        f.fixed = {{0, {11, 0, 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        if (f.check(t, status::success) && t.assert_equal(size_t(1), f.output.arenas.size())) {
            const auto & arena = f.output.arenas[0];
            if (t.assert_equal(size_t(1), arena.regions.size())) {
                t.assert_true(same_region(arena.regions[0], f.fixed[0].region));
                t.assert_equal(size_t(64), arena.used);
            }
        }
    });

    t.test("empty_stage_and_missing_stage", [](testing & t) {
        fixture f;
        f.plan.stages[0].requirements.clear();
        if (f.check(t, status::success) && t.assert_equal(size_t(1), f.output.arenas.size())) {
            t.assert_equal(size_t(128), f.output.arenas[0].unused);
            t.assert_true(f.output.arenas[0].regions.empty());
        }
        f.stage = 99;
        f.check(t, status::missing_stage);
        f.plan = {};
        f.stage = 0;
        f.check(t, status::missing_stage);
    });

    t.test("budget_validation", [](testing & t) {
        fixture f;
        f.budgets[0].domain = 99;
        f.check(t, status::invalid_budget, 0, 0);
        f.budgets[0].domain = 1;
        for (size_t alignment : {size_t(0), size_t(3)}) {
            f.budgets[0].alignment = alignment;
            f.check(t, status::invalid_budget, 0, 0);
        }
        f.budgets[0].alignment = 16;
        for (uint32_t allocation : {0u, 3u, uint32_t(LLAMA_MEMORY_ALLOCATION_HOST), 1u << 31}) {
            f.budgets[0].allocation_class = static_cast<llama_memory_allocation_class>(allocation);
            f.check(t, status::invalid_budget, 0, 0);
        }
        f.budgets[0].allocation_class = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
        f.budgets.push_back(f.budgets[0]);
        f.check(t, status::duplicate_budget, 0, 1);
    });

    t.test("missing_or_empty_budget_is_not_other_storage", [](testing & t) {
        fixture f;
        f.budgets.clear();
        f.check(t, status::missing_budget, 11);
        f.budgets = {{1, LLAMA_MEMORY_ALLOCATION_MANAGED, 1024, 16}};
        f.check(t, status::missing_budget, 11);
        f.budgets[0] = {1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 0, 16};
        f.check(t, status::placement_failed, 11, 0);
    });

    t.test("independent_domains_and_allocation_classes", [](testing & t) {
        fixture f;
        f.plan.resources[1].allocation_class = LLAMA_MEMORY_ALLOCATION_MANAGED;
        f.budgets = {
            {1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 64, 16},
            {1, LLAMA_MEMORY_ALLOCATION_MANAGED, 32, 16},
        };
        if (f.check(t, status::success) && t.assert_equal(size_t(2), f.output.arenas.size())) {
            for (const auto & arena : f.output.arenas) {
                if (t.assert_equal(size_t(1), arena.regions.size())) {
                    t.assert_equal(size_t(0), arena.regions[0].offset);
                    t.assert_equal(size_t(0), arena.unused);
                }
            }
        }
        f.plan.domains.push_back({2, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
        f.plan.resources[1].domain = 2;
        f.plan.resources[1].allocation_class = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
        f.budgets[1].domain = 2;
        f.budgets[1].allocation_class = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
        f.check(t, status::success);
        f.budgets[1].capacity = 31;
        f.check(t, status::placement_failed, 12, 1);
    });

    t.test("preserved_unread_resource_keeps_exact_region", [](testing & t) {
        fixture f;
        f.plan.stages[0].requirements.resize(1);
        auto req = f.plan.stages[0].requirements[0];
        req.access = LLAMA_MEMORY_ACCESS_NONE;
        f.plan.stages.push_back({20, {10}, {req}});
        f.plan.outputs = {11};
        f.stage = 20;
        f.fixed = {{0, {11, 32, 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        if (f.check(t, status::success) && t.assert_equal(size_t(1), f.output.arenas.size())) {
            const auto & regions = f.output.arenas[0].regions;
            if (t.assert_equal(size_t(1), regions.size())) {
                t.assert_true(same_region(regions[0], f.fixed[0].region));
            }
        }
    });

    t.test("fragmentation_and_alignment_failure_are_atomic", [](testing & t) {
        fixture f;
        f.budgets[0].capacity = 128;
        f.plan.stages[0].requirements[0].size_min = 80;
        f.fixed = {{0, {12, 64, 32, 32, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        f.check(t, status::placement_failed, 11, 0);
        f.plan.stages[0].requirements[0].size_min = 33;
        f.plan.stages[0].requirements[0].alignment = 64;
        f.fixed[0].region.offset = 0;
        f.budgets[0].capacity = 96;
        f.check(t, status::placement_failed, 11, 0);
    });

    t.test("large_ranges_do_not_allocate_device_bytes_or_overflow", [](testing & t) {
        fixture f;
        auto & req = f.plan.stages[0].requirements[0];
        req.size_min = req.size_preferred = std::numeric_limits<size_t>::max();
        req.alignment = 1;
        f.plan.stages[0].requirements.resize(1);
        f.budgets[0] = {1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, std::numeric_limits<size_t>::max(), 1};
        if (f.check(t, status::success) && t.assert_equal(size_t(1), f.output.arenas.size())) {
            t.assert_equal(std::numeric_limits<size_t>::max(), f.output.arenas[0].used);
            t.assert_equal(std::numeric_limits<size_t>::max(), f.output.arenas[0].high_water);
            t.assert_equal(size_t(0), f.output.arenas[0].unused);
        }
        f.plan.stages[0].requirements.push_back({12, 1, 1, 2, LLAMA_MEMORY_ACCESS_WRITE, 0});
        f.check(t, status::placement_failed, 12, 0);
    });

    t.test("plan_errors_precede_placement", [](testing & t) {
        fixture f;
        f.plan.stages[0].requirements[0].access = LLAMA_MEMORY_ACCESS_READ;
        f.check(t, status::plan_error, 11);
        const auto result = llama_memory_layout_minimum(f.plan, f.stage, f.budgets, f.fixed, f.output);
        t.assert_true(result.plan.status == llama_memory_plan_status::missing_producer);
        t.assert_equal(uint64_t(10), result.plan.stage);
    });

    t.test("repeated_planning_is_deterministic_and_read_only", [](testing & t) {
        fixture f;
        if (!f.check(t, status::success)) {
            return;
        }
        const auto original = f.output;
        for (int i = 0; i < 8; ++i) {
            f.check(t, status::success);
            t.assert_true(same_layout(original, f.output));
        }
        t.assert_equal(size_t(128), f.budgets[0].capacity);
        t.assert_equal(size_t(128), f.plan.stages[0].requirements[0].size_preferred);
        t.assert_true(f.fixed.empty());
    });

    return t.summary();
}
