#include "../src/llama-memory-layout.h"
#include "testing.h"

#include <limits>
#include <utility>

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

    bool elastic = false;

    // Remove alignment padding when checking the byte-sharing policy itself.
    void unit_alignment() {
        budgets[0].alignment = 1;
        for (auto & requirement : plan.stages[0].requirements) {
            requirement.alignment = 1;
        }
    }

    // Every rejected input must leave an existing usable output byte-for-byte equivalent by field.
    bool check(testing & t, status expected, uint64_t resource = 0, size_t arena = no_arena) {
        const auto before = output;
        const auto result = elastic
            ? llama_memory_layout_elastic(plan, stage, budgets, fixed, output)
            : llama_memory_layout_minimum(plan, stage, budgets, fixed, output);
        const bool matched = t.assert_equal(static_cast<int>(expected), static_cast<int>(result.status));
        t.assert_equal(resource, result.resource);
        t.assert_equal(arena, result.arena);
        if (expected != status::success) {
            t.assert_true(same_layout(before, output));
        }
        return matched;
    }
};

// Find grants by logical identity; output regions are address-sorted, not ID-sorted.
static void expect_grant(testing & t, const llama_memory_layout & layout, uint64_t id, size_t size, size_t offset) {
    for (const auto & arena : layout.arenas) {
        for (const auto & region : arena.regions) {
            if (region.id == id) {
                t.assert_equal(size, region.size);
                t.assert_equal(offset, region.offset);
                return;
            }
        }
    }
    t.assert_true("missing expected grant", false);
}

// Enumerate small layouts independently of the production forward-packing and size-search algorithms.
static bool brute_ordered_fit(size_t capacity, const ggml_backend_memory_region & a,
        const ggml_backend_memory_region & b, const ggml_backend_memory_region & fixed) {
    if (a.size > capacity || b.size > capacity) {
        return false;
    }
    const auto clear = [&](size_t offset, size_t size) {
        return fixed.size == 0 || offset + size <= fixed.offset || offset >= fixed.offset + fixed.size;
    };
    for (size_t x = 0; x <= capacity - a.size; ++x) {
        if (x % a.alignment != 0 || !clear(x, a.size)) {
            continue;
        }
        for (size_t y = x + a.size; y <= capacity - b.size; ++y) {
            if (y % b.alignment == 0 && clear(y, b.size)) {
                return true;
            }
        }
    }
    return false;
}

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


    t.test("elastic_equal_extras_above_minima", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 80, 0);
            expect_grant(t, f.output, 12, 48, 80);
            t.assert_equal(size_t(0), f.output.arenas[0].unused);
        }
    });

    t.test("elastic_remainder_uses_resource_id", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        f.budgets[0].capacity = 129;
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 81, 0);
            expect_grant(t, f.output, 12, 48, 81);
        }
        std::swap(f.plan.stages[0].requirements[0], f.plan.stages[0].requirements[1]);
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 81, 48);
            expect_grant(t, f.output, 12, 48, 0);
        }
    });

    t.test("elastic_caps_preferences_and_redistributes", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        f.plan.stages[0].requirements[1].size_preferred = 40;
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 88, 0);
            expect_grant(t, f.output, 12, 40, 88);
        }
        f.budgets[0].capacity = 512;
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 128, 0);
            expect_grant(t, f.output, 12, 40, 128);
            t.assert_equal(size_t(344), f.output.arenas[0].unused);
        }
    });

    t.test("elastic_can_move_unpinned_hard_workspace", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        f.plan.stages[0].requirements[1].size_preferred = 32;
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 96, 0);
            expect_grant(t, f.output, 12, 32, 96);
        }
        f.elastic = false;
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 64, 0);
            expect_grant(t, f.output, 12, 32, 64);
        }
    });

    t.test("elastic_fixed_region_never_grows_or_shrinks", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        f.fixed = {{0, {12, 0, 32, 32, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 96, 32);
            t.assert_true(same_region(f.output.arenas[0].regions[0], f.fixed[0].region));
        }
        f.plan.stages[0].requirements[0].size_min = 1;
        f.plan.stages[0].requirements[0].size_preferred = 2;
        f.fixed = {{0, {11, 0, 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 12, 64, 64);
            t.assert_true(same_region(f.output.arenas[0].regions[0], f.fixed[0].region));
        }
    });

    t.test("elastic_zero_minima_and_optional_budgets", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        for (auto & requirement : f.plan.stages[0].requirements) {
            requirement.size_min = 0;
            requirement.size_preferred = 10;
        }
        f.budgets[0].capacity = 15;
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 8, 0);
            expect_grant(t, f.output, 12, 7, 8);
        }
        f.budgets[0].capacity = 0;
        if (f.check(t, status::success)) {
            t.assert_true(f.output.arenas[0].regions.empty());
        }
        f.budgets[0] = {1, LLAMA_MEMORY_ALLOCATION_MANAGED, 15, 1};
        if (f.check(t, status::success)) {
            t.assert_true(f.output.arenas[0].regions.empty());
            t.assert_equal(size_t(15), f.output.arenas[0].unused);
        }
        f.budgets.clear();
        if (f.check(t, status::success)) {
            t.assert_true(f.output.arenas.empty());
        }
    });

    t.test("elastic_zero_preference_remains_absent", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        f.plan.stages[0].requirements[0].size_min = 0;
        f.plan.stages[0].requirements[0].size_preferred = 0;
        if (f.check(t, status::success)) {
            t.assert_equal(size_t(1), f.output.arenas[0].regions.size());
            expect_grant(t, f.output, 12, 96, 0);
        }
    });

    t.test("elastic_alignment_remainder_is_not_limited_to_one_byte", [](testing & t) {
        fixture f;
        f.elastic = true;
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 96, 0);
            expect_grant(t, f.output, 12, 32, 96);
            t.assert_equal(size_t(0), f.output.arenas[0].unused);
        }
        f.plan.stages[0].requirements[0].size_min = 17;
        f.plan.stages[0].requirements[0].size_preferred = 17;
        f.plan.stages[0].requirements[1].size_min = 1;
        f.plan.stages[0].requirements[1].size_preferred = 64;
        f.budgets[0].capacity = 65;
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 17, 0);
            expect_grant(t, f.output, 12, 33, 32);
            t.assert_equal(size_t(15), f.output.arenas[0].unused);
        }
    });

    t.test("elastic_ordered_packing_across_a_fixed_obstacle", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        f.budgets[0].capacity = 192;
        f.plan.stages[0].requirements[0].size_min = 16;
        f.plan.stages[0].requirements[1].size_min = 16;
        f.plan.stages[0].requirements[1].size_preferred = 16;
        f.plan.stages[0].requirements.push_back({13, 32, 32, 1, LLAMA_MEMORY_ACCESS_WRITE, 0});
        f.fixed = {{0, {13, 64, 32, 1, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 13, 32, 64);
            expect_grant(t, f.output, 11, 80, 96);
            expect_grant(t, f.output, 12, 16, 176);
            t.assert_equal(size_t(64), f.output.arenas[0].unused);
        }
    });

    t.test("elastic_independent_allocation_budgets", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        f.plan.resources[1].allocation_class = LLAMA_MEMORY_ALLOCATION_MANAGED;
        f.budgets = {
            {1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 96, 1},
            {1, LLAMA_MEMORY_ALLOCATION_MANAGED, 40, 1},
        };
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, 96, 0);
            expect_grant(t, f.output, 12, 40, 0);
        }
        f.budgets[1].capacity = 31;
        f.check(t, status::placement_failed, 12, 1);
    });

    t.test("elastic_overflow_safe_water_level_and_remainder", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        const size_t maximum = std::numeric_limits<size_t>::max();
        f.budgets[0].capacity = maximum;
        for (auto & requirement : f.plan.stages[0].requirements) {
            requirement.size_min = 0;
            requirement.size_preferred = maximum;
        }
        if (f.check(t, status::success)) {
            expect_grant(t, f.output, 11, maximum / 2 + 1, 0);
            expect_grant(t, f.output, 12, maximum / 2, maximum / 2 + 1);
            t.assert_equal(maximum, f.output.arenas[0].used);
            t.assert_equal(size_t(0), f.output.arenas[0].unused);
        }
    });

    t.test("elastic_no_preferences_matches_minimum", [](testing & t) {
        fixture f;
        for (auto & requirement : f.plan.stages[0].requirements) {
            requirement.size_preferred = requirement.size_min;
        }
        if (!f.check(t, status::success)) {
            return;
        }
        const auto minimum = f.output;
        f.elastic = true;
        f.check(t, status::success);
        t.assert_true(same_layout(minimum, f.output));
    });

    t.test("elastic_failure_and_repeated_success_preserve_contracts", [](testing & t) {
        fixture f;
        f.elastic = true;
        f.unit_alignment();
        if (!f.check(t, status::success)) {
            return;
        }
        const auto original = f.output;
        for (int i = 0; i < 8; ++i) {
            f.check(t, status::success);
            t.assert_true(same_layout(original, f.output));
        }
        t.assert_equal(size_t(64), f.plan.stages[0].requirements[0].size_min);
        t.assert_equal(size_t(128), f.plan.stages[0].requirements[0].size_preferred);
        f.budgets[0].capacity = 1;
        f.check(t, status::placement_failed, 11, 0);
        f.budgets[0].capacity = 128;
        f.fixed = {{0, {11, 1, 64, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        f.check(t, status::invalid_fixed_region, 11, 0);
    });


    t.test("elastic_small_layouts_match_exhaustive_growth_checks", [](testing & t) {
        for (size_t capacity = 4; capacity <= 32; capacity += 4) {
            for (size_t first_alignment : {size_t(1), size_t(2), size_t(4)}) {
                for (size_t second_alignment : {size_t(1), size_t(2), size_t(4)}) {
                    for (bool with_fixed : {false, true}) {
                        fixture f;
                        f.unit_alignment();
                        f.budgets[0].capacity = capacity;
                        f.plan.stages[0].requirements[0] = {11, 2, 16, first_alignment, LLAMA_MEMORY_ACCESS_WRITE, 0};
                        f.plan.stages[0].requirements[1] = {12, 1, 16, second_alignment, LLAMA_MEMORY_ACCESS_WRITE, 0};
                        ggml_backend_memory_region obstacle = {};
                        if (with_fixed) {
                            obstacle = {13, capacity / 2, 2, 1, GGML_BACKEND_MEMORY_REGION_PERSISTENT};
                            f.fixed = {{0, obstacle}};
                            f.plan.stages[0].requirements.push_back({13, 2, 2, 1, LLAMA_MEMORY_ACCESS_WRITE, 0});
                        }
                        llama_memory_layout minimum;
                        const auto baseline = llama_memory_layout_minimum(f.plan, f.stage, f.budgets, f.fixed, minimum);
                        f.elastic = true;
                        if (baseline.status != status::success) {
                            f.check(t, baseline.status, baseline.resource, baseline.arena);
                            continue;
                        }
                        if (!f.check(t, status::success)) {
                            return;
                        }
                        std::vector<ggml_backend_memory_region> ordered;
                        for (const auto & region : minimum.arenas[0].regions) {
                            if (region.flags & GGML_BACKEND_MEMORY_REGION_PERSISTENT) {
                                continue;
                            }
                            for (const auto & grant : f.output.arenas[0].regions) {
                                if (grant.id == region.id) {
                                    ordered.push_back(grant);
                                }
                            }
                        }
                        if (!t.assert_equal(size_t(2), ordered.size())) {
                            return;
                        }
                        t.assert_true(brute_ordered_fit(capacity, ordered[0], ordered[1], obstacle));
                        for (size_t i = 0; i < ordered.size(); ++i) {
                            t.assert_true(ordered[i].size >= (ordered[i].id == 11 ? size_t(2) : size_t(1)));
                            t.assert_true(ordered[i].size <= 16);
                            if (ordered[i].size < 16) {
                                auto enlarged = ordered;
                                ++enlarged[i].size;
                                t.assert_true("no further one-byte grant fits in this order",
                                    !brute_ordered_fit(capacity, enlarged[0], enlarged[1], obstacle));
                            }
                        }
                        const size_t used = ordered[0].size + ordered[1].size + obstacle.size;
                        t.assert_equal(used, f.output.arenas[0].used);
                        t.assert_equal(capacity - used, f.output.arenas[0].unused);
                    }
                }
            }
        }
    });

    return t.summary();
}
