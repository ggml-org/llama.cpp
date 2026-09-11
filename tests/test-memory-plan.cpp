#include "../src/llama-memory-plan.h"
#include "testing.h"

#include <limits>

using status = llama_memory_plan_status;

// Describe prefill, an inactive stage, and decode without loading any backend.
static llama_memory_execution_plan make_plan() {
    return {
        {{1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}},
        {{11, 1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, llama_memory_content::preserve}},
        {
            {10, {}, {{11, 128, 256, 64, LLAMA_MEMORY_ACCESS_WRITE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}}},
            {20, {10}, {{11, 128, 256, 64, LLAMA_MEMORY_ACCESS_NONE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}}},
            {30, {20}, {{11, 128, 256, 64, LLAMA_MEMORY_ACCESS_READ_WRITE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}}},
        },
        {},
        {11},
    };
}

// Check useful error identity as well as the success/failure status.
static void check(testing & t, const llama_memory_execution_plan & plan, status expected,
        uint64_t stage = 0, uint64_t resource = 0, uint64_t dependency = 0) {
    const auto result = llama_memory_plan_validate(plan);
    t.assert_equal(static_cast<int>(expected), static_cast<int>(result.status));
    t.assert_equal(stage, result.stage);
    t.assert_equal(resource, result.resource);
    t.assert_equal(dependency, result.dependency);
}

int main() {
    testing t;

    t.test("empty_and_serial_plans", [](testing & t) {
        check(t, {}, status::success);
        auto plan = make_plan();
        check(t, plan, status::success);
        plan.stages[2].dependencies = {10, 20};
        check(t, plan, status::success);
        // The declared serial order also orders stages without explicit dependency edges.
        plan.stages[1].dependencies.clear();
        plan.stages[2].dependencies.clear();
        check(t, plan, status::success);
    });

    t.test("stage_identity", [](testing & t) {
        auto plan = make_plan();
        plan.stages[0].id = 0;
        check(t, plan, status::invalid_stage);
        plan = make_plan();
        plan.stages[2].id = 10;
        check(t, plan, status::duplicate_stage, 10);
        plan = make_plan();
        plan.stages[2].id = std::numeric_limits<uint64_t>::max();
        check(t, plan, status::success);
    });

    t.test("dependencies", [](testing & t) {
        auto plan = make_plan();
        plan.stages[2].dependencies = {99};
        check(t, plan, status::missing_dependency, 30, 0, 99);
        plan.stages[2].dependencies = {0};
        check(t, plan, status::missing_dependency, 30);
        plan.stages[2].dependencies = {10, 10};
        check(t, plan, status::duplicate_dependency, 30, 0, 10);
        plan = make_plan();
        plan.stages[0].dependencies = {10};
        check(t, plan, status::invalid_dependency_order, 10, 0, 10);
        plan.stages[0].dependencies = {20};
        check(t, plan, status::invalid_dependency_order, 10, 0, 20);
        plan.stages[0].dependencies = {30};
        check(t, plan, status::invalid_dependency_order, 10, 0, 30);
        // An acyclic but out-of-order list is rejected, not silently reordered.
        plan.stages[1].dependencies.clear();
        plan.stages[2].dependencies.clear();
        check(t, plan, status::invalid_dependency_order, 10, 0, 30);
    });

    t.test("input_and_output_identity", [](testing & t) {
        auto plan = make_plan();
        plan.inputs = {99};
        check(t, plan, status::invalid_input, 0, 99);
        plan.inputs = {0};
        check(t, plan, status::invalid_input);
        plan.inputs = {11, 11};
        check(t, plan, status::duplicate_input, 0, 11);
        plan.inputs.clear();
        plan.outputs = {99};
        check(t, plan, status::invalid_output, 0, 99);
        plan.outputs = {0};
        check(t, plan, status::invalid_output);
        plan.outputs = {11, 11};
        check(t, plan, status::duplicate_output, 0, 11);
    });

    t.test("reads_need_initialized_contents", [](testing & t) {
        auto plan = make_plan();
        plan.stages[0].requirements[0].access = LLAMA_MEMORY_ACCESS_READ;
        check(t, plan, status::missing_producer, 10, 11);
        plan.stages[0].requirements[0].access = LLAMA_MEMORY_ACCESS_READ_WRITE;
        check(t, plan, status::missing_producer, 10, 11);
        plan.inputs = {11};
        check(t, plan, status::success);
        plan.inputs.clear();
        plan.stages[0].requirements[0].access = LLAMA_MEMORY_ACCESS_NONE;
        check(t, plan, status::missing_producer, 30, 11);
        plan.resources[0].content = llama_memory_content::reconstructible;
        check(t, plan, status::missing_producer, 30, 11);
    });

    t.test("scratch_write_initializes_contents", [](testing & t) {
        auto plan = make_plan();
        plan.resources[0].content = llama_memory_content::discardable;
        plan.stages[1].requirements.clear();
        plan.stages[2].requirements[0].access = LLAMA_MEMORY_ACCESS_WRITE;
        check(t, plan, status::success);
        plan.stages[2].requirements[0].access = LLAMA_MEMORY_ACCESS_READ_WRITE;
        check(t, plan, status::missing_producer, 30, 11);
    });

    t.test("preserved_but_unread_contents_stay_live", [](testing & t) {
        auto plan = make_plan();
        check(t, plan, status::success);
        plan.stages[1].requirements.clear();
        check(t, plan, status::missing_preservation, 20, 11);
        plan.resources[0].content = llama_memory_content::reconstructible;
        check(t, plan, status::success);
        plan.resources[0].content = llama_memory_content::discardable;
        check(t, plan, status::missing_producer, 30, 11);
    });

    t.test("imported_contents_need_preservation_before_first_use", [](testing & t) {
        auto plan = make_plan();
        plan.inputs = {11};
        plan.stages[0].requirements.clear();
        check(t, plan, status::missing_preservation, 10, 11);
        plan.resources[0].content = llama_memory_content::reconstructible;
        check(t, plan, status::success);
        plan.resources[0].content = llama_memory_content::discardable;
        check(t, plan, status::missing_producer, 30, 11);
    });

    t.test("last_use_and_export_extend_lifetimes", [](testing & t) {
        auto plan = make_plan();
        plan.stages[2].requirements.clear();
        check(t, plan, status::missing_preservation, 30, 11);
        plan.outputs.clear();
        check(t, plan, status::success);
        plan.stages[1].requirements.clear();
        check(t, plan, status::success);
        plan.outputs = {11};
        plan.resources[0].content = llama_memory_content::reconstructible;
        check(t, plan, status::success);
        plan.resources[0].content = llama_memory_content::discardable;
        check(t, plan, status::missing_producer, 0, 11);
    });

    t.test("uninitialized_outputs_are_rejected", [](testing & t) {
        auto plan = make_plan();
        for (auto & stage : plan.stages) {
            stage.requirements[0].access = LLAMA_MEMORY_ACCESS_NONE;
        }
        check(t, plan, status::missing_producer, 0, 11);
        plan.stages.clear();
        check(t, plan, status::missing_producer, 0, 11);
        plan.inputs = {11};
        check(t, plan, status::success);
    });

    t.test("unused_catalog_resources_do_not_require_producers", [](testing & t) {
        auto plan = make_plan();
        plan.resources.push_back({12, 1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, llama_memory_content::preserve});
        plan.inputs = {12};
        check(t, plan, status::success);
        plan.outputs.push_back(12);
        check(t, plan, status::missing_preservation, 10, 12);
    });

    t.test("preservation_is_conservative_until_last_declared_use", [](testing & t) {
        auto plan = make_plan();
        plan.stages[1].requirements.clear();
        plan.stages[2].requirements[0].access = LLAMA_MEMORY_ACCESS_WRITE;
        check(t, plan, status::missing_preservation, 20, 11);
        plan.resources[0].content = llama_memory_content::discardable;
        check(t, plan, status::success);
    });

    t.test("single_stage_contracts_are_reused", [](testing & t) {
        auto plan = make_plan();
        plan.stages[0].requirements[0].alignment = 3;
        check(t, plan, status::requirements_error, 10, 11);
        auto result = llama_memory_plan_validate(plan);
        t.assert_true(result.requirements.status == llama_memory_requirements_status::invalid_requirement);
        t.assert_equal(uint64_t(1), result.requirements.domain);
        plan = make_plan();
        plan.stages[1].requirements[0].resource = 99;
        check(t, plan, status::requirements_error, 20, 99);
        result = llama_memory_plan_validate(plan);
        t.assert_true(result.requirements.status == llama_memory_requirements_status::missing_resource);
        plan = make_plan();
        plan.resources[0].domain = 99;
        check(t, plan, status::requirements_error, 0, 11);
    });

    t.test("conflicting_declarations_in_one_stage_are_rejected", [](testing & t) {
        auto plan = make_plan();
        auto read = plan.stages[0].requirements[0];
        read.access = LLAMA_MEMORY_ACCESS_READ;
        plan.stages[0].requirements.push_back(read);
        check(t, plan, status::requirements_error, 10, 11);
        t.assert_true(llama_memory_plan_validate(plan).requirements.status ==
            llama_memory_requirements_status::duplicate_requirement);
        plan.stages[0].requirements.pop_back();
        plan.inputs = {11};
        plan.stages[0].requirements[0].access = LLAMA_MEMORY_ACCESS_READ_WRITE;
        check(t, plan, status::success);
    });

    t.test("malformed_later_stages_precede_unsupported_capabilities", [](testing & t) {
        auto plan = make_plan();
        plan.domains[0].capabilities = 0;
        check(t, plan, status::requirements_error, 10, 11);
        t.assert_true(llama_memory_plan_validate(plan).requirements.status ==
            llama_memory_requirements_status::unsupported_capability);
        plan.stages[2].requirements[0].alignment = 3;
        check(t, plan, status::requirements_error, 30, 11);
        t.assert_true(llama_memory_plan_validate(plan).requirements.status ==
            llama_memory_requirements_status::invalid_requirement);
        plan.stages[2].requirements[0].alignment = 64;
        plan.stages[1].requirements.clear();
        check(t, plan, status::missing_preservation, 20, 11);
    });

    t.test("validation_is_repeatable_and_read_only", [](testing & t) {
        const auto plan = make_plan();
        for (int i = 0; i < 8; ++i) {
            check(t, plan, status::success);
        }
        t.assert_equal(size_t(3), plan.stages.size());
        t.assert_equal(size_t(1), plan.stages[1].dependencies.size());
        t.assert_equal(uint64_t(10), plan.stages[1].dependencies[0]);
        t.assert_true(plan.stages[1].requirements[0].access == LLAMA_MEMORY_ACCESS_NONE);
        t.assert_equal(size_t(128), plan.stages[1].requirements[0].size_min);
    });

    return t.summary();
}
