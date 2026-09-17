#include "../src/llama-memory-budget.h"
#include "testing.h"

#include <limits>

using budget_kind     = llama_memory_budget_component_kind;
using budget_location = llama_memory_budget_location;
using budget_status   = llama_memory_budget_status;

static const llama_memory_budget_stage_report * stage(const llama_memory_budget_report & report,
                                                      llama_memory_stage_id              id) {
    for (const auto & candidate : report.stages) {
        if (candidate.stage == id) {
            return &candidate;
        }
    }
    return nullptr;
}

static llama_memory_budget_plan example() {
    llama_memory_budget_plan plan;
    plan.shared_allocation = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
    plan.stages            = { 10, 20, 30 };
    plan.components        = {
        { 1,
         budget_kind::compute_workspace,
         budget_location::shared_parent,
         LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 256,
         true,  0,
         { { 10, 700 }, { 20, 100 }, { 30, 0 } }      },
        { 2,
         budget_kind::kv_pool,
         budget_location::shared_parent,
         LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 128,
         true,  0,
         { { 10, 500 }, { 20, 900 }, { 30, 0 } }      },
        { 3,
         budget_kind::kv_attention_workspace,
         budget_location::shared_parent,
         LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 128,
         true,  0,
         { { 10, 400 }, { 20, 100 }, { 30, 0 } }      },
        { 4,
         budget_kind::kv_writer_workspace,
         budget_location::shared_parent,
         LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 128,
         true,  0,
         { { 10, 32 }, { 20, 32 }, { 30, 0 } }        },
        { 5,
         budget_kind::kv_graph_workspace,
         budget_location::alias,
         LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 1,
         true,  1,
         { { 10, 300 }, { 20, 100 }, { 30, 0 } }      },
        { 6,
         budget_kind::output,
         budget_location::external_device,
         LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 256,
         true,  0,
         { { 10, 64 }, { 20, 64 }, { 30, 64 } }       },
        { 7,
         budget_kind::host_kv,
         budget_location::external_host,
         LLAMA_MEMORY_ALLOCATION_HOST_PINNED,  128,
         true,  0,
         { { 10, 2048 }, { 20, 2048 }, { 30, 2048 } } },
        { 8,
         budget_kind::backend_scratch,
         budget_location::external_device,
         LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 256,
         false, 0,
         { { 10, 128 }, { 20, 64 }, { 30, 256 } }     },
        { 9,
         budget_kind::model_weights,
         budget_location::external_device,
         LLAMA_MEMORY_ALLOCATION_MANAGED,      256,
         true,  0,
         { { 10, 8192 }, { 20, 8192 }, { 30, 8192 } } },
        { 10,
         budget_kind::executable,
         budget_location::external_device,
         LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 256,
         false, 0,
         { { 10, 0 }, { 20, 0 }, { 30, 16 } }         },
    };
    return plan;
}

int main() {
    testing t;

    t.test("accounts_shared_external_alias_and_unknown_components", [](testing & t) {
        const auto                 plan = example();
        llama_memory_budget_report report;
        const auto                 result = llama_memory_budget_plan_make(plan, report);
        if (!t.assert_true(result.status == budget_status::success) ||
            !t.assert_equal(size_t(3), report.stages.size())) {
            return;
        }
        const auto * prefill    = stage(report, 10);
        const auto * decode     = stage(report, 20);
        const auto * transition = stage(report, 30);
        if (!t.assert_true(prefill && decode && transition)) {
            return;
        }

        t.assert_equal(size_t(1632), prefill->shared_payload_bytes);
        t.assert_equal(size_t(1824), prefill->shared_required_bytes);
        t.assert_equal(size_t(192), prefill->shared_alignment_bytes);
        t.assert_equal(size_t(1132), decode->shared_payload_bytes);
        t.assert_equal(size_t(1312), decode->shared_required_bytes);
        t.assert_equal(size_t(180), decode->shared_alignment_bytes);
        t.assert_equal(size_t(0), transition->shared_required_bytes);
        t.assert_equal(size_t(192), prefill->external_device_known_bytes);
        t.assert_equal(size_t(128), decode->external_device_known_bytes);
        t.assert_equal(size_t(336), transition->external_device_known_bytes);
        t.assert_equal(size_t(8192), prefill->external_managed_allocation_bytes);
        t.assert_equal(size_t(2048), prefill->external_host_known_bytes);
        t.assert_equal(size_t(3), prefill->unknown_external_device_components);
        t.assert_equal(size_t(3), transition->unknown_external_device_components);
        t.assert_equal(size_t(1824), report.shared_parent_minimum);
        t.assert_true(!report.device_residency_complete);
    });

    t.test("managed_weights_do_not_change_device_local_parent_class", [](testing & t) {
        auto                       plan = example();
        llama_memory_budget_report report;
        t.assert_true(llama_memory_budget_plan_make(plan, report).status == budget_status::success);
        t.assert_equal(size_t(8192), stage(report, 10)->external_managed_allocation_bytes);
        t.assert_equal(size_t(3), stage(report, 10)->unknown_external_device_components);
        plan.components[0].allocation_class = LLAMA_MEMORY_ALLOCATION_MANAGED;
        const auto result                   = llama_memory_budget_plan_make(plan, report);
        t.assert_true(result.status == budget_status::invalid_allocation);
        t.assert_equal(uint64_t(1), result.component);
    });

    t.test("explicit_transition_stage_reports_overlap_peak", [](testing & t) {
        auto plan = example();
        plan.components.erase(plan.components.begin() + 7, plan.components.end());
        plan.components.push_back({
            8,
            budget_kind::backend_scratch,
            budget_location::external_device,
            LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,
            256,
            true,
            0,
            { { 10, 100 }, { 20, 200 }, { 30, 500 } }
        });
        llama_memory_budget_report report;
        if (!t.assert_true(llama_memory_budget_plan_make(plan, report).status == budget_status::success)) {
            return;
        }
        t.assert_true(report.device_residency_complete);
        t.assert_equal(size_t(564), stage(report, 30)->external_device_known_bytes);
        t.assert_true(stage(report, 30)->external_device_known_bytes > stage(report, 20)->external_device_known_bytes);
    });

    t.test("rejects_missing_duplicate_and_unknown_stage_sizes", [](testing & t) {
        for (int variant = 0; variant < 4; ++variant) {
            auto plan = example();
            if (variant == 0) {
                plan.stages[1] = 10;
            } else if (variant == 1) {
                plan.components[1].id = plan.components[0].id;
            } else if (variant == 2) {
                plan.components[0].stages.pop_back();
            } else {
                plan.components[0].stages[2].stage = 99;
            }
            llama_memory_budget_report report;
            const auto                 result = llama_memory_budget_plan_make(plan, report);
            t.assert_true(result.status == (variant == 0 ? budget_status::duplicate_stage :
                                            variant == 1 ? budget_status::duplicate_component :
                                                           budget_status::invalid_stage));
        }
    });

    t.test("rejects_invalid_alignment_allocation_and_unknown_shared_bytes", [](testing & t) {
        for (int variant = 0; variant < 4; ++variant) {
            auto plan = example();
            if (variant == 0) {
                plan.components[0].alignment = 3;
            } else if (variant == 1) {
                plan.components[0].allocation_class = LLAMA_MEMORY_ALLOCATION_HOST;
            } else if (variant == 2) {
                plan.components[0].exact = false;
            } else {
                plan.components[6].allocation_class = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
            }
            llama_memory_budget_report report;
            const auto                 result = llama_memory_budget_plan_make(plan, report);
            const auto expected               = variant == 0                 ? budget_status::invalid_alignment :
                                                variant == 1 || variant == 3 ? budget_status::invalid_allocation :
                                                                               budget_status::inexact_shared_component;
            t.assert_true(result.status == expected);
        }
    });

    t.test("aliases_must_reference_earlier_compatible_capacity", [](testing & t) {
        for (int variant = 0; variant < 5; ++variant) {
            auto   plan  = example();
            auto & alias = plan.components[4];
            if (variant == 0) {
                alias.alias_of = 99;
            } else if (variant == 1) {
                alias.alias_of = 6;
            } else if (variant == 2) {
                alias.stages[0].bytes = 701;
            } else if (variant == 3) {
                alias.allocation_class = LLAMA_MEMORY_ALLOCATION_MANAGED;
            } else {
                alias.alignment = 512;
            }
            llama_memory_budget_report report;
            const auto                 result = llama_memory_budget_plan_make(plan, report);
            t.assert_true(result.status == budget_status::invalid_alias);
            t.assert_equal(alias.id, result.component);
        }
    });

    t.test("overflow_and_failure_leave_previous_report_unchanged", [](testing & t) {
        auto                       plan = example();
        llama_memory_budget_report report;
        if (!t.assert_true(llama_memory_budget_plan_make(plan, report).status == budget_status::success)) {
            return;
        }
        const auto before                  = report;
        plan.components[1].stages[0].bytes = std::numeric_limits<size_t>::max();
        const auto result                  = llama_memory_budget_plan_make(plan, report);
        t.assert_true(result.status == budget_status::overflow);
        t.assert_equal(before.shared_parent_minimum, report.shared_parent_minimum);
        t.assert_equal(before.stages.size(), report.stages.size());
        for (size_t i = 0; i < before.stages.size(); ++i) {
            t.assert_equal(before.stages[i].shared_required_bytes, report.stages[i].shared_required_bytes);
            t.assert_equal(before.stages[i].external_device_known_bytes, report.stages[i].external_device_known_bytes);
        }
    });

    t.test("text_manifest_classifies_current_compute_and_kv_owners", [](testing & t) {
        llama_text_memory_budget_input input;
        input.prefill_stage              = 10;
        input.decode_stage               = 20;
        input.transition_stage           = 30;
        input.compute_prefill_bytes      = 1024;
        input.compute_decode_bytes       = 256;
        input.compute_alignment          = 256;
        input.kv_pool_bytes              = 2048;
        input.kv_writer_bytes            = 32768;
        input.kv_attention_prefill_bytes = 4096;
        input.kv_attention_decode_bytes  = 512;
        input.kv_alignment               = 128;
        input.host_kv_bytes              = 4096;
        input.host_alignment             = 128;
        input.output_bytes               = 64;
        input.model_weight_bytes         = 8192;
        input.model_weight_allocation    = LLAMA_MEMORY_ALLOCATION_MANAGED;
        input.backend_scratch            = { 128, 64, 256 };
        input.executable                 = { 0, 0, 16 };
        input.driver                     = { 0, 0, 32 };

        llama_memory_budget_plan   plan;
        llama_memory_budget_report report;
        const auto                 result = llama_text_memory_budget_plan_make(input, plan, report);
        if (!t.assert_true(result.status == budget_status::success)) {
            return;
        }
        t.assert_equal(size_t(11), plan.components.size());
        t.assert_equal(size_t(39936), report.shared_parent_minimum);
        t.assert_equal(size_t(39936), stage(report, 10)->shared_required_bytes);
        t.assert_equal(size_t(35584), stage(report, 20)->shared_required_bytes);
        t.assert_equal(size_t(0), stage(report, 30)->shared_required_bytes);
        t.assert_equal(size_t(192), stage(report, 10)->external_device_known_bytes);
        t.assert_equal(size_t(368), stage(report, 30)->external_device_known_bytes);
        t.assert_equal(size_t(8192), stage(report, 10)->external_managed_allocation_bytes);
        t.assert_equal(size_t(4096), stage(report, 20)->external_host_known_bytes);
        t.assert_equal(size_t(4), stage(report, 10)->unknown_external_device_components);
        t.assert_true(!report.device_residency_complete);

        size_t aliases = 0;
        for (const auto & component : plan.components) {
            if (component.location == budget_location::alias) {
                ++aliases;
                t.assert_true(component.kind == budget_kind::kv_graph_workspace);
            }
            if (component.kind == budget_kind::model_weights) {
                t.assert_true(component.location == budget_location::external_device);
                t.assert_true(component.allocation_class == LLAMA_MEMORY_ALLOCATION_MANAGED);
            }
        }
        t.assert_equal(size_t(1), aliases);

        const auto before  = report;
        input.decode_stage = input.prefill_stage;
        t.assert_true(llama_text_memory_budget_plan_make(input, plan, report).status == budget_status::duplicate_stage);
        t.assert_equal(before.shared_parent_minimum, report.shared_parent_minimum);
    });

    t.test("reconciles_observed_device_peaks_without_hiding_residuals", [](testing & t) {
        llama_memory_budget_report report;
        if (!t.assert_true(llama_memory_budget_plan_make(example(), report).status == budget_status::success)) {
            return;
        }
        const std::vector<llama_memory_budget_observation> observations = {
            { 10, 11000 },
            { 20, 10500 },
            { 30, 12000 },
        };
        llama_memory_budget_reconciliation reconciliation;
        const auto result = llama_memory_budget_reconcile(report, 2000, observations, reconciliation);
        if (!t.assert_true(result.status == budget_status::success) ||
            !t.assert_equal(size_t(3), reconciliation.stages.size())) {
            return;
        }
        t.assert_equal(size_t(2192), reconciliation.stages[0].declared_device_bytes);
        t.assert_equal(size_t(8808), reconciliation.stages[0].unclassified_device_bytes);
        t.assert_equal(size_t(8372), reconciliation.stages[1].unclassified_device_bytes);
        t.assert_equal(size_t(9664), reconciliation.stages[2].unclassified_device_bytes);
        t.assert_equal(size_t(12000), reconciliation.maximum_observed_device_bytes);
        t.assert_equal(size_t(9664), reconciliation.maximum_unclassified_device_bytes);

        const auto before                = reconciliation;
        auto       invalid               = observations;
        invalid[0].observed_device_bytes = 100;
        t.assert_true(llama_memory_budget_reconcile(report, 2000, invalid, reconciliation).status ==
                      budget_status::invalid_measurement);
        t.assert_equal(before.maximum_unclassified_device_bytes, reconciliation.maximum_unclassified_device_bytes);
        t.assert_true(
            llama_memory_budget_reconcile(report, report.shared_parent_minimum - 1, observations, reconciliation)
                .status == budget_status::invalid_measurement);
    });
    return t.summary();
}
