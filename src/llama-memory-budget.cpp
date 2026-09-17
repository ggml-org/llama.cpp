#include "llama-memory-budget.h"

#include <algorithm>
#include <limits>
#include <new>

static bool budget_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static bool budget_allocation_class(llama_memory_allocation_class value) {
    return value == LLAMA_MEMORY_ALLOCATION_HOST || value == LLAMA_MEMORY_ALLOCATION_HOST_PINNED ||
           value == LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL || value == LLAMA_MEMORY_ALLOCATION_MANAGED;
}

static const llama_memory_budget_stage_size * budget_stage(const llama_memory_budget_component & component,
                                                           llama_memory_stage_id                 stage) {
    const auto found =
        std::find_if(component.stages.begin(), component.stages.end(),
                     [&](const llama_memory_budget_stage_size & candidate) { return candidate.stage == stage; });
    return found == component.stages.end() ? nullptr : &*found;
}

static bool budget_add(size_t & total, size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - total) {
        return false;
    }
    total += bytes;
    return true;
}

static bool budget_align(size_t value, size_t alignment, size_t & output) {
    const size_t mask = alignment - 1;
    if (value > std::numeric_limits<size_t>::max() - mask) {
        return false;
    }
    output = (value + mask) & ~mask;
    return true;
}

llama_memory_budget_result llama_memory_budget_plan_make(const llama_memory_budget_plan & plan,
                                                         llama_memory_budget_report &     output) {
    using status = llama_memory_budget_status;
    if (plan.stages.empty()) {
        return { status::invalid_stage, 0, 0 };
    }
    if (plan.shared_allocation != LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL) {
        return { status::invalid_allocation, 0, 0 };
    }
    for (size_t i = 0; i < plan.stages.size(); ++i) {
        if (!plan.stages[i]) {
            return { status::invalid_stage, 0, plan.stages[i] };
        }
        for (size_t j = 0; j < i; ++j) {
            if (plan.stages[j] == plan.stages[i]) {
                return { status::duplicate_stage, 0, plan.stages[i] };
            }
        }
    }

    try {
        llama_memory_budget_report next;
        next.stages.resize(plan.stages.size());
        for (size_t i = 0; i < plan.stages.size(); ++i) {
            next.stages[i].stage = plan.stages[i];
        }

        for (size_t i = 0; i < plan.components.size(); ++i) {
            const auto & component = plan.components[i];
            if (!component.id || component.kind == llama_memory_budget_component_kind::unspecified) {
                return { status::invalid_component, component.id, 0 };
            }
            if (component.stages.size() != plan.stages.size()) {
                return { status::invalid_stage, component.id, 0 };
            }
            for (size_t j = 0; j < i; ++j) {
                if (plan.components[j].id == component.id) {
                    return { status::duplicate_component, component.id, 0 };
                }
            }
            if (!budget_power_of_two(component.alignment)) {
                return { status::invalid_alignment, component.id, 0 };
            }
            if (!budget_allocation_class(component.allocation_class)) {
                return { status::invalid_allocation, component.id, 0 };
            }
            for (size_t j = 0; j < component.stages.size(); ++j) {
                const auto & value = component.stages[j];
                if (!value.stage) {
                    return { status::invalid_stage, component.id, value.stage };
                }
                for (size_t previous = 0; previous < j; ++previous) {
                    if (component.stages[previous].stage == value.stage) {
                        return { status::invalid_stage, component.id, value.stage };
                    }
                }
                if (std::find(plan.stages.begin(), plan.stages.end(), value.stage) == plan.stages.end()) {
                    return { status::invalid_stage, component.id, value.stage };
                }
            }

            if (component.location == llama_memory_budget_location::shared_parent) {
                if (component.allocation_class != plan.shared_allocation) {
                    return { status::invalid_allocation, component.id, 0 };
                }
                if (!component.exact) {
                    return { status::inexact_shared_component, component.id, 0 };
                }
                if (component.alias_of) {
                    return { status::invalid_alias, component.id, 0 };
                }
            } else if (component.location == llama_memory_budget_location::external_device) {
                if (component.allocation_class != LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL &&
                    component.allocation_class != LLAMA_MEMORY_ALLOCATION_MANAGED) {
                    return { status::invalid_allocation, component.id, 0 };
                }
                if (component.alias_of) {
                    return { status::invalid_alias, component.id, 0 };
                }
            } else if (component.location == llama_memory_budget_location::external_host) {
                if (component.allocation_class != LLAMA_MEMORY_ALLOCATION_HOST &&
                    component.allocation_class != LLAMA_MEMORY_ALLOCATION_HOST_PINNED) {
                    return { status::invalid_allocation, component.id, 0 };
                }
                if (component.alias_of) {
                    return { status::invalid_alias, component.id, 0 };
                }
            } else if (component.location == llama_memory_budget_location::alias) {
                if (!component.exact || !component.alias_of) {
                    return { status::invalid_alias, component.id, 0 };
                }
                const auto target = std::find_if(plan.components.begin(), plan.components.begin() + i,
                                                 [&](const llama_memory_budget_component & candidate) {
                                                     return candidate.id == component.alias_of;
                                                 });
                if (target == plan.components.begin() + i || target->location == llama_memory_budget_location::alias ||
                    !target->exact || target->allocation_class != component.allocation_class ||
                    component.alignment > target->alignment) {
                    return { status::invalid_alias, component.id, 0 };
                }
                for (const auto stage : plan.stages) {
                    const auto * alias_size  = budget_stage(component, stage);
                    const auto * target_size = budget_stage(*target, stage);
                    if (!alias_size || !target_size || alias_size->bytes > target_size->bytes) {
                        return { status::invalid_alias, component.id, stage };
                    }
                }
                continue;
            } else {
                return { status::invalid_component, component.id, 0 };
            }

            for (size_t stage_index = 0; stage_index < plan.stages.size(); ++stage_index) {
                const auto   stage_id = plan.stages[stage_index];
                const auto * value    = budget_stage(component, stage_id);
                if (!value) {
                    return { status::invalid_stage, component.id, stage_id };
                }
                auto & report = next.stages[stage_index];
                if (component.location == llama_memory_budget_location::shared_parent) {
                    size_t offset = 0;
                    if (!budget_add(report.shared_payload_bytes, value->bytes) ||
                        !budget_align(report.shared_required_bytes, component.alignment, offset) ||
                        value->bytes > std::numeric_limits<size_t>::max() - offset) {
                        return { status::overflow, component.id, stage_id };
                    }
                    report.shared_required_bytes = offset + value->bytes;
                } else if (component.location == llama_memory_budget_location::external_device) {
                    if (component.allocation_class == LLAMA_MEMORY_ALLOCATION_MANAGED) {
                        if (!budget_add(report.external_managed_allocation_bytes, value->bytes)) {
                            return { status::overflow, component.id, stage_id };
                        }
                        ++report.unknown_external_device_components;
                        next.device_residency_complete = false;
                    } else {
                        if (!budget_add(report.external_device_known_bytes, value->bytes)) {
                            return { status::overflow, component.id, stage_id };
                        }
                        if (!component.exact) {
                            ++report.unknown_external_device_components;
                            next.device_residency_complete = false;
                        }
                    }
                } else {
                    if (!budget_add(report.external_host_known_bytes, value->bytes)) {
                        return { status::overflow, component.id, stage_id };
                    }
                    if (!component.exact) {
                        ++report.unknown_external_host_components;
                        next.host_complete = false;
                    }
                }
            }
        }

        for (auto & report : next.stages) {
            report.shared_alignment_bytes = report.shared_required_bytes - report.shared_payload_bytes;
            next.shared_parent_minimum    = std::max(next.shared_parent_minimum, report.shared_required_bytes);
        }
        output = std::move(next);
        return {};
    } catch (const std::bad_alloc &) {
        return { status::allocation_failed, 0, 0 };
    }
}

llama_memory_budget_result llama_text_memory_budget_plan_make(const llama_text_memory_budget_input & input,
                                                              llama_memory_budget_plan &             plan,
                                                              llama_memory_budget_report &           report) {
    using kind     = llama_memory_budget_component_kind;
    using location = llama_memory_budget_location;
    using status   = llama_memory_budget_status;
    try {
        const auto sizes = [&](size_t prefill, size_t decode, size_t transition) {
            return std::vector<llama_memory_budget_stage_size>{
                { input.prefill_stage,    prefill    },
                { input.decode_stage,     decode     },
                { input.transition_stage, transition },
            };
        };
        const auto phase_sizes = [&](const llama_memory_budget_phase_bytes & value) {
            return sizes(value.prefill, value.decode, value.transition);
        };

        llama_memory_budget_plan next;
        next.shared_allocation = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
        next.stages            = { input.prefill_stage, input.decode_stage, input.transition_stage };
        next.components        = {
            { 1, kind::compute_workspace, location::shared_parent, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,
             input.compute_alignment, true, 0, sizes(input.compute_prefill_bytes, input.compute_decode_bytes, 0) },
            { 2, kind::kv_pool, location::shared_parent, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, input.kv_alignment, true,
             0, sizes(input.kv_pool_bytes, input.kv_pool_bytes, 0) },
            { 3, kind::kv_writer_workspace, location::shared_parent, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,
             input.kv_alignment, true, 0, sizes(input.kv_writer_bytes, input.kv_writer_bytes, 0) },
            { 4, kind::kv_attention_workspace, location::shared_parent, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,
             input.kv_alignment, true, 0,
             sizes(input.kv_attention_prefill_bytes, input.kv_attention_decode_bytes, 0) },
            { 5, kind::kv_graph_workspace, location::alias, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,
             input.compute_alignment, true, 1, sizes(input.compute_prefill_bytes, input.compute_decode_bytes, 0) },
            { 6, kind::host_kv, location::external_host, LLAMA_MEMORY_ALLOCATION_HOST_PINNED, input.host_alignment,
             true, 0, sizes(input.host_kv_bytes, input.host_kv_bytes, input.host_kv_bytes) },
            { 7, kind::output, location::external_device, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 1, true, 0,
             sizes(input.output_bytes, input.output_bytes, input.output_bytes) },
            { 8, kind::model_weights, location::external_device, input.model_weight_allocation, 1, true, 0,
             sizes(input.model_weight_bytes, input.model_weight_bytes, input.model_weight_bytes) },
            { 9, kind::backend_scratch, location::external_device, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 1, false, 0,
             phase_sizes(input.backend_scratch) },
            { 10, kind::executable, location::external_device, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 1, false, 0,
             phase_sizes(input.executable) },
            { 11, kind::driver, location::external_device, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, 1, false, 0,
             phase_sizes(input.driver) },
        };

        llama_memory_budget_report next_report;
        const auto                 result = llama_memory_budget_plan_make(next, next_report);
        if (result.status != status::success) {
            return result;
        }
        plan   = std::move(next);
        report = std::move(next_report);
        return {};
    } catch (const std::bad_alloc &) {
        return { status::allocation_failed, 0, 0 };
    }
}

llama_memory_budget_result llama_memory_budget_reconcile(
    const llama_memory_budget_report &                   report,
    size_t                                               shared_parent_capacity,
    const std::vector<llama_memory_budget_observation> & observations,
    llama_memory_budget_reconciliation &                 output) {
    using status = llama_memory_budget_status;
    if (report.stages.empty() || observations.size() != report.stages.size() ||
        shared_parent_capacity < report.shared_parent_minimum) {
        return { status::invalid_measurement, 0, 0 };
    }
    try {
        llama_memory_budget_reconciliation next;
        next.stages.reserve(report.stages.size());
        for (const auto & stage_report : report.stages) {
            const auto observation = std::find_if(observations.begin(), observations.end(),
                                                  [&](const llama_memory_budget_observation & candidate) {
                                                      return candidate.stage == stage_report.stage;
                                                  });
            if (!stage_report.stage || observation == observations.end()) {
                return { status::invalid_measurement, 0, stage_report.stage };
            }
            size_t matches = 0;
            for (const auto & candidate : observations) {
                matches += candidate.stage == stage_report.stage;
            }
            if (matches != 1 || stage_report.external_device_known_bytes >
                                    std::numeric_limits<size_t>::max() - shared_parent_capacity) {
                return { status::invalid_measurement, 0, stage_report.stage };
            }
            const size_t declared = shared_parent_capacity + stage_report.external_device_known_bytes;
            if (observation->observed_device_bytes < declared) {
                return { status::invalid_measurement, 0, stage_report.stage };
            }
            const size_t residual = observation->observed_device_bytes - declared;
            next.stages.push_back({
                stage_report.stage,
                observation->observed_device_bytes,
                declared,
                residual,
            });
            next.maximum_observed_device_bytes =
                std::max(next.maximum_observed_device_bytes, observation->observed_device_bytes);
            next.maximum_unclassified_device_bytes = std::max(next.maximum_unclassified_device_bytes, residual);
        }
        output = std::move(next);
        return {};
    } catch (const std::bad_alloc &) {
        return { status::allocation_failed, 0, 0 };
    }
}
