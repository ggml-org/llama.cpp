#include "llama-memory-plan.h"

// Look up an input/output ID without allocating temporary sets.
static bool contains_resource(
        const std::vector<llama_memory_resource_id> & ids, llama_memory_resource_id id) {
    for (auto candidate : ids) {
        if (candidate == id) {
            return true;
        }
    }
    return false;
}

// Resolve a catalog reference without interpreting its storage policy.
static bool has_resource(const llama_memory_execution_plan & plan, llama_memory_resource_id id) {
    for (const auto & resource : plan.resources) {
        if (resource.id == id) {
            return true;
        }
    }
    return false;
}

// Return the explicit binding declaration, including a preserved but unread binding.
static const llama_memory_requirement * find_requirement(
        const llama_memory_stage & stage, llama_memory_resource_id id) {
    for (const auto & requirement : stage.requirements) {
        if (requirement.resource == id) {
            return &requirement;
        }
    }
    return nullptr;
}

// Preserve the original contract diagnostic when adding stage identity.
static llama_memory_plan_result requirements_error(
        llama_memory_requirements_result result, llama_memory_stage_id stage = 0) {
    return {llama_memory_plan_status::requirements_error, stage, result.resource, 0, result};
}

// Validate structure and data lifetimes before returning a capability fallback result.
llama_memory_plan_result llama_memory_plan_validate(const llama_memory_execution_plan & plan) {
    using status = llama_memory_plan_status;
    using requirement_status = llama_memory_requirements_status;
    const auto catalog = llama_memory_requirements_validate(plan.domains, plan.resources, {});
    if (catalog.status != requirement_status::success) {
        return requirements_error(catalog);
    }

    llama_memory_plan_result unsupported;
    for (size_t i = 0; i < plan.stages.size(); ++i) {
        const auto & stage = plan.stages[i];
        if (stage.id == 0) {
            return {status::invalid_stage, 0, 0, 0, {}};
        }
        for (size_t j = 0; j < i; ++j) {
            if (plan.stages[j].id == stage.id) {
                return {status::duplicate_stage, stage.id, 0, 0, {}};
            }
        }
        const auto result = llama_memory_requirements_validate(plan.domains, plan.resources, stage.requirements);
        if (result.status == requirement_status::unsupported_allocation ||
                result.status == requirement_status::unsupported_capability) {
            if (unsupported.status == status::success) {
                unsupported = requirements_error(result, stage.id);
            }
        } else if (result.status != requirement_status::success) {
            return requirements_error(result, stage.id);
        }
    }

    for (size_t i = 0; i < plan.stages.size(); ++i) {
        const auto & stage = plan.stages[i];
        for (size_t j = 0; j < stage.dependencies.size(); ++j) {
            const auto dependency = stage.dependencies[j];
            for (size_t k = 0; k < j; ++k) {
                if (stage.dependencies[k] == dependency) {
                    return {status::duplicate_dependency, stage.id, 0, dependency, {}};
                }
            }
            size_t index = 0;
            while (index < plan.stages.size() && plan.stages[index].id != dependency) {
                ++index;
            }
            if (index == plan.stages.size()) {
                return {status::missing_dependency, stage.id, 0, dependency, {}};
            }
            // Strictly decreasing dependency indices rule out self-edges and every cycle.
            if (index >= i) {
                return {status::invalid_dependency_order, stage.id, 0, dependency, {}};
            }
        }
    }

    for (size_t i = 0; i < plan.inputs.size(); ++i) {
        const auto id = plan.inputs[i];
        if (!has_resource(plan, id)) {
            return {status::invalid_input, 0, id, 0, {}};
        }
        for (size_t j = 0; j < i; ++j) {
            if (plan.inputs[j] == id) {
                return {status::duplicate_input, 0, id, 0, {}};
            }
        }
    }
    for (size_t i = 0; i < plan.outputs.size(); ++i) {
        const auto id = plan.outputs[i];
        if (!has_resource(plan, id)) {
            return {status::invalid_output, 0, id, 0, {}};
        }
        for (size_t j = 0; j < i; ++j) {
            if (plan.outputs[j] == id) {
                return {status::duplicate_output, 0, id, 0, {}};
            }
        }
    }

    for (const auto & resource : plan.resources) {
        const bool output = contains_resource(plan.outputs, resource.id);
        bool used = output;
        size_t last_use = 0;
        for (size_t i = 0; i < plan.stages.size(); ++i) {
            if (find_requirement(plan.stages[i], resource.id) != nullptr) {
                used = true;
                last_use = i;
            }
        }
        if (output) {
            last_use = plan.stages.size();
        }

        bool initialized = contains_resource(plan.inputs, resource.id);
        for (size_t i = 0; i < plan.stages.size(); ++i) {
            const auto & stage = plan.stages[i];
            const auto * requirement = find_requirement(stage, resource.id);
            if (requirement == nullptr) {
                if (initialized && used && i <= last_use && resource.content == llama_memory_content::preserve) {
                    return {status::missing_preservation, stage.id, resource.id, 0, {}};
                }
                if (resource.content == llama_memory_content::discardable) {
                    initialized = false;
                }
                continue;
            }
            if ((requirement->access & LLAMA_MEMORY_ACCESS_READ) != 0 && !initialized) {
                return {status::missing_producer, stage.id, resource.id, 0, {}};
            }
            if ((requirement->access & LLAMA_MEMORY_ACCESS_WRITE) != 0) {
                initialized = true;
            }
        }
        if (output && !initialized) {
            return {status::missing_producer, 0, resource.id, 0, {}};
        }
    }

    return unsupported;
}
