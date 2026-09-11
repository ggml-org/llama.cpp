#pragma once

#include "llama-memory-requirements.h"

using llama_memory_stage_id = uint64_t;

struct llama_memory_stage {
    llama_memory_stage_id id = 0;
    std::vector<llama_memory_stage_id> dependencies;
    std::vector<llama_memory_requirement> requirements;
};

// Stages are in execution order, not an unordered DAG. Every listed stage executes once.
// Dependencies must refer to earlier stages; repeat decode with a new invocation, not a graph cycle.
struct llama_memory_execution_plan {
    std::vector<llama_memory_domain> domains;
    std::vector<llama_memory_resource> resources;
    std::vector<llama_memory_stage> stages;
    // Caller-supplied initialized contents and contents required after the final stage.
    std::vector<llama_memory_resource_id> inputs;
    std::vector<llama_memory_resource_id> outputs;
};

enum class llama_memory_plan_status {
    success,
    requirements_error,
    invalid_stage,
    duplicate_stage,
    missing_dependency,
    duplicate_dependency,
    invalid_dependency_order,
    invalid_input,
    duplicate_input,
    invalid_output,
    duplicate_output,
    missing_producer,
    missing_preservation,
};

struct llama_memory_plan_result {
    llama_memory_plan_status status = llama_memory_plan_status::success;
    // Zero stage denotes the plan boundary or a catalog error.
    llama_memory_stage_id stage = 0;
    llama_memory_resource_id resource = 0;
    llama_memory_stage_id dependency = 0;
    llama_memory_requirements_result requirements;
};

// Validate an ordered serial plan without allocation, backend calls, or input mutation.
// READ consumes incoming contents, WRITE defines outgoing contents, READ_WRITE requires both.
// NONE retains a binding but does not initialize data. Stage-local scratch should declare WRITE.
// Preserved contents require a binding through their last use/output; reconstructible contents may cross gaps.
// This checks declarations, not actual byte preservation, reconstruction, stage execution, or capacity.
llama_memory_plan_result llama_memory_plan_validate(const llama_memory_execution_plan & plan);
