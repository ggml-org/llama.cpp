#pragma once

#include "llama-memory-plan.h"
#include "../ggml/src/ggml-backend-memory.h"

#include <limits>

struct llama_memory_arena_budget {
    llama_memory_domain_id domain = 0;
    llama_memory_allocation_class allocation_class = LLAMA_MEMORY_ALLOCATION_HOST;
    size_t capacity = 0;
    size_t alignment = 1;
};

// The caller supplies snapshots of regions whose leases must survive unchanged.
// Arena indices refer to the budget vector, not backend ordinals.
struct llama_memory_fixed_region {
    size_t arena = 0;
    ggml_backend_memory_region region = {};
};

struct llama_memory_arena_layout {
    llama_memory_arena_budget budget;
    std::vector<ggml_backend_memory_region> regions;
    size_t used = 0;
    size_t high_water = 0;
    // Includes alignment holes, not just unused capacity after high_water.
    size_t unused = 0;
};

struct llama_memory_layout {
    std::vector<llama_memory_arena_layout> arenas;
};

enum class llama_memory_layout_status {
    success,
    plan_error,
    missing_stage,
    invalid_budget,
    duplicate_budget,
    invalid_fixed_region,
    missing_budget,
    placement_failed,
    allocation_failed,
};

struct llama_memory_layout_result {
    llama_memory_layout_status status = llama_memory_layout_status::success;
    llama_memory_resource_id resource = 0;
    size_t arena = std::numeric_limits<size_t>::max();
    llama_memory_plan_result plan;
};

// Plan one validated serial stage using only host-side metadata, never backend storage or leases.
// Reserve fixed persistent regions first, then first-fit minima in requirement order; do not grant preferred bytes.
// Sizes are exact byte minima; only offsets are aligned. Zero minima need no new region or budget.
// Existing fixed regions may exceed preferred sizes and are never shrunk or relabeled.
// Fixed regions must have a current requirement (NONE is valid); content preservation alone does not pin addresses.
// On failure, leave output unchanged. placement_failed covers no aligned fit or GGML metadata allocation failure.
llama_memory_layout_result llama_memory_layout_minimum(
        const llama_memory_execution_plan & plan,
        llama_memory_stage_id stage,
        const std::vector<llama_memory_arena_budget> & budgets,
        const std::vector<llama_memory_fixed_region> & fixed,
        llama_memory_layout & output);
