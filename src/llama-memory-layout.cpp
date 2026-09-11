#include "llama-memory-layout.h"

#include <algorithm>
#include <memory>
#include <new>
#include <utility>

using planner_ptr = std::unique_ptr<ggml_backend_memory_planner, decltype(&ggml_backend_memory_planner_free)>;
static constexpr size_t no_arena = std::numeric_limits<size_t>::max();

// Check an alignment without rounding or overflowing a byte count.
static bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

// Resolve the exact placement domain; backend names are not allocation identities.
static const llama_memory_domain * find_domain(
        const llama_memory_execution_plan & plan, llama_memory_domain_id id) {
    for (const auto & domain : plan.domains) {
        if (domain.id == id) {
            return &domain;
        }
    }
    return nullptr;
}

// Resolve storage properties for an already validated requirement.
static const llama_memory_resource * find_resource(
        const llama_memory_execution_plan & plan, llama_memory_resource_id id) {
    for (const auto & resource : plan.resources) {
        if (resource.id == id) {
            return &resource;
        }
    }
    return nullptr;
}

// Find a stage binding, including an explicitly retained but unread resource.
static const llama_memory_requirement * find_requirement(
        const llama_memory_stage & stage, llama_memory_resource_id id) {
    for (const auto & requirement : stage.requirements) {
        if (requirement.resource == id) {
            return &requirement;
        }
    }
    return nullptr;
}

// Match an exact allocation class rather than treating managed storage as a fallback.
static bool matches_budget(const llama_memory_resource & resource, const llama_memory_arena_budget & budget) {
    return resource.domain == budget.domain && resource.allocation_class == budget.allocation_class;
}

// Build a disposable metadata plan and publish it only after every arena succeeds.
llama_memory_layout_result llama_memory_layout_minimum(
        const llama_memory_execution_plan & plan,
        llama_memory_stage_id stage_id,
        const std::vector<llama_memory_arena_budget> & budgets,
        const std::vector<llama_memory_fixed_region> & fixed,
        llama_memory_layout & output) {
    using status = llama_memory_layout_status;
    const auto validation = llama_memory_plan_validate(plan);
    if (validation.status != llama_memory_plan_status::success) {
        return {status::plan_error, validation.resource, no_arena, validation};
    }
    const llama_memory_stage * stage = nullptr;
    for (const auto & candidate : plan.stages) {
        if (candidate.id == stage_id) {
            stage = &candidate;
            break;
        }
    }
    if (stage == nullptr) {
        return {status::missing_stage, 0, no_arena, {}};
    }

    for (size_t i = 0; i < budgets.size(); ++i) {
        const auto & budget = budgets[i];
        const auto * domain = find_domain(plan, budget.domain);
        if (domain == nullptr || !is_power_of_two(budget.alignment) ||
                !is_power_of_two(budget.allocation_class) ||
                (domain->allocation_classes & budget.allocation_class) == 0) {
            return {status::invalid_budget, 0, i, {}};
        }
        for (size_t j = 0; j < i; ++j) {
            if (budgets[j].domain == budget.domain && budgets[j].allocation_class == budget.allocation_class) {
                return {status::duplicate_budget, 0, i, {}};
            }
        }
    }

    for (size_t i = 0; i < fixed.size(); ++i) {
        const auto & binding = fixed[i];
        const auto & region = binding.region;
        const auto * resource = find_resource(plan, region.id);
        const auto * requirement = find_requirement(*stage, region.id);
        if (binding.arena >= budgets.size() || resource == nullptr || requirement == nullptr) {
            return {status::invalid_fixed_region, region.id, binding.arena, {}};
        }
        const auto & budget = budgets[binding.arena];
        if (!matches_budget(*resource, budget) ||
                region.flags != GGML_BACKEND_MEMORY_REGION_PERSISTENT ||
                region.size == 0 || region.size < requirement->size_min ||
                !is_power_of_two(region.alignment) ||
                region.alignment < std::max(budget.alignment, requirement->alignment) ||
                region.offset % region.alignment != 0 ||
                region.offset > budget.capacity || region.size > budget.capacity - region.offset) {
            return {status::invalid_fixed_region, region.id, binding.arena, {}};
        }
        for (size_t j = 0; j < i; ++j) {
            const auto & previous = fixed[j];
            if (previous.region.id == region.id ||
                    (previous.arena == binding.arena &&
                     region.offset < previous.region.offset + previous.region.size &&
                     previous.region.offset < region.offset + region.size)) {
                return {status::invalid_fixed_region, region.id, binding.arena, {}};
            }
        }
    }

    try {
        std::vector<planner_ptr> planners;
        planners.reserve(budgets.size());
        llama_memory_layout next;
        next.arenas.resize(budgets.size());
        for (size_t i = 0; i < budgets.size(); ++i) {
            const auto & budget = budgets[i];
            next.arenas[i].budget = budget;
            next.arenas[i].unused = budget.capacity;
            planners.emplace_back(
                budget.capacity == 0 ? nullptr : ggml_backend_memory_planner_new(budget.capacity, budget.alignment),
                ggml_backend_memory_planner_free);
            if (budget.capacity != 0 && (!planners.back() ||
                    !ggml_backend_memory_planner_begin(planners.back().get(), GGML_BACKEND_MEMORY_PLAN_NONE))) {
                return {status::allocation_failed, 0, i, {}};
            }
        }

        for (const auto & binding : fixed) {
            const auto & region = binding.region;
            if (!ggml_backend_memory_planner_reserve_at(planners[binding.arena].get(),
                    region.id, region.offset, region.size, region.alignment, region.flags, nullptr)) {
                return {status::allocation_failed, region.id, binding.arena, {}};
            }
        }

        for (const auto & requirement : stage->requirements) {
            if (requirement.size_min == 0) {
                continue;
            }
            bool retained = false;
            for (const auto & binding : fixed) {
                if (binding.region.id == requirement.resource) {
                    retained = true;
                    break;
                }
            }
            if (retained) {
                continue;
            }
            const auto & resource = *find_resource(plan, requirement.resource);
            size_t arena = 0;
            while (arena < budgets.size() && !matches_budget(resource, budgets[arena])) {
                ++arena;
            }
            if (arena == budgets.size()) {
                return {status::missing_budget, resource.id, no_arena, {}};
            }
            if (!planners[arena] || !ggml_backend_memory_planner_reserve(
                    planners[arena].get(), resource.id, requirement.size_min, requirement.alignment,
                    GGML_BACKEND_MEMORY_REGION_NONE, nullptr)) {
                return {status::placement_failed, resource.id, arena, {}};
            }
        }

        for (size_t i = 0; i < planners.size(); ++i) {
            if (!planners[i]) {
                continue;
            }
            auto * planner = planners[i].get();
            if (!ggml_backend_memory_planner_commit(planner)) {
                return {status::placement_failed, 0, i, {}};
            }
            auto & arena = next.arenas[i];
            arena.regions.resize(ggml_backend_memory_planner_region_count(planner));
            for (size_t j = 0; j < arena.regions.size(); ++j) {
                GGML_ASSERT(ggml_backend_memory_planner_get_region_at(planner, j, &arena.regions[j]));
            }
            arena.used = ggml_backend_memory_planner_used(planner);
            arena.high_water = ggml_backend_memory_planner_high_water(planner);
            arena.unused = arena.budget.capacity - arena.used;
        }
        output = std::move(next);
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed, 0, no_arena, {}};
    }
    return {};
}
