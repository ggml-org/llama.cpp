#include "transient-allocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ggml::hrx {
namespace {

static size_t align_up(size_t value, size_t alignment) {
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

struct TransientAllocationRequest {
    ValueId value;
    size_t  required_size = 0;
};

struct TransientAllocationInterval {
    TransientAllocation allocation;
    uint32_t            first_use = 0;
    uint32_t            last_use  = 0;
};

struct FreeTransientBlock {
    size_t offset = 0;
    size_t size   = 0;
};

struct ActiveTransientAllocation {
    size_t   offset   = 0;
    size_t   size     = 0;
    uint32_t last_use = 0;
};

static const CommandPlanTransient * find_plan_transient(const CommandPlan & plan, ValueId value) {
    const auto found = std::find_if(plan.transients.begin(), plan.transients.end(),
                                    [&](const CommandPlanTransient & transient) { return transient.value == value; });
    return found == plan.transients.end() ? nullptr : &*found;
}

static const CommandPlanCompletionCounterRequest * find_plan_completion_counter_request(const CommandPlan & plan,
                                                                                        ValueId             value) {
    const auto found =
        std::find_if(plan.completion_counter_requests.begin(), plan.completion_counter_requests.end(),
                     [&](const CommandPlanCompletionCounterRequest & request) { return request.value == value; });
    return found == plan.completion_counter_requests.end() ? nullptr : &*found;
}

static void add_transient_allocation_request(std::vector<TransientAllocationRequest> & requests,
                                             ValueId                                   value,
                                             size_t                                    required_size) {
    for (TransientAllocationRequest & request : requests) {
        if (request.value == value) {
            request.required_size = std::max(request.required_size, required_size);
            return;
        }
    }
    requests.push_back({ value, required_size });
}

static const TransientAllocationRequest * find_transient_allocation_request(
    const std::vector<TransientAllocationRequest> & requests,
    ValueId                                         value) {
    const auto found = std::find_if(requests.begin(), requests.end(),
                                    [&](const TransientAllocationRequest & request) { return request.value == value; });
    return found == requests.end() ? nullptr : &*found;
}

static bool contains_value(const std::vector<ValueId> & values, ValueId value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

static bool make_transient_allocation(const Graph &                      graph,
                                      const CommandPlan &                command_plan,
                                      const TransientAllocationRequest & request,
                                      TransientAllocation &              allocation,
                                      Status &                           errors) {
    const Value *                graph_value    = graph.values().find(request.value);
    const CommandPlanTransient * plan_transient = find_plan_transient(command_plan, request.value);
    if (graph_value == nullptr && plan_transient == nullptr) {
        errors.log("transient value %d is missing from graph values and command plan transients", request.value.value);
        return false;
    }
    if (graph_value != nullptr && graph_value->kind != ValueKind::Transient) {
        errors.log("transient value %d aliases a non-transient graph value", request.value.value);
        return false;
    }

    allocation.value = request.value;
    allocation.size =
        std::max(graph_value != nullptr ? graph_value->byte_count : plan_transient->size, request.required_size);
    allocation.alignment    = plan_transient != nullptr ? plan_transient->alignment : 256;
    allocation.arena_offset = 0;
    return true;
}

static void add_transient_allocation(const Graph &                      graph,
                                     const CommandPlan &                command_plan,
                                     const TransientAllocationRequest & request,
                                     TransientPlan &                    plan,
                                     Status &                           errors) {
    TransientAllocation allocation;
    if (!make_transient_allocation(graph, command_plan, request, allocation, errors)) {
        return;
    }
    allocation.arena_offset = align_up(plan.arena_size, allocation.alignment);
    plan.arena_size         = allocation.arena_offset + allocation.size;
    plan.allocations.push_back(allocation);
}

static void add_transient_interval(std::vector<TransientAllocationInterval> & intervals,
                                   TransientAllocation                        allocation,
                                   uint32_t                                   command_ordinal) {
    for (TransientAllocationInterval & interval : intervals) {
        if (interval.allocation.value == allocation.value) {
            interval.allocation.size      = std::max(interval.allocation.size, allocation.size);
            interval.allocation.alignment = std::max(interval.allocation.alignment, allocation.alignment);
            interval.first_use            = std::min(interval.first_use, command_ordinal);
            interval.last_use             = std::max(interval.last_use, command_ordinal);
            return;
        }
    }
    intervals.push_back({ allocation, command_ordinal, command_ordinal });
}

static void add_free_transient_block(std::vector<FreeTransientBlock> & free_blocks, size_t offset, size_t size) {
    if (size > 0) {
        free_blocks.push_back({ offset, size });
    }
}

static void coalesce_free_transient_blocks(std::vector<FreeTransientBlock> & free_blocks) {
    std::sort(free_blocks.begin(), free_blocks.end(),
              [](const FreeTransientBlock & lhs, const FreeTransientBlock & rhs) { return lhs.offset < rhs.offset; });
    size_t write_index = 0;
    for (const FreeTransientBlock & block : free_blocks) {
        if (block.size == 0) {
            continue;
        }
        if (write_index > 0) {
            FreeTransientBlock & previous     = free_blocks[write_index - 1];
            const size_t         previous_end = previous.offset + previous.size;
            if (previous_end == block.offset) {
                previous.size += block.size;
                continue;
            }
        }
        free_blocks[write_index++] = block;
    }
    free_blocks.resize(write_index);
}

static void release_completed_transient_allocations(std::vector<ActiveTransientAllocation> & active,
                                                    std::vector<FreeTransientBlock> &        free_blocks,
                                                    uint32_t                                 first_use) {
    size_t write_index = 0;
    for (size_t read_index = 0; read_index < active.size(); ++read_index) {
        if (active[read_index].last_use < first_use) {
            add_free_transient_block(free_blocks, active[read_index].offset, active[read_index].size);
            continue;
        }
        if (write_index != read_index) {
            active[write_index] = active[read_index];
        }
        ++write_index;
    }
    active.resize(write_index);
}

static bool try_allocate_from_free_blocks(std::vector<FreeTransientBlock> & free_blocks,
                                          size_t                            size,
                                          size_t                            alignment,
                                          size_t &                          offset) {
    coalesce_free_transient_blocks(free_blocks);
    for (size_t i = 0; i < free_blocks.size(); ++i) {
        const size_t block_begin      = free_blocks[i].offset;
        const size_t block_end        = free_blocks[i].offset + free_blocks[i].size;
        const size_t aligned_offset   = align_up(block_begin, alignment);
        const bool   aligned_in_block = aligned_offset >= block_begin && aligned_offset <= block_end;
        if (!aligned_in_block || size > block_end - aligned_offset) {
            continue;
        }

        offset                         = aligned_offset;
        const FreeTransientBlock block = free_blocks[i];
        free_blocks.erase(free_blocks.begin() + static_cast<std::ptrdiff_t>(i));
        add_free_transient_block(free_blocks, block.offset, aligned_offset - block.offset);
        add_free_transient_block(free_blocks, aligned_offset + size, block_end - (aligned_offset + size));
        return true;
    }
    return false;
}

static void pack_transient_intervals(std::vector<TransientAllocationInterval> & intervals, TransientPlan & plan) {
    std::sort(intervals.begin(), intervals.end(),
              [](const TransientAllocationInterval & lhs, const TransientAllocationInterval & rhs) {
                  if (lhs.first_use != rhs.first_use) {
                      return lhs.first_use < rhs.first_use;
                  }
                  if (lhs.last_use != rhs.last_use) {
                      return lhs.last_use < rhs.last_use;
                  }
                  return lhs.allocation.value.value < rhs.allocation.value.value;
              });

    std::vector<ActiveTransientAllocation> active;
    std::vector<FreeTransientBlock>        free_blocks;
    for (TransientAllocationInterval & interval : intervals) {
        release_completed_transient_allocations(active, free_blocks, interval.first_use);

        size_t offset = 0;
        if (!try_allocate_from_free_blocks(free_blocks, interval.allocation.size, interval.allocation.alignment,
                                           offset)) {
            offset          = align_up(plan.arena_size, interval.allocation.alignment);
            plan.arena_size = offset + interval.allocation.size;
        }

        interval.allocation.arena_offset = offset;
        active.push_back({ offset, interval.allocation.size, interval.last_use });
        plan.allocations.push_back(interval.allocation);
    }
}

static void add_completion_counter_allocations(const CommandPlan &                             command_plan,
                                               const std::vector<TransientAllocationRequest> & binding_requests,
                                               TransientPlan &                                 plan,
                                               CompletionCounterPlan &                         completion_counters,
                                               Status &                                        errors) {
    for (size_t i = 0; i < command_plan.completion_counter_requests.size(); ++i) {
        const CommandPlanCompletionCounterRequest & request = command_plan.completion_counter_requests[i];
        if (request.value.value < 0) {
            errors.log("completion counter request %s has invalid value %d", request.name.c_str(), request.value.value);
            continue;
        }
        if (request.count == 0) {
            errors.log("completion counter request %s has zero counters", request.name.c_str());
            continue;
        }
        for (size_t j = i + 1; j < command_plan.completion_counter_requests.size(); ++j) {
            if (request.value == command_plan.completion_counter_requests[j].value) {
                errors.log("duplicate completion counter request value %d", request.value.value);
            }
        }
        const size_t                       byte_count = static_cast<size_t>(request.count) * sizeof(int32_t);
        const TransientAllocationRequest * binding_request =
            find_transient_allocation_request(binding_requests, request.value);
        if (binding_request != nullptr && binding_request->required_size > byte_count) {
            errors.log("completion counter request %s requires %zu bytes but binding uses %zu bytes",
                       request.name.c_str(), byte_count, binding_request->required_size);
            continue;
        }
        if (completion_counters.count > std::numeric_limits<uint32_t>::max() - request.count) {
            errors.log("completion counter count overflows");
            continue;
        }

        TransientAllocation allocation;
        allocation.value        = request.value;
        allocation.size         = byte_count;
        allocation.alignment    = 16;
        allocation.arena_offset = align_up(plan.arena_size, allocation.alignment);
        if (completion_counters.byte_count == 0) {
            completion_counters.arena_offset = allocation.arena_offset;
        }
        plan.arena_size = allocation.arena_offset + allocation.size;
        completion_counters.byte_count =
            plan.arena_size > completion_counters.arena_offset ? plan.arena_size - completion_counters.arena_offset : 0;
        completion_counters.count += request.count;
        plan.allocations.push_back(allocation);
    }
}

static void record_transient_lifetime(std::vector<TransientAllocationInterval> & lifetimes,
                                      const CommandBinding &                     binding,
                                      uint32_t                                   command_ordinal) {
    for (TransientAllocationInterval & lifetime : lifetimes) {
        if (lifetime.allocation.value == binding.value) {
            lifetime.first_use = std::min(lifetime.first_use, command_ordinal);
            lifetime.last_use  = std::max(lifetime.last_use, command_ordinal);
            return;
        }
    }
    TransientAllocation allocation;
    allocation.value = binding.value;
    lifetimes.push_back({ allocation, command_ordinal, command_ordinal });
}

static const TransientAllocationInterval * find_transient_lifetime(
    const std::vector<TransientAllocationInterval> & lifetimes,
    ValueId                                          value) {
    const auto found =
        std::find_if(lifetimes.begin(), lifetimes.end(),
                     [&](const TransientAllocationInterval & lifetime) { return lifetime.allocation.value == value; });
    return found == lifetimes.end() ? nullptr : &*found;
}

static bool transient_allocation_overlaps_region(const TransientAllocation & allocation,
                                                 size_t                      region_offset,
                                                 size_t                      region_size) {
    if (region_size == 0) {
        return false;
    }
    return allocation.arena_offset < region_offset + region_size &&
           region_offset < allocation.arena_offset + allocation.size;
}

static bool transient_allocation_has_reserved_lifetime(const CommandProgram &      program,
                                                       const TransientAllocation & allocation) {
    if (transient_allocation_overlaps_region(allocation, program.completion_counters.arena_offset,
                                             program.completion_counters.byte_count)) {
        return true;
    }
    for (const Command & command : program.initialization_commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin == CommandBindingOrigin::Transient && binding.value == allocation.value) {
                return true;
            }
        }
    }
    for (const ConstantInitialization & initialization : program.constant_initializations) {
        if (initialization.value == allocation.value) {
            return true;
        }
    }
    return false;
}

static std::vector<TransientAllocationInterval> collect_main_transient_lifetimes(const CommandProgram & program) {
    std::vector<TransientAllocationInterval> lifetimes;
    for (const Command & command : program.commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin == CommandBindingOrigin::Transient) {
                record_transient_lifetime(lifetimes, binding, command.ordinal);
            }
        }
    }
    return lifetimes;
}

static bool transient_lifetimes_disjoint(const std::vector<TransientAllocationInterval> & lifetimes,
                                         const TransientAllocation &                      lhs,
                                         const TransientAllocation &                      rhs) {
    const TransientAllocationInterval * lhs_lifetime = find_transient_lifetime(lifetimes, lhs.value);
    const TransientAllocationInterval * rhs_lifetime = find_transient_lifetime(lifetimes, rhs.value);
    if (lhs_lifetime == nullptr || rhs_lifetime == nullptr) {
        return false;
    }
    return lhs_lifetime->last_use < rhs_lifetime->first_use || rhs_lifetime->last_use < lhs_lifetime->first_use;
}

}  // namespace

TransientPlan TransientAllocator::allocate(const Graph &                graph,
                                           const CommandPlan &          command_plan,
                                           const std::vector<Command> & initialization_commands,
                                           const std::vector<Command> & commands,
                                           CompletionCounterPlan &      completion_counters,
                                           Status &                     errors) {
    TransientPlan plan;
    plan.arena_alignment = 256;
    std::vector<ValueId>                    reserved_values;
    std::vector<TransientAllocationRequest> reserved_requests;
    std::vector<TransientAllocationRequest> packable_requests;
    std::vector<TransientAllocationRequest> completion_counter_binding_requests;

    for (const Command & command : initialization_commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin == CommandBindingOrigin::Transient &&
                find_plan_completion_counter_request(command_plan, binding.value) == nullptr &&
                !contains_value(reserved_values, binding.value)) {
                reserved_values.push_back(binding.value);
            }
        }
    }
    for (const CommandPlanConstantInitialization & initialization : command_plan.constant_initializations) {
        const bool completion_counter =
            find_plan_completion_counter_request(command_plan, initialization.value) != nullptr;
        if (!completion_counter && !contains_value(reserved_values, initialization.value)) {
            reserved_values.push_back(initialization.value);
        }
        if (initialization.offset > std::numeric_limits<size_t>::max() - initialization.data.size()) {
            errors.log("constant initialization %s range overflows", initialization.name.c_str());
            continue;
        }
        if (completion_counter) {
            add_transient_allocation_request(completion_counter_binding_requests, initialization.value,
                                             initialization.offset + initialization.data.size());
        } else {
            add_transient_allocation_request(reserved_requests, initialization.value,
                                             initialization.offset + initialization.data.size());
        }
    }

    auto append_command_bindings = [&](const std::vector<Command> & command_list, bool packable) {
        for (const Command & command : command_list) {
            for (const CommandBinding & binding : command.bindings) {
                if (binding.origin != CommandBindingOrigin::Transient) {
                    continue;
                }
                if (binding.offset > std::numeric_limits<size_t>::max() - binding.length) {
                    errors.log("transient value %d binding range overflows", binding.value.value);
                    continue;
                }
                if (find_plan_completion_counter_request(command_plan, binding.value) != nullptr) {
                    add_transient_allocation_request(completion_counter_binding_requests, binding.value,
                                                     binding.offset + binding.length);
                } else if (!packable || contains_value(reserved_values, binding.value)) {
                    add_transient_allocation_request(reserved_requests, binding.value, binding.offset + binding.length);
                } else {
                    add_transient_allocation_request(packable_requests, binding.value, binding.offset + binding.length);
                }
            }
        }
    };
    append_command_bindings(initialization_commands, false);
    append_command_bindings(commands, true);
    add_completion_counter_allocations(command_plan, completion_counter_binding_requests, plan, completion_counters,
                                       errors);
    for (const TransientAllocationRequest & request : reserved_requests) {
        add_transient_allocation(graph, command_plan, request, plan, errors);
    }

    std::vector<TransientAllocationInterval> intervals;
    for (const Command & command : commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != CommandBindingOrigin::Transient ||
                find_plan_completion_counter_request(command_plan, binding.value) != nullptr ||
                contains_value(reserved_values, binding.value)) {
                continue;
            }
            const TransientAllocationRequest * request =
                find_transient_allocation_request(packable_requests, binding.value);
            if (request == nullptr) {
                continue;
            }
            TransientAllocation allocation;
            if (make_transient_allocation(graph, command_plan, *request, allocation, errors)) {
                add_transient_interval(intervals, allocation, command.ordinal);
            }
        }
    }
    pack_transient_intervals(intervals, plan);
    plan.arena_size = align_up(plan.arena_size, plan.arena_alignment);
    return plan;
}

bool TransientAllocator::allocations_can_overlap(const CommandProgram &      program,
                                                 const TransientAllocation & lhs,
                                                 const TransientAllocation & rhs) {
    if (transient_allocation_has_reserved_lifetime(program, lhs) ||
        transient_allocation_has_reserved_lifetime(program, rhs)) {
        return false;
    }
    const std::vector<TransientAllocationInterval> lifetimes = collect_main_transient_lifetimes(program);
    return transient_lifetimes_disjoint(lifetimes, lhs, rhs);
}

}  // namespace ggml::hrx
