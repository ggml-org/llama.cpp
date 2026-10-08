#include "command-program.h"

#include "command-program-diagnostics.h"
#include "transient-allocator.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {
namespace {

static bool string_equal(const char * lhs, const char * rhs) {
    return std::strcmp(lhs != nullptr ? lhs : "", rhs != nullptr ? rhs : "") == 0;
}

static std::string string_value(const char * value) {
    return value != nullptr ? value : "";
}

static CommandBindingOrigin command_binding_origin(const Graph & graph, const Value & value) {
    if (value.kind == ValueKind::External) {
        return CommandBindingOrigin::GraphValue;
    }
    const Value * root = graph.values().find(value.storage_root);
    switch (root != nullptr ? root->kind : value.kind) {
        case ValueKind::External:
            return CommandBindingOrigin::GraphValue;
        case ValueKind::Transient:
            return CommandBindingOrigin::Transient;
    }
    return CommandBindingOrigin::GraphValue;
}

struct StorageBindingTarget {
    ValueId value;
    size_t  offset = 0;
};

static StorageBindingTarget storage_binding_target(const Graph & graph, ValueId value) {
    StorageBindingTarget target;
    target.value              = value;
    const Value * graph_value = graph.values().find(value);
    if (graph_value == nullptr || graph_value->kind != ValueKind::Transient) {
        return target;
    }
    const Value * root = graph.values().find(graph_value->storage_root);
    if (root == nullptr) {
        return target;
    }
    target.value  = root->id;
    target.offset = graph_value->storage_offset;
    return target;
}

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

static void append_command(const Graph &          graph,
                           const CommandPlan &    plan,
                           const KernelCorpus &   corpus,
                           const std::string &    target,
                           const Dispatch &       dispatch,
                           bool                   linear_dependency,
                           std::vector<Command> & commands,
                           Status &               status) {
    Command command;
    command.ordinal = static_cast<uint32_t>(commands.size());
    command.kind    = CommandKind::Kernel;
    command.kernel  = dispatch.kernel;
    // TODO: replace this linear ordinal dependency with real graph/resource dependency analysis.
    if (linear_dependency && command.ordinal > 0) {
        command.dependencies.push_back(command.ordinal - 1);
    }
    const KernelResolveResult resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
    const KernelDefinition *  definition = resolved.definition;
    if (!resolved.found()) {
        status.log("%s", format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
        definition = nullptr;
    } else if (dispatch.bindings.size() != definition->bindings.size()) {
        status.log("command %u kernel %s has %zu bindings but its ABI requires %zu", command.ordinal,
                   kernel_definition_name(*definition).c_str(), dispatch.bindings.size(), definition->bindings.size());
    }
    command.bindings.reserve(dispatch.bindings.size());
    for (size_t binding_index = 0; binding_index < dispatch.bindings.size(); ++binding_index) {
        const DispatchBinding & binding = dispatch.bindings[binding_index];
        CommandBinding          command_binding;
        command_binding.value                                      = binding.value;
        command_binding.offset                                     = binding.offset;
        command_binding.length                                     = binding.length;
        const Value *                               value          = graph.values().find(command_binding.value);
        const CommandPlanTransient *                plan_transient = find_plan_transient(plan, command_binding.value);
        const CommandPlanCompletionCounterRequest * completion_counter =
            find_plan_completion_counter_request(plan, command_binding.value);
        if (value == nullptr && plan_transient == nullptr && completion_counter == nullptr) {
            status.log("command %u binding %zu references missing value %d", command.ordinal, binding_index,
                       command_binding.value.value);
        } else if (value != nullptr) {
            command_binding.origin = command_binding_origin(graph, *value);
        } else {
            command_binding.origin = CommandBindingOrigin::Transient;
        }
        const StorageBindingTarget binding_target = storage_binding_target(graph, command_binding.value);
        command_binding.value                     = binding_target.value;
        if (binding_target.offset > 0) {
            if (binding_target.offset > std::numeric_limits<size_t>::max() - command_binding.offset) {
                status.log("command %u binding %zu storage alias offset overflows", command.ordinal, binding_index);
            } else {
                command_binding.offset += binding_target.offset;
            }
        }
        if (definition != nullptr && binding_index < definition->bindings.size()) {
            command_binding.name   = string_value(definition->bindings[binding_index].name);
            command_binding.access = definition->bindings[binding_index].access;
        }
        command.bindings.push_back(std::move(command_binding));
    }
    commands.push_back(std::move(command));
}

static void verify_command_list(const std::vector<Command> & commands,
                                const TransientPlan &        transients,
                                const KernelCorpus &         corpus,
                                const std::string &          target,
                                Status &                     status) {
    for (size_t i = 0; i < commands.size(); ++i) {
        const Command &   command         = commands[i];
        const std::string command_context = format_command(command);
        if (command.ordinal != i) {
            status.log("%s has non-contiguous ordinal at index %zu", command_context.c_str(), i);
        }
        if (command.kind != CommandKind::Kernel) {
            status.log("%s is not a kernel command", command_context.c_str());
        }
        KernelResolveResult      resolved;
        const KernelDefinition * definition = nullptr;
        if (command.kind == CommandKind::Kernel) {
            resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
            definition = resolved.definition;
        }
        if (command.kind == CommandKind::Kernel && !resolved.found()) {
            status.log("%s: %s", command_context.c_str(),
                       format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
        } else if (definition != nullptr) {
            if (command.bindings.size() != definition->bindings.size()) {
                status.log("%s kernel %s has %zu bindings but its ABI requires %zu", command_context.c_str(),
                           kernel_definition_name(*definition).c_str(), command.bindings.size(),
                           definition->bindings.size());
            }
            const size_t shared_count = std::min(command.bindings.size(), definition->bindings.size());
            for (size_t binding_index = 0; binding_index < shared_count; ++binding_index) {
                const CommandBinding &          binding = command.bindings[binding_index];
                const KernelBindingDefinition & abi     = definition->bindings[binding_index];
                if (!string_equal(binding.name.c_str(), abi.name) || binding.access != abi.access) {
                    status.log("%s %s does not match ABI binding %zu", command_context.c_str(),
                               format_command_binding(binding).c_str(), binding_index);
                }
            }
        }
        if (command.bindings.empty()) {
            status.log("%s has no bindings", command_context.c_str());
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) {
                status.log("%s has forward dependency %u", command_context.c_str(), dependency);
            }
        }
        for (const CommandBinding & binding : command.bindings) {
            const std::string binding_context = format_command_binding(binding);
            if (binding.origin != CommandBindingOrigin::GraphValue &&
                binding.origin != CommandBindingOrigin::Transient) {
                status.log("%s %s has an unsupported binding origin", command_context.c_str(), binding_context.c_str());
            }
            if (binding.origin == CommandBindingOrigin::Transient) {
                const TransientAllocation * allocation = find_transient_allocation(transients, binding.value);
                if (allocation == nullptr) {
                    status.log("%s %s has no transient allocation", command_context.c_str(), binding_context.c_str());
                } else if (binding.offset > allocation->size || binding.length > allocation->size - binding.offset) {
                    status.log("%s %s is outside transient allocation length %zu", command_context.c_str(),
                               binding_context.c_str(), allocation->size);
                }
            }
            if (binding.value.value < 0) {
                status.log("%s %s has an invalid value id", command_context.c_str(), binding_context.c_str());
            }
            if (binding.length == 0) {
                status.log("%s %s has an empty binding", command_context.c_str(), binding_context.c_str());
            }
        }
    }
}

struct VerifyTransientLifetime {
    bool     reserved      = false;
    bool     has_lifetime  = false;
    uint32_t first_command = 0;
    uint32_t last_command  = 0;
};

static size_t saturated_range_end(size_t offset, size_t size) {
    if (offset > std::numeric_limits<size_t>::max() - size) {
        return std::numeric_limits<size_t>::max();
    }
    return offset + size;
}

static bool allocation_overlaps_region(const TransientAllocation & allocation, size_t offset, size_t size) {
    if (size == 0) {
        return false;
    }
    return allocation.arena_offset < saturated_range_end(offset, size) &&
           offset < saturated_range_end(allocation.arena_offset, allocation.size);
}

static std::unordered_map<int32_t, VerifyTransientLifetime> collect_verify_transient_lifetimes(
    const CommandProgram & program) {
    std::unordered_map<int32_t, VerifyTransientLifetime> lifetimes;
    lifetimes.reserve(program.transients.allocations.size());
    for (const TransientAllocation & allocation : program.transients.allocations) {
        VerifyTransientLifetime & lifetime = lifetimes[allocation.value.value];
        if (allocation_overlaps_region(allocation, program.completion_counters.arena_offset,
                                       program.completion_counters.byte_count)) {
            lifetime.reserved = true;
        }
    }
    for (const Command & command : program.initialization_commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin == CommandBindingOrigin::Transient) {
                lifetimes[binding.value.value].reserved = true;
            }
        }
    }
    for (const ConstantInitialization & initialization : program.constant_initializations) {
        lifetimes[initialization.value.value].reserved = true;
    }
    for (const Command & command : program.commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != CommandBindingOrigin::Transient) {
                continue;
            }
            VerifyTransientLifetime & lifetime = lifetimes[binding.value.value];
            if (lifetime.has_lifetime) {
                lifetime.first_command = std::min(lifetime.first_command, command.ordinal);
                lifetime.last_command  = std::max(lifetime.last_command, command.ordinal);
            } else {
                lifetime.has_lifetime  = true;
                lifetime.first_command = command.ordinal;
                lifetime.last_command  = command.ordinal;
            }
        }
    }
    return lifetimes;
}

static bool verify_transient_allocations_can_overlap(
    const std::unordered_map<int32_t, VerifyTransientLifetime> & lifetimes,
    const TransientAllocation &                                  lhs,
    const TransientAllocation &                                  rhs) {
    const auto lhs_lifetime = lifetimes.find(lhs.value.value);
    const auto rhs_lifetime = lifetimes.find(rhs.value.value);
    if (lhs_lifetime == lifetimes.end() || rhs_lifetime == lifetimes.end() || lhs_lifetime->second.reserved ||
        rhs_lifetime->second.reserved || !lhs_lifetime->second.has_lifetime || !rhs_lifetime->second.has_lifetime) {
        return false;
    }
    return lhs_lifetime->second.last_command < rhs_lifetime->second.first_command ||
           rhs_lifetime->second.last_command < lhs_lifetime->second.first_command;
}

static void verify_transient_allocations(const CommandProgram & program, Status & status) {
    const std::unordered_map<int32_t, VerifyTransientLifetime> lifetimes = collect_verify_transient_lifetimes(program);
    std::vector<const TransientAllocation *>                   allocations_by_offset;
    allocations_by_offset.reserve(program.transients.allocations.size());
    for (const TransientAllocation & allocation : program.transients.allocations) {
        if (allocation.value.value < 0 || allocation.size == 0 || allocation.alignment == 0 ||
            allocation.arena_offset % allocation.alignment != 0 ||
            allocation.arena_offset > std::numeric_limits<size_t>::max() - allocation.size ||
            allocation.arena_offset + allocation.size > program.transients.arena_size) {
            status.log("invalid transient allocation for value %d", allocation.value.value);
        }
        allocations_by_offset.push_back(&allocation);
    }
    std::sort(allocations_by_offset.begin(), allocations_by_offset.end(),
              [](const TransientAllocation * lhs, const TransientAllocation * rhs) {
                  if (lhs->arena_offset != rhs->arena_offset) {
                      return lhs->arena_offset < rhs->arena_offset;
                  }
                  return lhs->value.value < rhs->value.value;
              });
    for (size_t i = 0; i < allocations_by_offset.size(); ++i) {
        const TransientAllocation & allocation = *allocations_by_offset[i];
        const size_t                end        = saturated_range_end(allocation.arena_offset, allocation.size);
        for (size_t j = i + 1; j < allocations_by_offset.size(); ++j) {
            const TransientAllocation & other = *allocations_by_offset[j];
            if (other.arena_offset >= end) {
                break;
            }
            if (!verify_transient_allocations_can_overlap(lifetimes, allocation, other)) {
                status.log("transient allocations overlap");
            }
        }
    }
}

}  // namespace

const TransientAllocation * find_transient_allocation(const TransientPlan & plan, ValueId value) {
    const auto found = std::find_if(plan.allocations.begin(), plan.allocations.end(),
                                    [&](const TransientAllocation & allocation) { return allocation.value == value; });
    return found == plan.allocations.end() ? nullptr : &*found;
}

CommandProgram build_command_program(const Graph &        graph,
                                     const CommandPlan &  plan,
                                     const KernelCorpus & corpus,
                                     const std::string &  target) {
    CommandProgram result;
    if (!plan.valid()) {
        result.status.append(plan.status);
        return result;
    }

    result.initialization_commands.reserve(plan.initialization_dispatches.size());
    for (const Dispatch & dispatch : plan.initialization_dispatches) {
        append_command(graph, plan, corpus, target, dispatch, false, result.initialization_commands, result.status);
    }
    for (const Dispatch & dispatch : plan.dispatches) {
        append_command(graph, plan, corpus, target, dispatch, true, result.commands, result.status);
    }
    result.transients = TransientAllocator::allocate(graph, plan, result.initialization_commands, result.commands,
                                                     result.completion_counters, result.status);
    result.constant_initializations.reserve(plan.constant_initializations.size());
    for (const CommandPlanConstantInitialization & initialization : plan.constant_initializations) {
        result.constant_initializations.push_back({
            initialization.value,
            initialization.name,
            initialization.offset,
            initialization.data,
        });
    }
    return result;
}

VerificationResult verify_command_program(const CommandProgram & program,
                                          const KernelCorpus &   corpus,
                                          const std::string &    target) {
    VerificationResult result;
    if (!program.valid()) {
        result.status.append(program.status);
    }
    verify_command_list(program.initialization_commands, program.transients, corpus, target, result.status);
    verify_command_list(program.commands, program.transients, corpus, target, result.status);
    if (program.transients.arena_alignment == 0) {
        result.status.log("transient arena has zero alignment");
    }
    if (program.completion_counters.count == 0) {
        if (program.completion_counters.byte_count != 0) {
            result.status.log("completion counter region has bytes but no counters");
        }
    } else {
        if (program.completion_counters.byte_count == 0) {
            result.status.log("completion counter region has counters but no bytes");
        }
        if (program.completion_counters.byte_count % sizeof(int32_t) != 0) {
            result.status.log("completion counter region byte count is not i32 aligned");
        }
        if (program.completion_counters.arena_offset % 16 != 0) {
            result.status.log("completion counter region is not 16-byte aligned");
        }
        if (program.completion_counters.arena_offset > program.transients.arena_size ||
            program.completion_counters.byte_count >
                program.transients.arena_size - program.completion_counters.arena_offset) {
            result.status.log("completion counter region is outside transient arena");
        }
    }
    verify_transient_allocations(program, result.status);
    for (const ConstantInitialization & initialization : program.constant_initializations) {
        const TransientAllocation * allocation = find_transient_allocation(program.transients, initialization.value);
        if (allocation == nullptr) {
            result.status.log("constant initialization %s references missing transient value %d",
                              initialization.name.c_str(), initialization.value.value);
            continue;
        }
        if (initialization.data.empty()) {
            result.status.log("constant initialization %s has no data", initialization.name.c_str());
        }
        if (initialization.offset > allocation->size ||
            initialization.data.size() > allocation->size - initialization.offset) {
            result.status.log("constant initialization %s is outside transient allocation length %zu",
                              initialization.name.c_str(), allocation->size);
        }
    }
    return result;
}

}  // namespace ggml::hrx
