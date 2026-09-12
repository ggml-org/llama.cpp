#include "command-program-resolver.h"

#include "command-program-diagnostics.h"

#include <string>

namespace ggml::hrx {
namespace {

static Status resolve_command_binding(const Command &                     command,
                                      const CommandProgram &              program,
                                      const CommandBinding &              binding,
                                      const CommandProgramBindings &      bindings,
                                      const TransientArenaAllocationRef * transient_arena,
                                      ResolvedBufferRef &                 ref) {
    Status            status;
    const std::string command_context = format_command(command);
    const std::string binding_context = format_command_binding(binding);
    if (binding.length == 0) {
        status.log("%s %s has an empty range", command_context.c_str(), binding_context.c_str());
        return status;
    }
    switch (binding.origin) {
        case CommandBindingOrigin::GraphValue:
            {
                const CommandProgramBinding * concrete = bindings.find(binding.value);
                if (concrete == nullptr) {
                    status.log("%s %s is not bound", command_context.c_str(), binding_context.c_str());
                    return status;
                }
                if (concrete->buffer == nullptr) {
                    status.log("%s %s has a null buffer", command_context.c_str(), binding_context.c_str());
                    return status;
                }
                if (binding.offset > concrete->length || binding.length > concrete->length - binding.offset) {
                    status.log("%s %s is outside runtime binding length %zu", command_context.c_str(),
                               binding_context.c_str(), concrete->length);
                    return status;
                }
                ref = { concrete->buffer, concrete->offset + binding.offset, binding.length };
                return status;
            }
        case CommandBindingOrigin::Transient:
            {
                const TransientAllocation * allocation = find_transient_allocation(program.transients, binding.value);
                if (allocation == nullptr) {
                    status.log("%s %s has no transient allocation", command_context.c_str(), binding_context.c_str());
                    return status;
                }
                if (transient_arena == nullptr || transient_arena->buffer == nullptr) {
                    status.log("%s %s has no transient arena", command_context.c_str(), binding_context.c_str());
                    return status;
                }
                if (transient_arena->allocation_id == kInvalidTransientArenaAllocationId) {
                    status.log("%s %s has no transient arena allocation id", command_context.c_str(),
                               binding_context.c_str());
                    return status;
                }
                if (program.transients.arena_size > transient_arena->capacity) {
                    status.log("%s %s requires transient arena size %zu but only %zu bytes are available",
                               command_context.c_str(), binding_context.c_str(), program.transients.arena_size,
                               transient_arena->capacity);
                    return status;
                }
                if (binding.offset > allocation->size || binding.length > allocation->size - binding.offset) {
                    status.log("%s %s is outside transient allocation length %zu", command_context.c_str(),
                               binding_context.c_str(), allocation->size);
                    return status;
                }
                ref = { transient_arena->buffer, allocation->arena_offset + binding.offset, binding.length };
                return status;
            }
        case CommandBindingOrigin::ProgramConstant:
            break;
    }
    status.log("%s %s has an unsupported binding origin", command_context.c_str(), binding_context.c_str());
    return status;
}

}  // namespace

static void resolve_command_list(const CommandProgram &              program,
                                 const std::vector<Command> &        commands,
                                 const CommandProgramBindings &      bindings,
                                 const TransientArenaAllocationRef * transient_arena,
                                 std::vector<ResolvedCommand> &      resolved_commands,
                                 Status &                            status) {
    resolved_commands.reserve(commands.size());
    for (const Command & command : commands) {
        ResolvedCommand resolved_command;
        resolved_command.ordinal = command.ordinal;
        resolved_command.kind    = command.kind;
        resolved_command.kernel  = command.kernel;
        resolved_command.bindings.reserve(command.bindings.size());

        for (const CommandBinding & binding : command.bindings) {
            ResolvedCommandBinding resolved_binding;
            resolved_binding.binding = binding;
            Status binding_status =
                resolve_command_binding(command, program, binding, bindings, transient_arena, resolved_binding.ref);
            if (binding_status.success()) {
                resolved_command.bindings.push_back(resolved_binding);
            } else {
                status.append(binding_status);
            }
        }
        resolved_commands.push_back(resolved_command);
    }
}

ResolvedCommandProgram resolve_command_program_bindings(const CommandProgram &              program,
                                                        const CommandProgramBindings &      bindings,
                                                        const TransientArenaAllocationRef * transient_arena) {
    ResolvedCommandProgram result;
    if (!program.valid()) {
        result.status.append(program.status);
    }
    if (!bindings.valid()) {
        result.status.append(bindings.status);
    }

    resolve_command_list(program, program.initialization_commands, bindings, transient_arena,
                         result.initialization_commands, result.status);
    resolve_command_list(program, program.commands, bindings, transient_arena, result.commands, result.status);
    return result;
}

}  // namespace ggml::hrx
