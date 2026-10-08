#include "command-program-diagnostics.h"

#include <sstream>

namespace ggml::hrx {
namespace {

template <typename Enum> static std::string unknown_enum_name(Enum value) {
    std::ostringstream out;
    out << "Unknown(" << static_cast<int>(value) << ")";
    return out.str();
}

static const char * binding_name(const CommandBinding & binding) {
    return binding.name.empty() ? "<unnamed>" : binding.name.c_str();
}

}  // namespace

std::string command_kind_name(CommandKind kind) {
    switch (kind) {
        case CommandKind::Invalid:
            return "Invalid";
        case CommandKind::Kernel:
            return "Kernel";
    }
    return unknown_enum_name(kind);
}

std::string command_binding_origin_name(CommandBindingOrigin origin) {
    switch (origin) {
        case CommandBindingOrigin::GraphValue:
            return "GraphValue";
        case CommandBindingOrigin::Transient:
            return "Transient";
        case CommandBindingOrigin::ProgramConstant:
            return "ProgramConstant";
    }
    return unknown_enum_name(origin);
}

std::string resource_access_name(ResourceAccess access) {
    switch (access) {
        case ResourceAccess::Read:
            return "Read";
        case ResourceAccess::Write:
            return "Write";
        case ResourceAccess::ReadWrite:
            return "ReadWrite";
    }
    return unknown_enum_name(access);
}

std::string format_command_binding(const CommandBinding & binding) {
    std::ostringstream out;
    out << "binding " << binding_name(binding) << " value=" << binding.value.value
        << " origin=" << command_binding_origin_name(binding.origin)
        << " access=" << resource_access_name(binding.access) << " range=[" << binding.offset << ", "
        << binding.offset + binding.length << ")";
    return out.str();
}

std::string format_command(const Command & command) {
    std::ostringstream out;
    out << "command " << command.ordinal << " kind=" << command_kind_name(command.kind)
        << " kernel_id=" << command.kernel.kernel_id << " bindings=" << command.bindings.size()
        << " deps=" << command.dependencies.size();
    return out.str();
}

std::string format_command_program(const CommandProgram & program) {
    std::ostringstream out;
    out << "command_program commands=" << program.commands.size()
        << " transient_arena=" << program.transients.arena_size
        << " transient_allocations=" << program.transients.allocations.size()
        << " completion_counters=" << program.completion_counters.count << " completion_counter_range=["
        << program.completion_counters.arena_offset << ", "
        << program.completion_counters.arena_offset + program.completion_counters.byte_count << ")";
    for (const Command & command : program.commands) {
        out << '\n' << format_command(command);
        for (const CommandBinding & binding : command.bindings) {
            out << "\n  " << format_command_binding(binding);
        }
    }
    for (const TransientAllocation & allocation : program.transients.allocations) {
        out << "\ntransient value=" << allocation.value.value << " range=[" << allocation.arena_offset << ", "
            << allocation.arena_offset + allocation.size << ") alignment=" << allocation.alignment;
    }
    return out.str();
}

}  // namespace ggml::hrx
