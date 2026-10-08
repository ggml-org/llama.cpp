#pragma once

#include "command-program-bindings.h"
#include "command-program.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml::hrx {

struct ResolvedBufferRef {
    hrx_buffer_t buffer = nullptr;
    size_t       offset = 0;
    size_t       length = 0;
};

static constexpr uint64_t kInvalidTransientArenaAllocationId = 0;

struct TransientArenaAllocationRef {
    hrx_buffer_t buffer        = nullptr;
    size_t       capacity      = 0;
    uint64_t     allocation_id = kInvalidTransientArenaAllocationId;
};

struct ResolvedCommandBinding {
    CommandBinding    binding;
    ResolvedBufferRef ref;
};

struct ResolvedCommand {
    uint32_t                            ordinal = 0;
    CommandKind                         kind    = CommandKind::Kernel;
    KernelSpecialization                kernel;
    std::vector<ResolvedCommandBinding> bindings;
};

struct ResolvedCommandProgram {
    std::vector<ResolvedCommand> initialization_commands;
    std::vector<ResolvedCommand> commands;
    Status                       status;

    bool valid() const { return status.success(); }
};

ResolvedCommandProgram resolve_command_program_bindings(const CommandProgram &              program,
                                                        const CommandProgramBindings &      bindings,
                                                        const TransientArenaAllocationRef * transient_arena = nullptr);

}  // namespace ggml::hrx
