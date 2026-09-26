#pragma once

#include "command-plan.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"
#include "kernel-corpus/kernel-types.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

enum class CommandKind : uint8_t {
    Invalid,
    Kernel,
};

enum class CommandBindingOrigin : uint8_t {
    GraphValue,
    Transient,
    ProgramConstant,
};

struct CommandBinding {
    std::string          name;
    ValueId              value;
    CommandBindingOrigin origin = CommandBindingOrigin::GraphValue;
    size_t               offset = 0;
    size_t               length = 0;
    ResourceAccess       access = ResourceAccess::Read;
};

struct Command {
    uint32_t                    ordinal = 0;
    CommandKind                 kind    = CommandKind::Kernel;
    KernelSpecialization        kernel;
    std::vector<CommandBinding> bindings;
    std::vector<uint32_t>       dependencies;
};

struct TransientAllocation {
    ValueId value;
    size_t  size         = 0;
    size_t  alignment    = 1;
    size_t  arena_offset = 0;
};

struct TransientPlan {
    size_t                           arena_size      = 0;
    size_t                           arena_alignment = 1;
    std::vector<TransientAllocation> allocations;
};

struct ConstantInitialization {
    ValueId              value;
    std::string          name;
    size_t               offset = 0;
    std::vector<uint8_t> data;
};

struct CompletionCounterPlan {
    size_t   arena_offset = 0;
    size_t   byte_count   = 0;
    uint32_t count        = 0;
};

struct CommandProgram {
    std::vector<Command>                initialization_commands;
    std::vector<Command>                commands;
    TransientPlan                       transients;
    CompletionCounterPlan               completion_counters;
    std::vector<ConstantInitialization> constant_initializations;
    Status                              status;

    bool valid() const { return status.success(); }
};

const TransientAllocation * find_transient_allocation(const TransientPlan & plan, ValueId value);

CommandProgram     build_command_program(const Graph &        graph,
                                         const CommandPlan &  plan,
                                         const KernelCorpus & corpus,
                                         const std::string &  target);
VerificationResult verify_command_program(const CommandProgram & program,
                                          const KernelCorpus &   corpus,
                                          const std::string &    target);

}  // namespace ggml::hrx
