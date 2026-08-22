#pragma once

#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program-resolver.h"
#include "dispatch/command-program.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/graph-replay.h"
#include "runtime/host-memory.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

typedef struct hrx_device_s *     hrx_device_t;
typedef struct hrx_stream_s *     hrx_stream_t;
typedef struct hrx_buffer_s *     hrx_buffer_t;
typedef struct hrx_graph_s *      hrx_graph_t;
typedef struct hrx_graph_exec_s * hrx_graph_exec_t;
struct ggml_hrx_loom_jit_amdgpu;

namespace ggml::hrx {

class KernelExecutableCache;
struct KernelExecutable;
class TransientArena;

struct CommandProgramExecutionContext {
    hrx_device_t            device             = nullptr;
    hrx_stream_t            stream             = nullptr;
    const char *            target             = nullptr;
    const KernelCorpus *    corpus             = nullptr;
    KernelExecutableCache * kernel_executables = nullptr;
    TransientArena *        transient_arena    = nullptr;
    HostTransferManager *   host_transfers     = nullptr;
    HostWeightCache *       host_weights       = nullptr;
};

struct PreparedCommandBinding {
    CommandBinding    binding;
    ResolvedBufferRef ref;
};

struct PreparedKernelCommand {
    KernelSpecialization                specialization;
    std::shared_ptr<KernelExecutable>   executable;
    std::vector<uint8_t>                constants;
    std::vector<PreparedCommandBinding> bindings;
};

struct PreparedCommand {
    uint32_t              ordinal = 0;
    CommandKind           kind    = CommandKind::Invalid;
    PreparedKernelCommand kernel;
};

struct PreparedProgramConstantBuffer {
    ValueId      value;
    std::string  name;
    hrx_buffer_t buffer = nullptr;
    size_t       size   = 0;

    PreparedProgramConstantBuffer() = default;
    ~PreparedProgramConstantBuffer();

    PreparedProgramConstantBuffer(PreparedProgramConstantBuffer && other) noexcept;
    PreparedProgramConstantBuffer & operator=(PreparedProgramConstantBuffer && other) noexcept;

    PreparedProgramConstantBuffer(const PreparedProgramConstantBuffer &)             = delete;
    PreparedProgramConstantBuffer & operator=(const PreparedProgramConstantBuffer &) = delete;
};

struct PreparedCommandProgram {
    std::vector<PreparedCommand>               initialization_commands;
    std::vector<PreparedCommand>               commands;
    std::vector<HostStagingBuffer>             host_staging;
    std::vector<HostWeightLease>               resident_host_weights;
    std::vector<PreparedProgramConstantBuffer> program_constants;
    Status                                     status;
    uint64_t                                   bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;

    bool valid() const { return status.success(); }
};

struct RecordedCommandGraph {
    hrx_graph_t      graph                               = nullptr;
    hrx_graph_exec_t exec                                = nullptr;
    uint64_t         bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;
    size_t           dispatch_count                      = 0;
    Status           status;

    RecordedCommandGraph() = default;
    ~RecordedCommandGraph();

    RecordedCommandGraph(RecordedCommandGraph && other) noexcept;
    RecordedCommandGraph & operator=(RecordedCommandGraph && other) noexcept;

    RecordedCommandGraph(const RecordedCommandGraph &)             = delete;
    RecordedCommandGraph & operator=(const RecordedCommandGraph &) = delete;

    bool valid() const { return status.success() && exec != nullptr; }
};

struct RecordedCommandGraphExecutionResult {
    bool                success = false;
    Status              status;
    HrxGraphReplayEvent event = HrxGraphReplayEvent::Disabled;
    std::string         ineligible_reason;
    size_t              dispatch_count               = 0;
    bool                transient_allocation_changed = false;
    uint64_t            build_ns                     = 0;
    uint64_t            launch_ns                    = 0;

    uint64_t total_ns() const { return build_ns + launch_ns; }
};

PreparedCommandProgram prepare_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings);

bool execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                      const PreparedCommandProgram &         commands);

bool bind_prepared_command_program_transients(const CommandProgram &              commands,
                                              const TransientArenaAllocationRef & transient_allocation,
                                              PreparedCommandProgram &            prepared);

bool bind_and_execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings,
                                               PreparedCommandProgram &               prepared);

RecordedCommandGraphExecutionResult bind_and_launch_recorded_command_graph(
    const CommandProgramExecutionContext & context,
    const CommandProgram &                 commands,
    const CommandProgramBindings &         bindings,
    PreparedCommandProgram &               prepared,
    RecordedCommandGraph &                 recorded);

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings);

}  // namespace ggml::hrx
