#include "command-program-executor.h"

#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/transient-arena.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {

PreparedProgramConstantBuffer::~PreparedProgramConstantBuffer() {
    if (buffer != nullptr) {
        hrx_buffer_release(buffer);
    }
}

PreparedProgramConstantBuffer::PreparedProgramConstantBuffer(PreparedProgramConstantBuffer && other) noexcept :
    value(other.value),
    name(std::move(other.name)),
    buffer(std::exchange(other.buffer, nullptr)),
    size(other.size) {
    other.size = 0;
}

PreparedProgramConstantBuffer &
PreparedProgramConstantBuffer::operator=(PreparedProgramConstantBuffer && other) noexcept {
    if (this != &other) {
        if (buffer != nullptr) {
            hrx_buffer_release(buffer);
        }
        value        = other.value;
        name         = std::move(other.name);
        buffer       = std::exchange(other.buffer, nullptr);
        size         = other.size;
        other.size   = 0;
    }
    return *this;
}

RecordedCommandGraph::~RecordedCommandGraph() {
    if (exec != nullptr) {
        hrx_graph_exec_release(exec);
    }
    if (graph != nullptr) {
        hrx_graph_release(graph);
    }
}

RecordedCommandGraph::RecordedCommandGraph(RecordedCommandGraph && other) noexcept :
    graph(std::exchange(other.graph, nullptr)),
    exec(std::exchange(other.exec, nullptr)),
    bound_transient_arena_allocation_id(other.bound_transient_arena_allocation_id),
    dispatch_count(other.dispatch_count),
    status(std::move(other.status)) {
    other.bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;
    other.dispatch_count                      = 0;
}

RecordedCommandGraph & RecordedCommandGraph::operator=(RecordedCommandGraph && other) noexcept {
    if (this != &other) {
        if (exec != nullptr) {
            hrx_graph_exec_release(exec);
        }
        if (graph != nullptr) {
            hrx_graph_release(graph);
        }
        graph                                = std::exchange(other.graph, nullptr);
        exec                                 = std::exchange(other.exec, nullptr);
        bound_transient_arena_allocation_id  = other.bound_transient_arena_allocation_id;
        dispatch_count                       = other.dispatch_count;
        status                               = std::move(other.status);
        other.bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;
        other.dispatch_count                      = 0;
    }
    return *this;
}

namespace {

static const char * status_first_error(const Status & status) {
    return status.errors().empty() ? "" : status.errors().front().c_str();
}

static bool resource_access_writes(ResourceAccess access) {
    return access == ResourceAccess::Write || access == ResourceAccess::ReadWrite;
}

static Status command_program_metadata_context_valid(const CommandProgramExecutionContext & context) {
    Status status;
    if (context.target == nullptr) {
        status.log("missing HRX target");
        return status;
    }
    if (context.corpus == nullptr) {
        status.log("missing HRX kernel corpus");
        return status;
    }
    return status;
}

static Status command_program_preparation_context_valid(const CommandProgramExecutionContext & context) {
    Status status;
    if (context.device == nullptr) {
        status.log("missing HRX device");
        return status;
    }
    if (context.kernel_executables == nullptr) {
        status.log("missing HRX kernel executable cache");
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    if (context.host_weights == nullptr) {
        status.log("missing HRX host weight cache");
        return status;
    }
    return status;
}

static Status command_program_transient_context_valid(const CommandProgramExecutionContext & context,
                                                      const CommandProgram &                 commands) {
    Status status;
    if (commands.transients.arena_size == 0) {
        return status;
    }
    if (context.transient_arena == nullptr) {
        status.log("missing HRX transient arena");
        return status;
    }
    if (context.stream == nullptr) {
        status.log("missing HRX stream for transient arena");
        return status;
    }
    return status;
}

static bool prepared_program_has_constant(const PreparedCommandProgram & prepared, ValueId value) {
    for (const PreparedProgramConstantBuffer & constant : prepared.program_constants) {
        if (constant.value == value) {
            return true;
        }
    }
    return false;
}

static bool prepared_execution_context_valid(const CommandProgramExecutionContext & context) {
    if (context.stream == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX stream\n", __func__);
        return false;
    }
    return true;
}

static Status ensure_transient_arena(const CommandProgramExecutionContext & context,
                                     const CommandProgram &                 commands,
                                     TransientArenaAllocationRef &          allocation) {
    allocation    = {};
    Status status = command_program_transient_context_valid(context, commands);
    if (!status.success()) {
        return status;
    }
    if (commands.transients.arena_size == 0) {
        return status;
    }
    status = context.transient_arena->ensure_capacity(context.device, context.stream, commands.transients.arena_size);
    if (!status.success()) {
        return status;
    }
    allocation = context.transient_arena->current_allocation();
    return status;
}

static Status initialize_command_program_constants(const CommandProgramExecutionContext & context,
                                                   const CommandProgram &                 commands,
                                                   const TransientArenaAllocationRef &    allocation,
                                                   const PreparedCommandProgram &         prepared) {
    Status status;
    if (commands.constant_initializations.empty()) {
        return status;
    }
    for (const ConstantInitialization & initialization : commands.constant_initializations) {
        if (prepared_program_has_constant(prepared, initialization.value)) {
            continue;
        }
        if (allocation.buffer == nullptr) {
            status.log("command program has constant initialization %s without a transient arena allocation",
                       initialization.name.c_str());
            continue;
        }
        const TransientAllocation * transient = find_transient_allocation(commands.transients, initialization.value);
        if (transient == nullptr) {
            status.log("constant initialization %s references missing transient value %d", initialization.name.c_str(),
                       initialization.value.value);
            continue;
        }
        if (initialization.offset > transient->size ||
            initialization.data.size() > transient->size - initialization.offset) {
            status.log("constant initialization %s is outside transient allocation length %zu",
                       initialization.name.c_str(), transient->size);
            continue;
        }
        // TODO: Track initialized transient arena allocation ids so constants are not transferred every invocation.
        if (context.host_transfers == nullptr) {
            status.log("constant initialization %s requires a host transfer manager", initialization.name.c_str());
            continue;
        }
        Status upload_status = context.host_transfers->upload_synchronous(
            context.stream, initialization.data.data(), allocation.buffer,
            transient->arena_offset + initialization.offset, initialization.data.size());
        if (!upload_status.success()) {
            status.log("failed to upload constant initialization %s", initialization.name.c_str());
            status.append(upload_status);
        }
    }
    return status;
}

static Status initialize_command_program_completion_counters(const CommandProgramExecutionContext & context,
                                                             const CommandProgram &                 commands,
                                                             const TransientArenaAllocationRef &    allocation) {
    Status status;
    if (commands.completion_counters.byte_count == 0) {
        return status;
    }
    if (allocation.buffer == nullptr) {
        status.log("command program has completion counters without a transient arena allocation");
        return status;
    }
    if (commands.completion_counters.arena_offset > commands.transients.arena_size ||
        commands.completion_counters.byte_count >
            commands.transients.arena_size - commands.completion_counters.arena_offset) {
        status.log("completion counter initialization is outside transient arena length %zu",
                   commands.transients.arena_size);
        return status;
    }
    const uint32_t zero_pattern = 0;
    if (ErrorResult error = take_status(
            hrx_stream_fill_buffer(context.stream, allocation.buffer, commands.completion_counters.arena_offset,
                                   commands.completion_counters.byte_count, &zero_pattern, sizeof(zero_pattern)))) {
        status.log("failed to initialize completion counters: %s", error->c_str());
    }
    return status;
}

static std::string format_resolved_command_context(const ResolvedCommand & command) {
    std::ostringstream out;
    out << "command " << command.ordinal << " kind=" << command_kind_name(command.kind)
        << " kernel_id=" << command.kernel.kernel_id << " bindings=" << command.bindings.size();
    return out.str();
}

static std::string format_prepared_command_context(const PreparedCommand & command) {
    std::ostringstream out;
    out << "command " << command.ordinal << " kind=" << command_kind_name(command.kind);
    if (command.kind == CommandKind::Kernel) {
        out << " kernel_id=" << command.kernel.specialization.kernel_id
            << " bindings=" << command.kernel.bindings.size();
    }
    return out.str();
}

static Dispatch build_dispatch(const ResolvedCommand & command) {
    Dispatch dispatch;
    dispatch.kernel = command.kernel;
    dispatch.bindings.reserve(command.bindings.size());
    for (const ResolvedCommandBinding & binding : command.bindings) {
        dispatch.bindings.push_back({ binding.binding.value, binding.binding.offset, binding.binding.length });
    }
    return dispatch;
}

struct GraphValueAccess {
    bool read  = false;
    bool write = false;
};

static std::unordered_map<int32_t, GraphValueAccess> collect_graph_value_access(const CommandProgram & commands) {
    std::unordered_map<int32_t, GraphValueAccess> access_by_value;
    auto append_command_list_access = [&](const std::vector<Command> & command_list) {
        for (const Command & command : command_list) {
            for (const CommandBinding & binding : command.bindings) {
                if (binding.origin != CommandBindingOrigin::GraphValue) {
                    continue;
                }
                GraphValueAccess & access = access_by_value[binding.value.value];
                switch (binding.access) {
                    case ResourceAccess::Read:
                        access.read = true;
                        break;
                    case ResourceAccess::Write:
                        access.write = true;
                        break;
                    case ResourceAccess::ReadWrite:
                        access.read  = true;
                        access.write = true;
                        break;
                }
            }
        }
    };
    append_command_list_access(commands.initialization_commands);
    append_command_list_access(commands.commands);
    return access_by_value;
}

static CommandProgramBindings materialize_host_bindings(const CommandProgramExecutionContext & context,
                                                        const CommandProgram &                 commands,
                                                        const CommandProgramBindings &         bindings,
                                                        PreparedCommandProgram &               prepared) {
    std::vector<CommandProgramBinding> materialized;
    Status                             status;
    materialized.reserve(bindings.bindings().size());
    const std::unordered_map<int32_t, GraphValueAccess> access_by_value = collect_graph_value_access(commands);
    for (const CommandProgramBinding & binding : bindings.bindings()) {
        if (!binding.requires_materialization()) {
            materialized.push_back(binding);
            continue;
        }
        const auto             found_access = access_by_value.find(binding.value.value);
        const GraphValueAccess access =
            found_access != access_by_value.end() ? found_access->second : GraphValueAccess{};
        if (binding.weight && access.read && !access.write) {
            HostWeightSource source;
            source.host_data  = binding.host_data;
            source.identity   = binding.identity;
            source.generation = binding.generation;
            source.capacity   = binding.capacity;
            source.offset     = binding.offset;
            source.length     = binding.length;
            HostWeightAcquireResult resident =
                context.host_weights->acquire(context.device, context.stream, *context.host_transfers, source);
            if (!resident.valid()) {
                status.log("materialize host weight value %d failed", binding.value.value);
                status.append(resident.status);
                materialized.push_back(binding);
                continue;
            }
            CommandProgramBinding device_binding = binding;
            device_binding.buffer                = resident.lease.buffer();
            device_binding.host_data             = nullptr;
            device_binding.offset                = 0;
            device_binding.capacity              = binding.length;
            materialized.push_back(device_binding);
            prepared.resident_host_weights.push_back(std::move(resident.lease));
            continue;
        }

        HostStagingBuffer staging;
        Status            allocation_status = allocate_host_staging_buffer(context.device, binding.length, staging);
        if (!allocation_status.success()) {
            status.log("allocate host staging for value %d failed", binding.value.value);
            status.append(allocation_status);
            materialized.push_back(binding);
            continue;
        }
        staging.value                        = binding.value.value;
        staging.host_data                    = static_cast<uint8_t *>(binding.host_data) + binding.offset;
        staging.upload                       = access.read;
        staging.download                     = access.write;
        CommandProgramBinding device_binding = binding;
        device_binding.buffer                = staging.buffer;
        device_binding.host_data             = nullptr;
        device_binding.offset                = 0;
        device_binding.capacity              = binding.length;
        materialized.push_back(device_binding);
        prepared.host_staging.push_back(std::move(staging));
    }
    return CommandProgramBindings::from_bindings(std::move(materialized), status);
}

struct ProgramConstantImage {
    ValueId              value;
    std::string          name;
    std::vector<uint8_t> data;
    bool                 read = false;
};

static Status collect_program_constant_images(const CommandProgram & commands,
                                              std::vector<ProgramConstantImage> & images) {
    Status                                  status;
    std::unordered_map<int32_t, size_t>     image_by_value;
    for (const ConstantInitialization & initialization : commands.constant_initializations) {
        const TransientAllocation * allocation = find_transient_allocation(commands.transients, initialization.value);
        if (allocation == nullptr) {
            status.log("constant initialization %s references missing transient value %d", initialization.name.c_str(),
                       initialization.value.value);
            continue;
        }
        if (initialization.offset > allocation->size ||
            initialization.data.size() > allocation->size - initialization.offset) {
            status.log("constant initialization %s is outside transient allocation length %zu",
                       initialization.name.c_str(), allocation->size);
            continue;
        }

        ProgramConstantImage * image = nullptr;
        const auto             found = image_by_value.find(initialization.value.value);
        if (found == image_by_value.end()) {
            ProgramConstantImage next;
            next.value = initialization.value;
            next.name  = initialization.name;
            next.data.resize(allocation->size);
            image_by_value.emplace(initialization.value.value, images.size());
            images.push_back(std::move(next));
            image = &images.back();
        } else {
            image = &images[found->second];
        }

        std::copy(initialization.data.begin(), initialization.data.end(),
                  image->data.begin() + initialization.offset);
    }
    return status;
}

static Status validate_program_constant_access(const CommandProgram &           commands,
                                               std::vector<ProgramConstantImage> & images) {
    Status                              status;
    std::unordered_map<int32_t, size_t> image_by_value;
    for (size_t i = 0; i < images.size(); ++i) {
        image_by_value.emplace(images[i].value.value, i);
    }

    auto validate_command_list = [&](const std::vector<Command> & command_list) {
        for (const Command & command : command_list) {
            for (const CommandBinding & binding : command.bindings) {
                const auto found = image_by_value.find(binding.value.value);
                if (found == image_by_value.end()) {
                    continue;
                }
                ProgramConstantImage & image = images[found->second];
                if (resource_access_writes(binding.access)) {
                    status.log("constant initialization %s is written by command %u; prepared constant buffer copy "
                               "support is required",
                               image.name.c_str(), command.ordinal);
                    continue;
                }
                image.read = true;
            }
        }
    };
    validate_command_list(commands.initialization_commands);
    validate_command_list(commands.commands);
    return status;
}

static PreparedProgramConstantBuffer make_program_constant_buffer(ValueId      value,
                                                                  std::string  name,
                                                                  hrx_buffer_t buffer,
                                                                  size_t       size) {
    PreparedProgramConstantBuffer result;
    result.value  = value;
    result.name   = std::move(name);
    result.buffer = buffer;
    result.size   = size;
    return result;
}

static Status bind_prepared_command_list_program_constants(
    const PreparedCommandProgram &              prepared,
    std::vector<PreparedCommand> &              prepared_commands) {
    Status status;
    for (PreparedCommand & command : prepared_commands) {
        for (PreparedCommandBinding & binding : command.kernel.bindings) {
            for (const PreparedProgramConstantBuffer & constant : prepared.program_constants) {
                if (binding.binding.origin != CommandBindingOrigin::Transient ||
                    binding.binding.value != constant.value) {
                    continue;
                }
                if (binding.binding.offset > constant.size ||
                    binding.binding.length > constant.size - binding.binding.offset) {
                    status.log("%s is outside prepared constant %s length %zu",
                               format_command_binding(binding.binding).c_str(), constant.name.c_str(), constant.size);
                    continue;
                }
                binding.ref = { constant.buffer, binding.binding.offset, binding.binding.length };
                binding.binding.origin = CommandBindingOrigin::ProgramConstant;
            }
        }
    }
    return status;
}

static Status prepare_program_constant_buffers(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               PreparedCommandProgram &               prepared) {
    Status status;
    if (commands.constant_initializations.empty()) {
        return status;
    }

    std::vector<ProgramConstantImage> images;
    status.append(collect_program_constant_images(commands, images));
    status.append(validate_program_constant_access(commands, images));
    if (!status.success()) {
        return status;
    }
    if (images.empty()) {
        return status;
    }
    if (context.device == nullptr) {
        status.log("missing HRX device for prepared constants");
        return status;
    }
    if (context.stream == nullptr) {
        status.log("missing HRX stream for prepared constants");
        return status;
    }

    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    for (const ProgramConstantImage & image : images) {
        if (!image.read) {
            continue;
        }
        hrx_buffer_t buffer = nullptr;
        if (ErrorResult error = take_status(
                hrx_allocator_allocate_buffer(hrx_device_allocator(context.device), params, image.data.size(),
                                              &buffer))) {
            status.log("allocate prepared constant %s: %s", image.name.c_str(), error->c_str());
            continue;
        }
        if (context.host_transfers == nullptr) {
            hrx_buffer_release(buffer);
            status.log("prepared constant %s requires a host transfer manager", image.name.c_str());
            continue;
        }
        Status upload_status =
            context.host_transfers->upload_synchronous(context.stream, image.data.data(), buffer, 0, image.data.size());
        if (!upload_status.success()) {
            hrx_buffer_release(buffer);
            status.log("upload prepared constant %s", image.name.c_str());
            status.append(upload_status);
            continue;
        }
        prepared.program_constants.push_back(
            make_program_constant_buffer(image.value, image.name, buffer, image.data.size()));
    }
    if (!status.success()) {
        return status;
    }

    status.append(bind_prepared_command_list_program_constants(prepared, prepared.initialization_commands));
    status.append(bind_prepared_command_list_program_constants(prepared, prepared.commands));
    return status;
}

static Status rebind_prepared_host_staging(const CommandProgramBindings & bindings, PreparedCommandProgram & prepared) {
    Status status;
    for (HostStagingBuffer & staging : prepared.host_staging) {
        const CommandProgramBinding * binding = bindings.find(ValueId(staging.value));
        if (binding == nullptr || binding->host_data == nullptr || binding->length != staging.length ||
            binding->offset > binding->capacity || binding->length > binding->capacity - binding->offset) {
            status.log("live host binding does not match prepared value %d", staging.value);
            continue;
        }
        staging.host_data = static_cast<uint8_t *>(binding->host_data) + binding->offset;
    }
    return status;
}

static Status upload_prepared_host_staging(const CommandProgramExecutionContext & context,
                                           const PreparedCommandProgram &         prepared) {
    Status status;
    if (prepared.host_staging.empty()) {
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        if (!staging.upload) {
            continue;
        }
        Status upload_status =
            context.host_transfers->upload_async(context.stream, staging.host_data, staging.buffer, 0, staging.length);
        status.append(upload_status);
    }
    return status;
}

static Status download_prepared_host_staging(const CommandProgramExecutionContext & context,
                                             const PreparedCommandProgram &         prepared) {
    Status status;
    if (prepared.host_staging.empty()) {
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        if (!staging.download) {
            continue;
        }
        Status download_status = context.host_transfers->download_synchronous(
            context.stream, staging.buffer, 0, staging.host_data, staging.length);
        status.append(download_status);
    }
    return status;
}

static PreparedCommand make_prepared_command_shape(const ResolvedCommand & command) {
    PreparedCommand prepared;
    prepared.ordinal               = command.ordinal;
    prepared.kind                  = command.kind;
    prepared.kernel.specialization = command.kernel;
    prepared.kernel.bindings.reserve(command.bindings.size());
    for (const ResolvedCommandBinding & binding : command.bindings) {
        prepared.kernel.bindings.push_back({
            binding.binding,
            { binding.ref.buffer, binding.ref.offset, binding.ref.length },
        });
    }
    return prepared;
}

static Status prepare_kernel_command(const CommandProgramExecutionContext & context,
                                     const ResolvedCommand &                command,
                                     PreparedCommand &                      prepared,
                                     KernelExecutableRef &                  executable_ref) {
    Status            status;
    const std::string command_context = format_resolved_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        status.log("unsupported command kind in %s", command_context.c_str());
        return status;
    }
    Dispatch dispatch = build_dispatch(command);

    KernelResolveResult resolved =
        resolve_kernel_definition(*context.corpus, context.target, dispatch.kernel.kernel_id);
    if (!resolved.found()) {
        status.log("%s: %s", command_context.c_str(),
                   format_kernel_resolve_error(resolved, dispatch.kernel.kernel_id).c_str());
        return status;
    }

    prepared       = make_prepared_command_shape(command);
    executable_ref = context.kernel_executables->get_or_compile(
        { context.device, context.target }, *resolved.definition, dispatch, prepared.kernel.constants);
    if (!executable_ref.valid()) {
        status.log("failed to prepare %s", command_context.c_str());
        return status;
    }
    return status;
}

static bool execute_prepared_kernel_command(const CommandProgramExecutionContext & context,
                                            const PreparedCommand &                command) {
    const std::string command_context = format_prepared_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        GGML_LOG_ERROR("%s: unsupported command kind in %s\n", __func__, command_context.c_str());
        return false;
    }
    if (command.kernel.executable == nullptr) {
        GGML_LOG_ERROR("%s: missing kernel executable for %s\n", __func__, command_context.c_str());
        return false;
    }

    std::vector<hrx_buffer_ref_t> refs;
    refs.reserve(command.kernel.bindings.size());
    for (const PreparedCommandBinding & binding : command.kernel.bindings) {
        refs.push_back({ binding.ref.buffer, binding.ref.offset, binding.ref.length });
    }

    const KernelExecutable & executable = *command.kernel.executable;
    hrx_dispatch_config_t    config     = {
        { executable.launch.workgroup_count[0], executable.launch.workgroup_count[1],
         executable.launch.workgroup_count[2] },
        { executable.launch.workgroup_size[0],  executable.launch.workgroup_size[1],
         executable.launch.workgroup_size[2]  },
        executable.launch.subgroup_size,
    };
    if (ErrorResult error = take_status(hrx_stream_dispatch(
            context.stream, executable.executable, executable.export_ordinal, &config, command.kernel.constants.data(),
            command.kernel.constants.size(), refs.data(), refs.size(), 0))) {
        GGML_LOG_ERROR("%s: failed to execute %s: %s\n", __func__, command_context.c_str(), error->c_str());
        return false;
    }
    return true;
}

static void prepare_command_list(const CommandProgramExecutionContext & context,
                                 const std::vector<ResolvedCommand> &   commands,
                                 std::vector<PreparedCommand> &         prepared_commands,
                                 std::vector<KernelExecutableRef> &     executable_refs,
                                 Status &                               status) {
    prepared_commands.reserve(commands.size());
    executable_refs.reserve(commands.size());
    for (const ResolvedCommand & command : commands) {
        PreparedCommand     prepared_command;
        KernelExecutableRef executable_ref;
        Status              command_status = prepare_kernel_command(context, command, prepared_command, executable_ref);
        if (command_status.success()) {
            prepared_commands.push_back(std::move(prepared_command));
            executable_refs.push_back(std::move(executable_ref));
        } else {
            status.append(command_status);
        }
    }
}

static void materialize_command_list_executables(const CommandProgramExecutionContext &   context,
                                                 std::vector<PreparedCommand> &           prepared_commands,
                                                 const std::vector<KernelExecutableRef> & executable_refs,
                                                 Status &                                 status) {
    for (size_t i = 0; i < prepared_commands.size(); ++i) {
        PreparedCommand & command = prepared_commands[i];
        command.kernel.executable = context.kernel_executables->materialize(
            { context.device, context.target }, executable_refs[i], command.kernel.constants);
        if (command.kernel.executable == nullptr) {
            status.log("failed to prepare %s", format_prepared_command_context(command).c_str());
        }
    }
}

static bool bind_prepared_command_list_transients(const CommandProgram &              commands,
                                                  const TransientArenaAllocationRef & transient_allocation,
                                                  std::vector<PreparedCommand> &      prepared_commands) {
    for (PreparedCommand & command : prepared_commands) {
        for (PreparedCommandBinding & binding : command.kernel.bindings) {
            if (binding.binding.origin != CommandBindingOrigin::Transient) {
                continue;
            }
            const TransientAllocation * allocation =
                find_transient_allocation(commands.transients, binding.binding.value);
            if (allocation == nullptr) {
                GGML_LOG_ERROR("%s: %s has no transient allocation\n", __func__,
                               format_command_binding(binding.binding).c_str());
                return false;
            }
            if (binding.binding.offset > allocation->size ||
                binding.binding.length > allocation->size - binding.binding.offset ||
                commands.transients.arena_size > transient_allocation.capacity) {
                GGML_LOG_ERROR("%s: %s is outside transient arena\n", __func__,
                               format_command_binding(binding.binding).c_str());
                return false;
            }
            binding.ref = {
                transient_allocation.buffer,
                allocation->arena_offset + binding.binding.offset,
                binding.binding.length,
            };
        }
    }
    return true;
}

static bool execute_prepared_command_list(const CommandProgramExecutionContext & context,
                                          const std::vector<PreparedCommand> &   commands) {
    for (const PreparedCommand & command : commands) {
        if (!execute_prepared_kernel_command(context, command)) {
            return false;
        }
    }
    return true;
}

struct GraphDependencyChain {
    hrx_graph_node_t last = nullptr;

    const hrx_graph_node_t * deps() const { return last == nullptr ? nullptr : &last; }
    size_t dep_count() const { return last == nullptr ? 0 : 1; }
    void update(hrx_graph_node_t node) { last = node; }
};

static Status record_completion_counter_fill(hrx_graph_t                         graph,
                                             GraphDependencyChain &              chain,
                                             const CommandProgram &              commands,
                                             const TransientArenaAllocationRef & allocation) {
    Status status;
    if (commands.completion_counters.byte_count == 0) {
        return status;
    }
    if (allocation.buffer == nullptr) {
        status.log("command program has completion counters without a transient arena allocation");
        return status;
    }
    if (commands.completion_counters.arena_offset > commands.transients.arena_size ||
        commands.completion_counters.byte_count >
            commands.transients.arena_size - commands.completion_counters.arena_offset) {
        status.log("completion counter graph fill is outside transient arena length %zu",
                   commands.transients.arena_size);
        return status;
    }

    hrx_graph_fill_buffer_node_attrs_t attrs = {
        { allocation.buffer, commands.completion_counters.arena_offset, commands.completion_counters.byte_count },
        0,
        sizeof(uint32_t),
    };
    hrx_graph_node_t node = nullptr;
    if (ErrorResult error =
            take_status(hrx_graph_add_fill_buffer_node(graph, chain.deps(), chain.dep_count(), &attrs, &node))) {
        status.log("record completion counter fill: %s", error->c_str());
        return status;
    }
    chain.update(node);
    return status;
}

static Status record_prepared_kernel_command(hrx_graph_t                  graph,
                                             GraphDependencyChain &       chain,
                                             const PreparedCommand &      command) {
    Status status;
    const std::string command_context = format_prepared_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        status.log("unsupported command kind in %s", command_context.c_str());
        return status;
    }
    if (command.kernel.executable == nullptr) {
        status.log("missing kernel executable for %s", command_context.c_str());
        return status;
    }

    std::vector<hrx_buffer_ref_t> refs;
    refs.reserve(command.kernel.bindings.size());
    for (const PreparedCommandBinding & binding : command.kernel.bindings) {
        if (binding.ref.buffer == nullptr) {
            status.log("%s has unbound buffer in %s", format_command_binding(binding.binding).c_str(),
                       command_context.c_str());
            continue;
        }
        refs.push_back({ binding.ref.buffer, binding.ref.offset, binding.ref.length });
    }
    if (!status.success()) {
        return status;
    }

    const KernelExecutable & executable = *command.kernel.executable;
    hrx_graph_kernel_node_attrs_t attrs = {
        executable.executable,
        executable.export_ordinal,
        {
            { executable.launch.workgroup_count[0], executable.launch.workgroup_count[1],
              executable.launch.workgroup_count[2] },
            { executable.launch.workgroup_size[0], executable.launch.workgroup_size[1],
              executable.launch.workgroup_size[2] },
            executable.launch.subgroup_size,
        },
        command.kernel.constants.data(),
        command.kernel.constants.size(),
        refs.data(),
        refs.size(),
        0,
    };
    hrx_graph_node_t node = nullptr;
    if (ErrorResult error =
            take_status(hrx_graph_add_kernel_node(graph, chain.deps(), chain.dep_count(), &attrs, &node))) {
        status.log("record %s: %s", command_context.c_str(), error->c_str());
        return status;
    }
    chain.update(node);
    return status;
}

static Status record_prepared_command_list(hrx_graph_t                        graph,
                                           GraphDependencyChain &             chain,
                                           const std::vector<PreparedCommand> & commands,
                                           size_t &                           dispatch_count) {
    Status status;
    for (const PreparedCommand & command : commands) {
        Status command_status = record_prepared_kernel_command(graph, chain, command);
        if (!command_status.success()) {
            status.append(command_status);
            return status;
        }
        ++dispatch_count;
    }
    return status;
}

static RecordedCommandGraph record_prepared_command_graph(const CommandProgramExecutionContext & context,
                                                          const CommandProgram &                 commands,
                                                          const PreparedCommandProgram &         prepared,
                                                          const TransientArenaAllocationRef &    allocation) {
    RecordedCommandGraph recorded;
    if (context.device == nullptr) {
        recorded.status.log("missing HRX device for graph replay");
        return recorded;
    }

    hrx_graph_t graph = nullptr;
    if (ErrorResult error = take_status(hrx_graph_create(context.device, 0, &graph))) {
        recorded.status.log("create HRX graph replay: %s", error->c_str());
        return recorded;
    }
    recorded.graph = graph;

    GraphDependencyChain chain;
    recorded.status.append(record_completion_counter_fill(recorded.graph, chain, commands, allocation));
    if (!recorded.status.success()) {
        return recorded;
    }
    recorded.status.append(record_prepared_command_list(recorded.graph, chain, prepared.initialization_commands,
                                                        recorded.dispatch_count));
    if (!recorded.status.success()) {
        return recorded;
    }
    recorded.status.append(record_prepared_command_list(recorded.graph, chain, prepared.commands,
                                                        recorded.dispatch_count));
    if (!recorded.status.success()) {
        return recorded;
    }

    hrx_graph_exec_t exec = nullptr;
    if (ErrorResult error = take_status(hrx_graph_instantiate(recorded.graph, 0, &exec))) {
        recorded.status.log("instantiate HRX graph replay: %s", error->c_str());
        return recorded;
    }
    recorded.exec = exec;
    recorded.bound_transient_arena_allocation_id = prepared.bound_transient_arena_allocation_id;
    return recorded;
}

}  // namespace

PreparedCommandProgram prepare_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings) {
    PreparedCommandProgram prepared;
    prepared.status = command_program_metadata_context_valid(context);
    if (!prepared.status.success()) {
        return prepared;
    }

    const VerificationResult verification = verify_command_program(commands, *context.corpus, context.target);
    if (!verification.valid()) {
        prepared.status.append(verification.status);
        return prepared;
    }
    if (!bindings.valid()) {
        prepared.status.append(bindings.status);
        return prepared;
    }

    TransientArenaAllocationRef transient_allocation;
    prepared.status = ensure_transient_arena(context, commands, transient_allocation);
    if (!prepared.status.success()) {
        return prepared;
    }

    prepared.status = command_program_preparation_context_valid(context);
    if (!prepared.status.success()) {
        return prepared;
    }

    const CommandProgramBindings materialized_bindings =
        materialize_host_bindings(context, commands, bindings, prepared);
    if (!materialized_bindings.valid()) {
        prepared.status.append(materialized_bindings.status);
        return prepared;
    }

    const TransientArenaAllocationRef * transient_allocation_ptr =
        commands.transients.arena_size == 0 ? nullptr : &transient_allocation;
    const ResolvedCommandProgram resolved =
        resolve_command_program_bindings(commands, materialized_bindings, transient_allocation_ptr);
    if (!resolved.valid()) {
        prepared.status.append(resolved.status);
        return prepared;
    }

    std::vector<KernelExecutableRef> initialization_executable_refs;
    std::vector<KernelExecutableRef> command_executable_refs;
    prepare_command_list(context, resolved.initialization_commands, prepared.initialization_commands,
                         initialization_executable_refs, prepared.status);
    prepare_command_list(context, resolved.commands, prepared.commands, command_executable_refs, prepared.status);
    materialize_command_list_executables(context, prepared.initialization_commands, initialization_executable_refs,
                                         prepared.status);
    materialize_command_list_executables(context, prepared.commands, command_executable_refs, prepared.status);
    prepared.bound_transient_arena_allocation_id = transient_allocation.allocation_id;
    if (prepared.status.success()) {
        prepared.status.append(prepare_program_constant_buffers(context, commands, prepared));
    }
    return prepared;
}

bool bind_prepared_command_program_transients(const CommandProgram &              commands,
                                              const TransientArenaAllocationRef & transient_allocation,
                                              PreparedCommandProgram &            prepared) {
    if (!prepared.valid()) {
        return false;
    }
    if (commands.transients.arena_size == 0) {
        prepared.bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;
        return true;
    }
    if (transient_allocation.buffer == nullptr ||
        transient_allocation.allocation_id == kInvalidTransientArenaAllocationId) {
        GGML_LOG_ERROR("%s: missing transient arena allocation\n", __func__);
        return false;
    }
    if (prepared.bound_transient_arena_allocation_id == transient_allocation.allocation_id) {
        return true;
    }
    if (!bind_prepared_command_list_transients(commands, transient_allocation, prepared.initialization_commands) ||
        !bind_prepared_command_list_transients(commands, transient_allocation, prepared.commands)) {
        return false;
    }
    prepared.bound_transient_arena_allocation_id = transient_allocation.allocation_id;
    return true;
}

bool bind_and_execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings,
                                               PreparedCommandProgram &               prepared) {
    if (!prepared.valid()) {
        return execute_prepared_command_program(context, prepared);
    }
    Status rebind_status = rebind_prepared_host_staging(bindings, prepared);
    if (!rebind_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(rebind_status));
        return false;
    }
    if (commands.transients.arena_size == 0) {
        Status status = initialize_command_program_constants(context, commands, {}, prepared);
        if (!status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
            return false;
        }
        status = initialize_command_program_completion_counters(context, commands, {});
        if (!status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
            return false;
        }
        return bind_prepared_command_program_transients(commands, {}, prepared) &&
               execute_prepared_command_program(context, prepared);
    }

    Status status = command_program_transient_context_valid(context, commands);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }

    TransientArena::AllocationLease lease = context.transient_arena->acquire_allocation_lease();
    status = lease.ensure_capacity(context.device, context.stream, commands.transients.arena_size);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    if (!bind_prepared_command_program_transients(commands, lease.current_allocation(), prepared)) {
        return false;
    }
    status = initialize_command_program_constants(context, commands, lease.current_allocation(), prepared);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    status = initialize_command_program_completion_counters(context, commands, lease.current_allocation());
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    return execute_prepared_command_program(context, prepared);
}

RecordedCommandGraphExecutionResult bind_and_launch_recorded_command_graph(
    const CommandProgramExecutionContext & context,
    const CommandProgram &                 commands,
    const CommandProgramBindings &         bindings,
    PreparedCommandProgram &               prepared,
    RecordedCommandGraph &                 recorded) {
    RecordedCommandGraphExecutionResult result;
    result.event = HrxGraphReplayEvent::Ineligible;

    if (!prepared.valid()) {
        result.status.append(prepared.status);
        if (result.status.success()) {
            result.status.log("invalid prepared command program");
        }
        return result;
    }
    if (!prepared_execution_context_valid(context)) {
        result.status.log("missing HRX stream");
        result.event = HrxGraphReplayEvent::BuildFailed;
        return result;
    }

    Status rebind_status = rebind_prepared_host_staging(bindings, prepared);
    if (!rebind_status.success()) {
        result.status.append(rebind_status);
        result.event = HrxGraphReplayEvent::BuildFailed;
        return result;
    }

    TransientArenaAllocationRef transient_allocation;
    TransientArena::AllocationLease lease;
    if (commands.transients.arena_size == 0) {
        if (!bind_prepared_command_program_transients(commands, {}, prepared)) {
            result.status.log("bind transient-free prepared command program failed");
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
    } else {
        Status status = command_program_transient_context_valid(context, commands);
        if (!status.success()) {
            result.status.append(status);
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
        lease  = context.transient_arena->acquire_allocation_lease();
        status = lease.ensure_capacity(context.device, context.stream, commands.transients.arena_size);
        if (!status.success()) {
            result.status.append(status);
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
        transient_allocation = lease.current_allocation();
        if (!bind_prepared_command_program_transients(commands, transient_allocation, prepared)) {
            result.status.log("bind prepared command program transients for graph replay failed");
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
    }

    const bool had_recorded = recorded.valid();
    result.transient_allocation_changed =
        had_recorded && recorded.bound_transient_arena_allocation_id != prepared.bound_transient_arena_allocation_id;
    if (!had_recorded || result.transient_allocation_changed) {
        result.event =
            result.transient_allocation_changed ? HrxGraphReplayEvent::RebuildTransient : HrxGraphReplayEvent::MissBuild;
        const uint64_t build_start_ns = hrx_graph_replay_now_ns();
        RecordedCommandGraph rebuilt = record_prepared_command_graph(context, commands, prepared, transient_allocation);
        result.build_ns              = hrx_graph_replay_now_ns() - build_start_ns;
        if (!rebuilt.valid()) {
            result.status.append(rebuilt.status);
            result.event = HrxGraphReplayEvent::BuildFailed;
            return result;
        }
        recorded = std::move(rebuilt);
    } else {
        result.event = HrxGraphReplayEvent::Hit;
    }

    const uint64_t launch_start_ns = hrx_graph_replay_now_ns();
    Status upload_status = upload_prepared_host_staging(context, prepared);
    if (!upload_status.success()) {
        result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
        result.status.append(upload_status);
        result.event = HrxGraphReplayEvent::LaunchFailed;
        return result;
    }
    if (ErrorResult error = take_status(hrx_graph_exec_launch(recorded.exec, context.stream))) {
        result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
        result.status.log("launch HRX graph replay: %s", error->c_str());
        result.event = HrxGraphReplayEvent::LaunchFailed;
        return result;
    }
    Status download_status = download_prepared_host_staging(context, prepared);
    if (!download_status.success()) {
        result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
        result.status.append(download_status);
        result.event = HrxGraphReplayEvent::LaunchFailed;
        return result;
    }
    result.launch_ns      = hrx_graph_replay_now_ns() - launch_start_ns;
    result.dispatch_count = recorded.dispatch_count;
    result.success        = true;
    return result;
}

bool execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                      const PreparedCommandProgram &         commands) {
    if (!commands.valid()) {
        GGML_LOG_ERROR("%s: invalid HRX prepared command program: %s\n", __func__, status_first_error(commands.status));
        return false;
    }
    if (!prepared_execution_context_valid(context)) {
        return false;
    }
    Status upload_status = upload_prepared_host_staging(context, commands);
    if (!upload_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(upload_status));
        return false;
    }
    if (!execute_prepared_command_list(context, commands.initialization_commands) ||
        !execute_prepared_command_list(context, commands.commands)) {
        return false;
    }
    Status download_status = download_prepared_host_staging(context, commands);
    if (!download_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(download_status));
        return false;
    }
    return true;
}

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings) {
    PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    return bind_and_execute_prepared_command_program(context, commands, bindings, prepared);
}

}  // namespace ggml::hrx
