#include "command-program-executor.h"

#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/transient-arena.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {

GraphReplayStreamState::~GraphReplayStreamState() {
    clear();
}

void GraphReplayStreamState::clear() {
    for (PendingHostWriteback & writeback : pending_host_writebacks_) {
        if (writeback.retained_buffer != nullptr) {
            hrx_buffer_release(writeback.retained_buffer);
            writeback.retained_buffer = nullptr;
        }
    }
    pending_host_writebacks_.clear();
}

void GraphReplayStreamState::mark_stream_synchronized() {
    for (PendingHostWriteback & writeback : pending_host_writebacks_) {
        if (writeback.host_destination != nullptr && writeback.mapped_source != nullptr && writeback.size > 0) {
            std::memcpy(writeback.host_destination, writeback.mapped_source, writeback.size);
        }
        if (writeback.retained_buffer != nullptr) {
            hrx_buffer_release(writeback.retained_buffer);
            writeback.retained_buffer = nullptr;
        }
    }
    pending_host_writebacks_.clear();
}

void GraphReplayStreamState::add_host_writeback(void *       host_destination,
                                                const void * mapped_source,
                                                size_t       size,
                                                hrx_buffer_t buffer) {
    if (buffer != nullptr) {
        hrx_buffer_retain(buffer);
    }
    pending_host_writebacks_.push_back({ host_destination, mapped_source, size, buffer });
}

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

PreparedProgramConstantBuffer & PreparedProgramConstantBuffer::operator=(
    PreparedProgramConstantBuffer && other) noexcept {
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
    for (hrx_buffer_t buffer : retained_buffers) {
        hrx_buffer_release(buffer);
    }
    for (hrx_executable_t executable : retained_executables) {
        hrx_executable_release(executable);
    }
}

RecordedCommandGraph::RecordedCommandGraph(RecordedCommandGraph && other) noexcept :
    graph(std::exchange(other.graph, nullptr)),
    exec(std::exchange(other.exec, nullptr)),
    retained_buffers(std::move(other.retained_buffers)),
    retained_executables(std::move(other.retained_executables)),
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
        for (hrx_buffer_t buffer : retained_buffers) {
            hrx_buffer_release(buffer);
        }
        for (hrx_executable_t executable : retained_executables) {
            hrx_executable_release(executable);
        }
        graph                                = std::exchange(other.graph, nullptr);
        exec                                 = std::exchange(other.exec, nullptr);
        retained_buffers                          = std::move(other.retained_buffers);
        retained_executables                      = std::move(other.retained_executables);
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

static bool environment_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

static const char * environment_string_value(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0') ? value : nullptr;
}

static size_t environment_size_value(const char * name, size_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return static_cast<size_t>(std::strtoull(value, nullptr, 0));
}

static void debug_serial_log(const char * event, const char * phase, const std::string & detail) {
    std::fprintf(stderr, "hrx debug serial: %s phase=%s", event, phase);
    if (!detail.empty()) {
        std::fprintf(stderr, " %s", detail.c_str());
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

class DebugSerialExecutionTrace {
  public:
    explicit DebugSerialExecutionTrace(const CommandProgramExecutionContext & context) :
        context_(context),
        enabled_(debug_serial_command_execution_enabled()) {}

    DebugSerialExecutionTrace(const CommandProgramExecutionContext & context, const CommandProgram & commands) :
        DebugSerialExecutionTrace(context) {
        if (enabled_) {
            std::ostringstream out;
            out << "main_commands=" << commands.commands.size()
                << " init_commands=" << commands.initialization_commands.size()
                << " transient_arena=" << commands.transients.arena_size;
            program_detail_ = out.str();
        }
    }

    DebugSerialExecutionTrace(const CommandProgramExecutionContext & context,
                              const PreparedCommandProgram &         commands) :
        DebugSerialExecutionTrace(context) {
        if (enabled_) {
            std::ostringstream out;
            out << "main_commands=" << commands.commands.size()
                << " init_commands=" << commands.initialization_commands.size();
            program_detail_ = out.str();
        }
    }

    bool enabled() const {
        return enabled_;
    }

    void log_program_begin() const {
        log("begin", "command-program", program_detail_);
    }

    void log_program_end(bool success) const {
        log(success ? "end" : "failed", "command-program", program_detail_);
    }

    bool sync_program_phase(const char * phase) const {
        return sync(phase, program_detail_);
    }

    bool sync(const char * phase, const std::string & detail = {}) const {
        if (!enabled_) {
            return true;
        }
        log("sync-begin", phase, detail);
        if (ErrorResult error = take_status(hrx_stream_synchronize(context_.stream))) {
            GGML_LOG_ERROR("HRX debug serial sync failed phase=%s %s: %s\n", phase, detail.c_str(), error->c_str());
            return false;
        }
        log("sync-end", phase, detail);
        return true;
    }

    bool should_trace_program(size_t command_count) const {
        const char * value = std::getenv("GGML_HRX_DEBUG_SERIAL_PROGRAM_COMMANDS");
        return value == nullptr || value[0] == '\0' || command_count == std::strtoull(value, nullptr, 0);
    }

    bool should_trace_command(size_t program_command_count, size_t index) const {
        if (!should_trace_program(program_command_count)) {
            return false;
        }
        const size_t start = environment_size_value("GGML_HRX_DEBUG_SERIAL_START_COMMAND", 0);
        const size_t end   = environment_size_value("GGML_HRX_DEBUG_SERIAL_END_COMMAND",
                                                    std::numeric_limits<size_t>::max());
        return index >= start && index <= end;
    }

    void log(const char * event, const char * phase, const std::string & detail = {}) const {
        if (enabled_) {
            debug_serial_log(event, phase, detail);
        }
    }

    static std::string command_detail(const char *           list_kind,
                                      size_t                 list_size,
                                      size_t                 index,
                                      const PreparedCommand & command) {
        std::ostringstream out;
        out << "list=" << list_kind
            << " index=" << index
            << " list_commands=" << list_size
            << " ordinal=" << command.ordinal
            << " kind=" << command_kind_name(command.kind);
        if (command.kind == CommandKind::Kernel) {
            out << " kernel_id=" << command.kernel.specialization.kernel_id
                << " bindings=" << command.kernel.bindings.size();
        }
        return out.str();
    }

    static std::string list_detail(const char * list_kind, size_t list_size, size_t program_command_count) {
        std::ostringstream out;
        out << "list=" << list_kind
            << " list_commands=" << list_size
            << " main_commands=" << program_command_count;
        return out.str();
    }

    static std::string host_staging_detail(const HostStagingBuffer & staging) {
        std::ostringstream out;
        out << "value=" << staging.value << " length=" << staging.length;
        return out.str();
    }

  private:
    const CommandProgramExecutionContext & context_;
    bool                                   enabled_ = false;
    std::string                            program_detail_;
};

#define HRX_DEBUG_SERIAL_SYNC(trace, phase) \
    do {                                    \
        if (!(trace).sync_program_phase(phase)) { \
            return false;                   \
        }                                   \
    } while (0)

static bool resource_access_writes(ResourceAccess access) {
    return access == ResourceAccess::Write || access == ResourceAccess::ReadWrite;
}

static bool resource_access_reads(ResourceAccess access) {
    return access == ResourceAccess::Read || access == ResourceAccess::ReadWrite;
}

static bool dispatch_trace_enabled() {
    return environment_string_value("GGML_HRX_TRACE_DISPATCH_DIR") != nullptr;
}

static bool dispatch_trace_blob_capture_enabled() {
    return environment_flag_enabled("GGML_HRX_TRACE_BLOBS");
}

static bool dispatch_trace_should_trace_program(size_t program_command_count) {
    const char * value = environment_string_value("GGML_HRX_TRACE_PROGRAM_COMMANDS");
    return value == nullptr || program_command_count == std::strtoull(value, nullptr, 0);
}

static bool dispatch_trace_should_trace_command(size_t program_command_count, size_t command_index) {
    if (!dispatch_trace_should_trace_program(program_command_count)) {
        return false;
    }
    const size_t start = environment_size_value("GGML_HRX_TRACE_START_COMMAND", 0);
    const size_t end =
        environment_size_value("GGML_HRX_TRACE_END_COMMAND", std::numeric_limits<size_t>::max());
    return command_index >= start && command_index <= end;
}

static uint64_t dispatch_trace_hash_bytes(const uint8_t * data, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static std::string dispatch_trace_hex(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

static std::string dispatch_trace_json_escape(const std::string & text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const unsigned char c : text) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped.push_back(kHex[c >> 4]);
                    escaped.push_back(kHex[c & 0xf]);
                } else {
                    escaped.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return escaped;
}

static std::string dispatch_trace_kernel_name(const CommandProgramExecutionContext & context,
                                              const PreparedCommand &                command) {
    if (context.corpus != nullptr && context.target != nullptr) {
        const KernelResolveResult resolved =
            resolve_kernel_definition(*context.corpus, context.target, command.kernel.specialization.kernel_id);
        if (resolved.found()) {
            return kernel_definition_name(*resolved.definition);
        }
    }
    std::ostringstream out;
    out << "kernel_id=" << command.kernel.specialization.kernel_id;
    return out.str();
}

static Status dispatch_trace_download_binding(const CommandProgramExecutionContext & context,
                                              const PreparedCommandBinding &         binding,
                                              std::vector<uint8_t> &                 data) {
    Status status;
    data.resize(binding.ref.length);
    if (binding.ref.length == 0) {
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager for dispatch trace");
        return status;
    }
    status.append(context.host_transfers->download_synchronous(context.stream, binding.ref.buffer, binding.ref.offset,
                                                               data.data(), binding.ref.length));
    return status;
}

static Status dispatch_trace_write_blob(const std::filesystem::path & path, const std::vector<uint8_t> & data) {
    Status status;
    try {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            status.log("failed to create dispatch trace blob %s", path.string().c_str());
            return status;
        }
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    } catch (const std::exception & error) {
        status.log("failed to write dispatch trace blob %s: %s", path.string().c_str(), error.what());
    }
    return status;
}

static Status dispatch_trace_command_bindings(const CommandProgramExecutionContext & context,
                                              const PreparedCommand &                command,
                                              const char *                           list_kind,
                                              size_t                                 list_size,
                                              size_t                                 command_index,
                                              size_t                                 program_command_count,
                                              const char *                           stage) {
    Status status;
    const char * directory_value = environment_string_value("GGML_HRX_TRACE_DISPATCH_DIR");
    if (directory_value == nullptr || !dispatch_trace_should_trace_command(program_command_count, command_index)) {
        return status;
    }
    const bool before_stage = std::strcmp(stage, "before") == 0;
    const bool after_stage  = std::strcmp(stage, "after") == 0;
    if (!before_stage && !after_stage) {
        return status;
    }
    const bool capture_blobs = dispatch_trace_blob_capture_enabled();
    if (before_stage && !capture_blobs) {
        return status;
    }
    const std::filesystem::path root(directory_value);
    const std::filesystem::path blobs = root / "blobs";
    const std::filesystem::path trace_path = root / "trace.jsonl";
    const std::string kernel_name = dispatch_trace_kernel_name(context, command);
    for (size_t binding_index = 0; binding_index < command.kernel.bindings.size(); ++binding_index) {
        const PreparedCommandBinding & binding = command.kernel.bindings[binding_index];
        const bool should_capture = before_stage ? resource_access_reads(binding.binding.access) :
                                                   resource_access_writes(binding.binding.access);
        if (!should_capture) {
            continue;
        }
        std::vector<uint8_t> data;
        Status download_status = dispatch_trace_download_binding(context, binding, data);
        if (!download_status.success()) {
            status.append(download_status);
            continue;
        }
        const uint64_t hash = dispatch_trace_hash_bytes(data.data(), data.size());
        std::string blob_relative;
        if (capture_blobs) {
            std::ostringstream blob_name;
            blob_name << stage << "-cmd" << command_index << "-ord" << command.ordinal << "-bind" << binding_index
                      << "-value" << binding.binding.value.value << "-off" << binding.binding.offset << "-"
                      << dispatch_trace_hex(hash) << ".bin";
            const std::filesystem::path blob_path = blobs / blob_name.str();
            status.append(dispatch_trace_write_blob(blob_path, data));
            blob_relative = std::filesystem::relative(blob_path, root).string();
        }
        try {
            std::filesystem::create_directories(root);
            std::ofstream out(trace_path, std::ios::app);
            if (!out) {
                status.log("failed to open dispatch trace %s", trace_path.string().c_str());
                continue;
            }
            out << "{"
                << "\"stage\":\"" << stage << "\","
                << "\"list_kind\":\"" << list_kind << "\","
                << "\"list_size\":" << list_size << ','
                << "\"program_command_count\":" << program_command_count << ','
                << "\"command_index\":" << command_index << ','
                << "\"ordinal\":" << command.ordinal << ','
                << "\"kernel_id\":" << command.kernel.specialization.kernel_id << ','
                << "\"kernel\":\"" << dispatch_trace_json_escape(kernel_name) << "\","
                << "\"binding_index\":" << binding_index << ','
                << "\"binding_name\":\"" << dispatch_trace_json_escape(binding.binding.name) << "\","
                << "\"value\":" << binding.binding.value.value << ','
                << "\"origin\":\"" << command_binding_origin_name(binding.binding.origin) << "\","
                << "\"access\":\"" << resource_access_name(binding.binding.access) << "\","
                << "\"offset\":" << binding.binding.offset << ','
                << "\"length\":" << binding.binding.length << ','
                << "\"ref_offset\":" << binding.ref.offset << ','
                << "\"ref_length\":" << binding.ref.length << ','
                << "\"hash\":\"" << dispatch_trace_hex(hash) << "\"";
            if (!blob_relative.empty()) {
                out << ",\"blob\":\"" << dispatch_trace_json_escape(blob_relative) << "\"";
            }
            out << "}\n";
        } catch (const std::exception & error) {
            status.log("failed to write dispatch trace %s: %s", trace_path.string().c_str(), error.what());
        }
    }
    return status;
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
        DispatchBinding dispatch_binding;
        dispatch_binding.value         = binding.binding.value;
        dispatch_binding.offset        = binding.binding.offset;
        dispatch_binding.length        = binding.binding.length;
        dispatch_binding.layout        = binding.binding.layout;
        dispatch_binding.source_type   = binding.binding.source_type;
        dispatch_binding.input_size    = binding.binding.input_size;
        dispatch_binding.output_size   = binding.binding.output_size;
        dispatch_binding.source_length = binding.binding.source_length;
        dispatch.bindings.push_back(std::move(dispatch_binding));
    }
    return dispatch;
}

struct GraphValueAccess {
    bool read  = false;
    bool write = false;
    bool        native_read         = false;
    bool        transformed_read    = false;
    bool        layout_conflict     = false;
    std::string layout              = kNativeWeightLayout;
    ggml_type   source_type         = GGML_TYPE_COUNT;
    int64_t     input_size          = 0;
    int64_t     output_size         = 0;
    size_t      source_length       = 0;
    size_t      materialized_length = 0;
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
                if (binding.access == ResourceAccess::Write) {
                    continue;
                }
                if (binding.layout == kNativeWeightLayout) {
                    access.native_read = true;
                    if (access.transformed_read) {
                        access.layout_conflict = true;
                    }
                    continue;
                }
                if (binding.offset != 0 || binding.source_length == 0 || binding.length == 0 ||
                    binding.source_type == GGML_TYPE_COUNT || binding.input_size <= 0 || binding.output_size <= 0) {
                    access.layout_conflict = true;
                    continue;
                }
                if (!access.transformed_read) {
                    access.transformed_read    = true;
                    access.layout              = binding.layout;
                    access.source_type         = binding.source_type;
                    access.input_size          = binding.input_size;
                    access.output_size         = binding.output_size;
                    access.source_length       = binding.source_length;
                    access.materialized_length = binding.length;
                } else if (access.layout != binding.layout || access.source_type != binding.source_type ||
                           access.input_size != binding.input_size || access.output_size != binding.output_size ||
                           access.source_length != binding.source_length ||
                           access.materialized_length != binding.length) {
                    access.layout_conflict = true;
                }
                if (access.native_read) {
                    access.layout_conflict = true;
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
        const auto             found_access = access_by_value.find(binding.value.value);
        const GraphValueAccess access =
            found_access != access_by_value.end() ? found_access->second : GraphValueAccess{};
        const bool materialize_weight = binding.weight && access.read && !access.write &&
                                        (binding.requires_materialization() || access.transformed_read);
        if (binding.length == 0 || (!binding.requires_materialization() && !materialize_weight)) {
            materialized.push_back(binding);
            continue;
        }
        if (materialize_weight) {
            if (access.layout_conflict) {
                status.log("weight value %d has conflicting resident layout requests", binding.value.value);
                materialized.push_back(binding);
                continue;
            }
            HostWeightSource source;
            source.host_data  = binding.host_data;
            source.device_buffer       = binding.host_data == nullptr ? binding.buffer : nullptr;
            source.identity   = binding.identity;
            source.generation = binding.generation;
            source.capacity   = binding.capacity;
            source.offset     = binding.offset;
            source.length              = access.transformed_read ? access.source_length : binding.length;
            source.materialized_length = access.transformed_read ? access.materialized_length : binding.length;
            source.layout              = access.layout;
            source.source_type         = access.source_type;
            source.input_size          = access.input_size;
            source.output_size         = access.output_size;
            if (source.length > binding.length) {
                status.log("weight value %d layout source length %zu exceeds runtime length %zu", binding.value.value,
                           source.length, binding.length);
                materialized.push_back(binding);
                continue;
            }
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
            device_binding.length                = resident.lease.length();
            device_binding.capacity              = resident.lease.length();
            materialized.push_back(device_binding);
            prepared.resident_host_weights.push_back(std::move(resident.lease));
            continue;
        }

        if (binding.host_data == nullptr) {
            materialized.push_back(binding);
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
        if (staging.upload && context.host_buffers != nullptr) {
            staging.source_host_buffer = context.host_buffers->find(staging.host_data, staging.length);
        }
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

        std::copy(initialization.data.begin(), initialization.data.end(), image->data.begin() + initialization.offset);
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
                    status.log(
                        "constant initialization %s is written by command %u; prepared constant buffer copy "
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

static Status bind_prepared_command_list_program_constants(const PreparedCommandProgram & prepared,
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
        if (ErrorResult error = take_status(hrx_allocator_allocate_buffer(hrx_device_allocator(context.device), params,
                                                                          image.data.size(), &buffer))) {
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

static Status rebind_prepared_host_staging(const CommandProgramExecutionContext & context,
                                           const CommandProgramBindings &         bindings,
                                           PreparedCommandProgram &               prepared) {
    Status status;
    for (HostStagingBuffer & staging : prepared.host_staging) {
        const CommandProgramBinding * binding = bindings.find(ValueId(staging.value));
        if (binding == nullptr || binding->host_data == nullptr || binding->length != staging.length ||
            binding->offset > binding->capacity || binding->length > binding->capacity - binding->offset) {
            status.log("live host binding does not match prepared value %d", staging.value);
            continue;
        }
        staging.host_data = static_cast<uint8_t *>(binding->host_data) + binding->offset;
        staging.source_host_buffer = HostBufferRef{};
        if (staging.upload && context.host_buffers != nullptr) {
            staging.source_host_buffer = context.host_buffers->find(staging.host_data, staging.length);
        }
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
    const DebugSerialExecutionTrace debug(context);
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        if (!staging.upload) {
            continue;
        }
        const std::string detail = debug.enabled() ? DebugSerialExecutionTrace::host_staging_detail(staging) :
                                                     std::string();
        if (!debug.sync("host-staging-upload-buffer-pre", detail)) {
            status.log("debug serial sync before host-staging upload failed for value %d", staging.value);
            return status;
        }
        Status upload_status;
        if (staging.source_host_buffer.valid()) {
            if (ErrorResult error = take_status(hrx_stream_copy_buffer(context.stream, staging.source_host_buffer.buffer(),
                                                                       staging.source_host_buffer.offset(),
                                                                       staging.buffer, 0, staging.length))) {
                upload_status.log("HRX host staging buffer copy failed: %s", error->c_str());
            }
        } else {
            upload_status =
                context.host_transfers->upload_async(context.stream, staging.host_data, staging.buffer, 0, staging.length);
        }
        status.append(upload_status);
        if (!debug.sync("host-staging-upload-buffer-post", detail)) {
            status.log("debug serial sync after host-staging upload failed for value %d", staging.value);
            return status;
        }
    }
    return status;
}

static void mark_graph_replay_synchronized(const CommandProgramExecutionContext & context) {
    if (context.graph_replay_state != nullptr) {
        context.graph_replay_state->mark_stream_synchronized();
    }
}

static Status enqueue_stream_execution_barrier(const CommandProgramExecutionContext & context, const char * label) {
    Status status;
    if (ErrorResult error = take_status(hrx_stream_execution_barrier(context.stream))) {
        status.log("%s failed: %s", label, error->c_str());
    }
    return status;
}

static Status flush_stream_commands(const CommandProgramExecutionContext & context, const char * label) {
    Status status;
    if (ErrorResult error = take_status(hrx_stream_flush(context.stream))) {
        status.log("%s failed: %s", label, error->c_str());
    }
    return status;
}

static Status wait_stream_commands(const CommandProgramExecutionContext & context, const char * label) {
    Status status;
    if (ErrorResult error = take_status(hrx_stream_wait(context.stream))) {
        status.log("%s failed: %s", label, error->c_str());
    }
    return status;
}

static Status download_prepared_host_staging(const CommandProgramExecutionContext & context,
                                             const PreparedCommandProgram &         prepared,
                                             bool                                   insert_barrier = true) {
    Status status;

    bool has_download = false;
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        has_download = has_download || staging.download;
    }
    if (!has_download) {
        return status;
    }
    if (context.graph_replay_state == nullptr) {
        status.log("missing HRX stream completion state");
        return status;
    }
    if (insert_barrier) {
        Status barrier_status = enqueue_stream_execution_barrier(context, "insert HRX host download barrier");
        if (!barrier_status.success()) {
            return barrier_status;
        }
    }
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        if (!staging.download) {
            continue;
        }
        hrx_buffer_t download_buffer = nullptr;
        void *       download_data   = nullptr;
        Status allocation_status =
            allocate_mapped_host_staging_buffer(context.device, staging.length, download_buffer, download_data);
        if (!allocation_status.success()) {
            status.log("allocate HRX host download staging buffer for value %d failed", staging.value);
            status.append(allocation_status);
            return status;
        }
        if (ErrorResult error = take_status(hrx_stream_copy_buffer(context.stream, staging.buffer, 0,
                                                                   download_buffer, 0, staging.length))) {
            status.log("HRX host download staging copy failed for value %d: %s", staging.value, error->c_str());
            hrx_buffer_release(download_buffer);
            return status;
        }
        context.graph_replay_state->add_host_writeback(staging.host_data, download_data, staging.length, download_buffer);
        hrx_buffer_release(download_buffer);
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
    executable_ref =
        context.kernel_executables->get_or_compile({ context.device, context.target }, *resolved.definition, dispatch);
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
        { command.kernel.launch.workgroup_count[0], command.kernel.launch.workgroup_count[1],
         command.kernel.launch.workgroup_count[2] },
        { command.kernel.launch.workgroup_size[0], command.kernel.launch.workgroup_size[1],
         command.kernel.launch.workgroup_size[2] },
        command.kernel.launch.subgroup_size,
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
            { context.device, context.target }, executable_refs[i], command.kernel.specialization,
            command.kernel.constants, command.kernel.launch);
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

static bool execute_prepared_command_list_serial(const CommandProgramExecutionContext & context,
                                                 const std::vector<PreparedCommand> &   commands,
                                                 const char *                           list_kind,
                                                 size_t                                 program_command_count) {
    const DebugSerialExecutionTrace debug(context);
    const bool trace_list = debug.enabled() && debug.should_trace_program(program_command_count);
    if (trace_list) {
        debug.log("begin", "command-list",
                  DebugSerialExecutionTrace::list_detail(list_kind, commands.size(), program_command_count));
    }
    for (size_t i = 0; i < commands.size(); ++i) {
        const PreparedCommand & command         = commands[i];
        const bool        trace_command  = debug.enabled() && debug.should_trace_command(program_command_count, i);
        const std::string command_detail = trace_command ?
                                               DebugSerialExecutionTrace::command_detail(list_kind, commands.size(), i,
                                                                                        command) :
                                               std::string();
        if (!debug.sync("command-pre-dispatch", command_detail)) {
            return false;
        }
        Status trace_status =
            dispatch_trace_command_bindings(context, command, list_kind, commands.size(), i, program_command_count,
                                            "before");
        if (!trace_status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(trace_status));
            return false;
        }
        if (trace_command) {
            debug.log("begin", "command-dispatch", command_detail);
        }
        if (!execute_prepared_kernel_command(context, command)) {
            return false;
        }
        if (trace_command) {
            debug.log("end", "command-dispatch", command_detail);
        }
        if (!debug.sync("command-post-dispatch", command_detail)) {
            return false;
        }
        trace_status =
            dispatch_trace_command_bindings(context, command, list_kind, commands.size(), i, program_command_count,
                                            "after");
        if (!trace_status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(trace_status));
            return false;
        }
    }
    if (trace_list) {
        debug.log("end", "command-list",
                  DebugSerialExecutionTrace::list_detail(list_kind, commands.size(), program_command_count));
    }
    return true;
}

static bool execute_prepared_command_list(const CommandProgramExecutionContext & context,
                                          const std::vector<PreparedCommand> &   commands) {
    return execute_prepared_command_list_serial(context, commands, "main", commands.size());
}

struct GraphResourceState {
    size_t                        end         = 0;
    hrx_graph_node_t              last_writer = nullptr;
    std::vector<hrx_graph_node_t> readers;
};

class GraphDependencyPlanner {
  public:
    std::vector<hrx_graph_node_t> dependencies(const std::vector<PreparedCommandBinding> & bindings) const {
        std::vector<hrx_graph_node_t> result;
        for (const PreparedCommandBinding & binding : bindings) {
            collect_dependencies(binding.ref.buffer, binding.ref.offset, binding.ref.length,
                                 resource_access_writes(binding.binding.access), result);
        }
        return result;
    }

    void record(hrx_graph_node_t node, const std::vector<PreparedCommandBinding> & bindings) {
        for (const PreparedCommandBinding & binding : bindings) {
            update(node, binding.ref.buffer, binding.ref.offset, binding.ref.length,
                   resource_access_writes(binding.binding.access));
        }
    }

    void record(hrx_graph_node_t node, hrx_buffer_t buffer, size_t offset, size_t length, bool writes) {
        update(node, buffer, offset, length, writes);
    }

  private:
    using ResourceMap = std::map<size_t, GraphResourceState>;

    static size_t range_end(size_t offset, size_t length) {
        return length > std::numeric_limits<size_t>::max() - offset ? std::numeric_limits<size_t>::max() :
                                                                      offset + length;
    }

    static void append_unique(std::vector<hrx_graph_node_t> & nodes, hrx_graph_node_t node) {
        if (node != nullptr && std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
            nodes.push_back(node);
        }
    }

    static void split(ResourceMap & ranges, size_t offset) {
        auto upper = ranges.upper_bound(offset);
        if (upper == ranges.begin()) {
            return;
        }
        auto current = std::prev(upper);
        if (offset <= current->first || offset >= current->second.end) {
            return;
        }
        GraphResourceState right = current->second;
        current->second.end      = offset;
        ranges.emplace(offset, std::move(right));
    }

    void collect_dependencies(hrx_buffer_t                    buffer,
                              size_t                          offset,
                              size_t                          length,
                              bool                            writes,
                              std::vector<hrx_graph_node_t> & result) const {
        if (buffer == nullptr || length == 0) {
            return;
        }
        const auto resource = resources_.find(buffer);
        if (resource == resources_.end()) {
            return;
        }
        const size_t end   = range_end(offset, length);
        auto         range = resource->second.upper_bound(offset);
        if (range != resource->second.begin()) {
            --range;
            if (range->second.end <= offset) {
                ++range;
            }
        }
        for (; range != resource->second.end() && range->first < end; ++range) {
            append_unique(result, range->second.last_writer);
            if (writes) {
                for (hrx_graph_node_t reader : range->second.readers) {
                    append_unique(result, reader);
                }
            }
        }
    }

    void update(hrx_graph_node_t node, hrx_buffer_t buffer, size_t offset, size_t length, bool writes) {
        if (buffer == nullptr || length == 0) {
            return;
        }
        const size_t  end    = range_end(offset, length);
        ResourceMap & ranges = resources_[buffer];
        split(ranges, offset);
        split(ranges, end);

        size_t cursor = offset;
        auto   range  = ranges.lower_bound(offset);
        while (cursor < end) {
            if (range == ranges.end() || range->first > cursor) {
                const size_t       gap_end = range == ranges.end() ? end : std::min(end, range->first);
                GraphResourceState state;
                state.end = gap_end;
                if (writes) {
                    state.last_writer = node;
                } else {
                    state.readers.push_back(node);
                }
                ranges.emplace(cursor, std::move(state));
                cursor = gap_end;
                continue;
            }

            if (writes) {
                range->second.last_writer = node;
                range->second.readers.clear();
            } else {
                append_unique(range->second.readers, node);
            }
            cursor = range->second.end;
            ++range;
        }
    }

    std::unordered_map<hrx_buffer_t, ResourceMap> resources_;
};

static Status record_completion_counter_fill(hrx_graph_t                         graph,
                                             GraphDependencyPlanner &            dependencies,
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
    if (ErrorResult error = take_status(hrx_graph_add_fill_buffer_node(graph, nullptr, 0, &attrs, &node))) {
        status.log("record completion counter fill: %s", error->c_str());
        return status;
    }
    dependencies.record(node, allocation.buffer, commands.completion_counters.arena_offset,
                        commands.completion_counters.byte_count, true);
    return status;
}

static Status record_prepared_kernel_command(hrx_graph_t                  graph,
                                             GraphDependencyPlanner & dependencies,
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
            { command.kernel.launch.workgroup_count[0], command.kernel.launch.workgroup_count[1],
              command.kernel.launch.workgroup_count[2] },
            { command.kernel.launch.workgroup_size[0], command.kernel.launch.workgroup_size[1],
              command.kernel.launch.workgroup_size[2] },
            command.kernel.launch.subgroup_size,
        },
        command.kernel.constants.data(),
        command.kernel.constants.size(),
        refs.data(),
        refs.size(),
        0,
    };
    const std::vector<hrx_graph_node_t> dependency_nodes = dependencies.dependencies(command.kernel.bindings);
    hrx_graph_node_t node = nullptr;
    if (ErrorResult error = take_status(
            hrx_graph_add_kernel_node(graph, dependency_nodes.data(), dependency_nodes.size(), &attrs, &node))) {
        status.log("record %s: %s", command_context.c_str(), error->c_str());
        return status;
    }
    dependencies.record(node, command.kernel.bindings);
    return status;
}

static Status record_prepared_command_list(hrx_graph_t                        graph,
                                           GraphDependencyPlanner &             dependencies,
                                           const std::vector<PreparedCommand> & commands,
                                           size_t &                           dispatch_count) {
    Status status;
    for (const PreparedCommand & command : commands) {
        Status command_status = record_prepared_kernel_command(graph, dependencies, command);
        if (!command_status.success()) {
            status.append(command_status);
            return status;
        }
        ++dispatch_count;
    }
    return status;
}

static void retain_prepared_command_list_resources(const std::vector<PreparedCommand> &         commands,
                                                   std::unordered_map<hrx_buffer_t, bool> &     retained_buffers,
                                                   std::unordered_map<hrx_executable_t, bool> & retained_executables,
                                                   RecordedCommandGraph &                       recorded) {
    for (const PreparedCommand & command : commands) {
        if (command.kernel.executable != nullptr && command.kernel.executable->executable != nullptr &&
            retained_executables.emplace(command.kernel.executable->executable, true).second) {
            hrx_executable_retain(command.kernel.executable->executable);
            recorded.retained_executables.push_back(command.kernel.executable->executable);
        }
        for (const PreparedCommandBinding & binding : command.kernel.bindings) {
            if (binding.ref.buffer != nullptr && retained_buffers.emplace(binding.ref.buffer, true).second) {
                hrx_buffer_retain(binding.ref.buffer);
                recorded.retained_buffers.push_back(binding.ref.buffer);
            }
        }
    }
}

static void retain_prepared_command_graph_resources(const PreparedCommandProgram &      prepared,
                                                    const TransientArenaAllocationRef & allocation,
                                                    RecordedCommandGraph &              recorded) {
    std::unordered_map<hrx_buffer_t, bool>     retained_buffers;
    std::unordered_map<hrx_executable_t, bool> retained_executables;
    if (allocation.buffer != nullptr) {
        hrx_buffer_retain(allocation.buffer);
        recorded.retained_buffers.push_back(allocation.buffer);
        retained_buffers.emplace(allocation.buffer, true);
    }
    retain_prepared_command_list_resources(prepared.initialization_commands, retained_buffers, retained_executables,
                                           recorded);
    retain_prepared_command_list_resources(prepared.commands, retained_buffers, retained_executables, recorded);
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

    GraphDependencyPlanner dependencies;
    recorded.status.append(record_completion_counter_fill(recorded.graph, dependencies, commands, allocation));
    if (!recorded.status.success()) {
        return recorded;
    }
    recorded.status.append(record_prepared_command_list(recorded.graph, dependencies, prepared.initialization_commands,
                                                        recorded.dispatch_count));
    if (!recorded.status.success()) {
        return recorded;
    }
    recorded.status.append(
        record_prepared_command_list(recorded.graph, dependencies, prepared.commands, recorded.dispatch_count));
    if (!recorded.status.success()) {
        return recorded;
    }

    // Unretained HRX command buffers require graph-lifetime resources.
    retain_prepared_command_graph_resources(prepared, allocation, recorded);

    hrx_graph_exec_t exec = nullptr;
    if (ErrorResult error = take_status(hrx_graph_instantiate(recorded.graph, 0, &exec))) {
        recorded.status.log("instantiate HRX graph replay: %s", error->c_str());
        return recorded;
    }
    recorded.exec = exec;
    recorded.bound_transient_arena_allocation_id = prepared.bound_transient_arena_allocation_id;
    return recorded;
}

static bool execute_prepared_kernel_command_via_graph(const CommandProgramExecutionContext & context,
                                                      const PreparedCommand &                command) {
    hrx_graph_t graph = nullptr;
    if (take_status(hrx_graph_create(context.device, 0, &graph))) {
        return false;
    }
    GraphDependencyPlanner dependencies;
    Status                 status = record_prepared_kernel_command(graph, dependencies, command);
    hrx_graph_exec_t       exec   = nullptr;
    if (status.success()) {
        if (ErrorResult error = take_status(hrx_graph_instantiate(graph, 0, &exec))) {
            status.log("instantiate diagnostic graph: %s", error->c_str());
        }
    }
    if (status.success()) {
        if (ErrorResult error = take_status(hrx_graph_exec_launch(exec, context.stream))) {
            status.log("launch diagnostic graph: %s", error->c_str());
        }
    }
    if (status.success()) {
        if (ErrorResult error = take_status(hrx_stream_synchronize(context.stream))) {
            status.log("synchronize diagnostic graph: %s", error->c_str());
        }
    }
    if (exec != nullptr) {
        hrx_graph_exec_release(exec);
    }
    hrx_graph_release(graph);
    return status.success();
}

static bool execute_prepared_command_prefix_via_graph(const CommandProgramExecutionContext & context,
                                                      const std::vector<PreparedCommand> &   commands,
                                                      size_t                                 prefix_count) {
    if (prefix_count == 0 || prefix_count > commands.size()) {
        return prefix_count == 0;
    }

    hrx_graph_t graph = nullptr;
    if (take_status(hrx_graph_create(context.device, 0, &graph))) {
        return false;
    }
    GraphDependencyPlanner dependencies;
    Status                 status;
    for (size_t i = 0; i < prefix_count && status.success(); ++i) {
        status.append(record_prepared_kernel_command(graph, dependencies, commands[i]));
    }

    hrx_graph_exec_t exec = nullptr;
    if (status.success()) {
        if (ErrorResult error = take_status(hrx_graph_instantiate(graph, 0, &exec))) {
            status.log("instantiate diagnostic prefix graph: %s", error->c_str());
        }
    }
    if (status.success()) {
        if (ErrorResult error = take_status(hrx_graph_exec_launch(exec, context.stream))) {
            status.log("launch diagnostic prefix graph: %s", error->c_str());
        }
    }
    if (status.success()) {
        if (ErrorResult error = take_status(hrx_stream_synchronize(context.stream))) {
            status.log("synchronize diagnostic prefix graph: %s", error->c_str());
        }
    }
    if (exec != nullptr) {
        hrx_graph_exec_release(exec);
    }
    hrx_graph_release(graph);
    return status.success();
}

}  // namespace

bool debug_serial_command_execution_enabled() {
    return environment_flag_enabled("GGML_HRX_DEBUG_SERIAL_EXECUTION") || dispatch_trace_enabled();
}

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
    const DebugSerialExecutionTrace debug(context, commands);
    debug.log_program_begin();
    HRX_DEBUG_SERIAL_SYNC(debug, "host-staging-rebind-pre");
    Status rebind_status = rebind_prepared_host_staging(context, bindings, prepared);
    if (!rebind_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(rebind_status));
        return false;
    }
    HRX_DEBUG_SERIAL_SYNC(debug, "host-staging-rebind-post");
    if (commands.transients.arena_size == 0) {
        HRX_DEBUG_SERIAL_SYNC(debug, "constant-initialization-pre");
        Status status = initialize_command_program_constants(context, commands, {}, prepared);
        if (!status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
            return false;
        }
        HRX_DEBUG_SERIAL_SYNC(debug, "constant-initialization-post");
        HRX_DEBUG_SERIAL_SYNC(debug, "completion-counter-initialization-pre");
        status = initialize_command_program_completion_counters(context, commands, {});
        if (!status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
            return false;
        }
        HRX_DEBUG_SERIAL_SYNC(debug, "completion-counter-initialization-post");
        HRX_DEBUG_SERIAL_SYNC(debug, "transient-bind-pre");
        if (!bind_prepared_command_program_transients(commands, {}, prepared)) {
            return false;
        }
        HRX_DEBUG_SERIAL_SYNC(debug, "transient-bind-post");
        const bool success = execute_prepared_command_program(context, prepared);
        debug.log_program_end(success);
        return success;
    }

    Status status = command_program_transient_context_valid(context, commands);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }

    HRX_DEBUG_SERIAL_SYNC(debug, "transient-arena-acquire-pre");
    TransientArena::AllocationLease lease = context.transient_arena->acquire_allocation_lease();
    status = lease.ensure_capacity(context.device, context.stream, commands.transients.arena_size);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    HRX_DEBUG_SERIAL_SYNC(debug, "transient-arena-acquire-post");
    HRX_DEBUG_SERIAL_SYNC(debug, "transient-bind-pre");
    if (!bind_prepared_command_program_transients(commands, lease.current_allocation(), prepared)) {
        return false;
    }
    HRX_DEBUG_SERIAL_SYNC(debug, "transient-bind-post");
    HRX_DEBUG_SERIAL_SYNC(debug, "constant-initialization-pre");
    status = initialize_command_program_constants(context, commands, lease.current_allocation(), prepared);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    HRX_DEBUG_SERIAL_SYNC(debug, "constant-initialization-post");
    HRX_DEBUG_SERIAL_SYNC(debug, "completion-counter-initialization-pre");
    status = initialize_command_program_completion_counters(context, commands, lease.current_allocation());
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    HRX_DEBUG_SERIAL_SYNC(debug, "completion-counter-initialization-post");
    const bool success = execute_prepared_command_program(context, prepared);
    debug.log_program_end(success);
    return success;
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

    Status rebind_status = rebind_prepared_host_staging(context, bindings, prepared);
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

    const char * diagnostic_kernel = std::getenv("GGML_HRX_DIAGNOSTIC_GRAPH_KERNEL_ID");
    if (diagnostic_kernel != nullptr && diagnostic_kernel[0] != '\0') {
        const uint64_t diagnostic_graph_kernel_id = std::strtoull(diagnostic_kernel, nullptr, 0);
        Status         diagnostic_upload_status   = upload_prepared_host_staging(context, prepared);
        if (!diagnostic_upload_status.success()) {
            result.status.append(diagnostic_upload_status);
            result.event = HrxGraphReplayEvent::LaunchFailed;
            return result;
        }
        auto execute_diagnostic_list = [&](const std::vector<PreparedCommand> & command_list) {
            for (const PreparedCommand & command : command_list) {
                const bool ok = (diagnostic_graph_kernel_id == 0 ||
                                 command.kernel.specialization.kernel_id == diagnostic_graph_kernel_id) ?
                                    execute_prepared_kernel_command_via_graph(context, command) :
                                    execute_prepared_kernel_command(context, command);
                if (!ok) {
                    return false;
                }
            }
            return true;
        };
        if (!execute_diagnostic_list(prepared.initialization_commands) || !execute_diagnostic_list(prepared.commands)) {
            result.status.log("diagnostic mixed graph execution failed");
            result.event = HrxGraphReplayEvent::LaunchFailed;
            return result;
        }
        result.success = true;
        return result;
    }

    const char * diagnostic_program_size = std::getenv("GGML_HRX_DIAGNOSTIC_GRAPH_PROGRAM_COMMANDS");
    const char * diagnostic_prefix       = std::getenv("GGML_HRX_DIAGNOSTIC_GRAPH_PREFIX_COMMANDS");
    if (diagnostic_program_size != nullptr && diagnostic_program_size[0] != '\0' && diagnostic_prefix != nullptr &&
        diagnostic_prefix[0] != '\0') {
        const size_t target_size   = std::strtoull(diagnostic_program_size, nullptr, 0);
        const size_t prefix_size   = std::strtoull(diagnostic_prefix, nullptr, 0);
        Status       upload_status = upload_prepared_host_staging(context, prepared);
        if (!upload_status.success()) {
            result.status.append(upload_status);
            result.event = HrxGraphReplayEvent::LaunchFailed;
            return result;
        }

        bool ok = execute_prepared_command_list(context, prepared.initialization_commands);
        if (ok && prepared.commands.size() == target_size) {
            ok = execute_prepared_command_prefix_via_graph(context, prepared.commands, prefix_size);
            for (size_t i = prefix_size; ok && i < prepared.commands.size(); ++i) {
                ok = execute_prepared_kernel_command(context, prepared.commands[i]);
            }
        } else if (ok) {
            ok = execute_prepared_command_list(context, prepared.commands);
        }
        if (!ok) {
            result.status.log("diagnostic prefix graph execution failed");
            result.event = HrxGraphReplayEvent::LaunchFailed;
            return result;
        }
        Status download_status = download_prepared_host_staging(context, prepared);
        if (!download_status.success()) {
            result.status.append(download_status);
            result.event = HrxGraphReplayEvent::LaunchFailed;
            return result;
        }
        result.success = true;
        return result;
    }

    result.transient_allocation_changed =
        had_recorded && recorded.bound_transient_arena_allocation_id != prepared.bound_transient_arena_allocation_id;
    if (!had_recorded || result.transient_allocation_changed) {
        result.event = result.transient_allocation_changed ? HrxGraphReplayEvent::RebuildTransient :
                                                             HrxGraphReplayEvent::MissBuild;
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
    Status replay_flush_status = flush_stream_commands(context, "flush HRX graph replay commands");
    if (!replay_flush_status.success()) {
        result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
        result.status.append(replay_flush_status);
        result.event = HrxGraphReplayEvent::LaunchFailed;
        return result;
    }
    Status replay_wait_status = wait_stream_commands(context, "wait for HRX graph replay commands");
    if (!replay_wait_status.success()) {
        result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
        result.status.append(replay_wait_status);
        result.event = HrxGraphReplayEvent::LaunchFailed;
        return result;
    }
    const char * diagnostic_sync_size = std::getenv("GGML_HRX_DIAGNOSTIC_GRAPH_SYNC_PROGRAM_COMMANDS");
    const bool   diagnostic_sync      = std::getenv("GGML_HRX_DIAGNOSTIC_GRAPH_SYNC") != nullptr ||
                                 (diagnostic_sync_size != nullptr && diagnostic_sync_size[0] != '\0' &&
                                  recorded.dispatch_count == std::strtoull(diagnostic_sync_size, nullptr, 0));
    if (diagnostic_sync) {
        if (ErrorResult error = take_status(hrx_stream_synchronize(context.stream))) {
            result.launch_ns = hrx_graph_replay_now_ns() - launch_start_ns;
            result.status.log("synchronize diagnostic graph replay: %s", error->c_str());
            result.event = HrxGraphReplayEvent::LaunchFailed;
            return result;
        }
        mark_graph_replay_synchronized(context);
    }
    Status download_status = download_prepared_host_staging(context, prepared, false);
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
    const DebugSerialExecutionTrace debug(context, commands);
    HRX_DEBUG_SERIAL_SYNC(debug, "host-staging-upload-pre");
    Status upload_status = upload_prepared_host_staging(context, commands);
    if (!upload_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(upload_status));
        return false;
    }
    HRX_DEBUG_SERIAL_SYNC(debug, "host-staging-upload-post");
    if (!execute_prepared_command_list_serial(context, commands.initialization_commands, "init", commands.commands.size()) ||
        !execute_prepared_command_list_serial(context, commands.commands, "main", commands.commands.size())) {
        return false;
    }
    HRX_DEBUG_SERIAL_SYNC(debug, "host-staging-download-pre");
    Status download_status = download_prepared_host_staging(context, commands);
    if (!download_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(download_status));
        return false;
    }
    HRX_DEBUG_SERIAL_SYNC(debug, "host-staging-download-post");
    return true;
}

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings) {
    PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    return bind_and_execute_prepared_command_program(context, commands, bindings, prepared);
}

}  // namespace ggml::hrx
