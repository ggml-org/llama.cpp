#include "graph-program-cache.h"

#include "dispatch/dispatch-scheduler.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

static bool tensor_metadata_matches(const Value & value, const ggml_tensor * tensor) {
    if (tensor == nullptr || value.type != tensor->type || value.element_count != ggml_nelements(tensor) ||
        value.byte_count != ggml_nbytes(tensor) || value.contiguous != ggml_is_contiguous(tensor)) {
        return false;
    }
    const bool tensor_alias = tensor->view_src != nullptr;
    const bool value_alias  = value.alias_source.value >= 0;
    if (tensor_alias && (!value_alias || value.storage_offset != tensor->view_offs)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] != tensor->ne[i] || value.nb[i] != tensor->nb[i]) {
            return false;
        }
    }
    return true;
}

static bool graph_node_params_match(const GraphNode & cached_node, const ggml_tensor * current_node) {
    if (current_node == nullptr) {
        return false;
    }
    return op_params_equivalent(cached_node.op, cached_node.params, *current_node);
}

static bool environment_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static void apply_graph_replay_result(PreparedCommandProgramCacheExecutionResult & result,
                                      const RecordedCommandGraphExecutionResult &  replay) {
    result.graph_replay_event                        = replay.event;
    result.graph_replay_ineligible_reason            = replay.ineligible_reason;
    result.graph_replay_build_ns                     = replay.build_ns;
    result.graph_replay_launch_ns                    = replay.launch_ns;
    result.graph_replay_total_ns                     = replay.total_ns();
    result.graph_replay_dispatches                   = replay.dispatch_count;
    result.graph_replay_transient_allocation_changed = replay.transient_allocation_changed;
}

static bool graph_replay_should_fallback(HrxGraphReplayEvent event) {
    return event == HrxGraphReplayEvent::Ineligible || event == HrxGraphReplayEvent::BuildFailed;
}

static Status bind_current_value(const ValueMap &                                   values,
                                 ValueId                                            expected,
                                 const ggml_tensor *                                tensor,
                                 std::vector<const ggml_tensor *> &                 tensor_by_value,
                                 std::unordered_map<const ggml_tensor *, int32_t> & value_by_tensor,
                                 const char *                                       role,
                                 size_t                                             node_index) {
    Status        status;
    const Value * value = values.find(expected);
    if (value == nullptr || expected.value < 0 || static_cast<size_t>(expected.value) >= tensor_by_value.size()) {
        status.log("node %zu %s references missing cached value %d", node_index, role, expected.value);
        return status;
    }
    if (tensor == nullptr) {
        status.log("node %zu %s value %d maps to a null tensor", node_index, role, expected.value);
        return status;
    }

    const ggml_tensor * existing_tensor = tensor_by_value[static_cast<size_t>(expected.value)];
    if (existing_tensor != nullptr && existing_tensor != tensor) {
        status.log("node %zu %s value %d maps to multiple current tensors", node_index, role, expected.value);
        return status;
    }

    const auto existing_value = value_by_tensor.find(tensor);
    if (existing_value != value_by_tensor.end() && existing_value->second != expected.value) {
        status.log("node %zu %s tensor maps to cached values %d and %d", node_index, role, existing_value->second,
                   expected.value);
        return status;
    }

    if (existing_tensor == nullptr && existing_value == value_by_tensor.end() &&
        !tensor_metadata_matches(*value, tensor)) {
        status.log("node %zu %s value %d metadata does not match current tensor", node_index, role, expected.value);
        return status;
    }

    tensor_by_value[static_cast<size_t>(expected.value)] = tensor;
    value_by_tensor.emplace(tensor, expected.value);
    return status;
}

static std::string command_program_shape_key(const CommandProgram & commands) {
    std::ostringstream out;
    out << "hrx-command-program-v1|commands=" << commands.commands.size();
    for (const Command & command : commands.commands) {
        out << "|ordinal=" << command.ordinal << "|kind=" << static_cast<int>(command.kind)
            << "|kernel=" << command.kernel.kernel_id;
        for (const auto & parameter : command.kernel.integer_parameters) {
            out << "|ip:" << parameter.first << '=' << parameter.second;
        }
        for (const auto & parameter : command.kernel.compile_parameters) {
            out << "|cp:" << parameter.first << '=' << parameter.second;
        }
        out << "|bindings=" << command.bindings.size();
        for (const CommandBinding & binding : command.bindings) {
            out << "|b:" << binding.name << ':' << binding.value.value << ':' << static_cast<int>(binding.origin) << ':'
                << binding.offset << ':' << binding.length << ':' << static_cast<int>(binding.access);
        }
        out << "|deps=" << command.dependencies.size();
        for (const uint32_t dependency : command.dependencies) {
            out << ':' << dependency;
        }
    }
    out << "|transients=" << commands.transients.allocations.size() << "|arena=" << commands.transients.arena_size
        << "|arena_alignment=" << commands.transients.arena_alignment;
    for (const TransientAllocation & allocation : commands.transients.allocations) {
        out << "|t:" << allocation.value.value << ':' << allocation.arena_offset << ':' << allocation.size << ':'
            << allocation.alignment;
    }
    return out.str();
}

}  // namespace

GraphProgram::GraphProgram(uint64_t                        uid,
                           std::string                     target,
                           std::unique_ptr<Graph>          graph,
                           std::unique_ptr<CommandProgram> commands,
                           std::string                     command_shape) :
    uid_(uid),
    target_(std::move(target)),
    graph_(std::move(graph)),
    commands_(std::move(commands)),
    command_shape_(std::move(command_shape)) {}

const GraphProgramExternalSlot * GraphProgram::find_external_slot(ValueId value) const {
    const auto found = external_slot_by_value_.find(value.value);
    if (found == external_slot_by_value_.end() || found->second >= external_slots_.size()) {
        return nullptr;
    }
    return &external_slots_[found->second];
}

const ggml_tensor * GraphProgram::resolve_external_slot(const ggml_cgraph &              graph,
                                                        const GraphProgramExternalSlot & slot,
                                                        Status &                         status) const {
    if (slot.node_index >= static_cast<size_t>(graph.n_nodes)) {
        status.log("external value %d references node slot %zu but current graph has %d nodes", slot.value.value,
                   slot.node_index, graph.n_nodes);
        return nullptr;
    }
    const ggml_tensor * node = graph.nodes[slot.node_index];
    if (node == nullptr) {
        status.log("external value %d references null node slot %zu", slot.value.value, slot.node_index);
        return nullptr;
    }
    if (slot.kind == GraphProgramExternalSlotKind::Node) {
        return node;
    }
    if (slot.source_index < 0 || slot.source_index >= GGML_MAX_SRC) {
        status.log("external value %d references invalid source slot %d", slot.value.value, slot.source_index);
        return nullptr;
    }
    const ggml_tensor * source = node->src[slot.source_index];
    if (source == nullptr) {
        status.log("external value %d references null source slot %zu:%d", slot.value.value, slot.node_index,
                   slot.source_index);
        return nullptr;
    }
    return source;
}

GraphProgramMatch GraphProgram::match_trusted_graph(const ggml_cgraph & current_graph, bool bind_external) const {
    GraphProgramMatch result;
    if (graph_ == nullptr || commands_ == nullptr) {
        result.status.log("missing cached HRX graph program");
        return result;
    }
    if (graph_->nodes().size() != static_cast<size_t>(current_graph.n_nodes)) {
        result.status.log("cached graph has %zu nodes but current graph has %d", graph_->nodes().size(),
                          current_graph.n_nodes);
        return result;
    }
    if (!graph_->nodes().empty()) {
        const ggml_tensor * first = current_graph.nodes[0];
        const ggml_tensor * last  = current_graph.nodes[current_graph.n_nodes - 1];
        if (first == nullptr || last == nullptr) {
            result.status.log("current graph has null sentinel nodes");
            return result;
        }
        if (first->op != graph_->nodes().front().op || last->op != graph_->nodes().back().op) {
            result.status.log("current graph sentinel ops do not match cached HRX graph");
            return result;
        }
    }
    if (!bind_external) {
        return result;
    }
    result.external_bindings.reserve(external_slots_.size());
    for (const GraphProgramExternalSlot & slot : external_slots_) {
        const ggml_tensor * tensor = resolve_external_slot(current_graph, slot, result.status);
        if (tensor == nullptr) {
            return result;
        }
        result.external_bindings.push_back({ slot.value, tensor });
    }
    return result;
}

GraphProgramMatch GraphProgram::match_host_staging_graph(const ggml_cgraph & current_graph) const {
    GraphProgramMatch           result;
    std::lock_guard<std::mutex> lock(prepared_mutex_);
    if (!has_prepared_) {
        result.status.log("missing prepared HRX command program");
        return result;
    }
    result.external_bindings.reserve(prepared_.host_staging.size());
    for (const HostStagingBuffer & staging : prepared_.host_staging) {
        const GraphProgramExternalSlot * slot = find_external_slot(ValueId(staging.value));
        if (slot == nullptr) {
            result.status.log("prepared host staging value %d has no external graph slot", staging.value);
            return result;
        }
        const ggml_tensor * tensor = resolve_external_slot(current_graph, *slot, result.status);
        if (tensor == nullptr) {
            return result;
        }
        result.external_bindings.push_back({ ValueId(staging.value), tensor });
    }
    return result;
}

Status GraphProgram::capture_external_slots(const ggml_cgraph & graph, const GraphProgramMatch & match) {
    Status status;
    external_slots_.clear();
    external_slot_by_value_.clear();
    fast_path_nodes_ = graph.nodes;
    external_slots_.reserve(match.external_bindings.size());
    for (const GraphProgramExternalBinding & binding : match.external_bindings) {
        GraphProgramExternalSlot slot;
        slot.value = binding.value;
        bool found = false;
        for (int i = 0; i < graph.n_nodes && !found; ++i) {
            const ggml_tensor * node = graph.nodes[i];
            if (node == nullptr) {
                continue;
            }
            if (node == binding.tensor) {
                slot.kind       = GraphProgramExternalSlotKind::Node;
                slot.node_index = static_cast<size_t>(i);
                found           = true;
                break;
            }
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (node->src[j] == binding.tensor) {
                    slot.kind         = GraphProgramExternalSlotKind::Source;
                    slot.node_index   = static_cast<size_t>(i);
                    slot.source_index = j;
                    found             = true;
                    break;
                }
            }
        }
        if (!found) {
            status.log("external value %d has no current graph slot", binding.value.value);
            continue;
        }
        external_slot_by_value_[binding.value.value] = external_slots_.size();
        external_slots_.push_back(slot);
    }
    return status;
}

bool GraphProgram::has_prepared_program() const {
    std::lock_guard<std::mutex> lock(prepared_mutex_);
    return has_prepared_;
}

bool GraphProgram::can_use_prepared_fast_path(const ggml_cgraph & graph) const {
    return graph.nodes == fast_path_nodes_;
}

PreparedCommandProgramCacheStats GraphProgram::prepared_stats() const {
    std::lock_guard<std::mutex> lock(prepared_mutex_);
    return prepared_stats_;
}

PreparedCommandProgramCacheExecutionResult GraphProgram::execute_with_result(
    const CommandProgramExecutionContext & context,
    const CommandProgramBindings &         bindings) {
    PreparedCommandProgramCacheExecutionResult result;
    if (commands_ == nullptr || !commands_->valid() || !bindings.valid()) {
        result.graph_replay_event             = HrxGraphReplayEvent::Ineligible;
        result.graph_replay_ineligible_reason = "invalid_graph_program";
        if (commands_ == nullptr) {
            result.status.log("missing cached HRX command program");
        }
        result.status.append(bindings.status);
        return result;
    }

    std::lock_guard<std::mutex> lock(prepared_mutex_);
    if (!has_prepared_) {
        prepared_ = prepare_command_program(context, *commands_, bindings);
        if (!prepared_.valid()) {
            result.status.append(prepared_.status);
            return result;
        }
        has_prepared_ = true;
        ++prepared_stats_.builds;
    } else {
        ++prepared_stats_.hits;
    }

    const RecordedCommandGraphExecutionResult replay =
        bind_and_launch_recorded_command_graph(context, *commands_, bindings, prepared_, recorded_);
    apply_graph_replay_result(result, replay);
    if (replay.success) {
        result.success = true;
        return result;
    }
    if (!graph_replay_should_fallback(replay.event)) {
        result.status.append(replay.status);
        if (result.status.success()) {
            result.status.log("execute cached HRX graph replay failed");
        }
        return result;
    }
    result.success = bind_and_execute_prepared_command_program(context, *commands_, bindings, prepared_);
    if (!result.success) {
        result.status.log("execute cached HRX command program failed");
    }
    return result;
}

GraphProgramMatch GraphProgram::match_current_graph(const ggml_cgraph & current_graph) const {
    GraphProgramMatch result;
    if (graph_ == nullptr) {
        result.status.log("missing cached HRX graph");
        return result;
    }
    if (commands_ == nullptr) {
        result.status.log("missing cached HRX command program");
        return result;
    }
    if (graph_->nodes().size() != static_cast<size_t>(current_graph.n_nodes)) {
        result.status.log("cached graph has %zu nodes but current graph has %d", graph_->nodes().size(),
                          current_graph.n_nodes);
        return result;
    }

    const ValueMap &                                 values = graph_->values();
    std::vector<const ggml_tensor *>                 tensor_by_value(values.size(), nullptr);
    std::unordered_map<const ggml_tensor *, int32_t> value_by_tensor;

    for (size_t node_index = 0; node_index < graph_->nodes().size(); ++node_index) {
        const GraphNode &   cached_node  = graph_->nodes()[node_index];
        const ggml_tensor * current_node = current_graph.nodes[node_index];
        if (current_node == nullptr) {
            result.status.log("current graph node %zu is null", node_index);
            return result;
        }
        if (cached_node.op != current_node->op) {
            result.status.log("node %zu cached op %s does not match current op %s", node_index,
                              ggml_op_name(cached_node.op), ggml_op_name(current_node->op));
            return result;
        }
        if (!graph_node_params_match(cached_node, current_node)) {
            result.status.log("node %zu cached op params do not match current graph", node_index);
            return result;
        }

        size_t input_index = 0;
        for (const ggml_tensor * source : current_node->src) {
            if (source == nullptr) {
                continue;
            }
            if (input_index >= cached_node.inputs.size()) {
                result.status.log("node %zu has more inputs than the cached graph", node_index);
                return result;
            }
            Status status = bind_current_value(values, cached_node.inputs[input_index], source, tensor_by_value,
                                               value_by_tensor, "input", node_index);
            if (!status.success()) {
                result.status.append(status);
                return result;
            }
            ++input_index;
        }
        if (input_index != cached_node.inputs.size()) {
            result.status.log("node %zu has %zu inputs but cached graph has %zu", node_index, input_index,
                              cached_node.inputs.size());
            return result;
        }
        Status status = bind_current_value(values, cached_node.output, current_node, tensor_by_value, value_by_tensor,
                                           "output", node_index);
        if (!status.success()) {
            result.status.append(status);
            return result;
        }
    }

    for (const ValueId id : values.external_value_ids()) {
        if (id.value < 0 || static_cast<size_t>(id.value) >= tensor_by_value.size() ||
            tensor_by_value[static_cast<size_t>(id.value)] == nullptr) {
            result.status.log("external value %d is missing from the current graph", id.value);
            return result;
        }
        result.external_bindings.push_back({ id, tensor_by_value[static_cast<size_t>(id.value)] });
    }
    return result;
}

bool GraphProgramCache::can_execute(const ggml_cgraph &  graph,
                                    const KernelCorpus & corpus,
                                    const std::string &  target) const {
    return check_support(graph, corpus, target).supported;
}

GraphProgramSupportResult GraphProgramCache::check_support(const ggml_cgraph &  graph,
                                                           const KernelCorpus & corpus,
                                                           const std::string &  target) const {
    GraphProgramSupportResult result;
    if (graph.n_nodes == 0) {
        result.supported = true;
        return result;
    }
    GraphImportResult imported = import_ggml_graph(graph);
    if (!imported.valid()) {
        result.status.append(imported.status);
        return result;
    }
    std::unique_ptr<GraphProgram> program =
        build_program_from_imported(graph.uid, std::move(imported.graph), corpus, target, result.status);
    result.supported = program != nullptr && result.status.success();
    return result;
}

GraphProgramLookup GraphProgramCache::build_from_imported(const ggml_cgraph &  graph,
                                                          Graph &&             imported_graph,
                                                          const KernelCorpus & corpus,
                                                          const std::string &  target) {
    GraphProgramLookup            result;
    std::unique_ptr<GraphProgram> program =
        build_program_from_imported(graph.uid, std::move(imported_graph), corpus, target, result.status);
    if (program == nullptr) {
        return result;
    }

    GraphProgramMatch match = program->match_current_graph(graph);
    if (!match.valid()) {
        result.status.append(match.status);
        return result;
    }
    Status slot_status = program->capture_external_slots(graph, match);
    if (!slot_status.success()) {
        result.status.append(slot_status);
        return result;
    }

    if (graph.uid == 0) {
        result.uncached_program = std::move(program);
        result.program          = result.uncached_program.get();
        result.match            = std::move(match);
        return result;
    }

    GraphProgram * cached_program = program.get();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        programs_[graph.uid] = std::move(program);
        cached_program       = programs_[graph.uid].get();
        last_program_        = cached_program;
        ++stats_.builds;
    }
    result.program = cached_program;
    result.match   = std::move(match);
    return result;
}

GraphProgramLookup GraphProgramCache::get_or_build(const ggml_cgraph &  graph,
                                                   const KernelCorpus & corpus,
                                                   const std::string &  target) {
    GraphProgramLookup result;
    if (graph.uid != 0) {
        const bool     disable_fast_path  = environment_flag_enabled("GGML_HRX_DISABLE_GRAPH_UID_FAST_PATH");
        const bool     validate_fast_path = environment_flag_enabled("GGML_HRX_VALIDATE_GRAPH_UID_CACHE");
        GraphProgram * cached_program     = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!disable_fast_path && last_program_ != nullptr && last_program_->uid() == graph.uid &&
                last_program_->target() == target) {
                cached_program = last_program_;
            } else {
                const auto found = programs_.find(graph.uid);
                if (found != programs_.end() && found->second->target() == target) {
                    cached_program = found->second.get();
                    last_program_  = cached_program;
                }
            }
        }
        if (cached_program != nullptr) {
            const bool bind_external =
                !cached_program->has_prepared_program() || !cached_program->can_use_prepared_fast_path(graph);
            GraphProgramMatch match = disable_fast_path || validate_fast_path ?
                                          cached_program->match_current_graph(graph) :
                                          cached_program->match_trusted_graph(graph, bind_external);
            if (match.valid() && validate_fast_path && !disable_fast_path) {
                GraphProgramMatch trusted_match = cached_program->match_trusted_graph(graph, bind_external);
                if (!trusted_match.valid()) {
                    result.status.append(trusted_match.status);
                    return result;
                }
            }
            if (match.valid()) {
                result.program = cached_program;
                result.match   = std::move(match);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++stats_.hits;
                }
                return result;
            }
            if (validate_fast_path) {
                result.status.append(match.status);
                return result;
            }
        }
    }

    GraphImportResult imported = import_ggml_graph(graph);
    if (!imported.valid()) {
        result.status.append(imported.status);
        return result;
    }
    return build_from_imported(graph, std::move(imported.graph), corpus, target);
}

GraphProgramCacheStats GraphProgramCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    GraphProgramCacheStats      stats = stats_;
    for (const auto & entry : programs_) {
        const PreparedCommandProgramCacheStats prepared = entry.second->prepared_stats();
        stats.prepared_program_builds += prepared.builds;
        stats.prepared_program_hits += prepared.hits;
    }
    return stats;
}

void GraphProgramCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    programs_.clear();
    last_program_ = nullptr;
}

std::unique_ptr<GraphProgram> GraphProgramCache::build_program_from_imported(uint64_t             uid,
                                                                             Graph &&             imported_graph,
                                                                             const KernelCorpus & corpus,
                                                                             const std::string &  target,
                                                                             Status &             errors) const {
    DispatchScheduler scheduler;
    if (!scheduler.schedule_graph(imported_graph, { target })) {
        errors.append(scheduler.plan().status);
        return nullptr;
    }

    CommandProgram commands = build_command_program(imported_graph, scheduler.plan(), corpus, target);
    if (!commands.valid()) {
        errors.append(commands.status);
        return nullptr;
    }

    std::string command_shape = command_program_shape_key(commands);
    return std::make_unique<GraphProgram>(uid, target, std::make_unique<Graph>(std::move(imported_graph)),
                                          std::make_unique<CommandProgram>(std::move(commands)),
                                          std::move(command_shape));
}

bool can_execute_standalone_op_as_graph(const ggml_tensor * op, const std::string & target) {
    if (op == nullptr) {
        return false;
    }
    Graph                graph;
    std::vector<ValueId> inputs;
    for (const ggml_tensor * source : op->src) {
        if (source == nullptr) {
            continue;
        }
        inputs.push_back(graph.values().get_or_add_tensor_value(source, ValueKind::External));
    }
    const ValueId output = graph.values().get_or_add_tensor_value(op, ValueKind::External);
    GraphNode &   node   = graph.add_node(op->op, output, std::move(inputs));
    node.params          = import_op_params(*op);
    if (!graph.build_index().success()) {
        return false;
    }
    return DispatchScheduler::supports_node(graph, &node, { target });
}

}  // namespace ggml::hrx
