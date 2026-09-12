#include "graph-executor.h"

#include "backend-buffer-binding.h"
#include "ggml-impl.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <utility>
#include <vector>

namespace ggml::hrx {

GraphExecutor::GraphExecutor(ggml_backend_hrx_context & context) : context_(context) {}

Status GraphExecutor::context_valid_for_graph_programs() const {
    Status status;
    if (context_.device == nullptr) {
        status.log("missing HRX device context");
    } else if (context_.device->architecture.empty()) {
        status.log("missing HRX target");
    }
    return status;
}

Status GraphExecutor::context_valid_for_execution() const {
    return context_valid_for_graph_programs();
}

GraphSupportResult GraphExecutor::can_execute(const ggml_cgraph & graph) const {
    GraphSupportResult result;
    if (graph.n_nodes == 0) {
        result.supported = true;
        return result;
    }
    result.status = context_valid_for_graph_programs();
    if (!result.status.success()) {
        return result;
    }
    const KernelCorpus &            corpus = get_qwen_kernel_corpus();
    const GraphProgramSupportResult support =
        context_.graph_programs.check_support(graph, corpus, context_.device->architecture);
    result.supported = support.supported;
    result.status.append(support.status);
    return result;
}

CommandProgramBindings GraphExecutor::bind_external_value_buffers(const GraphProgramMatch & match) const {
    std::vector<CommandProgramBinding> bindings;
    Status                             status;
    bindings.reserve(match.external_bindings.size());
    for (const GraphProgramExternalBinding & external : match.external_bindings) {
        ValueBufferBinding    value_binding;
        CommandProgramBinding binding;
        binding.value = external.value;
        if (ggml_backend_hrx_resolve_value_buffer(external.tensor, value_binding)) {
            binding.buffer     = value_binding.buffer;
            binding.host_data  = value_binding.host_data;
            binding.offset     = value_binding.offset;
            binding.length     = value_binding.length;
            binding.identity   = value_binding.identity;
            binding.generation = value_binding.generation;
            binding.capacity   = value_binding.capacity;
            binding.weight     = value_binding.weight;
        } else {
            status.log("external value %d is not bound", external.value.value);
        }
        bindings.push_back(binding);
    }
    return CommandProgramBindings::from_bindings(std::move(bindings), status);
}

GraphExecutionResult GraphExecutor::execute(const ggml_cgraph & graph) const {
    GraphExecutionResult result;
    if (graph.n_nodes == 0) {
        result.code = GGML_STATUS_SUCCESS;
        return result;
    }
    result.status = context_valid_for_execution();
    if (!result.status.success()) {
        return result;
    }

    const KernelCorpus & corpus = get_qwen_kernel_corpus();
    GraphProgramLookup   lookup = context_.graph_programs.get_or_build(graph, corpus, context_.device->architecture);
    if (!lookup.valid()) {
        result.status.append(lookup.status);
        result.status.append(lookup.match.status);
        if (result.status.success()) {
            result.status.log("build HRX graph program failed");
        }
        return result;
    }

    const bool use_graph_prepared =
        !lookup.program->has_prepared_program() || lookup.program->can_use_prepared_fast_path(graph);
    GraphProgramMatch binding_match = std::move(lookup.match);
    if (use_graph_prepared && lookup.program->has_prepared_program()) {
        binding_match = lookup.program->match_host_staging_graph(graph);
        if (!binding_match.valid()) {
            result.status.append(binding_match.status);
            return result;
        }
    }

    CommandProgramBindings bindings = bind_external_value_buffers(binding_match);
    if (!bindings.valid()) {
        result.status.append(bindings.status);
        return result;
    }
    const CommandProgramExecutionContext execution_context = {
        context_.device->device,
        context_.stream,
        context_.device->architecture.c_str(),
        &corpus,
        &context_.kernel_executables,
        &context_.transient_arena,
        &context_.host_transfers,
        &context_.host_weights,
    };
    const PreparedCommandProgramCacheExecutionResult execution =
        use_graph_prepared ? lookup.program->execute_with_result(execution_context, bindings) :
                             context_.prepared_programs.execute_with_result(execution_context, lookup.program->uid(),
                                                                            lookup.program->command_shape(),
                                                                            lookup.program->commands(), bindings);
    if (!execution.success) {
        result.status.append(execution.status);
        if (result.status.success()) {
            result.status.log("execute HRX command program failed");
        }
        return result;
    }

    result.code = GGML_STATUS_SUCCESS;
    return result;
}

}  // namespace ggml::hrx
