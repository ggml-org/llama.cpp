#pragma once

#include "backend-context.h"
#include "dispatch/command-program-bindings.h"
#include "ggml.h"
#include "runtime/graph-program-cache.h"
#include "status.h"

struct ggml_cgraph;

namespace ggml::hrx {

struct GraphSupportResult {
    bool   supported = false;
    Status status;

    bool success() const { return supported && status.success(); }
};

struct GraphExecutionResult {
    enum ggml_status code = GGML_STATUS_FAILED;
    Status           status;

    bool success() const { return code == GGML_STATUS_SUCCESS && status.success(); }
};

class GraphExecutor {
  public:
    explicit GraphExecutor(ggml_backend_hrx_context & context);

    GraphSupportResult   can_execute(const ggml_cgraph & graph) const;
    GraphExecutionResult execute(const ggml_cgraph & graph) const;

  private:
    Status context_valid_for_graph_programs() const;
    Status context_valid_for_execution() const;

    CommandProgramBindings bind_external_value_buffers(const GraphProgramMatch & match) const;

    ggml_backend_hrx_context & context_;
};

}  // namespace ggml::hrx
