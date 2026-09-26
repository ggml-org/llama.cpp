#pragma once

#include "command-plan.h"
#include "dispatch_registration/dispatch-registry.h"
#include "graph/graph.h"

#include <string>
#include <vector>

namespace ggml::hrx {

struct DispatchScheduleDiagnostics {
    size_t                   unsupported_node_index = 0;
    const GraphNode *        unsupported_node       = nullptr;
    std::string              unsupported_message;
    DispatchMatchDiagnostics match;
};

class DispatchScheduler {
  public:
    bool schedule_graph(Graph & graph, const DispatchTarget & target);
    bool schedule_graph(Graph & graph, const DispatchTarget & target, DispatchScheduleDiagnostics * diagnostics);

    const CommandPlan & plan() const { return plan_; }

    const std::vector<Dispatch> & dispatches() const { return plan_.dispatches; }

    const std::string & error() const {
        static const std::string empty;
        return plan_.status.errors().empty() ? empty : plan_.status.errors().front();
    }

    static bool supports_node(const Graph & graph, const GraphNode * node, const DispatchTarget & target);
    static bool can_schedule_graph(const Graph & graph, const DispatchTarget & target);

  private:
    CommandPlan plan_;
};

}  // namespace ggml::hrx
