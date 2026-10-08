#pragma once

#include "dispatch/dispatch-scheduler.h"
#include "graph.h"
#include "status.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace ggml::hrx {

struct GraphSnapshotLoadResult {
    uint64_t    uid = 0;
    std::string target;
    Graph       graph;
    Status      status;

    bool valid() const { return status.success(); }
};

std::string serialize_graph_snapshot_json(const Graph & graph, const std::string & target, uint64_t uid);
std::string format_graph_snapshot_text(const Graph & graph, const std::string & target, uint64_t uid);

GraphSnapshotLoadResult load_graph_snapshot_json(const std::string & contents);

Status write_graph_snapshot(const std::filesystem::path & directory,
                            const Graph &                 graph,
                            const std::string &           target,
                            uint64_t                      uid);

std::string format_schedule_diagnostics_text(const Graph &                       graph,
                                             const CommandPlan &                 plan,
                                             const DispatchScheduleDiagnostics & diagnostics);
std::string serialize_schedule_diagnostics_json(const Graph &                       graph,
                                                const CommandPlan &                 plan,
                                                const DispatchScheduleDiagnostics & diagnostics);

}  // namespace ggml::hrx
