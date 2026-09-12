#include "dispatch-scheduler.h"

#include "ggml.h"
#include "graph/graph-traversal.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static bool match_covers_root(const DispatchMatch & match, size_t root_index) {
    return std::find(match.covered_nodes.begin(), match.covered_nodes.end(), root_index) != match.covered_nodes.end();
}

static bool match_overlaps_covered_nodes(const DispatchMatch & match, const std::vector<bool> & covered_nodes) {
    for (const size_t node_index : match.covered_nodes) {
        if (node_index >= covered_nodes.size() || covered_nodes[node_index]) {
            return true;
        }
    }
    return false;
}

static bool try_match_registration(const Graph &              graph,
                                   const GraphNode *          node,
                                   size_t                     node_index,
                                   const std::vector<bool> &  covered_nodes,
                                   const CommandPlan &        plan,
                                   const DispatchRegistry &   registry,
                                   ValueId                    next_plan_value,
                                   DispatchMatch &            match,
                                   DispatchMatchDiagnostics * diagnostics) {
    const DispatchMatchContext context = {
        graph, node, node_index, covered_nodes, plan, next_plan_value,
    };
    return registry.match(context, match, diagnostics);
}

static void clear_plan_results(CommandPlan & plan) {
    plan.initialization_dispatches.clear();
    plan.dispatches.clear();
    plan.transients.clear();
    plan.constant_initializations.clear();
    plan.completion_counter_requests.clear();
    plan.metadata.clear();
}

static void append_value_summary(std::ostringstream & stream, const Graph & graph, ValueId value_id) {
    const Value * value = graph.values().find(value_id);
    if (value == nullptr) {
        stream << value_id.value << ":missing";
        return;
    }
    stream << value_id.value << ":" << ggml_type_name(value->type) << "[" << value->ne[0] << "," << value->ne[1] << ","
           << value->ne[2] << "," << value->ne[3] << "]";
    const GraphNode * producer = graph.index().producer(value_id);
    if (producer != nullptr) {
        stream << "<-" << ggml_op_name(producer->op);
    }
}

static void append_node_summary(std::ostringstream & stream, const Graph & graph, const GraphNode * node) {
    if (node == nullptr) {
        stream << "null";
        return;
    }
    size_t node_index = 0;
    if (graph.index().node_index(node, node_index)) {
        stream << node_index << ":";
    }
    stream << ggml_op_name(node->op);
}

static std::string unsupported_node_message(const Graph & graph, size_t index, const GraphNode & node) {
    std::ostringstream stream;
    stream << "unsupported HRX node " << index << ": " << ggml_op_name(node.op) << " output=";
    append_value_summary(stream, graph, node.output);
    stream << " inputs=[";
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        append_value_summary(stream, graph, node.inputs[i]);
    }
    stream << "]";
    stream << " consumers=[";
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(node.output);
    for (size_t i = 0; i < consumers.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        append_node_summary(stream, graph, consumers[i]);
    }
    stream << "]";
    return stream.str();
}

static bool value_is_available(const Graph & graph, ValueId value, const std::vector<bool> & covered_nodes) {
    const GraphNode * producer = graph.index().producer(value);
    if (producer == nullptr) {
        return true;
    }
    size_t producer_index = 0;
    return graph.index().node_index(producer, producer_index) && producer_index < covered_nodes.size() &&
           covered_nodes[producer_index];
}

static bool can_elide_layout_alias_node(const Graph &             graph,
                                        const GraphNode &         node,
                                        const std::vector<bool> & covered_nodes) {
    return is_layout_alias_node(graph, node) && value_is_available(graph, node.inputs[0], covered_nodes);
}

static bool apply_value_aliases(Graph & graph, const DispatchMatch & match, Status & status) {
    for (const DispatchValueAliasRequest & alias : match.value_aliases) {
        Status alias_status = graph.values().alias_storage(alias.target_value, alias.source_value);
        if (!alias_status.success()) {
            status.append(alias_status);
            return false;
        }
    }
    return true;
}

}  // namespace

bool DispatchScheduler::schedule_graph(Graph & graph, const DispatchTarget & target) {
    return this->schedule_graph(graph, target, nullptr);
}

bool DispatchScheduler::schedule_graph(Graph &                       graph,
                                       const DispatchTarget &        target,
                                       DispatchScheduleDiagnostics * diagnostics) {
    plan_ = {};
    if (diagnostics != nullptr) {
        *diagnostics = {};
    }
    const DispatchRegistry * registry = find_dispatch_registry(target);
    if (registry == nullptr) {
        plan_.status.log("no HRX dispatch registry for target %s", target.architecture.c_str());
        return false;
    }
    const std::vector<GraphNode> & nodes = graph.nodes();
    if (!graph.has_index()) {
        plan_.status.log("HRX graph is missing graph index");
        return false;
    }
    std::vector<bool>         covered_nodes(nodes.size(), false);
    Status                    pending_diagnostics;
    const GraphTraversalOrder traversal = GraphTraversalOrder::build(graph);
    for (const GraphNode * node : traversal.nodes()) {
        size_t i = 0;
        if (node == nullptr || !graph.index().node_index(node, i)) {
            plan_.status.log("HRX traversal references a node outside the graph");
            clear_plan_results(plan_);
            return false;
        }
        if (covered_nodes[i]) {
            continue;
        }
        DispatchMatch            match;
        const ValueId            next_plan_value(static_cast<int32_t>(graph.values().size() + plan_.transients.size() +
                                                                      plan_.completion_counter_requests.size()));
        DispatchMatchDiagnostics match_diagnostics;
        if (!try_match_registration(graph, node, i, covered_nodes, plan_, *registry, next_plan_value, match,
                                    &match_diagnostics)) {
            if (can_elide_layout_alias_node(graph, *node, covered_nodes)) {
                pending_diagnostics.append(match.status);
                covered_nodes[i] = true;
                continue;
            }
            plan_.status.append(pending_diagnostics);
            plan_.status.append(match.status);
            const std::string message = unsupported_node_message(graph, i, *node);
            plan_.status.log("%s", message.c_str());
            if (diagnostics != nullptr) {
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = node;
                diagnostics->unsupported_message    = message;
                diagnostics->match                  = std::move(match_diagnostics);
            }
            clear_plan_results(plan_);
            return false;
        }
        if (match.covered_nodes.empty() || match.dispatches.empty() || !match_covers_root(match, i) ||
            match_overlaps_covered_nodes(match, covered_nodes)) {
            plan_.status.log("invalid HRX dispatch match for node %zu: %s", i, ggml_op_name(node->op));
            if (diagnostics != nullptr) {
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = node;
                diagnostics->unsupported_message    = "invalid HRX dispatch match";
                diagnostics->match                  = std::move(match_diagnostics);
            }
            clear_plan_results(plan_);
            return false;
        }
        if (!apply_value_aliases(graph, match, plan_.status)) {
            if (diagnostics != nullptr) {
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = node;
                diagnostics->unsupported_message    = "invalid HRX value alias";
                diagnostics->match                  = std::move(match_diagnostics);
            }
            clear_plan_results(plan_);
            return false;
        }
        for (Dispatch & dispatch : match.initialization_dispatches) {
            plan_.initialization_dispatches.push_back(std::move(dispatch));
        }
        for (Dispatch & dispatch : match.dispatches) {
            plan_.dispatches.push_back(std::move(dispatch));
        }
        for (CommandPlanTransient & transient : match.transients) {
            plan_.transients.push_back(std::move(transient));
        }
        for (CommandPlanConstantInitialization & initialization : match.constant_initializations) {
            plan_.constant_initializations.push_back(std::move(initialization));
        }
        for (CommandPlanCompletionCounterRequest & request : match.completion_counter_requests) {
            plan_.completion_counter_requests.push_back(std::move(request));
        }
        if (!plan_.metadata.append(std::move(match.metadata), plan_.status)) {
            clear_plan_results(plan_);
            return false;
        }
        for (const size_t covered_node : match.covered_nodes) {
            covered_nodes[covered_node] = true;
        }
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!covered_nodes[i]) {
            if (can_elide_layout_alias_node(graph, nodes[i], covered_nodes)) {
                covered_nodes[i] = true;
                continue;
            }
            plan_.status.append(pending_diagnostics);
            const std::string message = unsupported_node_message(graph, i, nodes[i]);
            plan_.status.log("%s", message.c_str());
            if (diagnostics != nullptr) {
                DispatchMatch match;
                const ValueId next_plan_value(static_cast<int32_t>(graph.values().size() + plan_.transients.size() +
                                                                   plan_.completion_counter_requests.size()));
                DispatchMatchDiagnostics match_diagnostics;
                try_match_registration(graph, &nodes[i], i, covered_nodes, plan_, *registry, next_plan_value, match,
                                       &match_diagnostics);
                diagnostics->unsupported_node_index = i;
                diagnostics->unsupported_node       = &nodes[i];
                diagnostics->unsupported_message    = message;
                diagnostics->match                  = std::move(match_diagnostics);
            }
            clear_plan_results(plan_);
            return false;
        }
    }
    return true;
}

bool DispatchScheduler::supports_node(const Graph & graph, const GraphNode * node, const DispatchTarget & target) {
    const DispatchRegistry * registry = find_dispatch_registry(target);
    if (registry == nullptr) {
        return false;
    }
    if (node == nullptr || !graph.has_index()) {
        return false;
    }
    size_t node_index = 0;
    if (!graph.index().node_index(node, node_index)) {
        return false;
    }
    const std::vector<bool> covered_nodes(graph.nodes().size(), false);
    DispatchMatch           match;
    const ValueId           next_plan_value(static_cast<int32_t>(graph.values().size()));
    const CommandPlan       plan;
    return try_match_registration(graph, node, node_index, covered_nodes, plan, *registry, next_plan_value, match,
                                  nullptr);
}

bool DispatchScheduler::can_schedule_graph(const Graph & graph, const DispatchTarget & target) {
    Graph             graph_copy = graph;
    DispatchScheduler scheduler;
    return scheduler.schedule_graph(graph_copy, target);
}

}  // namespace ggml::hrx
