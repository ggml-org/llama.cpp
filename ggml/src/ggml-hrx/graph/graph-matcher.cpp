#include "graph-matcher.h"

namespace ggml::hrx {
namespace {

static bool node_in_list(const GraphNode * node, const std::vector<const GraphNode *> & nodes) {
    for (const GraphNode * candidate : nodes) {
        if (candidate == node) {
            return true;
        }
    }
    return false;
}

static void append_unique_node(std::vector<const GraphNode *> & nodes, const GraphNode * node) {
    if (node != nullptr && !node_in_list(node, nodes)) {
        nodes.push_back(node);
    }
}

}  // namespace

std::vector<const GraphNode *> layout_alias_consumers(const Graph & graph, ValueId value) {
    std::vector<const GraphNode *> matches;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && is_layout_alias_node(graph, *consumer)) {
            matches.push_back(consumer);
        }
    }
    return matches;
}

std::vector<const GraphNode *> layout_alias_consumers_with_op(const Graph & graph, ValueId value, ggml_op op) {
    std::vector<const GraphNode *> matches;
    for (const GraphNode * consumer : layout_alias_consumers(graph, value)) {
        if (consumer->op == op) {
            matches.push_back(consumer);
        }
    }
    return matches;
}

const GraphNode * find_single_layout_alias_consumer(const Graph & graph, ValueId value) {
    const std::vector<const GraphNode *> matches = layout_alias_consumers(graph, value);
    return matches.size() == 1 ? matches.front() : nullptr;
}

const GraphNode * find_single_layout_alias_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> matches = layout_alias_consumers_with_op(graph, value, op);
    return matches.size() == 1 ? matches.front() : nullptr;
}

std::vector<const GraphNode *> consumers_with_op_through_layout_aliases(const Graph & graph,
                                                                        ValueId       value,
                                                                        ggml_op       op) {
    std::vector<const GraphNode *> matches;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr) {
            continue;
        }
        if (consumer->op == op) {
            append_unique_node(matches, consumer);
        }
        if (!is_layout_alias_node(graph, *consumer)) {
            continue;
        }
        for (const GraphNode * alias_consumer : graph.index().consumers(consumer->output)) {
            if (alias_consumer != nullptr && alias_consumer->op == op) {
                append_unique_node(matches, alias_consumer);
            }
        }
    }
    return matches;
}

const GraphNode * find_single_consumer_with_op_through_layout_aliases(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> matches = consumers_with_op_through_layout_aliases(graph, value, op);
    return matches.size() == 1 ? matches.front() : nullptr;
}

bool node_has_input_or_alias(const Graph & graph, const GraphNode & node, ValueId input) {
    const Value * input_value = graph.values().find(input);
    for (ValueId candidate : node.inputs) {
        if (candidate == input) {
            return true;
        }
        const Value * candidate_value = graph.values().find(candidate);
        if (candidate_value != nullptr && candidate_value->alias_source == input) {
            return true;
        }
        if (input_value != nullptr && input_value->alias_source == candidate) {
            return true;
        }
    }
    return false;
}

bool append_covered_node_index_once(const Graph &             graph,
                                    const std::vector<bool> & covered_nodes,
                                    const GraphNode *         node,
                                    std::vector<size_t> &     covered_indices) {
    size_t index = 0;
    if (node == nullptr || !graph.index().node_index(node, index) || index >= covered_nodes.size() ||
        covered_nodes[index]) {
        return false;
    }
    for (const size_t covered : covered_indices) {
        if (covered == index) {
            return true;
        }
    }
    covered_indices.push_back(index);
    return true;
}

}  // namespace ggml::hrx
