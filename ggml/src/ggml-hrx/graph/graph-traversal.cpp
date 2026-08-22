#include "graph-traversal.h"

#include <set>
#include <vector>

namespace ggml::hrx {
namespace {

static bool is_root_op(ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_SSM_CONV:
            return true;
        default:
            return false;
    }
}

static bool is_fusable_followup_op(ggml_op op) {
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return true;
        default:
            return false;
    }
}

static void erase_ready(size_t node_index, std::set<size_t> & root_queue, std::set<size_t> & regular_queue) {
    root_queue.erase(node_index);
    regular_queue.erase(node_index);
}

static void add_ready_node(const GraphNode &         node,
                           size_t                    node_index,
                           const std::vector<bool> & selected,
                           std::set<size_t> &        root_queue,
                           std::set<size_t> &        regular_queue) {
    if (node_index >= selected.size() || selected[node_index]) {
        return;
    }
    if (is_root_op(node.op)) {
        root_queue.insert(node_index);
    } else {
        regular_queue.insert(node_index);
    }
}

static bool select_merge_candidate(const Graph &             graph,
                                   size_t                    selected_node,
                                   const std::vector<int> &  pending_inputs,
                                   const std::vector<bool> & selected,
                                   size_t &                  next_node) {
    const std::vector<GraphNode> & nodes = graph.nodes();
    if (selected_node >= nodes.size()) {
        return false;
    }

    bool   found      = false;
    size_t best_index = 0;
    for (const GraphNode * consumer : graph.index().consumers(nodes[selected_node].output)) {
        size_t consumer_index = 0;
        if (consumer == nullptr || !graph.index().node_index(consumer, consumer_index) ||
            consumer_index >= selected.size() || selected[consumer_index] || pending_inputs[consumer_index] != 0 ||
            !is_fusable_followup_op(consumer->op)) {
            continue;
        }
        if (!found || consumer_index < best_index) {
            found      = true;
            best_index = consumer_index;
        }
    }
    if (!found) {
        return false;
    }
    next_node = best_index;
    return true;
}

}  // namespace

GraphTraversalOrder GraphTraversalOrder::build(const Graph & graph) {
    GraphTraversalOrder            result;
    const std::vector<GraphNode> & nodes = graph.nodes();
    result.nodes_.reserve(nodes.size());
    if (!graph.has_index()) {
        for (const GraphNode & node : nodes) {
            result.nodes_.push_back(&node);
        }
        return result;
    }

    std::vector<int> pending_inputs(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (ValueId input : nodes[i].inputs) {
            if (graph.index().producer(input) != nullptr) {
                ++pending_inputs[i];
            }
        }
    }

    std::set<size_t>  root_queue;
    std::set<size_t>  regular_queue;
    std::vector<bool> selected(nodes.size(), false);
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (pending_inputs[i] == 0) {
            add_ready_node(nodes[i], i, selected, root_queue, regular_queue);
        }
    }

    bool   has_previous  = false;
    size_t previous_node = 0;
    while (result.nodes_.size() < nodes.size()) {
        size_t selected_index = 0;
        if (has_previous && select_merge_candidate(graph, previous_node, pending_inputs, selected, selected_index)) {
            erase_ready(selected_index, root_queue, regular_queue);
        } else if (!root_queue.empty()) {
            selected_index = *root_queue.begin();
            root_queue.erase(root_queue.begin());
        } else if (!regular_queue.empty()) {
            selected_index = *regular_queue.begin();
            regular_queue.erase(regular_queue.begin());
        } else {
            break;
        }

        if (selected[selected_index]) {
            continue;
        }
        selected[selected_index] = true;
        result.nodes_.push_back(&nodes[selected_index]);
        has_previous  = true;
        previous_node = selected_index;

        for (const GraphNode * consumer : graph.index().consumers(nodes[selected_index].output)) {
            size_t consumer_index = 0;
            if (consumer == nullptr || !graph.index().node_index(consumer, consumer_index) ||
                consumer_index >= pending_inputs.size() || selected[consumer_index]) {
                continue;
            }
            --pending_inputs[consumer_index];
            if (pending_inputs[consumer_index] == 0) {
                add_ready_node(*consumer, consumer_index, selected, root_queue, regular_queue);
            }
        }
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!selected[i]) {
            result.nodes_.push_back(&nodes[i]);
        }
    }
    return result;
}

}  // namespace ggml::hrx
