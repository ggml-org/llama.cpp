#include "graph.h"

#include "ggml-impl.h"

#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ggml::hrx {
namespace {

static const ggml_tensor * tensor_storage_root(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static int32_t graph_tensor_use_count(const ggml_cgraph & graph, const ggml_tensor * tensor) {
    if (tensor == nullptr || graph.use_counts == nullptr || graph.visited_hash_set.keys == nullptr) {
        return -1;
    }
    const size_t hash_pos = ggml_hash_find(&graph.visited_hash_set, tensor);
    if (hash_pos == GGML_HASHSET_FULL || !ggml_bitset_get(graph.visited_hash_set.used, hash_pos)) {
        return -1;
    }
    return graph.use_counts[hash_pos];
}

static bool tensor_is_external(const ggml_tensor *                                  tensor,
                               const std::unordered_map<const ggml_tensor *, int> & use_counts,
                               const std::unordered_set<const ggml_tensor *> &      graph_nodes,
                               const ggml_cgraph &                                  graph) {
    const ggml_tensor * root = tensor_storage_root(tensor);
    if (root == nullptr || root->op == GGML_OP_NONE || graph_nodes.find(root) == graph_nodes.end()) {
        return true;
    }
    if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return true;
    }
    const auto    found      = use_counts.find(tensor);
    const int     local_uses = found != use_counts.end() ? found->second : 0;
    const int32_t graph_uses = graph_tensor_use_count(graph, tensor);
    if (graph_uses > local_uses) {
        return true;
    }
    return local_uses == 0;
}

}  // namespace

GraphIndex GraphIndex::build(const Graph & graph) {
    GraphIndex                     index;
    const std::vector<GraphNode> & nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const GraphNode & node = nodes[i];
        index.node_indices_.emplace(&node, i);
        index.producers_.emplace(node.output.value, &node);
        for (ValueId input : node.inputs) {
            index.consumers_[input.value].push_back(&node);
        }
    }
    return index;
}

const GraphNode * GraphIndex::producer(ValueId value) const {
    const auto found = producers_.find(value.value);
    return found == producers_.end() ? nullptr : found->second;
}

const std::vector<const GraphNode *> & GraphIndex::consumers(ValueId value) const {
    static const std::vector<const GraphNode *> empty;
    const auto                                  found = consumers_.find(value.value);
    return found == consumers_.end() ? empty : found->second;
}

bool GraphIndex::has_single_consumer(ValueId value) const {
    return consumers(value).size() == 1;
}

bool GraphIndex::node_index(const GraphNode * node, size_t & index) const {
    const auto found = node_indices_.find(node);
    if (found == node_indices_.end()) {
        return false;
    }
    index = found->second;
    return true;
}

Graph::Graph(const Graph & other) : values_(other.values_), nodes_(other.nodes_) {
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
}

Graph & Graph::operator=(const Graph & other) {
    if (this == &other) {
        return *this;
    }
    values_ = other.values_;
    nodes_  = other.nodes_;
    index_.reset();
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
    return *this;
}

Graph::Graph(Graph && other) : values_(std::move(other.values_)), nodes_(std::move(other.nodes_)) {
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
}

Graph & Graph::operator=(Graph && other) {
    if (this == &other) {
        return *this;
    }
    values_ = std::move(other.values_);
    nodes_  = std::move(other.nodes_);
    index_.reset();
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
    return *this;
}

GraphNode & Graph::add_node(ggml_op op, ValueId output, std::vector<ValueId> inputs) {
    index_.reset();
    GraphNode node;
    node.op     = op;
    node.output = output;
    node.inputs = std::move(inputs);
    nodes_.push_back(std::move(node));
    return nodes_.back();
}

Status Graph::build_index() {
    index_ = GraphIndex::build(*this);
    return {};
}

const GraphIndex & Graph::index() const {
    assert(index_.has_value());
    return *index_;
}

GraphImportResult import_ggml_graph(const ggml_cgraph & graph) {
    GraphImportResult                            result;
    std::unordered_map<const ggml_tensor *, int> use_counts;
    std::unordered_set<const ggml_tensor *>      graph_nodes;
    graph_nodes.reserve(static_cast<size_t>(graph.n_nodes));
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        if (node == nullptr) {
            result.status.log("ggml graph contains a null node");
            return result;
        }
        graph_nodes.insert(node);
        for (const ggml_tensor * source : node->src) {
            if (source != nullptr) {
                ++use_counts[source];
            }
        }
    }

    ValueMap & values = result.graph.values();
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor *  node = graph.nodes[i];
        std::vector<ValueId> inputs;
        for (const ggml_tensor * source : node->src) {
            if (source == nullptr) {
                continue;
            }
            const ValueKind kind =
                tensor_is_external(source, use_counts, graph_nodes, graph) ? ValueKind::External : ValueKind::Transient;
            inputs.push_back(values.get_or_add_tensor_value(source, kind));
        }

        const ValueKind output_kind =
            tensor_is_external(node, use_counts, graph_nodes, graph) ? ValueKind::External : ValueKind::Transient;
        const ValueId output     = values.get_or_add_tensor_value(node, output_kind);
        GraphNode &   graph_node = result.graph.add_node(node->op, output, std::move(inputs));
        graph_node.params        = import_op_params(*node);
    }

    result.status.append(result.graph.build_index());
    return result;
}

bool is_layout_alias_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

bool is_layout_alias_node(const Graph & graph, const GraphNode & node) {
    return is_layout_alias_op(node.op) && node.inputs.size() == 1 &&
           graph.values().same_storage(node.output, node.inputs[0]);
}

}  // namespace ggml::hrx
