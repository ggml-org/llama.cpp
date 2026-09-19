#pragma once

#include "ggml.h"
#include "op-params.h"
#include "status.h"
#include "value-map.h"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace ggml::hrx {

struct GraphNode {
    ggml_op              op;
    ValueId              output;
    std::vector<ValueId> inputs;
    OpParams             params;
};

class Graph;

class GraphIndex {
  public:
    GraphIndex() = default;

    static GraphIndex build(const Graph & graph);

    const GraphNode *                      producer(ValueId value) const;
    const std::vector<const GraphNode *> & consumers(ValueId value) const;
    bool                                   has_single_consumer(ValueId value) const;
    bool                                   node_index(const GraphNode * node, size_t & index) const;

  private:
    std::unordered_map<int32_t, const GraphNode *>              producers_;
    std::unordered_map<int32_t, std::vector<const GraphNode *>> consumers_;
    std::unordered_map<const GraphNode *, size_t>               node_indices_;
};

class Graph {
  public:
    Graph() = default;
    Graph(const Graph & other);
    Graph & operator=(const Graph & other);
    Graph(Graph && other);
    Graph & operator=(Graph && other);

    GraphNode & add_node(ggml_op op, ValueId output, std::vector<ValueId> inputs);

    Status build_index();

    bool has_index() const { return index_.has_value(); }

    const GraphIndex & index() const;

    const std::vector<GraphNode> & nodes() const { return nodes_; }

    const ValueMap & values() const { return values_; }

    ValueMap & values() { return values_; }

  private:
    ValueMap                  values_;
    std::vector<GraphNode>    nodes_;
    std::optional<GraphIndex> index_;
};

struct GraphImportResult {
    Graph  graph;
    Status status;

    bool valid() const { return status.success(); }
};

GraphImportResult import_ggml_graph(const ggml_cgraph & graph);

bool is_layout_alias_op(ggml_op op);
bool is_layout_alias_node(const Graph & graph, const GraphNode & node);

}  // namespace ggml::hrx
