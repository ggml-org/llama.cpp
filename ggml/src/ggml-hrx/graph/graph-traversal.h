#pragma once

#include "graph.h"

#include <vector>

namespace ggml::hrx {

class GraphTraversalOrder {
  public:
    GraphTraversalOrder() = default;

    static GraphTraversalOrder build(const Graph & graph);

    const std::vector<const GraphNode *> & nodes() const { return nodes_; }

  private:
    std::vector<const GraphNode *> nodes_;
};

}  // namespace ggml::hrx
