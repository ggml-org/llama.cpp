#pragma once

#include "graph.h"

#include <cstddef>
#include <vector>

namespace ggml::hrx {

std::vector<const GraphNode *> layout_alias_consumers(const Graph & graph, ValueId value);
std::vector<const GraphNode *> layout_alias_consumers_with_op(const Graph & graph, ValueId value, ggml_op op);

const GraphNode * find_single_layout_alias_consumer(const Graph & graph, ValueId value);
const GraphNode * find_single_layout_alias_consumer_with_op(const Graph & graph, ValueId value, ggml_op op);

std::vector<const GraphNode *> consumers_with_op_through_layout_aliases(const Graph & graph, ValueId value, ggml_op op);
const GraphNode * find_single_consumer_with_op_through_layout_aliases(const Graph & graph, ValueId value, ggml_op op);

bool node_has_input_or_alias(const Graph & graph, const GraphNode & node, ValueId input);
bool append_covered_node_index_once(const Graph &             graph,
                                    const std::vector<bool> & covered_nodes,
                                    const GraphNode *         node,
                                    std::vector<size_t> &     covered_indices);

}  // namespace ggml::hrx
