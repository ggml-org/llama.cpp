#pragma once
#include "../ggml-decoder.h"
#include "node_context.h"

namespace ov::frontend::ggml {
// Match complete build_rs groups before replacing reset/reorder. Unmatched graphs retain
// their ordinary translators, including CPU, static and released-runtime execution.
class RecurrentCacheFusion {
public:
    RecurrentCacheFusion(const std::shared_ptr<GgmlDecoder> & decoder, const std::shared_ptr<TensorMap> & tensors);
    OutputVector translate(int index, const NodeContext & context);
private:
    struct Group {
        int reset = -1, gather = -1, remainder = -1, concat = -1, compute = -1, writeback = -1;
        std::vector<int> snapshots;
        Output<Node> old_cache, conv, updated;
    };

    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> m_tensors;
    std::vector<Group> m_groups;
    std::map<int, size_t> m_nodes;
};
}  // namespace ov::frontend::ggml
