#pragma once

#include "graph_iterator.h"
#include "ggml-decoder.h"
#include <ggml-impl.h>

// To remove tensorflow
namespace ov {
namespace frontend {
namespace tensorflow { 
namespace ggml {

class GgmlOvGraphIterator : public GgmlGraphIterator {

protected: 
    void initialize_decoders();

public:
    using Ptr = std::shared_ptr<GgmlOvGraphIterator>;
    GgmlOvGraphIterator(struct ggml_cgraph * cgraph);

    /// \brief Get a number of operation nodes in the sgraph
    virtual size_t size() const override;

    /// \brief Set iterator to the start position
    virtual void reset() override;

    /// \brief Move to the next node in the graph
    virtual void next() override;

    /// \brief Returns true if iterator goes out of the range of available nodes
    virtual bool is_end() const override;

    /// \brief Return a pointer to a decoder of the current node
    virtual std::shared_ptr<DecoderBase> get_decoder() const override;

    virtual std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string& func_name) const override {
        return nullptr;
        GGML_UNUSED(func_name);
    }

    /// \brief Returns a vector of input names in the original order
    virtual std::vector<std::string> get_input_names() const override;

    /// \brief Returns a vector of output names in the original order
    virtual std::vector<std::string> get_output_names() const override;

    virtual void dump_graph_iterator() const;

private:
    struct ggml_cgraph * m_cgraph;
    size_t node_index = 0;
    std::vector<std::shared_ptr<DecoderBase>> m_decoders;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

}
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
