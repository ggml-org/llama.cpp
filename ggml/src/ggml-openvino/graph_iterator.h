#pragma once

#include "openvino/frontend/graph_iterator.hpp"

namespace ov {
namespace frontend {
namespace tensorflow { // To be Removed
namespace ggml {

// TODO: Directly include from openvino
class GgmlGraphIterator : public GraphIterator {
public:

    virtual size_t size() const = 0;

    virtual void reset() = 0;

    virtual void next() = 0;

    virtual bool is_end() const = 0;

    virtual std::shared_ptr<DecoderBase> get_decoder() const = 0;

    virtual std::vector<std::string> get_input_names() const = 0;

    virtual std::vector<std::string> get_output_names() const = 0;

    virtual std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string& func_name) const = 0;

    virtual std::map<std::string, std::string> get_input_names_map() const {
        return {};
    }

    virtual std::map<std::string, std::string> get_output_names_map() const {
        return {};
    }
    
};

}
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
