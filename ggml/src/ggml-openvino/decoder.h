#pragma once

#include "openvino/core/node.hpp"
#include "openvino/frontend/decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

// TODO: Directly include from openvino
class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(size_t index) const = 0;

    virtual element::Type get_input_type(size_t index) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual bool is_graph_input(size_t index) const = 0;

    virtual std::string& get_input_name(size_t index) const = 0;

    virtual PartialShape get_output_shape(size_t index) const = 0;

    virtual element::Type get_output_type(size_t index) const = 0;

    virtual size_t get_output_size() const = 0;

    virtual bool is_graph_output(size_t index) const = 0;

    virtual int32_t* get_output_op_params(size_t index) const = 0;

    virtual std::string& get_output_name(size_t index) const = 0;

    virtual size_t get_output_size() const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_name() const = 0;

    // virtual const std::vector<size_t>& outputs() const = 0;

    // virtual size_t output(size_t index) const = 0;

};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
