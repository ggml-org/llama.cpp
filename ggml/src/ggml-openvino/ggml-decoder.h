#pragma once

#include "decoder.h"
#include "ggml.h"

class GgmlOvDecoder : public ov::frontend::ggml::GgmlDecoder {
public:
    using ov::frontend::ggml::GgmlDecoder::GgmlDecoder;
    GgmlOvDecoder(struct ggml_tensor * node, struct ggml_cgraph * cgraph);

    virtual ov::Any get_attribute(const std::string& name) const override {
        return nullptr;
        GGML_UNUSED(name);
    }

    virtual ov::PartialShape get_input_shape(size_t index) const override;

    virtual ov::element::Type get_input_type(size_t index) const override;

    virtual size_t get_input_size() const override;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const override {
        GGML_UNUSED(input_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    virtual bool is_graph_input(size_t index) const override;

    virtual std::string& get_input_name(size_t index) const override;

    virtual ov::PartialShape get_output_shape(size_t index) const override;

    virtual ov::element::Type get_output_type(size_t index) const override;

    virtual size_t get_output_size() const override; 

    virtual bool is_graph_output(size_t index) const override;

    virtual int32_t* get_output_op_params(size_t index) const override;

    virtual std::string& get_output_name(size_t index) const override;

    virtual size_t get_output_size() const override; 

    virtual bool is_graph_output(size_t index) const override;

    virtual std::string& get_output_name(size_t index) const override;

    virtual const std::string& get_op_type() const override;

    virtual const std::string& get_op_name() const override;

    const ggml_tensor* get_input_ggml_tensor(size_t index) const {
        return m_inputs[index];
    }

    // virtual const std::vector<size_t>& outputs() const override;

    // virtual size_t output(size_t index) const override;

private:
    size_t m_index;
    struct ggml_cgraph * m_cgraph;
    std::vector<ggml_tensor *> m_inputs;
    std::vector<ggml_tensor *> m_outputs;
    ggml_tensor * m_node;
    const std::string m_op_name;
    mutable std::string m_name;
};

