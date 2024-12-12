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

    virtual ov::PartialShape get_input_shape(const std::string& name) const override;

    virtual ov::element::Type get_input_type(const std::string& name) const override;

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

    virtual std::string& get_input_name(size_t index) const override;

    virtual std::vector<std::string> get_input_names() const override;

    virtual ov::PartialShape get_output_shape(const std::string& name) const override;

    virtual ov::element::Type get_output_type(const std::string& name) const override;

    virtual int32_t* get_output_op_params(const std::string& name) const override;

    virtual std::string& get_output_name(size_t index) const override;

    virtual size_t get_output_size() const override; 

    virtual bool is_graph_output(size_t index) const override;

    virtual std::string& get_output_name(size_t index) const override;

    virtual const std::string& get_op_type() const override;

    virtual const std::string& get_op_name() const override;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>)> node_visitor) const override;

    const ggml_tensor* get_input_ggml_tensor(std::string& name) const {
        return m_inputs.at(name);
    }

    const ggml_tensor* get_output_ggml_tensor(std::string& name) const {
        return m_outputs.at(name);
    }

private:
    void set_input_output(ggml_tensor* node, std::map<std::string, ggml_tensor *>& inputs, std::map<std::string, ggml_tensor *>& outputs);

    struct ggml_cgraph * m_cgraph;
    std::map<std::string, ggml_tensor *> m_inputs;
    std::vector<std::string> m_input_names;
    std::map<std::string, ggml_tensor *> m_outputs;
    std::vector<std::string> m_output_names;
    ggml_tensor* m_node;
    std::vector<ggml_tensor *> m_nodes;
    std::vector<std::shared_ptr<GgmlOvDecoder>> m_decoders;
    const std::string m_op_name;
    mutable std::string m_name;
};

