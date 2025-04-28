#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "decoder.h"
#include "ggml.h"

class GgmlOvDecoder : public ov::frontend::ggml::GgmlDecoder {
public:
    using ov::frontend::ggml::GgmlDecoder::GgmlDecoder;

    GgmlOvDecoder(struct ggml_tensor* node, struct ggml_cgraph* cgraph);

    virtual ov::Any get_attribute(const std::string& name) const override {
        return nullptr;
        GGML_UNUSED(name);
    }

    virtual ov::PartialShape get_input_shape(const std::string& name) const override;

    virtual std::vector<size_t> get_input_stride(const std::string& name) const override;

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

    virtual std::vector<size_t> get_output_stride(const std::string& name) const override;

    virtual ov::element::Type get_output_type(const std::string& name) const override;

    virtual int32_t* get_input_op_params(const std::string& name) const override;

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

    virtual bool check_if_continuous() const override {
        return m_continuous;
    }

    virtual const std::unordered_map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const override {
        return m_model_inputs;
    }
    virtual const std::unordered_map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const override {
        return m_model_weights;
    }
    virtual const std::vector<std::string>& get_model_output_names() const override {
        return m_model_output_names;
    }

private:
    void set_input_output(ggml_tensor* node);
    static void dump_cgraph(const struct ggml_cgraph* cgraph);
    static std::vector<size_t> get_shape(const ggml_tensor* tensor);
    static std::vector<size_t> get_stride(const ggml_tensor* tensor);
    static ov::element::Type get_ov_type(const ggml_tensor* tensor);
    static std::shared_ptr<ov::Node> create_weight_node(ggml_tensor* tensor);

    struct ggml_cgraph * m_cgraph;
    std::map<std::string, ggml_tensor *> m_inputs;
    std::vector<std::string> m_input_names;
    std::map<std::string, ggml_tensor *> m_outputs;
    std::vector<std::string> m_output_names;
    ggml_tensor* m_node;
    std::vector<ggml_tensor*> m_nodes;
    std::string m_op_name;
    mutable std::string m_name;
    bool m_continuous;
    std::vector<std::pair<std::string, std::string>> m_op_node_name;
    std::unordered_map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::unordered_map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::vector<std::string> m_model_output_names;
};
