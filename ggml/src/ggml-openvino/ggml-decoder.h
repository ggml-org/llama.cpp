#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "ggml.h"
#include "openvino/decoder.hpp"

class GgmlOvDecoder : public ov::frontend::ggml::GgmlDecoder {
public:
    using ov::frontend::ggml::GgmlDecoder::GgmlDecoder;

    GgmlOvDecoder(struct ggml_tensor* node, struct ggml_cgraph* cgraph, bool is_static, bool is_first_token);

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

    virtual std::vector<std::string> get_output_names() const override;

    virtual const std::string& get_op_type() const override;

    virtual const std::string& get_op_name() const override;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>)> node_visitor) const override;

    const ggml_tensor* get_input_ggml_tensor(const std::string& name) const {
        return m_inputs.at(name);
    }

    const ggml_tensor* get_output_ggml_tensor(const std::string& name) const {
        return m_outputs.at(name);
    }

    virtual int get_op_case() const override {
        return m_op_case;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const override {
        return m_model_inputs;
    }
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const override {
        return m_model_extra_inputs;
    }
    virtual const std::map<std::string, std::shared_ptr<ov::Tensor>>& get_model_extra_input_values() const {
        return m_model_extra_input_values;
    }
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const override {
        return m_model_weights;
    }
    virtual const std::vector<std::string>& get_model_output_names() const override {
        return m_model_output_names;
    }

    virtual bool is_static() const override {
        return m_is_static;
    }
    virtual bool is_first_token() const override {
        return m_is_first_token;
    }
    virtual int get_max_token_len() const override {
        return m_max_token_len;
    }

private:
    void set_input_output(ggml_tensor* node);
    void add_extra_inputs();
    static void dump_cgraph(const struct ggml_cgraph* cgraph);
    static std::vector<size_t> get_shape(const ggml_tensor* tensor);
    static std::vector<size_t> get_stride(const ggml_tensor* tensor);
    static ov::element::Type get_ov_type(const ggml_tensor* tensor);
    static std::shared_ptr<ov::Node> create_weight_node(ggml_tensor* tensor);

    void set_max_token_len();
    int m_max_token_len;

    void add_weight_const_parallel(std::map<std::string, std::shared_ptr<ov::Node>>& model_weights);

    struct ggml_cgraph* m_cgraph;
    std::map<std::string, ggml_tensor*> m_inputs;
    std::vector<std::string> m_input_names;
    std::map<std::string, ggml_tensor*> m_outputs;
    std::vector<std::string> m_output_names;
    ggml_tensor* m_node;
    std::vector<ggml_tensor*> m_nodes;
    std::string m_op_name;
    mutable std::string m_name;
    int m_op_case;
    std::vector<std::pair<std::string, std::string>> m_op_node_name;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::Tensor>> m_model_extra_input_values;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::vector<std::string> m_model_output_names;
    bool m_is_static;
    bool m_is_first_token;
};

void print_tensor_address_map(const struct ggml_cgraph* cgraph);
