#pragma once

#include "ggml-quants.hpp"
#include "ggml.h"
#include "openvino/decoder.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <openvino/core/partial_shape.hpp>
#include <optional>
#include <vector>

class GgmlOvDecoder : public ov::frontend::ggml::GgmlDecoder {
public:
    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                  bool is_static);

    // Node decoder, called in GgmlOvDecoder::visit_subgraph
    GgmlOvDecoder(ggml_tensor * node,
                  ggml_cgraph * cgraph,
                  bool is_static,
                  int context_size,
                  int context_size_swa,
                  int num_heads,
                  int num_heads_kv,
                  int head_size,
                  const std::vector<int> & swa_layers);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights);

    virtual ov::Any get_attribute(const std::string & name) const override {
        return nullptr;
        GGML_UNUSED(name);
    }

    virtual ov::PartialShape get_input_shape(const std::string & name) const override;

    virtual std::vector<size_t> get_input_stride(const std::string & name) const override;

    virtual ov::element::Type get_input_type(const std::string & name) const override;

    virtual size_t get_input_size() const override;

    virtual void get_input_node(size_t input_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) const override {
        GGML_UNUSED(input_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    virtual std::string & get_input_name(size_t index) const override;

    virtual std::vector<std::string> get_input_names() const override;

    virtual ov::PartialShape get_output_shape(const std::string & name) const override;

    virtual std::vector<size_t> get_output_stride(const std::string & name) const override;

    virtual ov::element::Type get_output_type(const std::string & name) const override;

    virtual int32_t * get_input_op_params(const std::string & name) const override;

    virtual int32_t * get_output_op_params(const std::string & name) const override;

    virtual std::string & get_output_name(size_t index) const override;

    virtual std::vector<std::string> get_output_names() const override;

    virtual const std::string & get_op_type() const override;

    virtual const std::string & get_op_name() const override;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>)> node_visitor) const override;

    ggml_tensor * get_input_ggml_tensor(const std::string & name) const { return m_inputs.at(name); }

    ggml_tensor * get_output_ggml_tensor(const std::string & name) const { return m_outputs.at(name); }

    virtual int get_op_case() const override { return m_op_case; }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_inputs() const override {
        return m_model_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_extra_inputs() const override {
        return m_model_extra_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Tensor>> & get_model_extra_input_values() const {
        return m_model_extra_input_values;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_weights() const override {
        return m_model_weights;
    }

    virtual const std::vector<std::string> & get_model_output_names() const override { return m_model_output_names; }

    virtual int get_ctx_size() const { return m_ctx; }

    virtual int get_ctx_swa_size() const { return m_ctx_swa; }

    virtual int get_ctx_per_seq() const { return m_ctx_per_seq; }

    virtual int get_ctx_per_seq_swa() const { return m_ctx_per_seq_swa; }

    virtual int get_n_seq() const { return m_n_seq; }

    virtual int is_swa_layer(int layer) const override {
        return std::find(m_swa_layers.begin(), m_swa_layers.end(), layer) != m_swa_layers.end();
    }

    int get_past_kv_len() const { return m_past_kv_len; }

    int get_input_len() const { return m_input_len; }

    virtual int32_t * get_rope_params() const override { return m_rope_params; }

    virtual std::map<std::string, std::string> get_kv_param_res_names() const override;

    virtual bool is_static() const override { return m_is_static; }

    ov::PartialShape get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const;

    static void dump_cgraph(const ggml_cgraph * cgraph, std::string & filename);

    static std::shared_ptr<ov::Node> create_weight_node(ggml_tensor * tensor,
                                                        std::optional<ExtraQuantType> requant_type = std::nullopt);

    static std::map<std::string, std::shared_ptr<ov::Node>> create_weight_nodes(
        ggml_cgraph * cgraph,
        std::map<ggml_type, ExtraQuantType> types_to_requantize = {});

    const ggml_tensor * get_tensor_used_op(const ggml_tensor * tensor) const;

    const ggml_tensor * get_tensor_from_name(const std::string & name) const;

    void clear_model_weights() { m_model_weights.clear(); }

private:
    void set_input_output(ggml_tensor * node, bool naive = false);
    void add_extra_inputs();
    static std::vector<size_t> get_shape(const ggml_tensor * tensor);
    static std::vector<size_t> get_stride(const ggml_tensor * tensor);
    static ov::element::Type get_ov_type(const ggml_tensor * tensor);

    void set_llm_params();
    void validate_cgraph() const;

    bool m_is_static = false;

    ggml_cgraph * m_cgraph = nullptr;
    ggml_tensor * m_node = nullptr;
    std::vector<ggml_tensor *> m_nodes;
    std::map<std::string, ggml_tensor *> m_inputs;
    std::vector<std::string> m_input_names;
    std::map<std::string, ggml_tensor *> m_outputs;
    std::vector<std::string> m_output_names;
    std::string m_op_name;
    mutable std::string m_name;
    int m_op_case = 0;
    std::vector<std::pair<std::string, std::string>> m_op_node_name;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::Tensor>> m_model_extra_input_values;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::vector<std::string> m_model_output_names;

    // Fixed for a model
    int m_ctx = -1;
    int m_ctx_swa = -1;
    int m_ctx_per_seq = -1;
    int m_ctx_per_seq_swa = -1;
    int m_n_seq = -1;
    int m_n_heads = -1;
    int m_n_heads_kv = -1;
    int m_head_size = -1;
    std::vector<int> m_swa_layers;
    std::vector<std::string> m_kv_names;

    // Changed per inference
    int m_n_seq_active = -1;
    int m_seq_active_start = -1;
    int m_attention_size = -1;
    int m_attention_size_swa = -1;
    int m_input_len = -1;
    int m_token_len_per_seq = -1;
    int m_past_kv_len = -1;
    int32_t * m_rope_params = nullptr;
};

void print_tensor_address_map(const ggml_cgraph * cgraph);

int extract_layer_from_name(const std::string & name);
