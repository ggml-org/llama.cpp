#pragma once

#include "decoder.h"

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                int node_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_node_idx(node_idx),
        m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession * get_translate_session() const { return m_translate_session; }

    const std::vector<std::string> & get_input_names() const { return m_input_names; }

    size_t get_input_size() const override { return m_decoder->get_input_size(m_node_idx); }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t * get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    size_t get_view_input_size(size_t index) const {
        return m_decoder->get_view_input_size(m_node_idx, m_input_names[index]);
    }

    size_t get_view_input_offset(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_offset(m_node_idx, m_input_names[index], view_index);
    }

    size_t get_view_input_src_offset(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_src_offset(m_node_idx, m_input_names[index], view_index);
    }

    std::vector<size_t> get_view_input_stride(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_stride(m_node_idx, m_input_names[index], view_index);
    }

    std::vector<size_t> get_view_input_src_stride(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_src_stride(m_node_idx, m_input_names[index], view_index);
    }

    ov::Shape get_view_input_ggml_shape(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_ggml_shape(m_node_idx, m_input_names[index], view_index);
    }

    ov::Shape get_view_input_src_ggml_shape(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_src_ggml_shape(m_node_idx, m_input_names[index], view_index);
    }

    ov::PartialShape get_view_input_ov_shape(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_ov_shape(m_node_idx, m_input_names[index], view_index);
    }

    ov::PartialShape get_view_input_src_ov_shape(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_src_ov_shape(m_node_idx, m_input_names[index], view_index);
    }

    std::string get_view_input_name(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_name(m_node_idx, m_input_names[index], view_index);
    }

    std::string get_view_input_src_name(size_t index, size_t view_index) const {
        return m_decoder->get_view_input_src_name(m_node_idx, m_input_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() const { return m_decoder->get_op_dynamic_dim(m_node_idx); }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    size_t get_output_op_offset() const { return m_decoder->get_output_op_offset(m_node_idx); }

    ov::element::Type get_output_type() const { return m_decoder->get_output_type(m_node_idx); }

    std::vector<size_t> get_output_stride() const { return m_decoder->get_output_stride(m_node_idx); }

    Output<Node> get_input(int idx) const override {
        // Check if this input is a VIEW
        size_t view_input_size = m_decoder->get_view_input_size(m_node_idx, m_input_names[idx]);
        if (view_input_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_input_src_name(m_node_idx, m_input_names[idx], view_input_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_input_names[idx]);
            if (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                if (base_it != m_tensor_map->end() &&
                    view_it->second.get_node_shared_ptr() != base_it->second.get_node_shared_ptr()) {
                    return view_it->second;
                }
                return base_it->second;
            }
            if (!base_name.empty()) {
                return m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string & name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string & name) const { return m_tensor_map->find(name) != m_tensor_map->end(); }

    const std::string & get_name() const override { return m_decoder->get_op_name(m_node_idx); }

    ov::Any get_attribute_as_any(const std::string & name) const override { return m_decoder->get_attribute(name); }

    int get_op_case() const { return m_decoder->get_op_case(m_node_idx); }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    int m_node_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
