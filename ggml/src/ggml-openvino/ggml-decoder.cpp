#include "ggml-decoder.h"

#include <ggml-impl.h>
#include <ggml.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <ostream>
#include <set>
#include <string>

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

GgmlOvDecoder::GgmlOvDecoder(struct ggml_tensor* node, struct ggml_cgraph* cgraph)
    : m_cgraph(cgraph),
      m_node(node),
      m_op_name(m_node ? std::string(m_node->name) : "NONE_OP") {
    static std::map<std::string, std::shared_ptr<ov::Node>> model_weights;

    if (m_node) {
        set_input_output(m_node, model_weights);
    } else {
        static bool printed = false;
        if (!printed && getenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            print_tensor_address_map(m_cgraph);
            printed = true;
        }

        for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
            auto* cur_node = m_cgraph->nodes[node_n];
            m_nodes.push_back(cur_node);
            set_input_output(cur_node, model_weights);
        }
        m_model_weights = model_weights;

        if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
            dump_cgraph(m_cgraph);
        }
    }
}

// Called in GgmlOvDecoder constructor. Two cases: 1. constructing a decoder for the whole graph;
// 2. constructing a decoder for a node.
void GgmlOvDecoder::set_input_output(ggml_tensor* node,
                                     std::map<std::string, std::shared_ptr<ov::Node>>& model_weights) {
    std::string node_name;
    if (node->op == GGML_OP_CPY) {
        // CPY updates the input tensor in place. For later ov op that uses the
        // input tensor of CPY, we need to make sure they get the updated tensor
        // by putting the src tensor name in the tensor_map in
        // <openvino>/src/frontends/ggml/src/translate_session.cpp
        node_name = std::string(node->view_src->name);
    } else {
        node_name = std::string(node->name);
    }

    m_output_names.push_back(node_name);
    m_outputs[node_name] = node;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto* src = node->src[i];
        if (src == nullptr) {
            continue;
        }
        std::string src_name = std::string(src->name);
        m_input_names.push_back(src_name);
        m_inputs[src_name] = src;
        m_op_node_name.emplace_back(src_name, ggml_op_name(node->op));

        // If called for the whole graph, create constant nodes for weights and param nodes for inputs
        if (!m_node && !src->view_src) {
            ggml_backend_buffer* buffer = src->buffer;

            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                bool weight_as_input = getenv("GGML_OPENVINO_WEIGHT_AS_INPUT");
                auto& weights_map = weight_as_input ? m_model_inputs : model_weights;
                if (weights_map.find(src_name) != weights_map.end()) {
                    continue;
                }

                std::shared_ptr<ov::Node> weight_node =
                    weight_as_input
                        ? std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), ov::Shape{get_shape(src)})
                        : create_weight_node(src);
                weight_node->set_friendly_name(src_name);
                weights_map[src_name] = weight_node;

            } else if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY || src->flags & GGML_TENSOR_FLAG_INPUT) {
                // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
                if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                    assert(src_name.find("cache_k") == 0 || src_name.find("cache_v") == 0);
                }
                if (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                    continue;
                }
                auto param_node = std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), ov::Shape{get_shape(src)});
                param_node->set_friendly_name(src_name);
                m_model_inputs[src_name] = param_node;
            }
        }
    }

    if (!m_node) {
        static std::set<std::string> debug_output_names = {};
        // Workaround: the final tensor "result_output" does not have GGML_TENSOR_FLAG_OUTPUT flag set in cgraph
        if (node->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY || node->flags & GGML_TENSOR_FLAG_OUTPUT ||
            std::string(node->name).find("result") == 0 || debug_output_names.count(node->name)) {
            auto name = node->view_src ? std::string(node->view_src->name) : std::string(node->name);
            if (node->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                assert(name.find("cache_k") == 0 || name.find("cache_v") == 0);
            }
            auto it = std::find(m_model_output_names.begin(), m_model_output_names.end(), name);
            if (it == m_model_output_names.end()) {
                m_model_output_names.push_back(name);
            }
        }
    }

    if (m_node) {
        switch (node->op) {
        case GGML_OP_CONT: {
            // Currently only two cases, either the input comes from a VIEW which is subtensor or from a PERMUTE
            m_continuous = ggml_nelements(node->src[0]) == ggml_nelements(node->src[0]->view_src);
            break;
        }
        case GGML_OP_CPY: {
            m_continuous = ggml_is_contiguous(node);
            break;
        }
        case GGML_OP_MUL_MAT: {
            m_continuous = node->src[0]->view_src == nullptr;
            break;
        }
        default:
            break;
        }
    }
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_node(ggml_tensor* tensor) {
    std::shared_ptr<ov::Node> weight_node;
    auto node_type = get_ov_type(tensor);
    auto node_shape = get_shape(tensor);
    auto ne_total = ggml_nelements(tensor);
    switch (tensor->type) {
    case GGML_TYPE_I32: {
        const auto* ptr = reinterpret_cast<const int32_t*>(tensor->data);
        std::vector<int32_t> data(ptr, ptr + ne_total);
        weight_node = std::make_shared<ov::op::v0::Constant>(node_type, node_shape, data);
        break;
    }
    case GGML_TYPE_I64: {
        const auto* ptr = reinterpret_cast<const int64_t*>(tensor->data);
        std::vector<int64_t> data(ptr, ptr + ne_total);
        weight_node = std::make_shared<ov::op::v0::Constant>(node_type, node_shape, data);
        break;
    }
    case GGML_TYPE_F32: {
        const auto* ptr = reinterpret_cast<const float*>(tensor->data);
        std::vector<float> data(ptr, ptr + ne_total);
        weight_node = std::make_shared<ov::op::v0::Constant>(node_type, node_shape, data);
        break;
    }
    case GGML_TYPE_F16: {
        const auto* ptr = reinterpret_cast<const uint16_t*>(tensor->data);
        std::vector<ov::float16> data_f16;
        data_f16.reserve(ne_total);
        for (int i = 0; i < ne_total; ++i) {
            data_f16.push_back(ov::float16::from_bits(ptr[i]));
        }
        weight_node = std::make_shared<ov::op::v0::Constant>(node_type, node_shape, data_f16);
        break;
    }
    default:
        throw std::invalid_argument("Unsupported tensor type");
    }
    return weight_node;
}

void GgmlOvDecoder::dump_cgraph(const struct ggml_cgraph* cgraph) {
    std::ofstream file("cgraph.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    file << "=== GRAPH ===\n";

    // clang-format off
    file << "n_nodes = " << cgraph->n_nodes << "\n";
    file << " " << std::setw(3) << "nodes"
                <<  std::setw(15) << "shape"
                << std::setw(20) << "op"
                << std::setw(20) << "name"
                << std::setw(3) << "    "
                << std::setw(50) << "stride"
                << "\n";
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << ", "
             << std::setw(5) << node->ne[2] << ", "
             << std::setw(5) << node->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(node->op) << std::right << " "
             << std::left << std::setw(44) << node->name << std::right
             << ((node->flags & GGML_TENSOR_FLAG_PARAM) ? "x" : node->grad ? "g" : " ")
             << std::setw(2) << "[ "
             << std::setw(0) << node->nb[0] << ", "
             << std::setw(5) << node->nb[1] << ", "
             << std::setw(5) << node->nb[2] << ", "
             << std::setw(5) << node->nb[3] << "] "
             << "\n";

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (auto* src = node->src[i]) {
                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::left << std::setw(12) << ggml_op_name(src->op) << std::right;
                file << std::left << std::setw(30) << src->name << std::right
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * node = cgraph->leafs[i];

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << "] "
             << std::setw(8) << ggml_op_name(node->op) << " "
             << std::setw(16) << ggml_get_name(node) << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

void print_tensor_address_map(const struct ggml_cgraph* cgraph) {
    std::map<void*, std::vector<std::string>> address_map;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto* node = cgraph->nodes[node_n];
        if (node->data) {
            auto it = address_map.find(node->data);
            if (it == address_map.end()) {
                address_map[node->data] = std::vector<std::string>();
            }
            address_map[node->data].push_back(node->name);
        }
    }
    for (const auto& pair : address_map) {
        std::cout << "Address: " << pair.first << std::endl;
        for (const auto& name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << std::endl << std::endl;
    }
}

std::vector<size_t> GgmlOvDecoder::get_shape(const ggml_tensor* tensor) {
    std::vector<size_t> shape;
    for (int i = GGML_MAX_DIMS - 2; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(const ggml_tensor* tensor) {
    std::vector<size_t> stride;
    for (int i = GGML_MAX_DIMS - 2; i >= 0 ; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    return stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(const ggml_tensor* tensor) {
    ov::element::Type type = ov::element::dynamic;
    switch (tensor->type) {
    case GGML_TYPE_F32:
        type = ov::element::f32;
        break;
    case GGML_TYPE_F16:
        type = ov::element::f16;
        break;
    case GGML_TYPE_I64:
        type = ov::element::i64;
        break;
    case GGML_TYPE_I32:
        type = ov::element::i32;
        break;
    default:
        break;
    }
    return type;
}

ov::PartialShape GgmlOvDecoder::get_input_shape(const std::string& name) const {
    return ov::PartialShape(get_shape(m_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_input_stride(const std::string& name) const {
    return get_stride(m_inputs.at(name));
}

ov::element::Type GgmlOvDecoder::get_input_type(const std::string& name) const {
    return get_ov_type(m_inputs.at(name));
}

size_t GgmlOvDecoder::get_input_size() const {
    return m_input_names.size();
}

std::string& GgmlOvDecoder::get_input_name(size_t index) const {
    m_name = m_input_names[index];
    return m_name;
}

std::vector<std::string> GgmlOvDecoder::get_input_names() const {
    return m_input_names;
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(const std::string& name) const {
    return get_stride(m_outputs.at(name));
}

ov::PartialShape GgmlOvDecoder::get_output_shape(const std::string& name) const {
    return ov::PartialShape(get_shape(m_outputs.at(name)));
}

ov::element::Type GgmlOvDecoder::get_output_type(const std::string& name) const {
    return get_ov_type(m_outputs.at(name));
}

std::string& GgmlOvDecoder::get_output_name(size_t index) const {
    m_name = std::string(m_output_names[index]);
    return m_name;
}

std::vector<std::string> GgmlOvDecoder::get_output_names() const {
    return m_output_names;
}

const std::string& GgmlOvDecoder::get_op_name() const {
    return m_op_name;
}

int32_t* GgmlOvDecoder::get_input_op_params(const std::string& name) const {
    return m_inputs.at(name)->op_params;
}

int32_t* GgmlOvDecoder::get_output_op_params(const std::string& name) const {
    return m_outputs.at(name)->op_params;
}

void GgmlOvDecoder::visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>)> node_visitor) const {
    for (const auto& node : m_nodes) {
        auto decoder = std::make_shared<GgmlOvDecoder>(node, m_cgraph);
        node_visitor(decoder);
    }
}

const std::string& GgmlOvDecoder::get_op_type() const {
    static const std::map<ggml_op, std::string> opTypeMap = {
        {GGML_OP_ACC, "GGML_OP_ACC"},
        {GGML_OP_ADD, "GGML_OP_ADD"},
        {GGML_OP_ADD1, "GGML_OP_ADD1"},
        {GGML_OP_CONT, "GGML_OP_CONT"},
        {GGML_OP_CPY, "GGML_OP_CPY"},
        {GGML_OP_DIV, "GGML_OP_DIV"},
        {GGML_OP_DUP, "GGML_OP_DUP"},
        {GGML_OP_GET_ROWS, "GGML_OP_GET_ROWS"},
        {GGML_OP_MUL, "GGML_OP_MUL"},
        {GGML_OP_MUL_MAT, "GGML_OP_MUL_MAT"},
        {GGML_OP_PERMUTE, "GGML_OP_PERMUTE"},
        {GGML_OP_RESHAPE, "GGML_OP_RESHAPE"},
        {GGML_OP_RMS_NORM, "GGML_OP_RMS_NORM"},
        {GGML_OP_ROPE, "GGML_OP_ROPE"},
        {GGML_OP_SCALE, "GGML_OP_SCALE"},
        {GGML_OP_SOFT_MAX, "GGML_OP_SOFT_MAX"},
        {GGML_OP_SUB, "GGML_OP_SUB"},
        {GGML_OP_TRANSPOSE, "GGML_OP_TRANSPOSE"},
        {GGML_OP_UNARY, "GGML_OP_UNARY"},
        {GGML_OP_VIEW, "GGML_OP_VIEW"}
    };
    static const std::map<ggml_unary_op, std::string> unaryOpTypeMap = {
        {GGML_UNARY_OP_ABS, "GGML_UNARY_OP_ABS"},
        {GGML_UNARY_OP_SGN, "GGML_UNARY_OP_SGN"},
        {GGML_UNARY_OP_NEG, "GGML_UNARY_OP_NEG"},
        {GGML_UNARY_OP_STEP, "GGML_UNARY_OP_STEP"},
        {GGML_UNARY_OP_TANH, "GGML_UNARY_OP_TANH"},
        {GGML_UNARY_OP_ELU, "GGML_UNARY_OP_ELU"},
        {GGML_UNARY_OP_RELU, "GGML_UNARY_OP_RELU"},
        {GGML_UNARY_OP_SIGMOID, "GGML_UNARY_OP_SIGMOID"},
        {GGML_UNARY_OP_GELU, "GGML_UNARY_OP_GELU"},
        {GGML_UNARY_OP_GELU_QUICK, "GGML_UNARY_OP_GELU_QUICK"},
        {GGML_UNARY_OP_SILU, "GGML_UNARY_OP_SILU"},
        {GGML_UNARY_OP_HARDSWISH, "GGML_UNARY_OP_HARDSWISH"},
        {GGML_UNARY_OP_HARDSIGMOID, "GGML_UNARY_OP_HARDSIGMOID"},
        {GGML_UNARY_OP_EXP, "GGML_UNARY_OP_EXP"},
        {GGML_UNARY_OP_COUNT, "GGML_UNARY_OP_COUNT"}
    };
    auto it = opTypeMap.find(m_node->op);
    if (it != opTypeMap.end()) {
        if (it->first == GGML_OP_UNARY) {
            auto unary_it = unaryOpTypeMap.find(ggml_get_unary_op(m_node));
            if (unary_it != unaryOpTypeMap.end()) {
                return unary_it->second;
            }
        }
        return it->second;
    }
    static const std::string unknown_op = "UNKNOWN_OP";
    return unknown_op;
}
