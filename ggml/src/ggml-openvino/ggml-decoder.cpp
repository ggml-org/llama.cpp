#include "ggml-decoder.h"
#include <ggml.h>
#include <ggml-impl.h>
#include <ggml-cpu-impl.h>
#include <iomanip>
#include <fstream>

void GgmlOvDecoder::set_input_output(ggml_tensor* node, std::map<std::string, ggml_tensor *>& inputs, std::map<std::string, ggml_tensor *>& outputs) {
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

    std::string src0_name = std::string(node->src[0]->name);
    inputs[src0_name] = node->src[0];
    outputs[node_name] = node;
    m_input_names.push_back(src0_name);
    m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
    if (node->op == GGML_OP_CPY && node->view_src) {
        m_output_names.push_back(node->view_src->name);
    } else {
        m_output_names.push_back(node_name);
    }

    if (node->src[1]) {
        std::string src1_name = std::string(node->src[1]->name);
        inputs[src1_name] = node->src[1];
        m_input_names.push_back(src1_name);
        m_op_node_name.emplace_back(src1_name, ggml_op_name(node->op));
    }
    if (node->src[2]) {
        std::string src2_name = std::string(node->src[2]->name);
        inputs[src2_name] = node->src[2];
        m_input_names.push_back(src2_name);
        m_op_node_name.emplace_back(src2_name, ggml_op_name(node->op));
    }

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

void ggml_graph_op_print(const struct ggml_cgraph * cgraph) {
    std::ofstream file("01_nodes.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    file << "=== GRAPH ===\n";

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

        if (node->src[0]) {
            file << std::setw(10) << " [ "
            << std::setw(5) << node->src[0]->ne[0] << ", "
            << std::setw(5) << node->src[0]->ne[1] << ", "
            << std::setw(5) << node->src[0]->ne[2] << ", "
            << std::setw(5) << node->src[0]->ne[3] << "] "
            << std::setw(12)
            << "0: " << std::left << std::setw(12) << ggml_op_name(node->src[0]->op) << std::right;
            // // Custom logic to handle '\000'
            // const char* name_ptr = node->src[0]->name;
            // while (*name_ptr != '\0' || *(name_ptr + 1) != '\0' || *(name_ptr + 2) != '\0') {
            //     file << *name_ptr;
            //     name_ptr++;
            // }
            file << std::left << std::setw(30) << node->src[0]->name << std::right
            << std::setw(16) << "[ "
            << std::setw(0) << node->src[0]->nb[0] << ", "
            << std::setw(5) << node->src[0]->nb[1] << ", "
            << std::setw(5) << node->src[0]->nb[2] << ", "
            << std::setw(5) << node->src[0]->nb[3] << "] "
            << "\n";
        }
        if (node->src[1]) {
            file << std::setw(10) << " [ "
            << std::setw(5) << node->src[1]->ne[0] << ", "
            << std::setw(5) << node->src[1]->ne[1] << ", "
            << std::setw(5) << node->src[1]->ne[2] << ", "
            << std::setw(5) << node->src[1]->ne[3] << "] "
            << std::setw(12)
            << "1: " << std::left << std::setw(12) << ggml_op_name(node->src[1]->op) << std::right;
            // // Custom logic to handle '\000'
            // const char* name_ptr = node->src[1]->name;
            // while (*name_ptr != '\0' || *(name_ptr + 1) != '\0' || *(name_ptr + 2) != '\0') {
            //     file << *name_ptr;
            //     name_ptr++;
            // }
            file << std::left << std::setw(30) << node->src[1]->name << std::right
            << std::setw(16) << "[ "
            << std::setw(0) << node->src[1]->nb[0] << ", "
            << std::setw(5) << node->src[1]->nb[1] << ", "
            << std::setw(5) << node->src[1]->nb[2] << ", "
            << std::setw(5) << node->src[1]->nb[3] << "] "
            << "\n";
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

    file << "========================================\n";

    file.close();
}

GgmlOvDecoder::GgmlOvDecoder(struct ggml_tensor * node, struct ggml_cgraph * cgraph, const int32_t start_index, const int32_t end_index)
                            :m_cgraph(cgraph),
                             m_node(node),
                             m_op_name(m_node ? std::string(m_node->name) : "NONE_OP") {
    m_inputs.clear();
    m_outputs.clear();
    m_input_names.clear();
    m_output_names.clear();
    m_params.clear();
    m_op_node_name.clear();
    m_decoders.clear();

    if (m_node) {
        set_input_output(m_node, m_inputs, m_outputs);
    } else {
        // for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        for (int node_n = start_index; node_n <= end_index; node_n++) {
            auto cur_node = m_cgraph->nodes[node_n];
            m_nodes.push_back(cur_node);
            // Init model input and output
            set_input_output(cur_node, m_inputs, m_outputs);
        }
        if (getenv("GGML_OPENVINO_DEBUG")) {
          ggml_graph_op_print(m_cgraph);
        }
    }
}

ov::PartialShape GgmlOvDecoder::get_input_shape(const std::string& name) const {
    ov::PartialShape input_shape;
    // Use input_node->ne
    ggml_tensor * node = m_inputs.at(name);
    std::vector<size_t> shape;

    for (int i = GGML_MAX_DIMS - 2; i >= 0 ; --i) {
        if (node->ne[i] == 0) {
            return input_shape;
        }
        shape.push_back(static_cast<size_t>(node->ne[i]));
    }
    input_shape = ov::PartialShape(shape);
    return input_shape;
}

std::vector<size_t> GgmlOvDecoder::get_input_stride(const std::string& name) const {
    std::vector<size_t> stride;
    ggml_tensor * node = m_inputs.at(name);
    for (int i = GGML_MAX_DIMS - 2; i >= 0 ; --i) {
        stride.push_back(static_cast<size_t>(node->nb[i]));
    }
    return stride;
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(const std::string& name) const {
    std::vector<size_t> stride;
    ggml_tensor * node = m_outputs.at(name);
    for (int i = GGML_MAX_DIMS - 2; i >= 0 ; --i) {
        stride.push_back(static_cast<size_t>(node->nb[i]));
    }
    return stride;
}

ov::element::Type GgmlOvDecoder::get_input_type(const std::string& name) const {
    ov::element::Type type = ov::element::dynamic;
    switch (m_inputs.at(name)->type) {
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

std::string& GgmlOvDecoder::get_op_node_name(const std::string& key_name, const int index) {
    if (index == -1) {
        for (size_t i = 0; i < m_op_node_name.size(); ++i) {
            if (m_op_node_name[i].first == key_name) {
                return m_op_node_name[i].second;
            }
        }
    } else {
        return m_op_node_name[index].second;
    }

    static std::string empty_string = "";
    return empty_string; // empty string
}

const std::vector<std::shared_ptr<ov::op::v0::Parameter>>& GgmlOvDecoder::get_params() const {
    return m_params;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(const std::string& name) const {
    ov::PartialShape output_shape;
    ggml_tensor * node = m_outputs.at(name);
    std::vector<size_t> shape;

    for (int i = GGML_MAX_DIMS - 2; i >= 0 ; --i) {
        if (node->ne[i] == 0 ) {
            // empty if any dimension has no elements
            return output_shape;
        }
        shape.push_back(static_cast<size_t>(node->ne[i]));
    }
    output_shape = ov::PartialShape(shape);
    return output_shape;
}

ov::element::Type GgmlOvDecoder::get_output_type(const std::string& name) const {
    // TODO: Change to Output
    ov::element::Type type = ov::element::dynamic;
    switch (m_outputs.at(name)->type) {
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

int32_t* GgmlOvDecoder::get_input_op_params(const std::string& name) const{
    return m_inputs.at(name)->op_params;
}

int32_t* GgmlOvDecoder::get_output_op_params(const std::string& name) const{
    return m_outputs.at(name)->op_params;
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

void GgmlOvDecoder::visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>)> node_visitor) const {
    for (const auto& node : m_nodes) {
        auto decoder = std::make_shared<GgmlOvDecoder>(node, m_cgraph);
        // m_decoders.push_back(decoder);
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
