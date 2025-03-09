#include "ggml-decoder.h"
#include <ggml.h>
#include <ggml-impl.h>
#include <ggml-cpu-impl.h>
#include <iomanip>
#include <fstream>

void GgmlOvDecoder::set_input_output(ggml_tensor* node, std::map<std::string, ggml_tensor *>& inputs, std::map<std::string, ggml_tensor *>& outputs) {
    // m_node_op_name[node->name] = ggml_op_name(node->op);

    // std::string src0_name = std::string(node->src[0]->name) + "_" + std::to_string(node->src[0]->view_offs) + "_input_" + ggml_op_name(node->src[0]->op);
    // std::string node_name = std::string(node->name) + "_" + std::to_string(node->view_offs) + "_output_" + ggml_op_name(node->op);

    // Execute singel CONT operator is OK
    // std::string src0_name = std::string(node->src[0]->name) + "_" + std::to_string(node->src[0]->view_offs) + "_" + ggml_op_name(node->src[0]->op);
    // std::string node_name = std::string(node->name) + "_" + std::to_string(node->view_offs) + "_" + ggml_op_name(node->op);

    // std::string src0_name = std::string(node->src[0]->name) + "_" + std::to_string(node->src[0]->view_offs);
    // std::string node_name = std::string(node->name) + "_" + std::to_string(node->view_offs);

    std::string src0_name = std::string(node->src[0]->name);
    std::string node_name = std::string(node->name);

    switch (node->op) {
        // Unary OPs 
        case GGML_OP_UNARY:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_RMS_NORM:
        {
            inputs[src0_name] = node->src[0];
            outputs[node_name] = node;
            m_input_names.push_back(src0_name);
            m_node_op_name[src0_name] = ggml_op_name(node->op);
            m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
            m_output_names.push_back(node_name);
            break;
        }
        case GGML_OP_CONT:
        {
            if (ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node)) {
                inputs[src0_name] = node->src[0];
                outputs[node_name] = node;
                m_input_names.push_back(src0_name);
                m_node_op_name[src0_name] = ggml_op_name(node->op);
                m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
                m_output_names.push_back(node_name);

                ov::Shape input_shape = { static_cast<size_t>(node->src[0]->ne[2]),
                                            static_cast<size_t>(node->src[0]->ne[1]),
                                            static_cast<size_t>(node->src[0]->ne[0])};
                auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
                m_params.push_back(input_param);

                m_continuous = true;
                break;
            }

            if (node->src[0]->type == node->type && node->src[0]->ne[0] == node->ne[0] &&
                node->src[0]->nb[0] == ggml_type_size(node->src[0]->type) &&
                node->nb[0] == ggml_type_size(node->src[0]->type)) {

                inputs[src0_name] = node->src[0];
                outputs[node_name] = node;
                m_input_names.push_back(src0_name);
                m_node_op_name[src0_name] = ggml_op_name(node->op);
                m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
                m_output_names.push_back(node_name);

                const size_t element_size = ggml_type_size(node->src[0]->type);
                size_t valid_elems = static_cast<size_t>(node->src[0]->ne[0]); // 3072
                size_t num_rows    = static_cast<size_t>(node->src[0]->ne[1]); // 7
                size_t dim2        = static_cast<size_t>(node->src[0]->ne[2]);  // 1
                size_t phys_stride = static_cast<size_t>(node->src[0]->nb[1]) / element_size; // 9216
                // size_t total_phys = (num_rows - 1) * phys_stride + valid_elems; // 6*9216 + 3072 = 58368
                size_t total_phys = num_rows * phys_stride; // 7 * 9216 = 64512
                ov::Shape input_shape = { dim2, num_rows, phys_stride };
                auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
                m_params.push_back(input_param);

                m_continuous = false;
                break;
            }

            if (ggml_is_contiguous(node)) {
                inputs[src0_name] = node->src[0];
                outputs[node_name] = node;
                m_input_names.push_back(src0_name);
                m_node_op_name[src0_name] = ggml_op_name(node->op);
                m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
                m_output_names.push_back(node_name);

                ov::Shape input_shape = { static_cast<size_t>(node->src[0]->ne[2]),
                                            static_cast<size_t>(node->src[0]->ne[1]),
                                            static_cast<size_t>(node->src[0]->ne[0])};
                auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
                m_params.push_back(input_param);

                m_continuous = false;
                break;
            }
        }
        case GGML_OP_CPY:
        {
            if (ggml_is_contiguous(node)) {
                inputs[src0_name] = node->src[0];
                outputs[node_name] = node;
                m_input_names.push_back(src0_name);
                m_node_op_name[src0_name] = ggml_op_name(node->op);
                m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
                m_output_names.push_back(node_name);
                m_continuous = true;

                // ov::Shape src_shape(node->src[0]->ne, node->src[0]->ne + 3);
                // auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, src_shape);
                // m_params.push_back(input_param);
                break;
            } else {
                for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {       // ne[1] = 3072
                    for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {   // ne[0] = 7
                        int64_t src_index = i0 * node->src[0]->nb[0] / sizeof(float) +  // stride in nb[0]
                                            i1 * node->src[0]->nb[1] / sizeof(float);   // stride in nb[1]
                        char *dst_ptr = static_cast<char *>(node->data) +
                                i0 * node->nb[0] + i1 * node->nb[1];
                        *(ggml_fp16_t *)dst_ptr = GGML_FP32_TO_FP16(((float*)node->src[0]->data)[src_index]);
                    }
                }
                // inputs[node->src[0]->name] = node->src[0];
                inputs[node_name] = node;
                outputs[node_name] = node;
                m_input_names.push_back(node_name);
                m_node_op_name[node_name] = ggml_op_name(node->op);
                m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
                m_output_names.push_back(node_name);
                m_continuous = false;

                break;
            }
        }
        // For view, input is node itself
        case GGML_OP_VIEW:
        {
            inputs[node_name] = node;
            outputs[node_name] = node;
            m_input_names.push_back(node_name);
            m_node_op_name[node_name] = ggml_op_name(node->op);
            m_op_node_name.emplace_back(node_name, ggml_op_name(node->op));
            m_output_names.push_back(node_name);
            break;
        }
        // SCALE
        case GGML_OP_SCALE:
        {
            inputs[src0_name] = node->src[0];
            outputs[node_name] = node;
            m_input_names.push_back(src0_name);
            m_node_op_name[src0_name] = ggml_op_name(node->op);
            m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
            m_output_names.push_back(node_name);
            break;
        }
        case GGML_OP_MUL_MAT:
        {
            if (!ggml_is_contiguous(node->src[1]) || node->src[1]->ne[0] * node->src[1]->nb[0] != node->src[1]->nb[1]) {
                m_continuous = false;
            } else {
                m_continuous = true;
            }
            // std::string src1_name = std::string(node->src[1]->name) + "_" + std::to_string(node->src[1]->view_offs) + "_input_" + ggml_op_name(node->src[1]->op);
            // std::string src1_name = std::string(node->src[1]->name) + "_" + std::to_string(node->src[1]->view_offs);
            std::string src1_name = std::string(node->src[1]->name);
            inputs[src0_name] = node->src[0];
            inputs[src1_name] = node->src[1];
            outputs[node_name] = node;
            m_input_names.push_back(src0_name);
            m_node_op_name[src0_name] = ggml_op_name(node->op);
            m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
            m_input_names.push_back(src1_name);
            m_node_op_name[src1_name] = ggml_op_name(node->op);
            m_op_node_name.emplace_back(src1_name, ggml_op_name(node->op));
            m_output_names.push_back(node_name);
            break;
        }
        // OPs with 2 inputs
        case GGML_OP_ADD:
        case GGML_OP_DIV:
        case GGML_OP_MUL:
        case GGML_OP_SUB:        
        case GGML_OP_GET_ROWS:
        case GGML_OP_SOFT_MAX:
        {
            inputs[src0_name] = node->src[0];
            outputs[node_name] = node;
            m_input_names.push_back(src0_name);
            m_node_op_name[src0_name] = ggml_op_name(node->op);
            m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
            m_output_names.push_back(node_name);
            if (node->src[1]) {
                // std::string src1_name = std::string(node->src[1]->name) + "_" + std::to_string(node->src[1]->view_offs) + "_input_" + ggml_op_name(node->src[1]->op);
                // std::string src1_name = std::string(node->src[1]->name) + "_" + std::to_string(node->src[1]->view_offs);
                std::string src1_name = std::string(node->src[1]->name);
                inputs[src1_name] = node->src[1];
                m_node_op_name[src1_name] = ggml_op_name(node->op);
                m_op_node_name.emplace_back(src1_name, ggml_op_name(node->op));
                m_input_names.push_back(src1_name);
            }
            break;
        } 
        // OPs with 3 inputs:
        case GGML_OP_ROPE:
        {
            // std::string src1_name = std::string(node->src[1]->name) + "_" + std::to_string(node->src[1]->view_offs) + "_input_" + ggml_op_name(node->src[1]->op);
            // std::string src1_name = std::string(node->src[1]->name) + "_" + std::to_string(node->src[1]->view_offs);
            std::string src1_name = std::string(node->src[1]->name);
            inputs[src0_name] = node->src[0];
            inputs[src1_name] = node->src[1];
            m_input_names.push_back(src0_name);
            m_node_op_name[src0_name] = ggml_op_name(node->op);
            m_op_node_name.emplace_back(src0_name, ggml_op_name(node->op));
            m_input_names.push_back(src1_name);
            m_node_op_name[src1_name] = ggml_op_name(node->op);
            m_op_node_name.emplace_back(src1_name, ggml_op_name(node->op));
            outputs[node_name] = node;
            m_output_names.push_back(node_name);
            if (node->src[2]) {
                // std::string src2_name = std::string(node->src[2]->name) + "_" + std::to_string(node->src[2]->view_offs) + "_input_" + ggml_op_name(node->src[2]->op);
                // std::string src2_name = std::string(node->src[2]->name) + "_" + std::to_string(node->src[2]->view_offs);
                std::string src2_name = std::string(node->src[2]->name);
                inputs[src2_name] = node->src[2];
                m_input_names.push_back(src2_name);
                m_node_op_name[src2_name] = ggml_op_name(node->op);
                m_op_node_name.emplace_back(src2_name, ggml_op_name(node->op));
            }
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
                << std::setw(16) << "op"
                << std::setw(20) << "name"
                << std::setw(3) << "    "
                << std::setw(50) << "stride"
                << "\n";
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << ", "
             << std::setw(5) << node->ne[2] << "] "
             << std::left << std::setw(20) << ggml_op_name(node->op) << std::right << " "
             << std::left << std::setw(44) << node->name << std::right
             << ((node->flags & GGML_TENSOR_FLAG_PARAM) ? "x" : node->grad ? "g" : " ")
             << std::setw(2) << "[ "
             << std::setw(0) << node->nb[0] << ", "
             << std::setw(5) << node->nb[1] << ", "
             << std::setw(5) << node->nb[2] << "] "
             << "\n";

        if (node->src[0]) {
            file << std::setw(10) << " [ "
            << std::setw(5) << node->src[0]->ne[0] << ", "
            << std::setw(5) << node->src[0]->ne[1] << ", "
            << std::setw(5) << node->src[0]->ne[2] << "] "
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
            << std::setw(5) << node->src[0]->nb[2] << "] "
            << "\n";
        }
        if (node->src[1]) {
            file << std::setw(10) << " [ "
            << std::setw(5) << node->src[1]->ne[0] << ", "
            << std::setw(5) << node->src[1]->ne[1] << ", "
            << std::setw(5) << node->src[1]->ne[2] << "] "
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
            << std::setw(5) << node->src[1]->nb[2] << "] "
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
    // If first init 
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
        #ifdef GGML_OPENVINO_DEBUG
            ggml_graph_op_print(m_cgraph);
        #endif
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

const std::string& GgmlOvDecoder::get_node_op_name(const std::string& name) const {
    auto it = m_node_op_name.find(name);
    static const std::string empty_str;
    return (it != m_node_op_name.end()) ? it->second : empty_str;
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
    // Use input_node->ne 
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
