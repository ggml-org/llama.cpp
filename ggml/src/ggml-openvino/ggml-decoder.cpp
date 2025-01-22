#include "ggml-decoder.h"
#include <ggml.h>
#include <ggml-impl.h>
#include <ggml-cpu-impl.h>

void GgmlOvDecoder::set_input_output(ggml_tensor* node, std::map<std::string, ggml_tensor *>& inputs, std::map<std::string, ggml_tensor *>& outputs) {
    switch (node->op) {
        // Unary OPs 
        case GGML_OP_UNARY:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_RMS_NORM:
        {
            inputs[node->src[0]->name] = node->src[0];
            outputs[node->name] = node;
            m_input_names.push_back(node->src[0]->name);
            m_output_names.push_back(node->name);
            break;
        }
        case GGML_OP_CONT:
        {
            if (ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node)) {
                inputs[node->src[0]->name] = node->src[0];
                outputs[node->name] = node;
                m_input_names.push_back(node->src[0]->name);
                m_output_names.push_back(node->name);
                m_continuous = true;
                break;
            }

            if (node->src[0]->type == node->type && node->src[0]->ne[0] == node->ne[0] &&
                node->src[0]->nb[0] == ggml_type_size(node->src[0]->type) && node->nb[0] == ggml_type_size(node->src[0]->type)) {

                for (size_t i01 = 0; i01 < node->src[0]->ne[1]; ++i01) {
                    const char *src_row = reinterpret_cast<const char *>(node->src[0]->data) + i01 * node->src[0]->nb[1];
                    char *dst_row = reinterpret_cast<char *>(node->data) + i01 * node->nb[1];
                    std::memcpy(dst_row, src_row, node->src[0]->ne[0]  * ggml_type_size(node->src[0]->type));
                }

                inputs[node->name] = node;
                outputs[node->name] = node;
                m_input_names.push_back(node->name);
                m_output_names.push_back(node->name);
                m_continuous = false;
                break;
            }

            // if (ggml_is_contiguous(node)) {
                const size_t rs = node->src[0]->ne[0] * ggml_type_size(node->src[0]->type); // Row size in bytes for dst

                // Create OpenVINO tensors for source and destination
                // The tensors are reshaped to a 2D structure (num_rows x ne00) for easier iteration and compatibility with the simplified loop.
                ov::Tensor src_tensor(ov::element::f32,
                                      ov::Shape{node->src[0]->ne[3] * node->src[0]->ne[2] * node->src[0]->ne[1], node->src[0]->ne[0]},
                                      node->src[0]->data);
                ov::Tensor dst_tensor(ov::element::f32,
                                      ov::Shape{node->src[0]->ne[3] * node->src[0]->ne[2] * node->src[0]->ne[1], node->src[0]->ne[0]}, 
                                      node->data);

                // Perform the copy in a single loop
                const size_t num_rows = node->src[0]->ne[3] * node->src[0]->ne[2] * node->src[0]->ne[1];
                for (size_t row = 0; row < num_rows; ++row) {
                    // Calculate the source row pointer based on original strides
                    // The source row pointer is calculated based on the combined index row and the strides nb03, nb02, and nb01.
                    const char* src0_ptr = (char*)src_tensor.data() +
                                            // Calculates which block of the i03 dimension the current row belongs to
                                           (row / (node->src[0]->ne[2] * node->src[0]->ne[1])) * node->src[0]->nb[3] +   // 0
                                            // Calculates which block of the i02 dimension the current row belongs to within the current i03 block.
                                           ((row / node->src[0]->ne[1]) % node->src[0]->ne[2]) * node->src[0]->nb[2] +   // 0,   0,......,    0,384,  384,......,  384,768,......, 2304
                                            // Calculates the position within the current i02 block in terms of the i01 index.
                                           (row % node->src[0]->ne[1]) * node->src[0]->nb[1];             // 0,2688,......,83328,  0, 2688,......,83328,  0,......, 83328

                // Destination row pointer is linear
                // Since dst is contiguous, its rows are accessed linearly using a single stride rs, simplifying the destination pointer calculation.
                char* dst_ptr = (char*)dst_tensor.data() + row * rs;

                // Copy row
                std::memcpy(dst_ptr, src0_ptr, rs);
                }

                inputs[node->name] = node;
                outputs[node->name] = node;
                m_input_names.push_back(node->name);
                m_output_names.push_back(node->name);
                m_continuous = false;
                break;
            //}
        }
        case GGML_OP_CPY:
        {
            if (ggml_is_contiguous(node)) {
                inputs[node->src[0]->name] = node->src[0];
                outputs[node->name] = node;
                m_input_names.push_back(node->src[0]->name);
                m_output_names.push_back(node->name);
                m_continuous = true;
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
                inputs[node->name] = node;
                outputs[node->name] = node;
                m_input_names.push_back(node->name);
                m_output_names.push_back(node->name);
                m_continuous = false;
                break;
            }
        }
        // For view, input is node itself
        case GGML_OP_VIEW:
        {
            inputs[node->name] = node;
            outputs[node->name] = node;
            m_input_names.push_back(node->name);
            m_output_names.push_back(node->name);
            break;
        }
        // SCALE
        case GGML_OP_SCALE:
        {
            inputs[node->src[0]->name] = node->src[0];
            outputs[node->name] = node;
            m_input_names.push_back(node->name);
            m_output_names.push_back(node->name);
            break;
        }
        // OPs with 2 inputs
        case GGML_OP_ADD:
        case GGML_OP_DIV:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_SUB:        
        case GGML_OP_GET_ROWS:
        case GGML_OP_SOFT_MAX:
        {
            inputs[node->src[0]->name] = node->src[0];
            outputs[node->name] = node;
            m_input_names.push_back(node->src[0]->name);
            m_output_names.push_back(node->name);
            if (node->src[1]) {
                inputs[node->src[1]->name] = node->src[1];
                m_input_names.push_back(node->src[1]->name);
            }
            break;
        } 
        // OPs with 3 inputs:
        case GGML_OP_ROPE:
        {
            inputs[node->src[0]->name] = node->src[0];
            inputs[node->src[1]->name] = node->src[1];
            m_input_names.push_back(node->src[0]->name);
            m_input_names.push_back(node->src[1]->name);
            outputs[node->name] = node;
            m_output_names.push_back(node->name);
            if (node->src[2]) {
                inputs[node->src[2]->name] = node->src[2];
                m_input_names.push_back(node->src[2]->name);
            }
            break;
        } 
        default:
            break;
    }
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
            ggml_graph_print(m_cgraph);
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
