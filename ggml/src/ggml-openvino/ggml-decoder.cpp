#include "ggml-decoder.h"
#include <ggml.h>
#include <ggml-impl.h>

GgmlOvDecoder::GgmlOvDecoder(struct ggml_tensor * node, struct ggml_cgraph * cgraph)
    :m_cgraph(cgraph),
     m_node(node),
     m_op_name(std::string(m_node->name)) {
    switch (m_node->op) {
        // Unary OPs 
        case GGML_OP_UNARY:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        {
            m_inputs.push_back(m_node->src[0]);
            m_outputs.push_back(m_node);
            #ifdef GGML_OPENVINO_DEBUG
                GGML_LOG_INFO("Decoder input 0: %f \n", *(float*)(m_node->src[0]->data));
            #endif
            break;
        }
        // SCALE
        case GGML_OP_SCALE:
        {
            m_inputs.push_back(m_node->src[0]);
            m_outputs.push_back(m_node);
            #ifdef GGML_OPENVINO_DEBUG
                float v;
                memcpy(&v, m_node->op_params, sizeof(float));
                GGML_LOG_INFO("Decoder input 0: %f \n", *(float*)(m_node->src[0]->data));
                GGML_LOG_INFO("Scale: %f \n", v);
            #endif
            break;
        }
        // OPs with 2 inputs
        case GGML_OP_ADD:
        case GGML_OP_DIV:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_SUB:        
        case GGML_OP_GET_ROWS:
        {
            m_inputs.push_back(m_node->src[0]);
            m_inputs.push_back(m_node->src[1]);
            m_outputs.push_back(m_node);
            #ifdef GGML_OPENVINO_DEBUG
                GGML_LOG_INFO("Decoder input 0: %f \n", *(float*)(m_node->src[0]->data));
                GGML_LOG_INFO("Decoder input 1: %f \n", *(float*)(m_node->src[1]->data));
            #endif
            break;
        } 
        default:
            break;
    }
}

ov::PartialShape GgmlOvDecoder::get_input_shape(size_t index) const {
    ov::PartialShape input_shape;
    // Use input_node->ne 
    ggml_tensor * node = m_inputs[index];
    std::vector<size_t> shape;
    // GGML_MAX_DIMS
    // for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    for (int i = GGML_MAX_DIMS - 2; i >= 0 ; --i) {
        if (node->ne[i] == 0) {
            return input_shape;
        }
        shape.push_back(static_cast<size_t>(node->ne[i]));
    }
    input_shape = ov::PartialShape(shape);
    return input_shape;
}

ov::element::Type GgmlOvDecoder::get_input_type(size_t index) const {
    ov::element::Type type = ov::element::dynamic;
    // GGML_LOG_DEBUG("%d\n", m_inputs[index]->type);
    switch (m_inputs[index]->type) {
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
    return m_inputs.size();
}

bool GgmlOvDecoder::is_graph_input(size_t index) const {
    if (m_inputs[index]->flags & GGML_TENSOR_FLAG_INPUT ) {
        return true;
    }
    return false;
}

std::string& GgmlOvDecoder::get_input_name(size_t index) const {
    m_name = std::string(m_inputs[index]->name);
    return m_name;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(size_t index) const {
    ov::PartialShape output_shape;
    // Use input_node->ne 
    ggml_tensor * node = m_outputs[index];
    std::vector<size_t> shape;
    // GGML_MAX_DIMS
    // for (int i = 0; i < GGML_MAX_DIMS; ++i) {
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

ov::element::Type GgmlOvDecoder::get_output_type(size_t index) const {
    // TODO: Change to Output
    ov::element::Type type = ov::element::dynamic;
    // GGML_LOG_DEBUG("%d\n", m_outputs[index]->type);
    switch (m_outputs[index]->type) {
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

bool GgmlOvDecoder::is_graph_output(size_t index) const {
    if (m_outputs[index]->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return true;
    }
    return false;
}

int32_t* GgmlOvDecoder::get_output_op_params(size_t index) const{
    return m_outputs[index]->op_params;
}

size_t GgmlOvDecoder::get_output_size() const {
    return m_outputs.size();
}

std::string& GgmlOvDecoder::get_output_name(size_t index) const {
    m_name = std::string(m_outputs[index]->name);
    return m_name;
}

const std::string& GgmlOvDecoder::get_op_name() const {
    return m_op_name;
}

const std::string& GgmlOvDecoder::get_op_type() const {
    static const std::map<ggml_op, std::string> opTypeMap = {
        {GGML_OP_ACC, "GGML_OP_ACC"},
        {GGML_OP_ADD, "GGML_OP_ADD"},
        {GGML_OP_ADD1, "GGML_OP_ADD1"},
        {GGML_OP_DIV, "GGML_OP_DIV"},
        {GGML_OP_DUP, "GGML_OP_DUP"},
        {GGML_OP_GET_ROWS, "GGML_OP_GET_ROWS"},
        {GGML_OP_MUL, "GGML_OP_MUL"},
        {GGML_OP_MUL_MAT, "GGML_OP_MUL_MAT"},
        {GGML_OP_PERMUTE, "GGML_OP_PERMUTE"},
        {GGML_OP_RESHAPE, "GGML_OP_RESHAPE"},
        {GGML_OP_SCALE, "GGML_OP_SCALE"},
        {GGML_OP_SUB, "GGML_OP_SUB"},
        {GGML_OP_UNARY, "GGML_OP_UNARY"},
        {GGML_OP_VIEW, "GGML_OP_VIEW"}
    };
    auto it = opTypeMap.find(m_node->op);
    if (it != opTypeMap.end()) {
        return it->second;
    } else {
        static const std::string unknown_op = "UNKNOWN_OP";
        return unknown_op;
    }
    // static std::string op_type = ggml_op_name(m_node->op);
    // return op_type;
}
