#include "ggml-decoder.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-quants.hpp"

#include <ggml-impl.h>
#include <ggml.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <openvino/core/dimension.hpp>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             bool is_static,
                             bool is_prefill,
                             int prefill_chunk_size) :
    m_is_static(is_static),
    m_is_prefill(is_prefill),
    m_prefill_chunk_size(prefill_chunk_size),
    m_cgraph(cgraph),
    m_model_weights(model_weights) {
    if (auto * env = getenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS"); env && std::string(env) != "0") {
        #ifdef _WIN32
		    _putenv_s("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS", "");
	    #else
		    unsetenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS");
	    #endif
        print_tensor_address_map(cgraph);
    }

    set_llm_params();
    validate_cgraph();

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto * cur_node = cgraph->nodes[node_n];
        m_nodes.push_back(cur_node);
        set_input_output(cur_node);
    }

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }

    add_extra_inputs();
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto * cur_node = cgraph->nodes[node_n];
        if (cur_node->op == GGML_OP_NONE) {
            continue;
        }
        m_nodes.push_back(cur_node);
        set_input_output(cur_node, true);
    }
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }
}

// Called in GgmlOvDecoder constructor. Two cases: 1. constructing a decoder for the whole graph;
// 2. constructing a decoder for a node;
// 3. constructing a decoder for the whole graph naively (op test case)
void GgmlOvDecoder::set_input_output(ggml_tensor * node, bool naive) {
    std::string node_name;
    NodeInfo current_node_info;
    if (node->op == GGML_OP_SET_ROWS) {
        // SET_ROWS updates the tensor in place. For later ov op that uses the
        // the view_src of SET_ROWS, we need to make sure they get the updated tensor
        // by putting the view_src name in the tensor_map in
        // <openvino>/src/frontends/ggml/src/translate_session.cpp
        node_name = std::string(node->view_src->name);
    } else {
        node_name = std::string(node->name);
    }

    m_output_names.push_back(node_name);
    m_outputs[node_name] = node;

    current_node_info.node = node;
    current_node_info.node_name = node_name;
    current_node_info.node_outputs[node_name] = node;
    current_node_info.node_outputs_names.push_back(node_name);
    current_node_info.node_op_case = 0;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto * src = node->src[i];
        if (src == nullptr) {
            continue;
        }
        std::string src_name = std::string(src->name);
        m_input_names.push_back(src_name);
        m_inputs[src_name] = src;
        current_node_info.node_inputs[src_name] = src;
        current_node_info.node_inputs_names.push_back(src_name);

        // Add model inputs and weights constants, if called for the whole graph
        if (naive) {
            if (m_model_weights.find(src_name) == m_model_weights.end()) {
                auto param_node =
                    std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), get_graph_input_shape(node, src));
                param_node->set_friendly_name(src_name);
                param_node->output(0).get_tensor().set_names({src_name});
                m_model_inputs[src_name] = param_node;
            }

        } else if (!src->view_src) {
            ggml_backend_buffer * buffer = src->buffer;

            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY || src->flags & GGML_TENSOR_FLAG_INPUT) {
                // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
                if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                    assert(src_name.find("cache_k") == 0 || src_name.find("cache_v") == 0);
                }
                if (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                    continue;
                }
                auto param_node =
                    std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), get_graph_input_shape(node, src));
                param_node->set_friendly_name(src_name);
                param_node->output(0).get_tensor().set_names({src_name});
                m_model_inputs[src_name] = param_node;
            }
        }
    }

    // Add model outputs, if called for the whole graph
    if (naive) {
        m_model_output_names.push_back(node_name);
    } else {
        // Model outputs are tensors with GGML_TENSOR_FLAG_OUTPUT flag and kv_caches
        static std::set<std::string> debug_output_names = {};
        // Workaround: the final tensor "result_output" does not have GGML_TENSOR_FLAG_OUTPUT flag set in cgraph
        if (node->op == GGML_OP_SET_ROWS || node->flags & GGML_TENSOR_FLAG_OUTPUT ||
            node_name.find("output") != std::string::npos || debug_output_names.count(node_name)) {
            if (node->op == GGML_OP_SET_ROWS) {
                assert(node_name.find("cache_k") == 0 || node_name.find("cache_v") == 0);
                if (auto it = std::find(m_kv_names.begin(), m_kv_names.end(), node_name); it == m_kv_names.end()) {
                    m_kv_names.push_back(node_name);
                }
            }
            if (auto it = std::find(m_model_output_names.begin(), m_model_output_names.end(), node_name);
                it == m_model_output_names.end()) {
                m_model_output_names.push_back(node_name);
            }
        }
    }

    m_node_info_list.push_back(current_node_info);
}

int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) {
    int op_case = 0;
    switch (node->op) {
    case GGML_OP_RESHAPE: {
        auto * src = node->src[0];
        if (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == node->ne[0] && src->src[0]->ne[1] == node->ne[1]) {
            op_case = 4;
        } else if (node->ne[0] * node->ne[1] == src->ne[0]) {
            op_case = 1;
        } else if (src->ne[0] * src->ne[1] == node->ne[0]) {
            op_case = 2;
            if (src->ne[2] * src->ne[3] == node->ne[1]) {
                op_case = 5;
            }
        } else if (src->ne[0] * src->ne[1] == node->ne[1]) {
            op_case = 3;
        } else if (src->ne[1] * src->ne[2] == node->ne[1]) {
            op_case = 6;
        }
        break;
    }
    case GGML_OP_CONT: {
        if (node->src[0]->op == GGML_OP_PERMUTE) {
            op_case = 1;
        } else if (node->src[0]->op == GGML_OP_TRANSPOSE) {
            op_case = 2;
        } else if (node->src[0]->op == GGML_OP_VIEW) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_PERMUTE: {
        if (node->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } else if (ggml_is_contiguous(node->src[0])) {
            std::string src_name(node->view_src->name);
            if (src_name.find("cache") == std::string::npos) {
                op_case = 4;
            } else {
                int layer = extract_layer_from_name(src_name);
                if (!is_swa_layer(layer)) {
                    op_case = 2;
                } else {
                    op_case = 3;
                }
            }
        }
        break;
    }
    case GGML_OP_MUL_MAT: {
        if (node->src[0]->op == GGML_OP_CONT && node->src[0]->src[0]->op == GGML_OP_TRANSPOSE) {
            op_case = 2;
        } else if (node->src[0]->op == GGML_OP_VIEW && node->src[1]->op == GGML_OP_VIEW) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_GET_ROWS: {
        if (node->src[1]->op == GGML_OP_VIEW) {
            op_case = 2;
        }
        break;
    }
    case GGML_OP_ROPE: {
        if (node->src[0]->op == GGML_OP_VIEW) {
            op_case = 2;
        }
        break;
    }
    case GGML_OP_VIEW: {
        if (node->src[0]->op == GGML_OP_VIEW) {
            auto * src = node->src[0];
            if (ggml_nelements(node) != ggml_nelements(src)) {
                throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 2;
        }
        break;
    }
    default:
        break;
    }
    return op_case;
}

int extract_layer_from_name(const std::string & name) {
    size_t pos1 = name.find("_l");
    assert(pos1 != std::string::npos);
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    if (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    int layer = std::stoi(layer_str);
    return layer;
}

void GgmlOvDecoder::set_llm_params() {
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        auto * node = m_cgraph->nodes[i];
        std::string name = std::string(node->name);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            auto * cache_k_perm = node->src[1];
            assert(cache_k_perm->op == GGML_OP_PERMUTE);
            auto * cache_k_view = cache_k_perm->src[0];
            assert(cache_k_view->op == GGML_OP_VIEW);

            auto * cache_k = cache_k_view->src[0];
            int layer = extract_layer_from_name(cache_k->name);
            auto * mask = node->src[3];
            std::string mask_name(mask->name);
            assert(mask_name.find("KQ_mask") == 0);

            if (std::string(node->src[3]->name).find("swa") != std::string::npos) {
                m_swa_layers.push_back(layer);
                m_ctx_per_seq_swa = cache_k->ne[1];
            } else {
                m_ctx_per_seq = cache_k->ne[1];
                m_n_seq = cache_k->ne[2];
            }

            m_n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, sizeof(size_t));
            m_seq_active_start = offset / seq_size;
            m_token_len_per_seq = node->ne[2];

            if (mask_name.find("swa") != std::string::npos) {
                m_attention_size_swa = mask->ne[0];
            } else {
                m_attention_size = mask->ne[0];
            }
            if (m_is_static) {
                m_attention_size = m_ctx_per_seq;
                m_attention_size_swa = m_ctx_per_seq_swa;
                m_token_len_per_seq = 1;
            }

        } else if (node->op == GGML_OP_ROPE) {
            if (name.find("Qcur-0") == 0 || std::string(node->src[0]->name).find("Qcur-0") == 0) {
                m_head_size = node->ne[0];
                m_n_heads = node->ne[1];
                m_rope_params = node->op_params;
                auto * inp_pos = node->src[1];
                m_input_len = inp_pos->ne[0];
            } else if (name.find("Kcur-0") == 0 || std::string(node->src[0]->name).find("Kcur-0") == 0) {
                m_n_heads_kv = node->ne[1];
            }
        }
    }
    m_ctx = m_ctx_per_seq * m_n_seq;
    m_ctx_swa = m_ctx_per_seq_swa * m_n_seq;
}

void GgmlOvDecoder::validate_cgraph() const {
    if (m_n_seq > 1 && m_is_static == true) {
        throw std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const {
    auto name = std::string(input->name);
    ov::PartialShape input_shape;

    if (name == "inp_tokens" || name == "inp_pos") {
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else if (name == "inp_out_ids") {
        input_shape = ov::PartialShape{1, 1, 1, m_is_static ? 1 : -1};

    } else if (name.find("KQ_mask") == 0) {
        if (m_is_static) {
            input_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_ctx};
        } else {
            input_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } else if (name.find("cache_") == 0) {
        input_shape = ov::PartialShape{get_shape(input)};
        if (!m_is_static) {
            // do not fix ctx size to make llama-bench work
            input_shape[2] = -1;
        }

    } else if (op && op->op == GGML_OP_SET_ROWS && op->src[1] == input) {
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else if (input->op == GGML_OP_VIEW) {
        // This case is added to make test-backend-ops work
        input_shape = ov::PartialShape{get_shape(input->view_src)};
    } else {
        input_shape = ov::PartialShape{get_shape(input)};
    }
    return input_shape;
}

void GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_input = [this](const std::string & name, int64_t value) {
        if (m_is_static) {
            auto constant =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{value});
            constant->set_friendly_name(name);
            m_model_extra_inputs[name] = constant;
        } else {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
            param_node->set_friendly_name(name);
            param_node->output(0).get_tensor().set_names({name});
            m_model_extra_inputs[name] = param_node;

            auto tensor = std::make_shared<ov::Tensor>(ov::element::i64, ov::Shape{1});
            *tensor->data<int64_t>() = value;
            m_model_extra_input_values[name] = tensor;
        }
    };

    create_1d_input("attention_size", m_attention_size);
    if (m_attention_size_swa != -1) {
        create_1d_input("attention_size_swa", m_attention_size_swa);
    }
    create_1d_input("n_seq_active", m_n_seq_active);
    create_1d_input("seq_active_start", m_seq_active_start);
    create_1d_input("seq_active_end", m_seq_active_start + m_n_seq_active);
    create_1d_input("token_len_per_seq", m_token_len_per_seq);
    // create_1d_input("token_len", m_token_len_per_seq * m_n_seq_active);
}

const ggml_tensor * GgmlOvDecoder::get_tensor_used_op(const ggml_tensor * tensor) const {
    if (tensor == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] == tensor) {
                return node;
            }
        }
    }
    return nullptr;
}

const ggml_tensor * GgmlOvDecoder::get_tensor_from_name(const std::string & name) const {
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const auto * src = node->src[j];
            if (src == nullptr) {
                break;
            }
            if (std::string(src->name) == name) {
                return src;
            }
        }
    }
    return nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() const {
    std::map<std::string, std::string> kv_param_res_names;
    for (const auto & name : m_kv_names) {
        if (name.find("cache_k") == 0 || name.find("cache_v") == 0) {
            kv_param_res_names[name] = name;
        }
    }
    return kv_param_res_names;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(
    ggml_cgraph * cgraph,
    std::map<ggml_type, ExtraQuantType> types_to_requantize) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    static std::mutex weights_mutex;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    std::for_each(std::execution::par, nodes, nodes + n_nodes, [&](ggml_tensor * node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }

            std::string src_name(src->name);
            if (!src->view_src) {
                ggml_backend_buffer * buffer = src->buffer;
                if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type)) {
                    bool should_create = false;
                    {
                        std::lock_guard<std::mutex> lock(weights_mutex);
                        if (model_weights.find(src_name) == model_weights.end()) {
                            model_weights[src_name] = nullptr;
                            should_create = true;
                        }
                    }
                    if (should_create) {
                        auto requant_type = types_to_requantize.count(src->type) ?
                                                std::optional<ExtraQuantType>(types_to_requantize.at(src->type)) :
                                                std::nullopt;
                        auto weight_node = create_weight_node(src, requant_type);
                        weight_node->set_friendly_name(src_name);
                        {
                            std::lock_guard<std::mutex> lock(weights_mutex);
                            model_weights[src_name] = weight_node;
                        }
                    }
                }
            }
        }
    });
    return model_weights;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_node(ggml_tensor * tensor,
                                                            std::optional<ExtraQuantType> requant_type) {
    std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
                                        GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
    if (weight_types.find(tensor->type) == weight_types.end()) {
        throw std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    auto node_type = get_ov_type(tensor);
    auto node_shape = get_shape(tensor);
    auto ne_total = ggml_nelements(tensor);

    OPENVINO_ASSERT(node_shape[0] == 1, "Got 4D weights, expect all weights to be 2D: ", tensor->name);
    node_shape.erase(node_shape.begin());
    OPENVINO_ASSERT(node_shape[0] == 1, "Got 3D weights, expect all weights to be 2D: ", tensor->name);
    node_shape.erase(node_shape.begin());

    // F16 and F32 case
    if (node_type != ov::element::dynamic) {
        ov::Tensor weights(node_type, node_shape);
        memcpy(weights.data(), tensor->data, ne_total * node_type.size());
        std::shared_ptr<ov::Node> weight_node = std::make_shared<ov::op::v0::Constant>(weights);
        // Disabled because it triggers a bug in NPUW, no performance impact on CPU GPU
        // if (node_type == ov::element::f16) {
        //     weight_node = std::make_shared<ov::op::v0::Convert>(weight_node, ov::element::f32);
        // }
        weight_node->set_friendly_name(tensor->name);
        return weight_node;
    }

    // Quantized case
    OPENVINO_ASSERT(tensor->extra == nullptr, "Unsupported weight tensor: " + std::string(tensor->name) +
                                                  " Possibly this is a repacked quantized weights");

    if (requant_type.has_value()) {
        return requantize(tensor, requant_type.value());
    }

    ov::element::Type weight_type;
    if (tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_1 || tensor->type == GGML_TYPE_Q4_K) {
        weight_type = ov::element::u4;
    } else {  // tensor.type == GGUF_TYPE_Q8_0 || tensor.type == GGUF_TYPE_Q6_K || tensor.type == GGUF_TYPE_Q5_K
        weight_type = ov::element::u8;
    }

    uint64_t weights_per_block;
    // here we only consider sub block, q6k:16 q4k:32 q5k:32
    if (tensor->type == GGML_TYPE_Q6_K) {
        weights_per_block = 16;
    } else {
        weights_per_block = 32;
    }

    OPENVINO_ASSERT(node_shape.back() % weights_per_block == 0, "[load_gguf] tensor ", tensor->name,
                    " has incompatible last dim shape: ", node_shape.back());

    ov::Tensor weights(weight_type, node_shape);
    // For scales and biases
    node_shape[node_shape.size() - 1] = node_shape[node_shape.size() - 1] / weights_per_block;
    ov::Tensor scales(ov::element::f16, node_shape);
    ov::Tensor biases(ov::element::f16, node_shape);

    ov::Output<ov::Node> weight_node;
    if (tensor->type == GGML_TYPE_Q4_0) {
        extract_q4_0_data(tensor, weights, scales, biases);
        weight_node = make_int4_weights(weights, scales, biases, weights_per_block);
    } else if (tensor->type == GGML_TYPE_Q4_1) {
        extract_q4_1_data(tensor, weights, scales, biases);
        weight_node = make_int4_weights(weights, scales, biases, weights_per_block);
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        extract_q8_0_data(tensor, weights, scales, biases);
        weight_node = make_int8_weights(weights, scales, biases, weights_per_block);
    } else if (tensor->type == GGML_TYPE_Q6_K) {
        extract_q6_k_data(tensor, weights, scales, biases);
        weight_node = make_int8_weights(weights, scales, biases, weights_per_block);
    } else if (tensor->type == GGML_TYPE_Q4_K) {
        extract_q4_k_data(tensor, weights, scales, biases);
        weight_node = make_int4_weights(weights, scales, biases, weights_per_block);
    } else if (tensor->type == GGML_TYPE_Q5_K) {
        extract_q5_k_data(tensor, weights, scales, biases);
        weight_node = make_int8_weights(weights, scales, biases, weights_per_block);
    }

    OPENVINO_ASSERT(weight_node.get_shape().size() == 2, "Weight should be 2D");

    weight_node.get_node_shared_ptr()->set_friendly_name(tensor->name);
    return weight_node.get_node_shared_ptr();
}

void GgmlOvDecoder::dump_cgraph(const ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
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
        ggml_tensor * node = cgraph->nodes[i];

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << ", "
             << std::setw(5) << node->ne[2] << ", "
             << std::setw(5) << node->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(node->op) << std::right << " "
             << std::left << std::setw(45) << node->name << std::right
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
        ggml_tensor * node = cgraph->leafs[i];

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

void print_tensor_address_map(const ggml_cgraph * cgraph) {
    std::map<void *, std::vector<std::string>> address_map;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto * node = cgraph->nodes[node_n];
        if (node->data) {
            auto it = address_map.find(node->data);
            if (it == address_map.end()) {
                address_map[node->data] = std::vector<std::string>();
            }
            address_map[node->data].push_back(node->name);
        }
    }
    for (const auto & pair : address_map) {
        std::cout << "Address: " << pair.first << std::endl;
        for (const auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << std::endl << std::endl;
    }
}

std::vector<size_t> GgmlOvDecoder::get_shape(const ggml_tensor * tensor) {
    std::vector<size_t> shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(const ggml_tensor * tensor) {
    std::vector<size_t> stride;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    return stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(const ggml_tensor * tensor) {
    switch (tensor->type) {
    case GGML_TYPE_F64:
        return ov::element::f64;
    case GGML_TYPE_F32:
        return ov::element::f32;
    case GGML_TYPE_F16:
        return ov::element::f16;
    case GGML_TYPE_BF16:
        return ov::element::bf16;
    case GGML_TYPE_I8:
        return ov::element::i8;
    case GGML_TYPE_I16:
        return ov::element::i16;
    case GGML_TYPE_I32:
        return ov::element::i32;
    case GGML_TYPE_I64:
        return ov::element::i64;
    default:
        return ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_input_shape(const std::string & name) const {
    return ov::PartialShape(get_shape(m_inputs.at(name)));
}

ov::PartialShape GgmlOvDecoder::get_input_shape(int node_idx, const std::string & name) const {
    return ov::PartialShape(get_shape(m_node_info_list[node_idx].node_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_input_stride(const std::string & name) const {
    return get_stride(m_inputs.at(name));
}

std::vector<size_t> GgmlOvDecoder::get_input_stride(int node_idx, const std::string & name) const {
    return get_stride(m_node_info_list[node_idx].node_inputs.at(name));
}

ov::element::Type GgmlOvDecoder::get_input_type(const std::string & name) const {
    return get_ov_type(m_inputs.at(name));
}

size_t GgmlOvDecoder::get_input_size() const {
    return m_input_names.size();
}

size_t GgmlOvDecoder::get_input_size(int node_idx) const {
    return m_node_info_list[node_idx].node_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_input_names() const {
    return m_input_names;
}

std::vector<std::string> GgmlOvDecoder::get_input_names(int node_idx) const {
    return m_node_info_list[node_idx].node_inputs_names;
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(const std::string & name) const {
    return get_stride(m_outputs.at(name));
}

ov::PartialShape GgmlOvDecoder::get_output_shape(const std::string & name) const {
    auto * ggml_tensor = m_outputs.at(name);
    if (ggml_tensor->op == GGML_OP_SET_ROWS) {
        ggml_tensor = ggml_tensor->view_src;
    }
    return ov::PartialShape(get_shape(ggml_tensor));
}

ov::PartialShape GgmlOvDecoder::get_output_shape(int node_idx, const std::string & name) const {
    auto * ggml_tensor = m_node_info_list[node_idx].node_outputs.at(name);
    if (ggml_tensor->op == GGML_OP_SET_ROWS) {
        ggml_tensor = ggml_tensor->view_src;
    }
    return ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(const std::string & name) const {
    return get_ov_type(m_outputs.at(name));
}

std::vector<std::string> GgmlOvDecoder::get_output_names() const {
    return m_output_names;
}

std::vector<std::string> GgmlOvDecoder::get_output_names(int node_idx) const {
    return m_node_info_list[node_idx].node_outputs_names;
}

const std::string & GgmlOvDecoder::get_op_name() const {
    static const std::string unknown_name = "UNKNOWN_OP_NAME";
    return unknown_name;
}

const std::string & GgmlOvDecoder::get_op_name(int node_idx) const {
    return m_node_info_list[node_idx].node_name;
}

int32_t * GgmlOvDecoder::get_input_op_params(const std::string & name) const {
    return m_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_input_op_params(int node_idx, const std::string & name) const {
    return m_node_info_list[node_idx].node_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(const std::string & name) const {
    return m_outputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(int node_idx, const std::string & name) const {
    return m_node_info_list[node_idx].node_outputs.at(name)->op_params;
}

void GgmlOvDecoder::visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const {
    for (int node_idx = 0; node_idx < m_cgraph->n_nodes; node_idx++) {
        node_visitor(std::make_shared<GgmlOvDecoder>(*this), node_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(const ggml_tensor * node) {
    static const std::map<ggml_op, std::string> ops = {
        {GGML_OP_NONE,           "GGML_OP_NONE"          },
        {GGML_OP_ACC,            "GGML_OP_ACC"           },
        {GGML_OP_ADD,            "GGML_OP_ADD"           },
        {GGML_OP_ADD1,           "GGML_OP_ADD1"          },
        {GGML_OP_CONT,           "GGML_OP_CONT"          },
        {GGML_OP_DIV,            "GGML_OP_DIV"           },
        {GGML_OP_DUP,            "GGML_OP_DUP"           },
        {GGML_OP_GET_ROWS,       "GGML_OP_GET_ROWS"      },
        {GGML_OP_MUL,            "GGML_OP_MUL"           },
        {GGML_OP_MUL_MAT,        "GGML_OP_MUL_MAT"       },
        {GGML_OP_PERMUTE,        "GGML_OP_PERMUTE"       },
        {GGML_OP_RESHAPE,        "GGML_OP_RESHAPE"       },
        {GGML_OP_RMS_NORM,       "GGML_OP_RMS_NORM"      },
        {GGML_OP_ROPE,           "GGML_OP_ROPE"          },
        {GGML_OP_SCALE,          "GGML_OP_SCALE"         },
        {GGML_OP_SOFT_MAX,       "GGML_OP_SOFT_MAX"      },
        {GGML_OP_SUB,            "GGML_OP_SUB"           },
        {GGML_OP_TRANSPOSE,      "GGML_OP_TRANSPOSE"     },
        {GGML_OP_VIEW,           "GGML_OP_VIEW"          },
        {GGML_OP_SET_ROWS,       "GGML_OP_SET_ROWS"      },
        {GGML_OP_CPY,            "GGML_OP_CPY"           },
        {GGML_OP_FLASH_ATTN_EXT, "GGML_OP_FLASH_ATTN_EXT"},
    };
    static const std::map<ggml_unary_op, std::string> unary_ops = {
        {GGML_UNARY_OP_ABS,         "GGML_UNARY_OP_ABS"        },
        {GGML_UNARY_OP_SGN,         "GGML_UNARY_OP_SGN"        },
        {GGML_UNARY_OP_NEG,         "GGML_UNARY_OP_NEG"        },
        {GGML_UNARY_OP_STEP,        "GGML_UNARY_OP_STEP"       },
        {GGML_UNARY_OP_TANH,        "GGML_UNARY_OP_TANH"       },
        {GGML_UNARY_OP_ELU,         "GGML_UNARY_OP_ELU"        },
        {GGML_UNARY_OP_RELU,        "GGML_UNARY_OP_RELU"       },
        {GGML_UNARY_OP_SIGMOID,     "GGML_UNARY_OP_SIGMOID"    },
        {GGML_UNARY_OP_GELU,        "GGML_UNARY_OP_GELU"       },
        {GGML_UNARY_OP_GELU_QUICK,  "GGML_UNARY_OP_GELU_QUICK" },
        {GGML_UNARY_OP_SILU,        "GGML_UNARY_OP_SILU"       },
        {GGML_UNARY_OP_HARDSWISH,   "GGML_UNARY_OP_HARDSWISH"  },
        {GGML_UNARY_OP_HARDSIGMOID, "GGML_UNARY_OP_HARDSIGMOID"},
        {GGML_UNARY_OP_EXP,         "GGML_UNARY_OP_EXP"        },
        {GGML_UNARY_OP_COUNT,       "GGML_UNARY_OP_COUNT"      }
    };
    static const std::map<ggml_glu_op, std::string> glu_ops = {
        {GGML_GLU_OP_SWIGLU, "GGML_GLU_OP_SWIGLU"},
        {GGML_GLU_OP_GEGLU,  "GGML_GLU_OP_GEGLU" },
        {GGML_GLU_OP_REGLU,  "GGML_GLU_OP_REGLU" }
    };

    switch (node->op) {
    case GGML_OP_UNARY:
        return unary_ops.at(ggml_get_unary_op(node));
    case GGML_OP_GLU:
        return glu_ops.at(ggml_get_glu_op(node));
    default:
        return ops.at(node->op);
    }
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}

const std::string & GgmlOvDecoder::get_op_type(int node_idx) const {
    return m_node_info_list[node_idx].node_op_type;
}

const std::string & GgmlOvDecoder::get_op_type() const {
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}
