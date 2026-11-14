#ifdef GGML_OPENVINO_HAS_CACHE_UPDATE
#    include "recurrent_cache_fusion.h"

#    include "../ggml-openvino-extra.h"
#    include "ggml-impl.h"
#    include "op_table.h"
#    ifdef GGML_OPENVINO_HAS_CACHE_UPDATE
#        include "op/recurrent_cache_update.hpp"
#    endif
#    include "utils.h"

#    include <openvino/op/concat.hpp>
#    include <openvino/op/constant.hpp>
#    include <openvino/op/convert.hpp>
#    include <openvino/op/parameter.hpp>
#    include <openvino/op/reshape.hpp>
#    include <openvino/op/slice.hpp>

namespace ov::frontend::ggml {
RecurrentCacheFusion::RecurrentCacheFusion(const std::shared_ptr<GgmlDecoder> & decoder,
                                           const std::shared_ptr<TensorMap> & tensors) :
    m_decoder(decoder),
    m_tensors(tensors) {
    if (!ggml_openvino_is_gpu() || decoder->is_static() || decoder->is_stateful() || decoder->is_splited_model() ||
        ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_FUSED_RECURRENCE")) {
        return;
    }
    auto actual = std::dynamic_pointer_cast<GgmlOvDecoder>(decoder);
    if (!actual || actual->m_naive || !tensors->count("cache_rs_reset_idx") || !tensors->count("cache_rs_reset_len") ||
        !tensors->count("s_copy_active_slot_len")) {
        return;
    }
    const auto graph = actual->get_cgraph();
    std::map<const ggml_tensor *, int> indices;
    for (int i = 0; i < graph->n_nodes; ++i) {
        indices[graph->nodes[i]] = i;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto cp = graph->nodes[i];
        if (cp->op != GGML_OP_CPY || decoder->get_op_case(i) != 3 || !cp->src[0] || cp->src[0]->op != GGML_OP_VIEW ||
            !cp->src[0]->src[0] || cp->src[0]->src[0]->op != GGML_OP_GET_ROWS) {
            continue;
        }
        const auto gather = cp->src[0]->src[0];
        const auto root = cp->view_src;
        if (!root || (root->type != GGML_TYPE_F32 && root->type != GGML_TYPE_F16) || root->ne[2] != 1 ||
            root->ne[3] != 1 || gather->src[0]->op != GGML_OP_RESHAPE || gather->src[0]->src[0] != root ||
            gather->src[1]->op != GGML_OP_NONE || gather->src[1]->type != GGML_TYPE_I32) {
            continue;
        }
        Group group;
        group.gather = indices.at(gather);
        group.remainder = i;
        for (int j = 0; j < group.gather; ++j) {
            const auto node = graph->nodes[j];
            if (GgmlOvDecoder::is_recurrent_cache_reset(node) &&
                node->view_src == root && decoder->get_op_case(j) == 1) {
                // SCALE (scale, bias) or FILL (value, unused 0)
                float params[2];
                memcpy(params, node->op_params, sizeof(params));
                if (params[0] == 0 && params[1] == 0) {
                    group.reset = j;
                }
            }
        }
        if (group.reset < 0) {
            continue;
        }
        // build_conv_state_store emits one CPY without rollback or one SET_ROWS with rollback.
        for (int j = i + 1; j < graph->n_nodes; ++j) {
            const auto node = graph->nodes[j];
            if (node->op != GGML_OP_CONCAT || node->op_params[0] != 0 || !node->src[0] ||
                node->src[0]->op != GGML_OP_RESHAPE) {
                continue;
            }
            const auto active = node->src[0]->src[0];
            if (active && active->op == GGML_OP_VIEW && active->src[0] == gather && active->view_offs == 0 &&
                node->src[0]->ne[3] == 1 && node->src[0]->ne[0] * node->src[0]->ne[1] == root->ne[0]) {
                group.concat = j;
                for (int k = j + 1; k < graph->n_nodes; ++k) {
                    const auto write = graph->nodes[k];
                    const bool copy = GgmlOvDecoder::is_conv_state_writeback(write);
                    const bool slots = GgmlOvDecoder::is_conv_state_snapshot_writeback(write);
                    if ((copy || slots) && write->view_src == root && write->src[0]->src[0] == node &&
                        write->src[0]->ne[0] == node->src[0]->ne[0] &&
                        write->src[0]->ne[slots ? 2 : 1] == node->src[0]->ne[1] &&
                        (slots ? write->src[2] : write->src[1])->view_offs % root->nb[1] == 0) {
                        if (group.writeback >= 0) {
                            group.writeback = -1;
                            break;
                        }
                        group.writeback = k;
                    }
                }
                break;
            }
        }
        if (group.concat >= 0) {
            if (group.writeback < 0) {
                continue;
            }
            for (int j = group.concat + 1; j < graph->n_nodes; ++j) {
                const auto node = graph->nodes[j];
                if (node->op == GGML_OP_SSM_CONV && node->src[0] == graph->nodes[group.concat] && node->src[1] &&
                    node->src[1]->op == GGML_OP_NONE &&
                    node->src[1]->ne[0] == graph->nodes[group.concat]->src[0]->ne[0] + 1 &&
                    node->src[1]->ne[1] == graph->nodes[group.concat]->src[0]->ne[1] && node->src[1]->ne[2] == 1 &&
                    node->src[1]->ne[3] == 1) {
                    if (group.compute >= 0) {
                        group.compute = -1;
                        break;
                    }
                    group.compute = j;
                }
            }
        } else {
            for (int j = i + 1; j < graph->n_nodes; ++j) {
                const auto node = graph->nodes[j];
                if (node->op != GGML_OP_GATED_DELTA_NET || !node->src[5] || node->src[5]->op != GGML_OP_RESHAPE) {
                    continue;
                }
                const auto active = node->src[5]->src[0];
                if (!active || active->op != GGML_OP_VIEW || active->src[0] != gather || active->view_offs != 0 ||
                    node->src[5]->ne[0] * node->src[5]->ne[1] * node->src[5]->ne[2] != root->ne[0] ||
                    decoder->get_output_names(j).size() < 2) {
                    continue;
                }
                group.compute = j;
                for (int k = j + 1; k < graph->n_nodes; ++k) {
                    const auto write = graph->nodes[k];
                    if (write->op == GGML_OP_CPY && write->view_src == root && write->src[0] &&
                        write->src[0]->op == GGML_OP_VIEW && write->src[0]->src[0] == node &&
                        (decoder->get_op_case(k) == 10 || decoder->get_op_case(k) == 12)) {
                        if (group.writeback >= 0) {
                            group.writeback = -1;
                            break;
                        }
                        group.writeback = k;
                    }
                }
                break;
            }
            if (group.writeback < 0 || !tensors->count("rs_update_" + decoder->get_op_name(group.writeback))) {
                continue;
            }
        }
        if (group.compute < 0) {
            continue;
        }
        bool complete = tensors->count("rs_slot_begin_" + decoder->get_op_name(i)) &&
                        tensors->count("rs_src_begin_" + decoder->get_op_name(i));
        if (group.concat >= 0) {
            complete &= tensors->count("rs_windows_" + decoder->get_op_name(group.concat));
            const auto & name = decoder->get_op_name(group.writeback);
            complete &= tensors->count("rs_slot_begin_" + name) && tensors->count("rs_src_begin_" + name);
        }
        for (int j = 0; j < graph->n_nodes && complete; ++j) {
            const auto node = graph->nodes[j];
            // Other cache readers or writers require functional translation.
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                const auto input = node->src[source];
                if (!input || (input != root && input->view_src != root)) {
                    continue;
                }
                const bool alias = node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE;
                const bool matched_read = j == group.gather || j == group.reset;
                const bool destination = (node->op == GGML_OP_CPY && source == 1) ||
                                         (GgmlOvDecoder::is_conv_state_snapshot_writeback(node) && source == 2);
                if (!alias && !matched_read && !destination) {
                    complete = false;
                }
            }
            if (node->view_src == root &&
                (node->op == GGML_OP_SCALE || node->op == GGML_OP_CPY || node->op == GGML_OP_SET_ROWS) &&
                j != group.reset && j != group.remainder && j != group.writeback) {
                complete = false;
            }
        }
        // Every read of the gathered state or convolution input must belong to
        // this recurrence. Shared readers retain the ordinary GGML translation.
        const auto compute = graph->nodes[group.compute];
        const auto active = group.concat >= 0 ? graph->nodes[group.concat]->src[0]->src[0] : compute->src[5]->src[0];
        const auto reshaped = group.concat >= 0 ? graph->nodes[group.concat]->src[0] : compute->src[5];
        for (int j = group.gather + 1; j < graph->n_nodes && complete; ++j) {
            const auto node = graph->nodes[j];
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                const auto input = node->src[source];
                if (!input) {
                    continue;
                }
                if (input == gather && node != active && node != graph->nodes[group.remainder]->src[0]) {
                    complete = false;
                }
                if (input == active && node != reshaped) {
                    complete = false;
                }
                if (input == reshaped && node != (group.concat >= 0 ? graph->nodes[group.concat] : compute)) {
                    complete = false;
                }
                if (group.concat >= 0 && input == graph->nodes[group.concat] && node != compute) {
                    complete &= node == graph->nodes[group.writeback]->src[0];
                }
                if (input == compute && group.concat < 0 && node->op != GGML_OP_VIEW) {
                    complete = false;
                }
                if (group.concat < 0 && input == graph->nodes[group.writeback]->src[0] &&
                    node != graph->nodes[group.writeback]) {
                    complete = false;
                }
            }
        }
        if (!complete) {
            continue;
        }
        group.root_name = GgmlOvDecoder::get_tensor_name(graph, root);
        const auto id = m_groups.size();
        for (int n : {group.reset, group.gather, group.remainder}) {
            m_nodes[n] = id;
        }
        if (group.concat >= 0) {
            m_nodes[group.concat] = id;
        }
        m_nodes[group.compute] = id;
        if (group.writeback >= 0) {
            m_nodes[group.writeback] = id;
        }
        m_groups.push_back(std::move(group));
    }
}

OutputVector RecurrentCacheFusion::translate(int index, const NodeContext & context) {
    const auto it = m_nodes.find(index);
    if (it == m_nodes.end()) {
        return {};
    }
    auto & group = m_groups[it->second];
    using namespace ov::op;
    auto constant = [](std::initializer_list<size_t> shape, std::initializer_list<int64_t> values) {
        return v0::Constant::create(element::i64, Shape(shape), std::vector<int64_t>(values));
    };
    auto metadata = [&](const std::string & name) -> Output<Node> {
        return std::make_shared<v1::Reshape>(context.get_input(name), constant({4}, {1, 1, 1, 1}), false);
    };
    if (index == group.reset) {
        group.old_cache = context.get_input(0);
        auto root = group.old_cache.get_node_shared_ptr();
        while (ov::is_type<v1::Reshape>(root)) {
            root = root->input_value(0).get_node_shared_ptr();
        }
        group.enabled = ov::is_type<v0::Parameter>(root);
        if (!group.enabled) {
            return {};
        }
        return {group.old_cache};
    }
    if (!group.enabled) {
        return {};
    }
    NodeContext gather_context(m_decoder, m_tensors, group.gather);
    auto rows = process_view_input_new(gather_context, 1);
    const auto remainder_name = m_decoder->get_op_name(group.remainder);
    auto offsets = std::make_shared<v0::Concat>(OutputVector{context.get_input("rs_slot_begin_" + remainder_name),
                                                             context.get_input("rs_src_begin_" + remainder_name)},
                                                0);
    OutputVector inputs{group.old_cache,
                        rows,
                        rows,
                        metadata("cache_rs_reset_idx"),
                        metadata("cache_rs_reset_len"),
                        context.has_input("rs_offsets_" + remainder_name) ?
                            context.get_input("rs_offsets_" + remainder_name) :
                            std::make_shared<v1::Reshape>(offsets, constant({4}, {1, 1, 1, 2}), false)};
    inputs[1] = context.has_input("rs_active_rows") ?
                    context.get_input("rs_active_rows") :
                    std::make_shared<v8::Slice>(rows, constant({1}, {0}), context.get_input("s_copy_active_slot_len"),
                                                constant({1}, {1}), constant({1}, {3}));
    if (index == group.gather) {
        // Ordinary nodes keep intermediate translation shapes valid. They become
        // unreachable once the complete recurrence is replaced by its fused op.
        return op::translate_get_rows(context);
    }
    if (index == group.remainder) {
        return {group.old_cache};
    }
    if (index == group.concat) {
        inputs.push_back(process_view_input_new(context, 1));
        // Runtime pairs [source column, destination slot] cover the current CPY or SET_ROWS store.
        inputs.push_back(context.get_input("rs_windows_" + context.get_name()));
        const auto history = context.get_input_shape(0)[3].get_length();
        NodeContext compute_context(m_decoder, m_tensors, group.compute);
        const auto channels = context.get_input_shape(0)[2].get_length();
        inputs.push_back(std::make_shared<v1::Reshape>(compute_context.get_input(1),
                                                     constant({4}, {1, 1, channels, history + 1}), false));
        auto fused = std::make_shared<internal::RecurrentConvCacheUpdate>(inputs, history);
        fused->set_friendly_name(compute_context.get_name() + "/RecurrentConvCacheUpdate");
        group.conv = fused->output(0);
        return op::translate_concat(context);
    }
    if (index == group.compute) {
        if (group.concat >= 0) {
            return {group.conv};
        }
        inputs.push_back(context.get_input("rs_update_" + m_decoder->get_op_name(group.writeback)));
        return op::translate_recurrent_gated_delta_net(context, inputs);
    }
    return {group.old_cache};
}

bool RecurrentCacheFusion::is_inplace_output(const std::string & name) const {
    return std::any_of(m_groups.begin(), m_groups.end(),
                       [&](const Group & group) { return group.enabled && group.root_name == name; });
}
}  // namespace ov::frontend::ggml
#endif
