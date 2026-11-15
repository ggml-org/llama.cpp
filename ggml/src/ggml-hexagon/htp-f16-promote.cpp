#include "htp-f16-promote.h"

#include "htp-common.h"
#include "htp-kparams.h"
#include "htp-opnode.h"

#include <unordered_map>
#include <unordered_set>

static const ggml_tensor * htp_f16_root(const ggml_tensor * tensor) {
    while (tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static bool htp_f16_is_view(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE;
}

static bool htp_f16_is_root_candidate(const ggml_tensor * root, const std::unordered_set<const ggml_tensor *> & produced) {
    return produced.count(root) && root->type == GGML_TYPE_F32 && root->buffer &&
           ggml_backend_buffer_get_usage(root->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
           (root->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_PARAM)) == 0;
}

static bool htp_f16_is_partial_write(const ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_SET || tensor->op == GGML_OP_SET_ROWS) {
        return true;
    }
    return tensor->op == GGML_OP_CPY && !ggml_is_contiguous(tensor);
}

static bool htp_f16_safe_class(const ggml_cgraph * graph, const ggml_tensor * root, const std::vector<const ggml_tensor *> & tensors, const std::unordered_map<const ggml_tensor *, int> & node_idx) {
    int internal_uses = 0;
    for (const ggml_tensor * tensor : tensors) {
        if (tensor != root && !htp_f16_is_view(tensor)) {
            return false;
        }
        const ptrdiff_t offset = (const uint8_t *) tensor->data - (const uint8_t *) root->data;
        if (offset % 4 != 0 || (tensor->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->nb[d] % 4 != 0) {
                return false;
            }
        }
        auto it = node_idx.find(tensor);
        if (it == node_idx.end() || htp_f16_is_partial_write(tensor)) {
            return false;
        }
        const int total_uses = ggml_node_get_use_count(graph, it->second);
        int uses = 0;
        for (int i = 0; i < graph->n_nodes; ++i) {
            const ggml_tensor * node = graph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                uses += node->src[j] == tensor;
            }
        }
        if (uses != total_uses) {
            return false;
        }
        internal_uses += uses;
    }
    return internal_uses >= 1;
}

static bool htp_f16_node_has_output(const htp_opnode & node, const ggml_tensor * tensor) {
    for (const ggml_tensor * output : node.outputs) {
        if (output == tensor) {
            return true;
        }
    }
    return false;
}

static bool htp_f16_is_mm_root(const std::vector<htp_opnode> & nodes, const ggml_tensor * root) {
    for (const htp_opnode & node : nodes) {
        if (htp_f16_node_has_output(node, root)) {
            return node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ADD ||
                   node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_MUL_MAT_NX ||
                   node.opcode == HTP_OP_MUL_MAT_ID_NX;
        }
    }
    return false;
}

static bool htp_f16_is_rms_root(const std::vector<htp_opnode> & nodes, const ggml_tensor * root) {
    for (const htp_opnode & node : nodes) {
        if (htp_f16_node_has_output(node, root)) {
            return node.node && node.node->op == GGML_OP_RMS_NORM;
        }
    }
    return false;
}

static bool htp_f16_safe_gdn_output(const htp_opnode & owner, const ggml_tensor * root, const std::vector<htp_opnode> & nodes) {
    if (owner.outputs.size() != 2 || root != owner.outputs[0] || owner.inputs.size() < 3) {
        return false;
    }
    const ggml_tensor * v = owner.inputs[2];
    const size_t tail_off = (size_t) v->ne[0] * v->ne[1] * v->ne[2] * v->ne[3] * sizeof(float);
    bool has_consumer = false;
    for (const htp_opnode & node : nodes) {
        for (const ggml_tensor * input : node.inputs) {
            if (!input || htp_f16_root(input) != root) {
                continue;
            }
            const ptrdiff_t offset = (const uint8_t *) input->data - (const uint8_t *) root->data;
            if (offset < 0 || (size_t) offset + ggml_nbytes(input) > tail_off) {
                return false;
            }
            has_consumer = true;
        }
    }
    return has_consumer;
}

void htp_f16_promote(const ggml_hexagon_session * sess, const ggml_cgraph * graph, std::vector<htp_opnode> & nodes) {
    if (opt_mm_f16_out == 0 && opt_f16_test == 0) {
        return;
    }

    std::vector<const ggml_tensor *> roots;
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> classes;
    std::unordered_set<const ggml_tensor *> seen;
    std::unordered_map<const ggml_tensor *, int> node_idx;
    std::unordered_set<const ggml_tensor *> produced;

    auto add_to_class = [&](const ggml_tensor * tensor) {
        if (!tensor || !seen.insert(tensor).second) {
            return;
        }
        const ggml_tensor * root = htp_f16_root(tensor);
        auto & cls = classes[root];
        if (cls.empty()) {
            roots.push_back(root);
        }
        cls.push_back(tensor);
    };

    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * tensor = graph->nodes[i];
        node_idx[tensor] = i;
        produced.insert(tensor);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            add_to_class(tensor->src[j]);
        }
        add_to_class(tensor);
    }

    std::unordered_set<const ggml_tensor *> promoted;

    auto find_owner = [&](const ggml_tensor * r) -> htp_opnode * {
        for (htp_opnode & node : nodes) {
            if (htp_f16_node_has_output(node, r)) {
                return &node;
            }
        }
        return nullptr;
    };

    auto can_promote = [&](const ggml_tensor * r) -> bool {
        auto it = classes.find(r);
        if (it == classes.end() || promoted.count(r)) {
            return false;
        }
        if (!htp_f16_is_root_candidate(r, produced) || !htp_f16_safe_class(graph, r, it->second, node_idx)) {
            return false;
        }
        htp_opnode * owner = find_owner(r);
        if (!owner) {
            return false;
        }
        if (owner->opcode == HTP_OP_GATED_DELTA_NET && !htp_f16_safe_gdn_output(*owner, r, nodes)) {
            return false;
        }
        return true;
    };

    auto is_selected_root = [&](const ggml_tensor * r) -> bool {
        if (opt_f16_test != 0 || htp_f16_is_mm_root(nodes, r) || (opt_mm_f16_out >= 2 && htp_f16_is_rms_root(nodes, r))) {
            return true;
        }
        if (opt_mm_f16_out) {
            for (const ggml_tensor * tensor : classes[r]) {
                for (const htp_opnode & node : nodes) {
                    if (htp_f16_node_has_output(node, tensor)) {
                        for (const ggml_tensor * input : node.inputs) {
                            if (input && promoted.count(input) != 0) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    };

    struct replacement { const ggml_tensor ** slot; const ggml_tensor * old; };
    struct saved_node {
        htp_opnode * node;
        int32_t      kernel_params[HTP_OP_MAX_KERN_PARAMS];
    };

    for (const ggml_tensor * root : roots) {
        if (!can_promote(root) || !is_selected_root(root)) {
            continue;
        }

        std::unordered_set<const ggml_tensor *>            tx_roots;
        std::unordered_map<const ggml_tensor *, const ggml_tensor *> dummies;
        std::vector< std::pair<htp_opnode *, size_t> >     saved_dummies;
        std::vector<replacement>                           replacements;
        std::vector<saved_node>                            touched;

        auto apply_root = [&](const ggml_tensor * r) -> bool {
            if (!tx_roots.insert(r).second) {
                return false;
            }
            htp_opnode * owner = find_owner(r);
            bool owner_saved = false;
            for (const auto & sd : saved_dummies) {
                if (sd.first == owner) {
                    owner_saved = true;
                    break;
                }
            }
            if (!owner_saved) {
                saved_dummies.push_back({ owner, owner->dummy.size() });
            }

            std::unordered_map<const ggml_tensor *, const ggml_tensor *> cls_dummies;
            for (const ggml_tensor * tensor : classes[r]) {
                ggml_tensor dummy = *tensor;
                dummy.type = GGML_TYPE_F16;
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    dummy.nb[d] /= 2;
                }
                dummy.data = (uint8_t *) r->data + ((const uint8_t *) tensor->data - (const uint8_t *) r->data) / 2;
                const ggml_tensor * d_ptr = owner->add_dummy(dummy);
                cls_dummies[tensor] = d_ptr;
                dummies[tensor]     = d_ptr;
            }

            for (htp_opnode & node : nodes) {
                bool node_changed = false;
                for (const ggml_tensor * & input : node.inputs) {
                    auto it = cls_dummies.find(input);
                    if (it != cls_dummies.end()) {
                        replacements.push_back({ &input, input });
                        input = it->second;
                        node_changed = true;
                    }
                }
                for (const ggml_tensor * & output : node.outputs) {
                    auto it = cls_dummies.find(output);
                    if (it != cls_dummies.end()) {
                        replacements.push_back({ &output, output });
                        output = it->second;
                        node_changed = true;
                    }
                }
                if (node_changed) {
                    bool already_touched = false;
                    for (const saved_node & s : touched) {
                        if (s.node == &node) {
                            already_touched = true;
                            break;
                        }
                    }
                    if (!already_touched) {
                        saved_node s;
                        s.node = &node;
                        memcpy(s.kernel_params, node.kernel_params, sizeof(s.kernel_params));
                        touched.push_back(s);
                    }
                }
            }
            return true;
        };

        apply_root(root);

        bool solved = false;
        while (true) {
            solved = true;
            bool expanded = false;
            for (size_t i = 0; i < touched.size(); ++i) {
                htp_opnode * tnode = touched[i].node;
                if (htp_kparams_solve(sess, *tnode, true)) {
                    continue;
                }
                solved = false;
                for (const ggml_tensor * out : tnode->outputs) {
                    const ggml_tensor * out_root = out ? htp_f16_root(out) : nullptr;
                    if (out_root && can_promote(out_root) && apply_root(out_root)) {
                        expanded = true;
                    }
                }
                for (const ggml_tensor * inp : tnode->inputs) {
                    const ggml_tensor * inp_root = inp ? htp_f16_root(inp) : nullptr;
                    if (inp_root && can_promote(inp_root) && is_selected_root(inp_root) && apply_root(inp_root)) {
                        expanded = true;
                    }
                }
                if (expanded) {
                    break;
                }
            }
            if (solved || !expanded) {
                break;
            }
        }

        if (!solved) {
            for (auto it = replacements.rbegin(); it != replacements.rend(); ++it) {
                *it->slot = it->old;
            }
            for (saved_node & s : touched) {
                memcpy(s.node->kernel_params, s.kernel_params, sizeof(s.kernel_params));
            }
            for (const auto & sd : saved_dummies) {
                sd.first->dummy.resize(sd.second);
            }
            continue;
        }
        for (const ggml_tensor * r : tx_roots) {
            const auto & cls = classes[r];
            promoted.insert(cls.begin(), cls.end());
        }
        for (const auto & entry : dummies) {
            promoted.insert(entry.second);
        }
    }
}
