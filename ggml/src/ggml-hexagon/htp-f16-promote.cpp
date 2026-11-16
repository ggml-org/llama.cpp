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

static bool htp_f16_inplace_types_ok(const htp_opnode & node) {
    for (const ggml_tensor * inp : node.inputs) {
        if (!inp || (inp->type != GGML_TYPE_F32 && inp->type != GGML_TYPE_F16)) {
            continue;
        }
        for (const ggml_tensor * out : node.outputs) {
            if (!out || (out->type != GGML_TYPE_F32 && out->type != GGML_TYPE_F16)) {
                continue;
            }
            if (inp->type != out->type && htp_tensors_overlap(inp, out)) {
                return false;
            }
        }
    }
    return true;
}

static bool htp_f16_is_mm_opcode(uint32_t opcode) {
    return opcode == HTP_OP_MUL_MAT || opcode == HTP_OP_MUL_MAT_ADD ||
           opcode == HTP_OP_MUL_MAT_ID || opcode == HTP_OP_MUL_MAT_NX ||
           opcode == HTP_OP_MUL_MAT_ID_NX;
}

static const ggml_tensor * htp_f16_orig_mm_for_output(const htp_opnode & node, const ggml_tensor * tensor) {
    if (!htp_f16_is_mm_opcode(node.opcode)) {
        return nullptr;
    }
    if (node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX) {
        for (size_t p = 0; p < node.outputs.size(); ++p) {
            if (node.outputs[p] == tensor) {
                if (p == 0) {
                    return node.node;
                }
                return (p - 1 < node.fused.size()) ? node.fused[p - 1] : nullptr;
            }
        }
        return nullptr;
    }
    return node.node;
}

static bool htp_f16_mm_blocks_output_f16(const ggml_tensor * orig_mm) {
    if (!orig_mm) {
        return false;
    }
    if (ggml_prec_get_acc(orig_mm) == GGML_PREC_F32) {
        return true;
    }
    const ggml_prec src1_prec = ggml_prec_get_src(orig_mm, 1);
    return src1_prec == GGML_PREC_BF16 || src1_prec == GGML_PREC_F32;
}

static bool htp_f16_respect_prec_hints(const htp_opnode & owner, const ggml_tensor * r, const std::vector<const ggml_tensor *> & cls, const std::vector<htp_opnode> & nodes) {
    if (htp_f16_is_mm_opcode(owner.opcode)) {
        const ggml_tensor * orig_mm = htp_f16_orig_mm_for_output(owner, r);
        if (htp_f16_mm_blocks_output_f16(orig_mm)) {
            return false;
        }
    }
    if (owner.opcode == HTP_OP_SOFTMAX && !owner.inputs.empty() && owner.inputs[0]) {
        const ggml_tensor * in_root = htp_f16_root(owner.inputs[0]);
        for (const htp_opnode & node : nodes) {
            if (htp_f16_node_has_output(node, in_root) && htp_f16_is_mm_opcode(node.opcode)) {
                if (htp_f16_mm_blocks_output_f16(htp_f16_orig_mm_for_output(node, in_root))) {
                    return false;
                }
            }
        }
    }
    for (const htp_opnode & node : nodes) {
        if (!htp_f16_is_mm_opcode(node.opcode)) {
            continue;
        }
        const size_t act_idx = (node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX)
                                   ? node.outputs.size()
                                   : 1;
        if (act_idx >= node.inputs.size()) {
            continue;
        }
        const ggml_tensor * act = node.inputs[act_idx];
        bool uses_cls = false;
        for (const ggml_tensor * tensor : cls) {
            if (act == tensor) {
                uses_cls = true;
                break;
            }
        }
        if (!uses_cls && (!act || htp_f16_root(act) != r)) {
            continue;
        }
        auto blocks_src1 = [](const ggml_tensor * orig_mm) -> bool {
            if (!orig_mm) {
                return false;
            }
            const ggml_prec src1_prec = ggml_prec_get_src(orig_mm, 1);
            return src1_prec == GGML_PREC_BF16 || src1_prec == GGML_PREC_F32;
        };
        if (blocks_src1(node.node)) {
            return false;
        }
        if (node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX) {
            for (const ggml_tensor * f : node.fused) {
                if (f && (f->op == GGML_OP_MUL_MAT || f->op == GGML_OP_MUL_MAT_ID) && blocks_src1(f)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool htp_f16_is_mm_root(const std::vector<htp_opnode> & nodes, const ggml_tensor * root) {
    for (const htp_opnode & node : nodes) {
        if (htp_f16_node_has_output(node, root)) {
            return htp_f16_is_mm_opcode(node.opcode);
        }
    }
    return false;
}

static bool htp_f16_is_fa_root(const std::vector<htp_opnode> & nodes, const ggml_tensor * root) {
    for (const htp_opnode & node : nodes) {
        if (htp_f16_node_has_output(node, root)) {
            return node.opcode == HTP_OP_FLASH_ATTN_EXT;
        }
    }
    return false;
}

static bool htp_f16_is_rms_root(const std::vector<htp_opnode> & nodes, const ggml_tensor * root) {
    for (const htp_opnode & node : nodes) {
        if (htp_f16_node_has_output(node, root)) {
            return node.opcode == HTP_OP_RMS_NORM || node.opcode == HTP_OP_RMS_NORM_MUL ||
                   node.opcode == HTP_OP_RMS_NORM_MUL_ADD;
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
    if (opt_opf16 <= 0 && opt_f16_test == 0) {
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
    size_t n_candidates     = 0;
    size_t n_promoted_roots = 0;
    size_t n_promoted_mm    = 0;
    size_t bytes_saved      = 0;

    auto find_owner = [&](const ggml_tensor * r) -> htp_opnode * {
        for (htp_opnode & node : nodes) {
            if (htp_f16_node_has_output(node, r)) {
                return &node;
            }
        }
        return nullptr;
    };

    auto is_f16_tensor = [&](const ggml_tensor * t) -> bool {
        if (!t) {
            return false;
        }
        return t->type == GGML_TYPE_F16 || promoted.count(t) != 0 || promoted.count(htp_f16_root(t)) != 0;
    };

    auto is_weight_or_bcast = [](const ggml_tensor * inp, const ggml_tensor * dst) -> bool {
        if (!inp) {
            return true;
        }
        if ((inp->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
            return true;
        }
        if (inp->buffer && ggml_backend_buffer_get_usage(inp->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return true;
        }
        return ggml_nelements(inp) < ggml_nelements(dst);
    };

    // Under OPF16=1, keep cross-layer residual spine merges in F32 when the residual input is F32.
    auto is_residual_spine_stop = [&](const htp_opnode & owner, const ggml_tensor * r) -> bool {
        if (opt_opf16 != 1 || opt_f16_test != 0) {
            return false;
        }
        if (owner.opcode == HTP_OP_ADD) {
            for (const ggml_tensor * inp : owner.inputs) {
                if (inp && !is_f16_tensor(inp) && !is_weight_or_bcast(inp, r)) {
                    return true;
                }
            }
        } else if (owner.opcode == HTP_OP_RMS_NORM_MUL_ADD || owner.opcode == HTP_OP_MUL_MAT_ADD) {
            if (owner.inputs.size() >= 3 && owner.inputs[2]) {
                const ggml_tensor * res = owner.inputs[2];
                if (!is_f16_tensor(res) && !is_weight_or_bcast(res, r)) {
                    return true;
                }
            }
        }
        return false;
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
        if (!htp_f16_respect_prec_hints(*owner, r, it->second, nodes)) {
            return false;
        }
        if (is_residual_spine_stop(*owner, r)) {
            return false;
        }
        if (owner->opcode == HTP_OP_GATED_DELTA_NET && !htp_f16_safe_gdn_output(*owner, r, nodes)) {
            return false;
        }
        return true;
    };

    auto is_selected_root = [&](const ggml_tensor * r) -> bool {
        if (opt_f16_test != 0 || htp_f16_is_mm_root(nodes, r) || htp_f16_is_fa_root(nodes, r) ||
            (opt_opf16 >= 2 && htp_f16_is_rms_root(nodes, r))) {
            return true;
        }
        if (opt_opf16 >= 1) {
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

    for (const ggml_tensor * root : roots) {
        auto it = classes.find(root);
        if (it != classes.end() && htp_f16_is_root_candidate(root, produced) &&
            htp_f16_safe_class(graph, root, it->second, node_idx) && find_owner(root)) {
            ++n_candidates;
        }
    }

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
                if (htp_f16_inplace_types_ok(*tnode) && htp_kparams_solve(sess, *tnode, true)) {
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
                    if (inp_root && can_promote(inp_root) &&
                        (is_selected_root(inp_root) || htp_f16_is_rms_root(nodes, inp_root)) &&
                        apply_root(inp_root)) {
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
            ++n_promoted_roots;
            bytes_saved += ggml_nbytes(r) / 2;
            auto dit = dummies.find(r);
            if (dit != dummies.end() && htp_f16_is_mm_root(nodes, dit->second)) {
                ++n_promoted_mm;
            }
        }
        for (const auto & entry : dummies) {
            promoted.insert(entry.second);
        }
    }

    if (n_promoted_roots > 0 || opt_opf16 > 0) {
        HEX_VERBOSE("ggml-hex: %s f16-promote opf16=%d roots=%zu/%zu mm_out=%zu saved=%zuB\n",
                    htp_session_name(sess), opt_opf16, n_promoted_roots, n_candidates, n_promoted_mm, bytes_saved);
    }
}
