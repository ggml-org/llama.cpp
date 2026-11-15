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
            return node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX;
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
    for (const ggml_tensor * root : roots) {
        const std::vector<const ggml_tensor *> & tensors = classes[root];
        if (!htp_f16_is_root_candidate(root, produced) || !htp_f16_safe_class(graph, root, tensors, node_idx)) {
            continue;
        }

        bool selected = opt_f16_test != 0 || htp_f16_is_mm_root(nodes, root) || (opt_mm_f16_out >= 2 && htp_f16_is_rms_root(nodes, root));
        if (!selected && opt_mm_f16_out) {
            for (const ggml_tensor * tensor : tensors) {
                for (const htp_opnode & node : nodes) {
                    if (htp_f16_node_has_output(node, tensor)) {
                        for (const ggml_tensor * input : node.inputs) {
                            selected |= input && promoted.count(input) != 0;
                        }
                    }
                }
            }
        }
        if (!selected) {
            continue;
        }

        htp_opnode * owner = nullptr;
        for (htp_opnode & node : nodes) {
            if (htp_f16_node_has_output(node, root)) {
                owner = &node;
                break;
            }
        }
        if (!owner) {
            continue;
        }

        const size_t old_dummy_size = owner->dummy.size();
        std::unordered_map<const ggml_tensor *, const ggml_tensor *> dummies;
        for (const ggml_tensor * tensor : tensors) {
            ggml_tensor dummy = *tensor;
            dummy.type = GGML_TYPE_F16;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                dummy.nb[d] /= 2;
            }
            dummy.data = (uint8_t *) root->data + ((const uint8_t *) tensor->data - (const uint8_t *) root->data) / 2;
            dummies[tensor] = owner->add_dummy(dummy);
        }

        struct replacement { const ggml_tensor ** slot; const ggml_tensor * old; };
        struct saved_node {
            htp_opnode * node;
            int32_t      kernel_params[HTP_OP_MAX_KERN_PARAMS];
        };

        std::vector<replacement> replacements;
        std::vector<saved_node>  touched;

        for (htp_opnode & node : nodes) {
            bool node_changed = false;
            for (const ggml_tensor * & input : node.inputs) {
                auto it = dummies.find(input);
                if (it != dummies.end()) {
                    replacements.push_back({ &input, input });
                    input = it->second;
                    node_changed = true;
                }
            }
            for (const ggml_tensor * & output : node.outputs) {
                auto it = dummies.find(output);
                if (it != dummies.end()) {
                    replacements.push_back({ &output, output });
                    output = it->second;
                    node_changed = true;
                }
            }
            if (node_changed) {
                saved_node s;
                s.node = &node;
                memcpy(s.kernel_params, node.kernel_params, sizeof(s.kernel_params));
                touched.push_back(s);
            }
        }

        bool solved = true;
        for (saved_node & s : touched) {
            if (!htp_kparams_solve(sess, *s.node, true)) {
                solved = false;
                break;
            }
        }
        if (!solved) {
            for (const replacement & replacement : replacements) {
                *replacement.slot = replacement.old;
            }
            for (saved_node & s : touched) {
                memcpy(s.node->kernel_params, s.kernel_params, sizeof(s.kernel_params));
            }
            owner->dummy.resize(old_dummy_size);
            continue;
        }
        promoted.insert(tensors.begin(), tensors.end());
        for (const auto & entry : dummies) {
            promoted.insert(entry.second);
        }
    }
}
