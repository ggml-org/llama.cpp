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

static bool htp_f16_is_mm_root(const std::vector<htp_opnode> & nodes, const ggml_tensor * root) {
    for (const htp_opnode & node : nodes) {
        if (node.node == root) {
            return node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX;
        }
    }
    return false;
}

static bool htp_f16_is_rms_root(const std::vector<htp_opnode> & nodes, const ggml_tensor * root) {
    for (const htp_opnode & node : nodes) {
        if (node.node == root) {
            return node.node->op == GGML_OP_RMS_NORM;
        }
    }
    return false;
}

void htp_f16_promote(const ggml_hexagon_session * sess, const ggml_cgraph * graph, std::vector<htp_opnode> & nodes) {
    if (opt_mm_f16_out == 0 && opt_f16_test == 0) {
        return;
    }

    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> classes;
    std::unordered_map<const ggml_tensor *, int> node_idx;
    std::unordered_set<const ggml_tensor *> produced;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * tensor = graph->nodes[i];
        node_idx[tensor] = i;
        produced.insert(tensor);
        classes[htp_f16_root(tensor)].push_back(tensor);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (tensor->src[j]) {
                classes[htp_f16_root(tensor->src[j])].push_back(tensor->src[j]);
            }
        }
    }

    std::unordered_set<const ggml_tensor *> promoted;
    for (const auto & entry : classes) {
        const ggml_tensor * root = entry.first;
        const std::vector<const ggml_tensor *> & tensors = entry.second;
        if (!htp_f16_is_root_candidate(root, produced) || !htp_f16_safe_class(graph, root, tensors, node_idx)) {
            continue;
        }

        bool selected = opt_f16_test != 0 || htp_f16_is_mm_root(nodes, root) || (opt_mm_f16_out >= 2 && htp_f16_is_rms_root(nodes, root));
        if (!selected && opt_mm_f16_out) {
            for (const ggml_tensor * tensor : tensors) {
                for (const htp_opnode & node : nodes) {
                    if (node.node == tensor) {
                        for (const ggml_tensor * input : node.inputs) {
                            selected |= promoted.count(input) != 0;
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
            for (const ggml_tensor * output : node.outputs) {
                if (output == root) {
                    owner = &node;
                    break;
                }
            }
            if (owner) break;
        }
        if (!owner) {
            continue;
        }

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
        std::vector<replacement> replacements;
        for (htp_opnode & node : nodes) {
            for (const ggml_tensor * & input : node.inputs) {
                auto it = dummies.find(input);
                if (it != dummies.end()) {
                    replacements.push_back({ &input, input });
                    input = it->second;
                }
            }
            for (const ggml_tensor * & output : node.outputs) {
                auto it = dummies.find(output);
                if (it != dummies.end()) {
                    replacements.push_back({ &output, output });
                    output = it->second;
                }
            }
        }

        bool solved = true;
        for (htp_opnode & node : nodes) {
            solved &= htp_kparams_solve(sess, node);
        }
        if (!solved) {
            for (const replacement & replacement : replacements) {
                *replacement.slot = replacement.old;
            }
            for (htp_opnode & node : nodes) {
                GGML_ASSERT(htp_kparams_solve(sess, node));
            }
            continue;
        }
        promoted.insert(tensors.begin(), tensors.end());
    }
}
