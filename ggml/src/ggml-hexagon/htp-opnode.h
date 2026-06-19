#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"

#include <algorithm>
#include <string>
#include <vector>
#include <stdio.h>
#include "htp-ops.h"
#include "htp/matmul-ops.h"

struct htp_opnode {
    ggml_tensor * node = nullptr;

    std::vector<ggml_tensor *> fused;

    htp_op_code opcode = HTP_OP_INVALID;

    std::vector<ggml_tensor *> extra_dsts;

    int32_t kernel_params[HTP_OP_MAX_KERN_PARAMS] = {0};

    htp_opnode(ggml_tensor * node = nullptr, std::vector<ggml_tensor *> fused = {}, htp_op_code opcode = HTP_OP_INVALID, std::vector<ggml_tensor *> extra_dsts = {})
        : node(node), fused(std::move(fused)), opcode(opcode), extra_dsts(std::move(extra_dsts)) {}

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    void add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.push_back(t);
        if (extra_dst) {
            extra_dsts.push_back(t);
        }
    }

    std::vector<const ggml_tensor *> get_outputs() const {
        std::vector<const ggml_tensor *> res;
        if (extra_dsts.empty()) {
            res.push_back(dst());
        } else {
            res.push_back(node);
            for (const auto * x : extra_dsts) {
                res.push_back(x);
            }
        }
        return res;
    }

    const ggml_tensor * src0() const {
        return node->src[0];
    }

    const ggml_tensor * src1() const {
        return node->src[1];
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    bool stackable() const {
        switch (this->op()) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
                return ggml_is_quantized(this->src0()->type);
            default:
                return false;
        }
    }

    bool same_input(const htp_opnode& n) const {
        return n.src1() == this->src1();
    }

    std::vector<const ggml_tensor *> get_inputs() const {
        if (fused.empty()) {
            int last_non_null = -1;
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (node->src[i]) {
                    last_non_null = i;
                }
            }
            std::vector<const ggml_tensor *> inputs(last_non_null + 1, nullptr);
            for (int i = 0; i <= last_non_null; i++) {
                inputs[i] = node->src[i];
            }
            return inputs;
        }

        std::vector<const ggml_tensor *> inputs(GGML_MAX_SRC, nullptr);
        std::vector<const ggml_tensor *> outputs;
        outputs.push_back(node);
        for (const auto * f : fused) {
            outputs.push_back(f);
        }

        auto contains = [&](const std::vector<const ggml_tensor *> & vec, const ggml_tensor * t) {
            for (const auto * x : vec) {
                if (x == t) return true;
            }
            return false;
        };

        int count = 0;
        auto add_input = [&](const ggml_tensor * t) {
            if (t && !contains(outputs, t) && !contains(inputs, t)) {
                if (count < (int)inputs.size()) {
                    inputs[count++] = t;
                } else {
                    inputs.push_back(t);
                }
            }
        };

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node->src[i]) {
                add_input(node->src[i]);
            }
        }
        for (const auto * f : fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (f->src[i]) {
                    add_input(f->src[i]);
                }
            }
        }

        inputs.resize(count);
        return inputs;
    }

    std::string op_name() const {
        if (fused.empty()) {
            return ggml_op_desc(node);
        }
        std::string name = ggml_op_desc(node);
        for (const auto * f : fused) {
            name += "+";
            name += ggml_op_desc(f);
        }
        return name;
    }
};

struct htp_opformat {
    char strides[64 * GGML_MAX_SRC];
    char dims[64 * GGML_MAX_SRC];
    char types[16 * GGML_MAX_SRC];
    char buffs[64 * GGML_MAX_SRC];
    char names[64 * GGML_MAX_SRC];
    char kparams[128];

    int format_tensor_dims(char * str, size_t max_size, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_size, "NONE");
        }
        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, max_size, "%d:%d", (int) t->ne[0], (int) t->ne[1]);
        } else {
            return snprintf(str, max_size, "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
        }
    }

    void format_op_dims(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        char self[64];
        format_tensor_dims(self, sizeof(self), node.dst());
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    int format_tensor_strides(char * str, size_t max_size, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_size, "NONE");
        }
        const char * c = ggml_is_contiguous(t) ? "" : "!";

        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } else {
            return snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    void format_op_strides(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        char self[64];
        format_tensor_strides(self, sizeof(self), node.dst());
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    void format_op_types(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(node.dst()->type)), (size_t)(p_end - p));
        }
    }

    const char * tensor_buff_name(const struct ggml_tensor * t) {
        if (t && t->buffer) {
            return ggml_backend_buffer_name(t->buffer);
        }
        return "NONE";
    }

    void format_op_buffs(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(node.dst())), (size_t)(p_end - p));
        }
    }

    void format_op_names(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", node.dst()->name), (size_t)(p_end - p));
        }
    }
    void format_kernel_params(char * str, size_t max_size, const htp_opnode & node) {
        if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
            node.opcode == HTP_OP_MUL_MAT_QKV || node.opcode == HTP_OP_MUL_MAT_FFN) {
            const auto * kparams = (const struct htp_mm_kernel_params *) node.kernel_params;
            const char * path = "unknown";
            int32_t type = kparams->kernel_type;
            if (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
                path = "hmx-tiled";
            } else if (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_QUANT_ROW    || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            } else if (type == HTP_MM_KERNEL_HVX_F16_F16_DDR  || type == HTP_MM_KERNEL_HVX_F16_F32_DDR ||
                       type == HTP_MM_KERNEL_HVX_F32_F32_DDR  || type == HTP_MM_KERNEL_HVX_F32_F16_DDR ||
                       type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
                path = "hvx-flat";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
        } else {
            snprintf(str, max_size, "----");
        }
    }

    void format(const htp_opnode & node) {
        format_op_dims(dims, sizeof(dims), node);
        format_op_strides(strides, sizeof(strides), node);
        format_op_types(types, sizeof(types), node);
        format_op_buffs(buffs, sizeof(buffs), node);
        format_op_names(names, sizeof(names), node);
        format_kernel_params(kparams, sizeof(kparams), node);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(const htp_opnode & node) { format(node); }
};

#endif // HTP_OPNODE_H
