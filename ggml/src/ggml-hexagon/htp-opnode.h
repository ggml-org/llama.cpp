#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"

#include <string>
#include <vector>
#include <stdio.h>
#include "htp-ops.h"

struct htp_opnode {
    ggml_tensor * node = nullptr;

    std::vector<ggml_tensor *> fused;

    htp_op_code opcode = HTP_OP_INVALID;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
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

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
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
        std::vector<const ggml_tensor *> inputs;
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

        auto add_input = [&](const ggml_tensor * t) {
            if (t && !contains(outputs, t) && !contains(inputs, t)) {
                inputs.push_back(t);
            }
        };

        for (int i = 0; i < GGML_MAX_SRC && node->src[i]; i++) {
            add_input(node->src[i]);
        }
        for (const auto * f : fused) {
            for (int i = 0; i < GGML_MAX_SRC && f->src[i]; i++) {
                add_input(f->src[i]);
            }
        }
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

    int format_tensor_dims(char * str, const struct ggml_tensor * t) {
        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return sprintf(str, "%d:%d", (int) t->ne[0], (int) t->ne[1]);
        } else {
            return sprintf(str, "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
        }
    }

    void format_op_dims(char * str, const htp_opnode & node) {
        char * p = str;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += format_tensor_dims(p, inputs[0]);

            for (size_t i = 1; i < inputs.size(); i++) {
                p += sprintf(p, " x ");
                p += format_tensor_dims(p, inputs[i]);
            }

            p += sprintf(p, " -> ");
        }

        char self[64];
        format_tensor_dims(self, node.dst());
        p += sprintf(p, "%s", self);
    }

    int format_tensor_strides(char * str, const struct ggml_tensor * t) {
        const char * c = ggml_is_contiguous(t) ? "" : "!";

        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return sprintf(str, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } else {
            return sprintf(str, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    void format_op_strides(char * str, const htp_opnode & node) {
        char * p = str;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += format_tensor_strides(p, inputs[0]);

            for (size_t i = 1; i < inputs.size(); i++) {
                p += sprintf(p, " x ");
                p += format_tensor_strides(p, inputs[i]);
            }

            p += sprintf(p, " -> ");
        }

        char self[64];
        format_tensor_strides(self, node.dst());
        p += sprintf(p, "%s", self);
    }

    void format_op_types(char * str, const htp_opnode & node) {
        char * p = str;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += sprintf(p, "%s", ggml_type_name(inputs[0]->type));

            for (size_t i = 1; i < inputs.size(); i++) {
                p += sprintf(p, " x ");
                p += sprintf(p, "%s", ggml_type_name(inputs[i]->type));
            }

            p += sprintf(p, " -> ");
        }

        p += sprintf(p, "%s", ggml_type_name(node.dst()->type));
    }

    const char * tensor_buff_name(const struct ggml_tensor * t) {
        if (t->buffer) {
            return ggml_backend_buffer_name(t->buffer);
        }
        return "NONE";
    }

    void format_op_buffs(char * str, const htp_opnode & node) {
        char * p = str;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += sprintf(p, "%s", tensor_buff_name(inputs[0]));

            for (size_t i = 1; i < inputs.size(); i++) {
                p += sprintf(p, " x ");
                p += sprintf(p, "%s", tensor_buff_name(inputs[i]));
            }

            p += sprintf(p, " -> ");
        }

        p += sprintf(p, "%s", tensor_buff_name(node.dst()));
    }

    void format_op_names(char * str, const htp_opnode & node) {
        char * p = str;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += sprintf(p, "%s", inputs[0]->name);

            for (size_t i = 1; i < inputs.size(); i++) {
                p += sprintf(p, " x ");
                p += sprintf(p, "%s", inputs[i]->name);
            }

            p += sprintf(p, " -> ");
        }

        p += sprintf(p, "%s", node.dst()->name);
    }

    void format(const htp_opnode & node) {
        format_op_dims(dims, node);
        format_op_strides(strides, node);
        format_op_types(types, node);
        format_op_buffs(buffs, node);
        format_op_names(names, node);
    }

    htp_opformat() {}
    htp_opformat(const htp_opnode & node) { format(node); }
};

#endif // HTP_OPNODE_H
