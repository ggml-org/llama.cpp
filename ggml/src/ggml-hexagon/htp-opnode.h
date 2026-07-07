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

    int format_tensor_dims(char * str, size_t max_len, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_len, "NONE");
        }
        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, max_len, "%d:%d", (int) t->ne[0], (int) t->ne[1]);
        } else {
            return snprintf(str, max_len, "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
        }
    }

    void format_op_dims(char * str, size_t max_len, const htp_opnode & node) {
        char * p = str;
        size_t remaining = max_len;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            int written = format_tensor_dims(p, remaining, inputs[0]);
            if (written > 0) {
                p += written;
                remaining -= written;
            }

            for (size_t i = 1; i < inputs.size() && remaining > 0; i++) {
                int w_sep = snprintf(p, remaining, " x ");
                if (w_sep > 0) {
                    p += w_sep;
                    remaining -= w_sep;
                }
                int w_val = format_tensor_dims(p, remaining, inputs[i]);
                if (w_val > 0) {
                    p += w_val;
                    remaining -= w_val;
                }
            }
            int w_arrow = snprintf(p, remaining, " -> ");
            if (w_arrow > 0) {
                p += w_arrow;
                remaining -= w_arrow;
            }
        }

        char self[64];
        format_tensor_dims(self, sizeof(self), node.dst());
        snprintf(p, remaining, "%s", self);
    }

    int format_tensor_strides(char * str, size_t max_len, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_len, "NONE");
        }
        const char * c = ggml_is_contiguous(t) ? "" : "!";
        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, max_len, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } else {
            return snprintf(str, max_len, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    void format_op_strides(char * str, size_t max_len, const htp_opnode & node) {
        char * p = str;
        size_t remaining = max_len;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            int written = format_tensor_strides(p, remaining, inputs[0]);
            if (written > 0) {
                p += written;
                remaining -= written;
            }

            for (size_t i = 1; i < inputs.size() && remaining > 0; i++) {
                int w_sep = snprintf(p, remaining, " x ");
                if (w_sep > 0) {
                    p += w_sep;
                    remaining -= w_sep;
                }
                int w_val = format_tensor_strides(p, remaining, inputs[i]);
                if (w_val > 0) {
                    p += w_val;
                    remaining -= w_val;
                }
            }
            int w_arrow = snprintf(p, remaining, " -> ");
            if (w_arrow > 0) {
                p += w_arrow;
                remaining -= w_arrow;
            }
        }

        char self[64];
        format_tensor_strides(self, sizeof(self), node.dst());
        snprintf(p, remaining, "%s", self);
    }

    int format_tensor_types(char * str, size_t max_len, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_len, "NONE");
        }
        return snprintf(str, max_len, "%s", ggml_type_name(t->type));
    }

    void format_op_types(char * str, size_t max_len, const htp_opnode & node) {
        char * p = str;
        size_t remaining = max_len;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            int written = format_tensor_types(p, remaining, inputs[0]);
            if (written > 0) {
                p += written;
                remaining -= written;
            }

            for (size_t i = 1; i < inputs.size() && remaining > 0; i++) {
                int w_sep = snprintf(p, remaining, " x ");
                if (w_sep > 0) {
                    p += w_sep;
                    remaining -= w_sep;
                }
                int w_val = format_tensor_types(p, remaining, inputs[i]);
                if (w_val > 0) {
                    p += w_val;
                    remaining -= w_val;
                }
            }
            int w_arrow = snprintf(p, remaining, " -> ");
            if (w_arrow > 0) {
                p += w_arrow;
                remaining -= w_arrow;
            }
        }

        int written_dst = format_tensor_types(p, remaining, node.dst());
        if (written_dst > 0) {
            p += written_dst;
            remaining -= written_dst;
        }
    }

    int format_tensor_buff_name(char * str, size_t max_len, const struct ggml_tensor * t) {
        if (t && t->buffer) {
            return snprintf(str, max_len, "%s", ggml_backend_buffer_name(t->buffer));
        }
        return snprintf(str, max_len, "NONE");
    }

    void format_op_buffs(char * str, size_t max_len, const htp_opnode & node) {
        char * p = str;
        size_t remaining = max_len;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            int written = format_tensor_buff_name(p, remaining, inputs[0]);
            if (written > 0) {
                p += written;
                remaining -= written;
            }

            for (size_t i = 1; i < inputs.size() && remaining > 0; i++) {
                int w_sep = snprintf(p, remaining, " x ");
                if (w_sep > 0) {
                    p += w_sep;
                    remaining -= w_sep;
                }
                int w_val = format_tensor_buff_name(p, remaining, inputs[i]);
                if (w_val > 0) {
                    p += w_val;
                    remaining -= w_val;
                }
            }
            int w_arrow = snprintf(p, remaining, " -> ");
            if (w_arrow > 0) {
                p += w_arrow;
                remaining -= w_arrow;
            }
        }

        int written_dst = format_tensor_buff_name(p, remaining, node.dst());
        if (written_dst > 0) {
            p += written_dst;
            remaining -= written_dst;
        }
    }

    int format_tensor_name(char * str, size_t max_len, const struct ggml_tensor * t) {
        if (t) {
            return snprintf(str, max_len, "%s", t->name);
        }
        return snprintf(str, max_len, "NONE");
    }

    void format_op_names(char * str, size_t max_len, const htp_opnode & node) {
        char * p = str;
        size_t remaining = max_len;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            int written = format_tensor_name(p, remaining, inputs[0]);
            if (written > 0) {
                p += written;
                remaining -= written;
            }

            for (size_t i = 1; i < inputs.size() && remaining > 0; i++) {
                int w_sep = snprintf(p, remaining, " x ");
                if (w_sep > 0) {
                    p += w_sep;
                    remaining -= w_sep;
                }
                int w_val = format_tensor_name(p, remaining, inputs[i]);
                if (w_val > 0) {
                    p += w_val;
                    remaining -= w_val;
                }
            }
            int w_arrow = snprintf(p, remaining, " -> ");
            if (w_arrow > 0) {
                p += w_arrow;
                remaining -= w_arrow;
            }
        }

        int written_dst = format_tensor_name(p, remaining, node.dst());
        if (written_dst > 0) {
            p += written_dst;
            remaining -= written_dst;
        }
    }

    void format(const htp_opnode & node) {
        format_op_dims(dims, sizeof(dims), node);
        format_op_strides(strides, sizeof(strides), node);
        format_op_types(types, sizeof(types), node);
        format_op_buffs(buffs, sizeof(buffs), node);
        format_op_names(names, sizeof(names), node);
    }

    htp_opformat() {}
    htp_opformat(const htp_opnode & node) { format(node); }
};

#endif // HTP_OPNODE_H
