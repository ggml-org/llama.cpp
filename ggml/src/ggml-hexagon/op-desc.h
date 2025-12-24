#ifndef OP_DESC_H
#define OP_DESC_H

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"

#include <string>
#include <stdio.h>

struct op_desc {
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

    void format_op_dims(char * str, const struct ggml_tensor * t) {
        char * p = str;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            p += format_tensor_dims(p, t->src[0]);

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                p += sprintf(p, " x ");
                p += format_tensor_dims(p, t->src[i]);
            }

            p += sprintf(p, " -> ");
        }

        // format self dims separately for better visual alignment
        char self[64];
        format_tensor_dims(self, t);

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

    void format_op_strides(char * str, const struct ggml_tensor * t) {
        char * p = str;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            p += format_tensor_strides(p, t->src[0]);

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                p += sprintf(p, " x ");
                p += format_tensor_strides(p, t->src[i]);
            }

            p += sprintf(p, " -> ");
        }

        // format self dims separately for better visual alignment
        char self[64];
        format_tensor_strides(self, t);

        p += sprintf(p, "%s", self);
    }

    void format_op_types(char * str, const struct ggml_tensor * t) {
        char * p = str;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            p += sprintf(p, "%s", ggml_type_name(t->src[0]->type));

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                p += sprintf(p, " x ");
                p += sprintf(p, "%s", ggml_type_name(t->src[i]->type));
            }

            p += sprintf(p, " -> ");
        }

        p += sprintf(p, "%s", ggml_type_name(t->type));
    }

    const char * tensor_buff_name(const struct ggml_tensor * t) {
        if (t->buffer) {
            return ggml_backend_buffer_name(t->buffer);
        }
        return "NONE";
    }

    void format_op_buffs(char * str, const struct ggml_tensor * t) {
        char * p = str;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            p += sprintf(p, "%s", tensor_buff_name(t->src[0]));

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                p += sprintf(p, " x ");
                p += sprintf(p, "%s", tensor_buff_name(t->src[i]));
            }

            p += sprintf(p, " -> ");
        }

        p += sprintf(p, "%s", tensor_buff_name(t));
    }

    void format_op_names(char * str, const struct ggml_tensor * t) {
        char * p = str;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            p += sprintf(p, "%s", t->src[0]->name);

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                p += sprintf(p, " x ");
                p += sprintf(p, "%s", t->src[i]->name);
            }

            p += sprintf(p, " -> ");
        }

        p += sprintf(p, "%s", t->name);
    }

    void format(const ggml_tensor * op) {
        format_op_dims(dims, op);
        format_op_strides(strides, op);
        format_op_types(types, op);
        format_op_buffs(buffs, op);
        format_op_names(names, op);
    }

    op_desc() {}
    op_desc(const ggml_tensor * op) { format(op); }
};

#endif // OP_DESC_H
