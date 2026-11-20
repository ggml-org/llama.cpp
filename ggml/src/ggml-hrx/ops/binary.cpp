#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Must match the op codes in elementwise/binary_f32.loom.
static int ggml_hrx_binary_code(ggml_op op) {
    switch (op) {
        case GGML_OP_ADD: return 0;
        case GGML_OP_MUL: return 2;
        default:          return -1;
    }
}

// Row r of dst uses row (r % rows) of x, at element (r % rows) * rs.
static bool ggml_hrx_binary_rows_form(const ggml_tensor * x, const ggml_tensor * dst, int64_t * rows, int64_t * rs) {
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(x, s)) {
        return false;
    }
    *rows = 1;
    *rs   = 0;
    int64_t next  = 0;
    bool    bcast = false;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (dst->ne[i] == 1) {
            continue;
        }
        if (x->ne[i] == 1) {
            bcast = true;
            continue;
        }
        if (bcast || x->ne[i] != dst->ne[i] || (*rows > 1 && s[i] != next)) {
            return false;
        }
        if (*rows == 1) {
            *rs = s[i];
        }
        *rows *= x->ne[i];
        next = s[i] * x->ne[i];
    }
    return *rs < INT_MAX;
}

bool ggml_hrx_supports_binary(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (ggml_hrx_binary_code(op->op) < 0 || a == nullptr || b == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    // The kernel does 4 elements for each work item.
    if (!ggml_is_contiguous(op) || !ggml_are_same_shape(a, op) || !ggml_can_repeat(b, op) || op->ne[0] % 4 != 0 ||
        a->nb[0] != sizeof(float)) {
        return false;
    }
    if (b->ne[0] != 1 && (b->ne[0] != op->ne[0] || b->nb[0] != sizeof(float))) {
        return false;
    }
    // The last workgroup can go 4092 elements past n (1024 lanes). All offsets must stay below 2^31 - 8.
    const int64_t max_elems = INT_MAX - 4096;
    if (ggml_nelements(op) > max_elems || (int64_t) (ggml_nbytes(a) / 4) > max_elems || (int64_t) (ggml_nbytes(b) / 4) > max_elems) {
        return false;
    }
    int64_t rows;
    int64_t rs;
    return ggml_hrx_binary_rows_form(a, op, &rows, &rs) && ggml_hrx_binary_rows_form(b, op, &rows, &rs);
}

bool ggml_hrx_op_binary(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (ggml_nelements(node) == 0) {
        return true;
    }
    int64_t a_rows;
    int64_t a_rs;
    int64_t b_rows;
    int64_t b_rs;
    ggml_hrx_binary_rows_form(a, node, &a_rows, &a_rs);
    ggml_hrx_binary_rows_form(b, node, &b_rows, &b_rs);

    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/binary_f32.loom";
    desc.symbol = "ggml_binary_f32_rows";
    desc.config = {
        { "ggml.binary.op", std::to_string(ggml_hrx_binary_code(node->op)) },
        { "ggml.binary.b_bcast0", b->ne[0] == 1 ? "1" : "0" },
    };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n = ggml_nelements(node);
    return ggml_hrx_dispatch(ctx, kernel, { n },
        { n, node->ne[0], a_rs, (int64_t) (ggml_nbytes(a) / 4), b_rows, b_rs, (int64_t) (ggml_nbytes(b) / 4) },
        { a, b, node });
}
