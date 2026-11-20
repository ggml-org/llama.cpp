#include "ggml-hrx-ops.h"

#include <climits>

// Counts and spans are 32-bit launch scalars.
static bool ggml_hrx_rows_fits(const ggml_tensor * t) {
    return ggml_nelements(t) < INT_MAX && ggml_nbytes(t) / ggml_type_size(t->type) < INT_MAX;
}

static int64_t ggml_hrx_rows_span(const ggml_tensor * t) {
    return (int64_t) (ggml_nbytes(t) / ggml_type_size(t->type));
}

bool ggml_hrx_supports_get_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * ids = op->src[1];
    if (src->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(op)) {
        return false;
    }
    int64_t is[GGML_MAX_DIMS];
    int64_t ss[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(ids, is) || !ggml_hrx_rows_fits(ids) || !ggml_hrx_elem_strides(src, ss)) {
        return false;
    }
    // The kernels move 4 values of one row. All indices must be below 2^31 - 8.
    return op->ne[0] % 4 == 0 && ss[0] == 1 && ggml_nelements(op) <= INT_MAX - 4096 && ggml_hrx_rows_span(src) <= INT_MAX - 4096;
}

bool ggml_hrx_op_get_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const ggml_tensor * ids = node->src[1];

    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t is[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(ids, is);

    const int64_t groups = ggml_hrx_stream_groups(ctx->device, n);

    ggml_hrx_kernel_desc desc;
    desc.file = "rows/get_rows_f32.loom";
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    if (groups > 0) {
        desc.symbol = "ggml_get_rows_f32_stream";
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { groups },
            {
                n, groups, node->ne[0], node->ne[1], node->ne[2],
                src->ne[1], ss[1], ss[2], ss[3], ggml_hrx_rows_span(src),
                is[0], is[1], is[2], ggml_hrx_rows_span(ids),
            },
            { src, ids, node });
    }
    desc.symbol = "ggml_get_rows_f32_rows";
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, node->ne[0], node->ne[1], node->ne[2],
            src->ne[1], ss[0], ss[1], ss[2], ss[3], ggml_hrx_rows_span(src),
            is[0], is[1], is[2], ggml_hrx_rows_span(ids),
        },
        { src, ids, node });
}

bool ggml_hrx_supports_set_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * ids = op->src[1];
    if (src->type != GGML_TYPE_F32 || (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) || ids->type != GGML_TYPE_I64) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, s) || !ggml_hrx_elem_strides(ids, s) || !ggml_hrx_elem_strides(op, s)) {
        return false;
    }
    return ggml_hrx_rows_fits(op) && ggml_hrx_rows_fits(src) && ggml_hrx_rows_fits(ids);
}

bool ggml_hrx_op_set_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const ggml_tensor * ids = node->src[1];

    const int64_t n = ggml_nelements(src);
    if (n == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t is[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(ids, is);
    ggml_hrx_elem_strides(node, ds);

    ggml_hrx_kernel_desc desc;
    desc.file   = "rows/set_rows_f32.loom";
    desc.symbol = "ggml_set_rows_f32";
    desc.config = { { "ggml.set_rows.dst_type", node->type == GGML_TYPE_F16 ? "1" : "0" } };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // node is the dst view. Its binding covers only the rows that the kernel can write.
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, src->ne[0], src->ne[1], src->ne[2],
            ss[0], ss[1], ss[2], ss[3], ggml_hrx_rows_span(src),
            ids->ne[1], ids->ne[2], is[0], is[1], is[2], ggml_hrx_rows_span(ids),
            node->ne[1], ds[0], ds[1], ds[2], ds[3], ggml_hrx_rows_span(node),
        },
        { src, ids, node });
}
