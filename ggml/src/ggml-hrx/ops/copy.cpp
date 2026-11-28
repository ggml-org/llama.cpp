#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Must match the dst type codes in copy/cpy_f32.loom and rows/set_rows_f32.loom.
static int ggml_hrx_cpy_dst_code(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 0;
        case GGML_TYPE_F16: return 1;
        default:            return -1;
    }
}

static int64_t ggml_hrx_span(const ggml_tensor * t) {
    return (int64_t) (ggml_nbytes(t) / ggml_type_size(t->type));
}

// Indices and spans are 32-bit launch scalars.
static bool ggml_hrx_fits_i32(const ggml_tensor * t) {
    return ggml_nelements(t) < INT_MAX && ggml_hrx_span(t) < INT_MAX;
}

// The kernel does not use the stride of a size-1 dim. Set it to 0.
static bool ggml_hrx_launch_strides(const ggml_tensor * t, int64_t s[GGML_MAX_DIMS]) {
    if (!ggml_hrx_elem_strides(t, s)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] == 1) {
            s[i] = 0;
        }
        if (s[i] >= INT_MAX) {
            return false;
        }
    }
    return true;
}

bool ggml_hrx_supports_cpy(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src == nullptr || src->type != GGML_TYPE_F32 || ggml_hrx_cpy_dst_code(op->type) < 0 ||
        ggml_nelements(src) != ggml_nelements(op)) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_launch_strides(src, s) || !ggml_hrx_launch_strides(op, s)) {
        return false;
    }
    return ggml_hrx_fits_i32(src) && ggml_hrx_fits_i32(op);
}

bool ggml_hrx_op_cpy(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src    = node->src[0];
    const ggml_tensor * dst    = node;
    const int64_t       n      = ggml_nelements(dst);
    const bool          stream = dst->type == GGML_TYPE_F32 && n % 4 == 0 && ggml_is_contiguous(src) && ggml_is_contiguous(dst);
    const int64_t       groups = stream ? ggml_hrx_stream_groups(ctx->device, n) : 0;
    if (groups > 0) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "copy/cpy_f32.loom";
        desc.symbol = "ggml_cpy_f32_stream";
        desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
        return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, desc), { groups }, { n, groups }, { src, dst });
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ts[GGML_MAX_DIMS];
    ggml_hrx_launch_strides(src, ss);
    ggml_hrx_launch_strides(dst, ts);

    ggml_hrx_kernel_desc desc;
    desc.file   = "copy/cpy_f32.loom";
    desc.symbol = "ggml_cpy_f32";
    desc.config = { { "ggml.cpy.dst_type", std::to_string(ggml_hrx_cpy_dst_code(dst->type)) } };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // src and dst can have different shapes. Both use logical element order.
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n,
            src->ne[0], src->ne[1], src->ne[2], ss[0], ss[1], ss[2], ss[3], ggml_hrx_span(src),
            dst->ne[0], dst->ne[1], dst->ne[2], ts[0], ts[1], ts[2], ts[3], ggml_hrx_span(dst),
        },
        { src, dst });
}

bool ggml_hrx_supports_set_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * ids = op->src[1];
    if (src->type != GGML_TYPE_F32 || ggml_hrx_cpy_dst_code(op->type) < 0 || ids->type != GGML_TYPE_I64) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, s) || !ggml_hrx_elem_strides(ids, s) || !ggml_hrx_elem_strides(op, s)) {
        return false;
    }
    return ggml_hrx_fits_i32(op) && ggml_hrx_fits_i32(src) && ggml_hrx_fits_i32(ids);
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
    desc.config = { { "ggml.set_rows.dst_type", std::to_string(ggml_hrx_cpy_dst_code(node->type)) } };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // node is the dst view. Its binding covers only the rows that the kernel can write.
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, src->ne[0], src->ne[1], src->ne[2],
            ss[0], ss[1], ss[2], ss[3], ggml_hrx_span(src),
            ids->ne[1], ids->ne[2], is[0], is[1], is[2], ggml_hrx_span(ids),
            node->ne[1], ds[0], ds[1], ds[2], ds[3], ggml_hrx_span(node),
        },
        { src, ids, node });
}
