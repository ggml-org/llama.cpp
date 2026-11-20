#include "ggml-hrx-ops.h"

#include <climits>
#include <cstring>

// llama.cpp runs SCALE in place on recurrent state in device buffers. The CPU cannot run it there.
bool ggml_hrx_supports_scale(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    return src != nullptr && src->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_are_same_shape(src, op) &&
           ggml_is_contiguous(src) && ggml_is_contiguous(op) && ggml_nelements(op) < INT_MAX;
}

bool ggml_hrx_op_scale(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }
    float    p[2] = { ggml_get_op_params_f32(node, 0), ggml_get_op_params_f32(node, 1) };
    uint32_t bits[2];
    std::memcpy(bits, p, sizeof(bits));

    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/scale_f32.loom";
    desc.symbol = "ggml_scale_f32";
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    return ggml_hrx_dispatch(ctx, kernel, { n }, { n, (int64_t) bits[0], (int64_t) bits[1] }, { node->src[0], node });
}

// The last workgroup can go 4092 elements past n (1024 lanes). All offsets must stay below 2^31 - 8.
static bool ggml_hrx_row_stride(const ggml_tensor * t, int64_t * rs) {
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(t, s) || s[0] != 1 || ggml_nbytes(t) / 4 > (size_t) INT_MAX - 4096) {
        return false;
    }
    *rs = 0;
    int64_t next = -1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] == 1) {
            continue;
        }
        if (next >= 0 && s[i] != next) {
            return false;
        }
        if (next < 0) {
            *rs = s[i];
        }
        next = s[i] * t->ne[i];
    }
    return *rs < INT_MAX;
}

bool ggml_hrx_supports_glu(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU || a == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || (b != nullptr && b->type != GGML_TYPE_F32)) {
        return false;
    }
    // The kernel does 4 elements for each work item.
    if (!ggml_is_contiguous(op) || op->ne[0] % 4 != 0) {
        return false;
    }
    if (b != nullptr) {
        if (!ggml_are_same_shape(a, op) || !ggml_are_same_shape(b, op)) {
            return false;
        }
    } else if ((a->ne[0] != 2 * op->ne[0] && a->ne[0] != 2 * op->ne[0] + 1) || a->ne[1] != op->ne[1] ||
               a->ne[2] != op->ne[2] || a->ne[3] != op->ne[3]) {
        // With an odd row length, the last element is not used. The CPU backend does the same.
        return false;
    }
    int64_t rs;
    return ggml_hrx_row_stride(a, &rs) && (b == nullptr || ggml_hrx_row_stride(b, &rs));
}

bool ggml_hrx_op_glu(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];

    int64_t a_off = 0;
    int64_t b_off = 0;
    // The split form ignores swap. src0 is always the activated input. The CPU backend does the same.
    if (b == nullptr) {
        b = a;
        if (ggml_get_op_params_i32(node, 1) != 0) {
            a_off = node->ne[0];
        } else {
            b_off = node->ne[0];
        }
    }

    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }

    int64_t a_rs;
    int64_t b_rs;
    ggml_hrx_row_stride(a, &a_rs);
    ggml_hrx_row_stride(b, &b_rs);

    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/swiglu_f32.loom";
    desc.symbol = "ggml_swiglu_f32_rows";
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, node->ne[0],
            a_off, a_rs, (int64_t) (ggml_nbytes(a) / 4),
            b_off, b_rs, (int64_t) (ggml_nbytes(b) / 4),
        },
        { a, b, node });
}
