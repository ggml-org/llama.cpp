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
