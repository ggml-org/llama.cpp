#include "ggml-hrx-ops.h"

#include <climits>
#include <cmath>
#include <cstring>
#include <string>

static int64_t ggml_hrx_rope_float_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (int64_t) bits;
}

bool ggml_hrx_supports_rope(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * pos = op->src[1];
    const ggml_tensor * ff  = op->src[2];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32) {
        return false;
    }
    if (!ggml_is_contiguous(pos)) {
        return false;
    }

    const int32_t * params = (const int32_t *) op->op_params;
    const int n_dims = params[1];
    const int mode   = params[2];
    const int n_offs = params[15];
    if (mode != GGML_ROPE_TYPE_NORMAL) {
        return false;
    }
    if (src->ne[0] % 2 != 0 || n_dims <= 0 || n_dims % 2 != 0 || n_offs < 0 || n_offs % 2 != 0 ||
        n_offs + n_dims > src->ne[0]) {
        return false;
    }
    if (ff != nullptr && (ff->type != GGML_TYPE_F32 || !ggml_is_contiguous(ff) || ff->ne[0] < n_dims / 2)) {
        return false;
    }

    // The kernel reads and writes each pair as two adjacent elements.
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds) || ss[0] != 1 || ds[0] != 1) {
        return false;
    }
    // Counts and spans are 32-bit launch scalars.
    return ggml_nelements(op) < INT_MAX && ggml_nbytes(src) / 4 < INT_MAX && ggml_nbytes(op) / 4 < INT_MAX &&
           ggml_nelements(pos) < INT_MAX && (ff == nullptr || ggml_nelements(ff) < INT_MAX);
}

bool ggml_hrx_op_rope(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const ggml_tensor * pos = node->src[1];
    const ggml_tensor * ff  = node->src[2];

    const int32_t * params = (const int32_t *) node->op_params;
    const int n_dims     = params[1];
    const int n_ctx_orig = params[4];
    const int n_offs     = params[15];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   params +  5, sizeof(float));
    memcpy(&freq_scale,  params +  6, sizeof(float));
    memcpy(&ext_factor,  params +  7, sizeof(float));
    memcpy(&attn_factor, params +  8, sizeof(float));
    memcpy(&beta_fast,   params +  9, sizeof(float));
    memcpy(&beta_slow,   params + 10, sizeof(float));

    if (ggml_nelements(node) == 0) {
        return true;
    }

    const float log2_theta_scale = (float) (-2.0 / n_dims * log2((double) freq_base));
    float corr_dims[2] = { 0.0f, 0.0f };
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(node), corr_dims);
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    ggml_hrx_kernel_desc desc;
    desc.file   = "rope/rope_f32.loom";
    desc.symbol = "ggml_rope_f32";
    desc.config = { { "ggml.rope.freq_factors", ff != nullptr ? "1" : "0" } };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // Without freq factors, the kernel does not read this binding. Bind pos instead.
    const ggml_tensor * ff_buf = ff != nullptr ? ff : pos;

    const int64_t n = ggml_nelements(node) / 2;
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, src->ne[0] / 2, src->ne[1], src->ne[2], n_dims, n_offs,
            ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
            ggml_nelements(pos), ggml_nelements(ff_buf),
            ggml_hrx_rope_float_bits(log2_theta_scale), ggml_hrx_rope_float_bits(freq_scale),
            ggml_hrx_rope_float_bits(ext_factor), ggml_hrx_rope_float_bits(corr_dims[0]),
            ggml_hrx_rope_float_bits(corr_dims[1]), ggml_hrx_rope_float_bits(mscale),
        },
        { src, pos, ff_buf, node });
}
