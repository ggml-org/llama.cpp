#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>


// utility class that provides translation between rope and rope_comp
// use for testing purposes
struct rope_utils {
    bool use_comp = false;
    rope_utils() = default;
    rope_utils(bool use_comp) : use_comp(use_comp) {}

    ggml_tensor * rope_multi(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                struct ggml_tensor  * c,
                int                   n_dims,
                int                   sections[GGML_MROPE_SECTIONS],
                int                   mode,
                int                   n_ctx_orig,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow) {
        if (use_comp) {
            int legacy_type = GGML_ROPE_TYPE_NEOX; // the only mode supported by legacy m-rope kernel
            int n_dims_corrected = n_dims;
            float freq_base_corrected = freq_base;
            ggml_tensor * pos = ggml_cast(ctx, b, GGML_TYPE_F32); // pos must be F32

            // pos must be 4D
            pos = ggml_reshape_2d(ctx, pos, b->ne[0]/4, 4);

            if (mode == GGML_ROPE_TYPE_VISION) {
                // correct n_dims for vision
                // instead of rotating n_dims/2, we actually rotate all dims
                // in other words, there are no dimensions that are not rotated
                n_dims_corrected = n_dims * 2;
                // correct theta_scale for vision
                // only n_dims/2 are rotated for each dimension (we have 2 dimensions: x, y)
                // to adjust theta_scale, we need to adjust freq_base accordingly
                freq_base_corrected = powf(freq_base, 2.0f);
            }

            ggml_tensor * x = this->rope_ext(
                ctx, a, pos, c, n_dims_corrected, legacy_type, n_ctx_orig,
                freq_base_corrected, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
            GGML_ASSERT(x->op == GGML_OP_ROPE_COMP);
            x = ggml_rope_comp_set_multi(ctx, x, mode, sections);
            return x;
        } else {
            return ggml_rope_multi(
                ctx, a, b, c, n_dims, sections, mode, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
        }
    }

    struct ggml_tensor * rope_ext(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                struct ggml_tensor  * c,
                int                   n_dims,
                int                   mode,
                int                   n_ctx_orig,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow) {
        if (use_comp) {
            b = ggml_cast(ctx, b, GGML_TYPE_F32); // pos must be F32
            bool is_neox = (mode == GGML_ROPE_TYPE_NEOX);
            float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
            ggml_tensor * x = ggml_rope_comp(
                ctx, a, b, n_dims,
                theta_scale,
                is_neox ? GGML_ROPE_ORDERING_NEOX : GGML_ROPE_ORDERING_NORMAL);
            if (ext_factor != 0.0f) {
                // apply yarn mscale
                attn_factor *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            x = ggml_rope_comp_set_yarn(ctx, x, n_ctx_orig, n_dims,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
            if (c) {
                x = ggml_rope_comp_set_freq_factors(ctx, x, c);
            }
            return x;
        } else {
            return ggml_rope_ext(
                ctx, a, b, c, n_dims, mode, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
        }
    }

    //
    // inplace version
    //

    ggml_tensor * rope_multi_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                struct ggml_tensor  * c,
                int                   n_dims,
                int                   sections[GGML_MROPE_SECTIONS],
                int                   mode,
                int                   n_ctx_orig,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow) {
        if (use_comp) {
            // no-op for now
            return rope_multi(
                ctx, a, b, c, n_dims, sections, mode, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
        } else {
            return ggml_rope_multi_inplace(
                ctx, a, b, c, n_dims, sections, mode, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
        }
    }

    struct ggml_tensor * rope_ext_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                struct ggml_tensor  * c,
                int                   n_dims,
                int                   mode,
                int                   n_ctx_orig,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow) {
        if (use_comp) {
            // no-op for now
            return rope_ext(
                ctx, a, b, c, n_dims, mode, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
        } else {
            return ggml_rope_ext(
                ctx, a, b, c, n_dims, mode, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
        }
    }
};
