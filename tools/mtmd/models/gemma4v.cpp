#include "models.h"
#include <cmath>

ggml_cgraph * clip_graph_gemma4v::build() {
    ggml_tensor * inp_raw = build_inp_raw();

    // patches = 2 * (patches - 0.5)
    // equivalent to: patches * 2 - 1
    inp_raw = ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f);
    ggml_set_name(inp_raw, "inp_raw_scaled");

    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    ggml_set_name(inp, "inp");
    // note: no patch bias

    ggml_tensor * pos_x = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_x, "pos_x");
    ggml_set_input(pos_x);

    ggml_tensor * pos_y = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_y, "pos_y");
    ggml_set_input(pos_y);

    {
        const int64_t pos_size = model.position_embeddings->ne[1];
        const size_t  nb1      = ggml_row_size(model.position_embeddings->type, n_embd);

        // positional embeddings are stored as lookup tables (one for x, one for y)
        ggml_tensor * tbl_x = ggml_view_2d(ctx0, model.position_embeddings,
                                             n_embd, pos_size, nb1, 0);
        ggml_tensor * tbl_y = ggml_view_2d(ctx0, model.position_embeddings,
                                             n_embd, pos_size, nb1, pos_size * nb1);

        // ggml_get_rows: [n_embd, n_patches]
        ggml_tensor * emb_x = ggml_get_rows(ctx0, tbl_x, pos_x);
        ggml_tensor * emb_y = ggml_get_rows(ctx0, tbl_y, pos_y);

        inp = ggml_add(ctx0, inp, emb_x);
        inp = ggml_add(ctx0, inp, emb_y);
        cb(inp, "pos_embd", -1);
    }

    // similar to build_rope_2d, but use neox ordering
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        const int64_t n_dim  = cur->ne[0];
        const int64_t n_head = cur->ne[1];
        const int64_t n_pos  = cur->ne[2];

        // first half
        ggml_tensor * first;
        {
            first = ggml_view_3d(ctx0, cur,
                n_dim/2, n_head, n_pos,
                cur->nb[1],
                cur->nb[2],
                0);
            first = ggml_rope_ext(
                ctx0,
                first,
                pos_x,      // positions
                nullptr,    // freq factors
                n_dim/2,    // n_dims
                GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f
            );
        }

        // second half
        ggml_tensor * second;
        {
            second = ggml_view_3d(ctx0, cur,
                n_dim/2, n_head, n_pos,
                cur->nb[1],
                cur->nb[2],
                n_dim/2 * ggml_element_size(cur));
            second = ggml_rope_ext(
                ctx0,
                second,
                pos_y,      // positions
                nullptr,    // freq factors
                n_dim/2,    // n_dims
                GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f
            );
        }

        cur = ggml_concat(ctx0, first, second, 0);
        return cur;
    };

    kq_scale = 1.0f;
    ggml_tensor * cur = build_vit(
                        inp, n_patches,
                        NORM_TYPE_RMS,
                        hparams.ffn_op,
                        nullptr, // pos embd is already handled above
                        add_pos);

    // Gemma4VisionPooler
    {
        const int kernel_size = hparams.n_merge;
        GGML_ASSERT(kernel_size > 0);

        // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, 1]
        cur = ggml_cont_4d(ctx0, ggml_transpose(ctx0, cur), n_patches_x, n_patches_y, n_embd, 1);
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG,
                           kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        const int out_x = n_patches_x / kernel_size;
        const int out_y = n_patches_y / kernel_size;
        // [out_x, out_y, n_embd, 1] -> [n_embd, out_x * out_y]
        cur = ggml_reshape_3d(ctx0, cur, out_x * out_y, n_embd, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_scale(ctx0, cur, sqrtf((float)n_embd));
        cb(cur, "pooled", -1);
    }

    // hidden_states = (hidden_states - self.std_bias) * self.std_scale
    if (model.std_bias && model.std_scale) {
        cur = ggml_sub(ctx0, cur, model.std_bias);
        cur = ggml_mul(ctx0, cur, model.std_scale);
        cb(cur, "std_scaled", -1);
    }

    // Gemma4MultimodalEmbedder
    cur = build_mm(model.mm_input_proj_w, cur);
    cb(cur, "projected", -1);

    // embedding_post_projection_norm
    cur = ggml_rms_norm(ctx0, cur, hparams.eps);
    cb(cur, "projected_normed", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_tensor * clip_graph_gemma4v::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    // Gemma4ClippableLinear

    auto it = model.clamp_info_map.find(w->name);
    if (it == model.clamp_info_map.end()) {
        return ggml_mul_mat(ctx0, w, x);
    } else {
        const auto & clamp_info = it->second;
        ggml_tensor * clamped = ggml_clamp(ctx0, x, clamp_info.inp_min, clamp_info.inp_max);
        ggml_tensor * out = ggml_mul_mat(ctx0, w, clamped);
        out = ggml_clamp(ctx0, out, clamp_info.out_min, clamp_info.out_max);
        return out;
    }
}
