#include "models.h"
#include <cmath>

ggml_cgraph * clip_graph_gemma4uv::build() {
    ggml_tensor * inp_raw = build_inp_raw();

    // Gemma4UnifiedVisionEmbedder uses default pytorch LayerNorm, not RMSNorm
    float eps = 1e-5f; // default eps for pytorch LayerNorm

    ggml_tensor * inp = nullptr;
    {
        // note: we cannot use ggml_conv_2d here because we need to apply norm after im2col
        auto c = inp_raw->ne[2];
        ggml_tensor * kernel = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, patch_size, patch_size, c);
        inp = ggml_im2col(ctx0, kernel, inp_raw, patch_size, patch_size, 0, 0, 1, 1, true, inp_raw->type);
        // inp shape: [patch_size * patch_size * c, n_patches_w, n_patches_h]

        inp = ggml_reshape_2d(ctx0, inp, inp->ne[0], inp->ne[1] * inp->ne[2] * inp->ne[3]);
        inp = build_norm(inp, model.patch_norm_1_w, model.patch_norm_1_b, NORM_TYPE_NORMAL, eps, -1);
        // inp shape: [patch_size * patch_size * c, n_patches]

        inp = ggml_mul_mat(ctx0, model.patch_embeddings_0, inp);
        inp = ggml_add(ctx0, inp, model.patch_bias);
        // inp shape: [n_embd, n_patches]

        inp = build_norm(inp, model.patch_norm_2_w, model.patch_norm_2_b, NORM_TYPE_NORMAL, eps, -1);
    }

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

        // pos_norm
        inp = build_norm(inp, model.patch_norm_3_w, model.patch_norm_3_b, NORM_TYPE_NORMAL, eps, -1);
    }

    auto cur = inp;

    // Gemma4UnifiedMultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = build_mm(model.mm_input_proj_w, cur);
        cb(cur, "projected", -1);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
