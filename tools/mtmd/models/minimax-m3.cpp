#include "models.h"

// MiniMax-M3 vision graph

ggml_tensor * clip_graph_minimax_m3::apply_rope(
        ggml_tensor * x, ggml_tensor * pos_t, ggml_tensor * pos_h, ggml_tensor * pos_w) {
    const int64_t Hn  = x->ne[1];
    const int64_t P   = x->ne[2];
    const size_t  es  = ggml_element_size(x);
    const int     dh  = (int) x->ne[0];
    const int     axd = 2 * ((2 * (dh / 2) / 3) / 2);
    
    GGML_ASSERT(x->nb[0] == es);
    GGML_ASSERT(3 * axd <= dh);
    
    const float   th  = hparams.rope_theta;
    auto sl = [&](int off, int n) {
        return ggml_cont(ctx0, ggml_view_3d(ctx0, x, n, Hn, P, x->nb[1], x->nb[2], (size_t) off * es));
    };
    ggml_tensor * t   = sl(0,        axd);
    ggml_tensor * h   = sl(axd,      axd);
    ggml_tensor * w   = sl(2 * axd,  axd);
    ggml_tensor * pad = sl(3 * axd,  dh - 3 * axd);
    
    t = ggml_rope_ext(ctx0, t, pos_t, nullptr, axd, GGML_ROPE_TYPE_NEOX, 0, th, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    h = ggml_rope_ext(ctx0, h, pos_h, nullptr, axd, GGML_ROPE_TYPE_NEOX, 0, th, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    w = ggml_rope_ext(ctx0, w, pos_w, nullptr, axd, GGML_ROPE_TYPE_NEOX, 0, th, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_concat(ctx0, ggml_concat(ctx0, ggml_concat(ctx0, t, h, 0), w, 0), pad, 0);
}

ggml_cgraph * clip_graph_minimax_m3::build() {
    GGML_ASSERT(model.patch_bias     == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(model.patch_embeddings_0 && model.patch_embeddings_1);
    GGML_ASSERT(model.mm_1_w && model.mm_2_w);
    GGML_ASSERT(model.mm_merge_fc1_w && model.mm_merge_fc2_w);

    const int batch_size = 1;
    const int n_pos      = n_patches;
    const int merge      = 2;

    // patch embedding
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_add(ctx0,
        ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1),
        ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));

    // spatial merge
    {
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);
        inp = ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    ggml_tensor * pos_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_t, "minimax_pos_t"); ggml_set_input(pos_t);
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_h, "minimax_pos_h"); ggml_set_input(pos_h);
    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_w, "minimax_pos_w"); ggml_set_input(pos_w);

    ggml_tensor * inpL = build_vit(
        inp, n_pos, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr,
        [&](ggml_tensor * c, const clip_layer &) {
            return apply_rope(c, pos_t, pos_h, pos_w);
        });

    // projector
    ggml_tensor * emb = inpL;
    emb = build_ffn(emb, model.mm_1_w, model.mm_1_b,
                    nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b, FFN_GELU_ERF, -1);

    const int64_t proj = emb->ne[0];
    emb = ggml_reshape_2d(ctx0, emb, proj * merge * merge, n_pos / (merge * merge));

    emb = build_ffn(emb, model.mm_merge_fc1_w, model.mm_merge_fc1_b,
                    nullptr, nullptr,
                    model.mm_merge_fc2_w, model.mm_merge_fc2_b, FFN_GELU_ERF, -1);

    ggml_build_forward_expand(gf, emb);
    return gf;
}
