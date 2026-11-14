#include "models.h"

ggml_cgraph * mtmd_graph_inkling2::build() {
    const int merge = hparams.n_merge;
    GGML_ASSERT(merge == 2);
    GGML_ASSERT(n_patches_x % merge == 0 && n_patches_y % merge == 0);
    GGML_ASSERT(model.class_embedding && model.inkling2_cls_pos && model.position_embeddings);

    const int64_t n_pos = n_patches + 1;

    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * inp = build_inp();

    ggml_tensor * pos_embd = model.position_embeddings;
    GGML_ASSERT(pos_embd->ne[0] == n_embd);
    if (pos_embd->ne[1] != n_patches_x || pos_embd->ne[2] != n_patches_y) {
        pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
        pos_embd = ggml_interpolate(ctx0, pos_embd, n_patches_y, n_patches_x, n_embd, 1, GGML_SCALE_MODE_BILINEAR);
        pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    }
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, n_patches);
    inp = ggml_add(ctx0, inp, pos_embd);

    ggml_tensor * cls = ggml_add(ctx0, model.class_embedding, model.inkling2_cls_pos);
    cls = ggml_reshape_2d(ctx0, cls, n_embd, 1);
    inp = ggml_concat(ctx0, cls, inp, 1);
    cb(inp, "inp_pos", -1);

    auto add_pos = [&](ggml_tensor * cur, const mtmd_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, add_pos);
    cb(cur, "vit_out", -1);

    cur = ggml_view_2d(ctx0, cur, n_embd, n_patches, cur->nb[1], cur->nb[1]);
    cur = ggml_cont_4d(ctx0, cur, n_embd * merge, n_patches_x / merge, n_patches_y, 1);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_4d(ctx0, cur, n_embd * merge * merge, n_patches_y / merge, n_patches_x / merge, 1);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);

    cur = build_ffn(cur,
        model.mm_ffn_up_w, nullptr,
        nullptr, nullptr,
        model.mm_ffn_down_w, nullptr,
        FFN_GELU_ERF, -1);
    cb(cur, "proj_out", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
