#include "models.h"

ggml_cgraph * mtmd_graph_hy_v4::build() {
    const int merge = hparams.n_merge;
    const int pw    = n_patches_x;
    const int ph    = n_patches_y;

    // filled on CPU by mtmd_subcontext::encode(), same as hunyuanvl
    ggml_tensor * pos_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ph * pw);
    ggml_set_name(pos_embd, "hunyuanvl_pos_embd");
    ggml_set_input(pos_embd);

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(inp, n_patches, NORM_TYPE_NORMAL, hparams.ffn_op, pos_embd, nullptr);

    cur = build_norm(cur, model.mm_pre_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);

    // [C, W*H] -> [W, H, C] for conv2d
    cur = ggml_reshape_3d(ctx0, cur, n_embd, pw, ph);
    cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
    cur = ggml_cont(ctx0, cur);

    cur = ggml_conv_2d(ctx0, model.mm_0_w, cur, merge, merge, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.mm_0_b, 1, 1, model.mm_0_b->ne[0]));
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_conv_2d(ctx0, model.mm_1_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.mm_1_b, 1, 1, model.mm_1_b->ne[0]));

    const int ow   = pw / merge;
    const int oh   = ph / merge;
    const int idim = (int)cur->ne[2];

    ggml_tensor * nl = ggml_reshape_4d(ctx0, model.image_newline, 1, 1, idim, 1);
    nl = ggml_repeat_4d(ctx0, nl, 1, oh, idim, 1);
    cur = ggml_concat(ctx0, cur, nl, 0);

    // [OW+1, OH, OC] -> [OC, (OW+1)*OH]
    cur = ggml_permute(ctx0, cur, 1, 2, 0, 3);
    cur = ggml_cont_2d(ctx0, cur, idim, (ow + 1) * oh);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
