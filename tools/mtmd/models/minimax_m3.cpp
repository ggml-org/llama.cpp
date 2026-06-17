#include "models.h"

// MiniMax-M3 vision tower: CLIP-style blocks (separate biased q/k/v/out, LayerNorm,
// gelu fc1/fc2, a pre_layrnorm, no post-LN / class token / abs-pos) over a Conv3D
// (temporal-split) patch embed, with a custom 3D (T/H/W) RoPE and a two-stage
// projector. Image-only (n_batch == 1); video is out of scope for now.

// Apply MiniMax-M3 3D RoPE using host-precomputed cos/sin (filled in set_input).
//   x        : [d_head, n_head, n_pos]
//   rope_cos : [rope_dim, 1, n_pos]   (rope_dim = 3*axis_dim = 78, broadcasts over heads)
// First rope_dim dims are rotated (HF block rotate_half); the tail passes through.
ggml_tensor * clip_graph_minimax_m3::apply_rope(
        ggml_tensor * x, ggml_tensor * rope_cos, ggml_tensor * rope_sin) {
    const int64_t d    = x->ne[0];          // 80
    const int64_t rd   = rope_cos->ne[0];   // 78
    const int64_t half = rd / 2;            // 39
    const size_t  es   = ggml_element_size(x);

    ggml_tensor * x_rot  = ggml_cont(ctx0, ggml_view_3d(ctx0, x, rd,    x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0));
    ggml_tensor * x_pass = ggml_cont(ctx0, ggml_view_3d(ctx0, x, d - rd, x->ne[1], x->ne[2], x->nb[1], x->nb[2], rd * es));

    const size_t es_r = ggml_element_size(x_rot);
    ggml_tensor * x1 = ggml_cont(ctx0, ggml_view_3d(ctx0, x_rot, half, x_rot->ne[1], x_rot->ne[2], x_rot->nb[1], x_rot->nb[2], 0));
    ggml_tensor * x2 = ggml_cont(ctx0, ggml_view_3d(ctx0, x_rot, half, x_rot->ne[1], x_rot->ne[2], x_rot->nb[1], x_rot->nb[2], half * es_r));
    ggml_tensor * rot = ggml_concat(ctx0, ggml_neg(ctx0, x2), x1, 0);   // [-x2 ; x1]

    ggml_tensor * out = ggml_add(ctx0,
        ggml_mul(ctx0, x_rot, rope_cos),    // cos/sin broadcast over n_head (ne1 == 1)
        ggml_mul(ctx0, rot,   rope_sin));
    return ggml_concat(ctx0, out, x_pass, 0);   // [d_head, n_head, n_pos]
}

ggml_cgraph * clip_graph_minimax_m3::build() {
    GGML_ASSERT(model.patch_bias     == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(model.patch_embeddings_0 && model.patch_embeddings_1);
    GGML_ASSERT(model.mm_1_w && model.mm_2_w);
    GGML_ASSERT(model.mm_merge_fc1_w && model.mm_merge_fc2_w);

    const int batch_size = 1;
    const int n_pos      = n_patches;     // gh*gw (full patch grid)
    const int merge      = 2;

    // ---- Conv3D patch embed: two Conv2D temporal slices, summed (still image) ----
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_add(ctx0,
        ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1),
        ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));

    // ---- 2x2 spatial-merge reorder (raster -> block order), matches HF flatten ----
    {
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);
        inp = ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    ggml_tensor * inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
    }

    // ---- 3D RoPE cos/sin inputs (host-filled in set_input) ----
    const int axis_dim = 2 * ((2 * (d_head / 2) / 3) / 2);   // 26
    const int rope_dim = 3 * axis_dim;                        // 78
    ggml_tensor * rope_cos = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, rope_dim, 1, n_pos);
    ggml_set_name(rope_cos, "minimax_cos"); ggml_set_input(rope_cos);
    ggml_tensor * rope_sin = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, rope_dim, 1, n_pos);
    ggml_set_name(rope_sin, "minimax_sin"); ggml_set_input(rope_sin);

    // ---- encoder ----
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ln1", il);

        ggml_tensor * Q = ggml_add(ctx0, build_mm(layer.q_w, cur), layer.q_b);
        ggml_tensor * K = ggml_add(ctx0, build_mm(layer.k_w, cur), layer.k_b);
        ggml_tensor * V = ggml_add(ctx0, build_mm(layer.v_w, cur), layer.v_b);

        Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, n_pos);
        K = ggml_reshape_3d(ctx0, K, d_head, n_head, n_pos);
        V = ggml_reshape_3d(ctx0, V, d_head, n_head, n_pos);

        Q = apply_rope(Q, rope_cos, rope_sin);
        K = apply_rope(K, rope_cos, rope_sin);
        cb(Q, "Qcur_rope", il);
        cb(K, "Kcur_rope", il);

        // full bidirectional attention (no mask, no windowing)
        cur = build_attn(layer.o_w, layer.o_b, Q, K, V, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        cur  = ggml_add(ctx0, cur, inpL);   // residual 1
        inpL = cur;

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur,
            layer.ff_up_w,   layer.ff_up_b,
            nullptr,         nullptr,
            layer.ff_down_w, layer.ff_down_b,
            FFN_GELU, il);
        cb(cur, "ffn_out", il);

        cur  = ggml_add(ctx0, inpL, cur);   // residual 2
        inpL = cur;
    }
    // NOTE: MiniMax-M3 has no post-layernorm.

    // ---- projector: per-patch MLP -> group-4 -> merge MLP ----
    ggml_tensor * emb = inpL;                                   // [n_embd, n_pos]
    emb = build_ffn(emb, model.mm_1_w, model.mm_1_b,
                    nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b, FFN_GELU, -1);   // [proj_dim, n_pos]

    const int64_t proj = emb->ne[0];                            // 6144
    emb = ggml_reshape_2d(ctx0, emb, proj * merge * merge, n_pos / (merge * merge)); // [4*proj, n_pos/4]

    emb = build_ffn(emb, model.mm_merge_fc1_w, model.mm_merge_fc1_b,
                    nullptr, nullptr,
                    model.mm_merge_fc2_w, model.mm_merge_fc2_b, FFN_GELU, -1);        // [proj_dim, n_pos/4]

    ggml_build_forward_expand(gf, emb);
    return gf;
}
