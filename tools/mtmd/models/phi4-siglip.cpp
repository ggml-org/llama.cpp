#include "models.h"

ggml_cgraph * clip_graph_phi4_siglip::build() {
    GGML_ASSERT(model.class_embedding == nullptr);

    ggml_tensor * cur = build_inp();
    ggml_tensor * learned_pos_embd = resize_position_embeddings();

    if (learned_pos_embd) {
        cur = ggml_add(ctx0, cur, learned_pos_embd);
        cb(cur, "pos_embed", -1);
    }

    if (model.pre_ln_w) {
        cur = build_norm(cur, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
        cb(cur, "pre_ln", -1);
    }

    ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer - 1; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * hidden = inpL;

        hidden = build_norm(hidden, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(hidden, "layer_inp_normed", il);

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.q_w, hidden);
        if (layer.q_b) {
            q = ggml_add(ctx0, q, layer.q_b);
        }

        ggml_tensor * k = ggml_mul_mat(ctx0, layer.k_w, hidden);
        if (layer.k_b) {
            k = ggml_add(ctx0, k, layer.k_b);
        }

        ggml_tensor * v = ggml_mul_mat(ctx0, layer.v_w, hidden);
        if (layer.v_b) {
            v = ggml_add(ctx0, v, layer.v_b);
        }

        q = ggml_reshape_3d(ctx0, q, d_head, n_head, n_patches);
        k = ggml_reshape_3d(ctx0, k, d_head, n_head, n_patches);
        v = ggml_reshape_3d(ctx0, v, d_head, n_head, n_patches);

        cb(q, "Qcur", il);
        cb(k, "Kcur", il);
        cb(v, "Vcur", il);

        hidden = build_attn(layer.o_w, layer.o_b, q, k, v, nullptr, kq_scale, il);
        cb(hidden, "attn_out", il);

        hidden = ggml_add(ctx0, hidden, inpL);
        cb(hidden, "ffn_inp", il);

        ggml_tensor * ffn = build_norm(hidden, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(ffn, "ffn_inp_normed", il);

        ffn = build_ffn(
            ffn,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op,
            il);
        cb(ffn, "ffn_out", il);

        inpL = ggml_add(ctx0, hidden, ffn);
        cb(inpL, "layer_out", il);
    }

    cur = build_ffn(
        inpL,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU,
        -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
