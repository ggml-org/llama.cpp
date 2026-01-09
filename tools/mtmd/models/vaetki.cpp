#include "models.h"

ggml_cgraph * clip_graph_vaetki::build() {
    GGML_ASSERT(model.class_embedding != nullptr);

    const int batch_size = 1;
    const int n_pos = n_patches + 1;
    const int n_pos_patches = n_patches;
    const int num_position_ids = n_pos_patches * 4;

    norm_type norm_t = NORM_TYPE_NORMAL;
    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = build_inp();

    // add CLS token
    inp = ggml_concat(ctx0, model.class_embedding, inp, 1);
    cb(inp, "inp_with_cls", -1);

    ggml_tensor * inpL = inp;

    // position IDs for 2D RoPE (patch tokens only)
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // precompute CLS position embedding cos/sin
    ggml_tensor * cls_cos = nullptr;
    ggml_tensor * cls_sin = nullptr;
    if (model.class_pos_emb) {
        // class_pos_emb: [head_dim/2] -> concat to [head_dim]
        ggml_tensor * cls_pos = ggml_concat(ctx0, model.class_pos_emb, model.class_pos_emb, 0);
        cls_cos = ggml_cos(ctx0, cls_pos);
        cls_sin = ggml_sin(ctx0, cls_pos);
    }

    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        // self-attention with 2D RoPE
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
            if (layer.q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            }

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
            if (layer.k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.k_b);
            }

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
            if (layer.v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // split CLS and patch tokens for RoPE
            ggml_tensor * Q_cls = ggml_view_3d(ctx0, Qcur, d_head, n_head, 1,
                ggml_row_size(Qcur->type, d_head),
                ggml_row_size(Qcur->type, d_head * n_head), 0);
            ggml_tensor * K_cls = ggml_view_3d(ctx0, Kcur, d_head, n_head, 1,
                ggml_row_size(Kcur->type, d_head),
                ggml_row_size(Kcur->type, d_head * n_head), 0);

            ggml_tensor * Q_patch = ggml_view_3d(ctx0, Qcur, d_head, n_head, n_pos_patches,
                ggml_row_size(Qcur->type, d_head),
                ggml_row_size(Qcur->type, d_head * n_head),
                ggml_row_size(Qcur->type, d_head * n_head));
            ggml_tensor * K_patch = ggml_view_3d(ctx0, Kcur, d_head, n_head, n_pos_patches,
                ggml_row_size(Kcur->type, d_head),
                ggml_row_size(Kcur->type, d_head * n_head),
                ggml_row_size(Kcur->type, d_head * n_head));

            // apply RoPE to CLS token using class_pos_emb
            if (cls_cos && cls_sin) {
                // rotate_half: split into two halves, negate second, swap order
                ggml_tensor * Q_cls_1 = ggml_view_3d(ctx0, Q_cls, d_head/2, n_head, 1,
                    ggml_row_size(Q_cls->type, d_head),
                    ggml_row_size(Q_cls->type, d_head * n_head), 0);
                ggml_tensor * Q_cls_2 = ggml_view_3d(ctx0, Q_cls, d_head/2, n_head, 1,
                    ggml_row_size(Q_cls->type, d_head),
                    ggml_row_size(Q_cls->type, d_head * n_head),
                    ggml_row_size(Q_cls->type, d_head/2));
                ggml_tensor * Q_cls_rot = ggml_concat(ctx0, ggml_neg(ctx0, Q_cls_2), Q_cls_1, 0);

                ggml_tensor * K_cls_1 = ggml_view_3d(ctx0, K_cls, d_head/2, n_head, 1,
                    ggml_row_size(K_cls->type, d_head),
                    ggml_row_size(K_cls->type, d_head * n_head), 0);
                ggml_tensor * K_cls_2 = ggml_view_3d(ctx0, K_cls, d_head/2, n_head, 1,
                    ggml_row_size(K_cls->type, d_head),
                    ggml_row_size(K_cls->type, d_head * n_head),
                    ggml_row_size(K_cls->type, d_head/2));
                ggml_tensor * K_cls_rot = ggml_concat(ctx0, ggml_neg(ctx0, K_cls_2), K_cls_1, 0);

                // RoPE: x * cos + rotate_half(x) * sin
                Q_cls = ggml_add(ctx0,
                    ggml_mul(ctx0, Q_cls, cls_cos),
                    ggml_mul(ctx0, Q_cls_rot, cls_sin));
                K_cls = ggml_add(ctx0,
                    ggml_mul(ctx0, K_cls, cls_cos),
                    ggml_mul(ctx0, K_cls_rot, cls_sin));
            }

            // apply 2D RoPE to patch tokens
            Q_patch = ggml_rope_multi(ctx0, Q_patch, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            K_patch = ggml_rope_multi(ctx0, K_patch, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            Qcur = ggml_concat(ctx0, Q_cls, Q_patch, 2);
            Kcur = ggml_concat(ctx0, K_cls, K_patch, 2);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ln2", il);

        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // remove CLS token
    ggml_tensor * embeddings = ggml_view_2d(ctx0, inpL,
        n_embd, n_pos_patches,
        ggml_row_size(inpL->type, n_embd),
        ggml_row_size(inpL->type, n_embd));
    cb(embeddings, "patches_only", -1);

    // merger
    embeddings = build_norm(embeddings, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
    cb(embeddings, "merger_normed", -1);

    // pixel shuffle
    const int scale_factor = hparams.n_merge;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * scale_factor * scale_factor, n_pos_patches / (scale_factor * scale_factor), batch_size);
    cb(embeddings, "merger_reshaped", -1);

    embeddings = build_ffn(embeddings,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_3_w, model.mm_3_b,
        FFN_GELU,
        -1);
    cb(embeddings, "merger_out", -1);

    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
