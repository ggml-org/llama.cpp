// similar to qwen2vl, except for GQA attention
#include "models.h"

ggml_cgraph * clip_graph_exaone4_5::build() {
    GGML_ASSERT(model.patch_bias == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size = 1;
    const bool use_window_attn = hparams.n_wa_pattern > 0;
    const int n_wa_pattern     = hparams.n_wa_pattern;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4;

    const norm_type norm_t = NORM_TYPE_RMS;

    const int64_t n_kv_head = hparams.n_head_kv > 0 ? hparams.n_head_kv : n_head;
    GGML_ASSERT(n_head % n_kv_head == 0);

    int rope_sections[4] = { d_head / 4, d_head / 4, d_head / 4, d_head / 4 };
    const float rope_freq_base = hparams.rope_theta > 0.0f ? hparams.rope_theta : 10000.0f;

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    GGML_ASSERT(img.nx() % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny() % (patch_size * 2) == 0);

    {
        ggml_tensor * inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * window_mask    = nullptr;
    ggml_tensor * window_idx     = nullptr;
    ggml_tensor * inv_window_idx = nullptr;

    if (use_window_attn) {
        window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(window_idx, "window_idx");
        ggml_set_input(window_idx);

        inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(inv_window_idx, "inv_window_idx");
        ggml_set_input(inv_window_idx);

        window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
        ggml_set_name(window_mask, "window_mask");
        ggml_set_input(window_mask);

        if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
            window_mask = ggml_cast(ctx0, window_mask, GGML_TYPE_F16);
        }
    }

    ggml_tensor * inpL = inp;

    if (use_window_attn) {
        GGML_ASSERT(batch_size == 1);
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
        inpL = ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const bool full_attn = use_window_attn ? (il + 1) % n_wa_pattern == 0 : true;
        ggml_tensor * cur = inpL;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        {
            GGML_ASSERT(layer.qkv_w != nullptr);
            cur = build_mm(layer.qkv_w, cur);
            if (layer.qkv_b) {
                cur = ggml_add(ctx0, cur, layer.qkv_b);
            }

            const int64_t n_embd_kv = d_head * n_kv_head;
            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_patches,
                ggml_row_size(cur->type, d_head),
                cur->nb[1],
                0);
            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, d_head, n_kv_head, n_patches,
                ggml_row_size(cur->type, d_head),
                cur->nb[1],
                ggml_row_size(cur->type, n_embd));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, d_head, n_kv_head, n_patches,
                ggml_row_size(cur->type, d_head),
                cur->nb[1],
                ggml_row_size(cur->type, n_embd + n_embd_kv));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head / 2, rope_sections, GGML_ROPE_TYPE_VISION, 32768, rope_freq_base, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head / 2, rope_sections, GGML_ROPE_TYPE_VISION, 32768, rope_freq_base, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);
            cb(Vcur, "Vcur", il);

            ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;
            cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cb(cur, "ffn_inp", il);

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    ggml_tensor * embeddings = inpL;
    embeddings = build_norm(embeddings, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
    embeddings = build_ffn(embeddings,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        FFN_GELU,
        -1);

    if (use_window_attn) {
        GGML_ASSERT(batch_size == 1);
        embeddings = ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4);
        embeddings = ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4, batch_size);
    }

    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
