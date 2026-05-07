#include "models.h"

ggml_cgraph * clip_graph_minicpmv::build() {
    GGML_ASSERT(model.class_embedding == nullptr);
    const int n_pos       = n_patches;
    const int n_embd_proj = n_mmproj_embd;

    // position embeddings for the projector (not for ViT)
    // see: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/resampler.py#L70
    // base frequency omega
    ggml_tensor * omega = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_embd_proj / 4);
    ggml_set_name(omega, "omega");
    ggml_set_input(omega);

    // 2D input positions (using float for sinusoidal embeddings)
    ggml_tensor * pos_h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_pos);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);
    ggml_tensor * pos_w = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_pos);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    // for selecting learned pos embd, used by ViT
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);

    ggml_tensor * inp = build_inp();
    ggml_tensor * embeddings = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            nullptr);

    // resampler projector (it is just another transformer)

    ggml_tensor * q = model.mm_model_query;
    ggml_tensor * v = build_mm(model.mm_model_kv_proj, embeddings);

    // norm
    q = build_norm(q, model.mm_model_ln_q_w,  model.mm_model_ln_q_b,  NORM_TYPE_NORMAL, eps, -1);
    v = build_norm(v, model.mm_model_ln_kv_w, model.mm_model_ln_kv_b, NORM_TYPE_NORMAL, eps, -1);

    // calculate sinusoidal pos embd
    ggml_tensor * pos_embed = nullptr;
    {
        // outer product
        ggml_tensor * omega_b = ggml_repeat_4d(ctx0, omega, omega->ne[0], n_pos, 1, 1); // n_pos rows
        ggml_tensor * theta_x = ggml_mul(ctx0, omega_b, pos_w);
        ggml_tensor * theta_y = ggml_mul(ctx0, omega_b, pos_h);
        // sin and cos
        ggml_tensor * pos_embd_x = ggml_concat(
            ctx0,
            ggml_sin(ctx0, theta_x),
            ggml_cos(ctx0, theta_x),
            0 // concat on first dim
        );
        ggml_tensor * pos_embd_y = ggml_concat(
            ctx0,
            ggml_sin(ctx0, theta_y),
            ggml_cos(ctx0, theta_y),
            0 // concat on first dim
        );
        pos_embed = ggml_concat(ctx0, pos_embd_x, pos_embd_y, 0);
    }

    // k = v + pos_embed
    ggml_tensor * k = ggml_add(ctx0, v, pos_embed);

    // attention
    {
        const int d_head = 128;
        int n_head = n_embd_proj/d_head;
        // Use actual config value if available, otherwise fall back to hardcoded values
        int num_query = hparams.minicpmv_query_num;
        ggml_tensor * Q = ggml_add(ctx0,
            build_mm(model.mm_model_attn_q_w, q),
            model.mm_model_attn_q_b);
        ggml_tensor * K = ggml_add(ctx0,
            build_mm(model.mm_model_attn_k_w, k),
            model.mm_model_attn_k_b);
        ggml_tensor * V = ggml_add(ctx0,
            build_mm(model.mm_model_attn_v_w, v),
            model.mm_model_attn_v_b);

        Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, num_query);
        K = ggml_reshape_3d(ctx0, K, d_head, n_head, n_pos);
        V = ggml_reshape_3d(ctx0, V, d_head, n_head, n_pos);

        cb(Q, "resampler_Q", -1);
        cb(K, "resampler_K", -1);
        cb(V, "resampler_V", -1);

        float resampler_kq_scale = 1.0f/ sqrtf(float(d_head));
        embeddings = build_attn(
            model.mm_model_attn_o_w,
            model.mm_model_attn_o_b,
            Q, K, V, nullptr, resampler_kq_scale, -1);
        cb(embeddings, "resampler_attn_out", -1);
    }
    // layernorm
    embeddings = build_norm(embeddings, model.mm_model_ln_post_w, model.mm_model_ln_post_b, NORM_TYPE_NORMAL, eps, -1);

    // projection
    embeddings = build_mm(model.mm_model_proj, embeddings);

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}

ggml_cgraph * clip_graph_minicpmv4_6::build() {
    const int insert_lid = hparams.insert_layer_id;
    const int n_pos      = n_patches;
    const int half_h     = n_patches_y / 2;
    const int half_w     = n_patches_x / 2;
    const int n_ds       = half_h * half_w;     // after ViT merger 2x2 downsample
    const int qh         = half_h / 2;
    const int qw         = half_w / 2;
    const int n_ds2      = qh * qw;             // after final merger 2x2 downsample

    auto add_i32_input = [&](const char * name, int n) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };

    // position indices for ViT learned positional embeddings
    ggml_tensor * positions = add_i32_input("positions", n_pos);
    ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);

    // ViT merger window reorder indices + block-diagonal mask
    // (mask layout follows qwen2vl: -inf except for 4x4 blocks on the diagonal,
    // so each window-major group of 4 tokens only attends to itself)
    ggml_tensor * vit_merger_window_idx     = add_i32_input("vit_merger_window_idx", n_pos);
    ggml_tensor * vit_merger_inv_window_idx = add_i32_input("vit_merger_inv_window_idx", n_pos);
    ggml_tensor * vit_merger_window_mask    = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
    ggml_set_name(vit_merger_window_mask, "vit_merger_window_mask");
    ggml_set_input(vit_merger_window_mask);
    if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
        vit_merger_window_mask = ggml_cast(ctx0, vit_merger_window_mask, GGML_TYPE_F16);
    }

    // ViT merger 2x2 downsample gather indices
    ggml_tensor * vit_merger_ds_idx_0 = add_i32_input("vit_merger_ds_idx_0", n_ds);
    ggml_tensor * vit_merger_ds_idx_1 = add_i32_input("vit_merger_ds_idx_1", n_ds);
    ggml_tensor * vit_merger_ds_idx_2 = add_i32_input("vit_merger_ds_idx_2", n_ds);
    ggml_tensor * vit_merger_ds_idx_3 = add_i32_input("vit_merger_ds_idx_3", n_ds);

    // final merger 2x2 downsample gather indices
    ggml_tensor * merger_ds_idx_0 = add_i32_input("merger_ds_idx_0", n_ds2);
    ggml_tensor * merger_ds_idx_1 = add_i32_input("merger_ds_idx_1", n_ds2);
    ggml_tensor * merger_ds_idx_2 = add_i32_input("merger_ds_idx_2", n_ds2);
    ggml_tensor * merger_ds_idx_3 = add_i32_input("merger_ds_idx_3", n_ds2);

    // patch embedding + positional embedding
    ggml_tensor * inp = build_inp();
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(inp, "pos_embed", -1);

    ggml_tensor * inpL = inp;
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    // ViT layers 0..insert_layer_id (inclusive)
    // Mirrors the separate-qkv path of clip_graph::build_vit so the two manually
    // unrolled segments around the ViT merger read like build_vit() expansions.
    for (int il = 0; il <= insert_lid; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "layer_inp_normed", il);

        {
            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            if (layer.q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            }
            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            if (layer.k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.k_b);
            }
            ggml_tensor * Vcur = build_mm(layer.v_w, cur);
            if (layer.v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (layer.ls_1_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            cb(cur, "attn_out_scaled", il);
        }
        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_inp_normed", il);

        cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b,
                        layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
        cb(cur, "ffn_out", il);

        if (layer.ls_2_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            cb(cur, "ffn_out_scaled", il);
        }
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // ViT merger: window self-attention
    // Tokens are reordered to window-major (4 tokens per window are contiguous),
    // and a block-diagonal mask restricts attention to within each window. This
    // mirrors the qwen2vl windowed-attention pattern so build_attn() can pick the
    // flash-attention path when available.
    {
        ggml_tensor * residual = inpL;
        ggml_tensor * cur = build_norm(inpL,
            model.vit_merger_ln1_w, model.vit_merger_ln1_b,
            NORM_TYPE_NORMAL, eps, -1);
        cb(cur, "vit_merger_attn_inp_normed", -1);

        cur = ggml_get_rows(ctx0, cur, vit_merger_window_idx);
        cb(cur, "vit_merger_window_reorder", -1);

        ggml_tensor * Qcur = build_mm(model.vit_merger_attn_q_w, cur);
        if (model.vit_merger_attn_q_b) {
            Qcur = ggml_add(ctx0, Qcur, model.vit_merger_attn_q_b);
        }
        ggml_tensor * Kcur = build_mm(model.vit_merger_attn_k_w, cur);
        if (model.vit_merger_attn_k_b) {
            Kcur = ggml_add(ctx0, Kcur, model.vit_merger_attn_k_b);
        }
        ggml_tensor * Vcur = build_mm(model.vit_merger_attn_v_w, cur);
        if (model.vit_merger_attn_v_b) {
            Vcur = ggml_add(ctx0, Vcur, model.vit_merger_attn_v_b);
        }

        Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
        Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
        Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);
        cb(Qcur, "vit_merger_Qcur", -1);
        cb(Kcur, "vit_merger_Kcur", -1);
        cb(Vcur, "vit_merger_Vcur", -1);

        cur = build_attn(model.vit_merger_attn_o_w, model.vit_merger_attn_o_b,
                         Qcur, Kcur, Vcur, vit_merger_window_mask, kq_scale, -1);
        cb(cur, "vit_merger_attn_out", -1);

        cur = ggml_get_rows(ctx0, cur, vit_merger_inv_window_idx);
        inpL = ggml_add(ctx0, cur, residual);
        cb(inpL, "vit_merger_attn_residual", -1);
    }

    // ViT merger: 2x2 spatial downsample + MLP (4 tokens -> 1)
    {
        ggml_tensor * p0 = ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_0);
        ggml_tensor * p1 = ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_1);
        ggml_tensor * p2 = ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_2);
        ggml_tensor * p3 = ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_3);

        ggml_tensor * mean_res = ggml_add(ctx0, p0, p1);
        mean_res = ggml_add(ctx0, mean_res, p2);
        mean_res = ggml_add(ctx0, mean_res, p3);
        mean_res = ggml_scale(ctx0, mean_res, 0.25f);
        cb(mean_res, "vit_merger_ds_mean_res", -1);

        ggml_tensor * cat = ggml_concat(ctx0, p0, p1, 0);
        cat = ggml_concat(ctx0, cat, p2, 0);
        cat = ggml_concat(ctx0, cat, p3, 0);

        ggml_tensor * cur = build_norm(cat,
            model.vit_merger_ds_ln_w, model.vit_merger_ds_ln_b,
            NORM_TYPE_NORMAL, eps, -1);
        cb(cur, "vit_merger_ds_normed", -1);

        // ViTWindowAttentionMerger downsample MLP uses gelu_pytorch_tanh (FFN_GELU)
        cur = build_ffn(cur,
            model.vit_merger_ds_up_w,   model.vit_merger_ds_up_b,
            nullptr, nullptr,
            model.vit_merger_ds_down_w, model.vit_merger_ds_down_b,
            FFN_GELU, -1);
        cb(cur, "vit_merger_ds_mlp_out", -1);

        inpL = ggml_add(ctx0, cur, mean_res);
        cb(inpL, "vit_merger_ds_out", -1);
    }

    // ViT layers (insert_layer_id+1)..n_layer-1, operating on the downsampled tokens
    {
        const int64_t n_pos_ds = n_ds;
        for (int il = insert_lid + 1; il < n_layer; il++) {
            auto & layer = model.layers[il];
            ggml_tensor * cur = inpL;

            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "layer_inp_normed", il);

            {
                ggml_tensor * Qcur = build_mm(layer.q_w, cur);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }
                ggml_tensor * Kcur = build_mm(layer.k_w, cur);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }
                ggml_tensor * Vcur = build_mm(layer.v_w, cur);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos_ds);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos_ds);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos_ds);
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (layer.ls_1_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_1_w);
                cb(cur, "attn_out_scaled", il);
            }
            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;
            cb(cur, "ffn_inp", il);

            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "ffn_inp_normed", il);

            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b,
                            layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
            cb(cur, "ffn_out", il);

            if (layer.ls_2_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_2_w);
                cb(cur, "ffn_out_scaled", il);
            }
            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "layer_out", il);

            inpL = cur;
        }
    }

    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
        cb(inpL, "post_ln", -1);
    }

    // Final Merger (DownsampleMLP): another 2x2 spatial merge -> projector embedding
    {
        ggml_tensor * p0 = ggml_get_rows(ctx0, inpL, merger_ds_idx_0);
        ggml_tensor * p1 = ggml_get_rows(ctx0, inpL, merger_ds_idx_1);
        ggml_tensor * p2 = ggml_get_rows(ctx0, inpL, merger_ds_idx_2);
        ggml_tensor * p3 = ggml_get_rows(ctx0, inpL, merger_ds_idx_3);

        ggml_tensor * cat = ggml_concat(ctx0, p0, p1, 0);
        cat = ggml_concat(ctx0, cat, p2, 0);
        cat = ggml_concat(ctx0, cat, p3, 0);

        ggml_tensor * cur = build_norm(cat,
            model.mm_input_norm_w, model.mm_input_norm_b,
            NORM_TYPE_NORMAL, eps, -1);
        cb(cur, "merger_normed", -1);

        // MiniCPMV4_6DownsampleMLP uses nn.GELU() (erf-based, FFN_GELU_ERF)
        cur = build_ffn(cur,
            model.mm_ffn_up_w,   model.mm_ffn_up_b,
            nullptr, nullptr,
            model.mm_ffn_down_w, model.mm_ffn_down_b,
            FFN_GELU_ERF, -1);
        cb(cur, "merger_out", -1);

        inpL = cur;
    }

    ggml_build_forward_expand(gf, inpL);
    return gf;
}
