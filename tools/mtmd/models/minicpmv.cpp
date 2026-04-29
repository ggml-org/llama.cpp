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

ggml_cgraph * clip_graph_minicpmv_merger::build() {
    const int insert_lid = hparams.insert_layer_id;
    const int n_pos      = n_patches;
    const int half_h     = n_patches_y / 2;
    const int half_w     = n_patches_x / 2;
    const int n_ds       = half_h * half_w;     // after insert merger 2x2 downsample
    const int qh         = half_h / 2;
    const int qw         = half_w / 2;
    const int n_ds2      = qh * qw;             // after final merger 2x2 downsample

    // position indices for ViT learned positional embeddings
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);

    // insert merger window reorder indices
    struct ggml_tensor * im_window_idx     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(im_window_idx, "im_window_idx");     ggml_set_input(im_window_idx);
    struct ggml_tensor * im_inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(im_inv_window_idx, "im_inv_window_idx"); ggml_set_input(im_inv_window_idx);

    // insert merger 2x2 downsample gather indices
    struct ggml_tensor * im_ds_idx_0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ds);
    ggml_set_name(im_ds_idx_0, "im_ds_idx_0"); ggml_set_input(im_ds_idx_0);
    struct ggml_tensor * im_ds_idx_1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ds);
    ggml_set_name(im_ds_idx_1, "im_ds_idx_1"); ggml_set_input(im_ds_idx_1);
    struct ggml_tensor * im_ds_idx_2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ds);
    ggml_set_name(im_ds_idx_2, "im_ds_idx_2"); ggml_set_input(im_ds_idx_2);
    struct ggml_tensor * im_ds_idx_3 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ds);
    ggml_set_name(im_ds_idx_3, "im_ds_idx_3"); ggml_set_input(im_ds_idx_3);

    // final merger 2x2 downsample gather indices
    struct ggml_tensor * merger_ds_idx_0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ds2);
    ggml_set_name(merger_ds_idx_0, "merger_ds_idx_0"); ggml_set_input(merger_ds_idx_0);
    struct ggml_tensor * merger_ds_idx_1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ds2);
    ggml_set_name(merger_ds_idx_1, "merger_ds_idx_1"); ggml_set_input(merger_ds_idx_1);
    struct ggml_tensor * merger_ds_idx_2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ds2);
    ggml_set_name(merger_ds_idx_2, "merger_ds_idx_2"); ggml_set_input(merger_ds_idx_2);
    struct ggml_tensor * merger_ds_idx_3 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ds2);
    ggml_set_name(merger_ds_idx_3, "merger_ds_idx_3"); ggml_set_input(merger_ds_idx_3);

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
    for (int il = 0; il <= insert_lid; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
            if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
            if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);
            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
            if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);
            cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
        }
        if (layer.ls_1_w) cur = ggml_mul(ctx0, cur, layer.ls_1_w);
        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b,
                        layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
        if (layer.ls_2_w) cur = ggml_mul(ctx0, cur, layer.ls_2_w);
        cur = ggml_add(ctx0, inpL, cur);
        inpL = cur;
    }

    // Insert Merger: Window Self-Attention
    {
        ggml_tensor * residual = inpL;
        ggml_tensor * cur = build_norm(inpL,
            model.insert_merger_ln1_w, model.insert_merger_ln1_b,
            NORM_TYPE_NORMAL, eps, -1);

        cur = ggml_get_rows(ctx0, cur, im_window_idx);
        const int n_windows = n_patches / 4;
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 4, n_windows);

        ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.insert_merger_attn_q_w, cur);
        if (model.insert_merger_attn_q_b) Qcur = ggml_add(ctx0, Qcur, model.insert_merger_attn_q_b);
        ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.insert_merger_attn_k_w, cur);
        if (model.insert_merger_attn_k_b) Kcur = ggml_add(ctx0, Kcur, model.insert_merger_attn_k_b);
        ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.insert_merger_attn_v_w, cur);
        if (model.insert_merger_attn_v_b) Vcur = ggml_add(ctx0, Vcur, model.insert_merger_attn_v_b);

        Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, 4, n_windows);
        Kcur = ggml_reshape_4d(ctx0, Kcur, d_head, n_head, 4, n_windows);
        Vcur = ggml_reshape_4d(ctx0, Vcur, d_head, n_head, 4, n_windows);

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        ggml_tensor * v = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_soft_max_ext(ctx0, kq, nullptr, kq_scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, n_embd, n_patches);

        cur = ggml_mul_mat(ctx0, model.insert_merger_attn_o_w, cur);
        if (model.insert_merger_attn_o_b) cur = ggml_add(ctx0, cur, model.insert_merger_attn_o_b);

        cur = ggml_get_rows(ctx0, cur, im_inv_window_idx);
        inpL = ggml_add(ctx0, cur, residual);
    }

    // Insert Merger: ViT MLP Downsample (4 tokens -> 1)
    {
        ggml_tensor * p0 = ggml_get_rows(ctx0, inpL, im_ds_idx_0);
        ggml_tensor * p1 = ggml_get_rows(ctx0, inpL, im_ds_idx_1);
        ggml_tensor * p2 = ggml_get_rows(ctx0, inpL, im_ds_idx_2);
        ggml_tensor * p3 = ggml_get_rows(ctx0, inpL, im_ds_idx_3);

        ggml_tensor * mean_res = ggml_add(ctx0, p0, p1);
        mean_res = ggml_add(ctx0, mean_res, p2);
        mean_res = ggml_add(ctx0, mean_res, p3);
        mean_res = ggml_scale(ctx0, mean_res, 0.25f);

        ggml_tensor * cat = ggml_concat(ctx0, p0, p1, 0);
        cat = ggml_concat(ctx0, cat, p2, 0);
        cat = ggml_concat(ctx0, cat, p3, 0);

        ggml_tensor * cur = build_norm(cat,
            model.insert_merger_ds_ln_w, model.insert_merger_ds_ln_b,
            NORM_TYPE_NORMAL, eps, -1);
        cur = ggml_mul_mat(ctx0, model.insert_merger_ds_up_w, cur);
        if (model.insert_merger_ds_up_b) cur = ggml_add(ctx0, cur, model.insert_merger_ds_up_b);
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, model.insert_merger_ds_down_w, cur);
        if (model.insert_merger_ds_down_b) cur = ggml_add(ctx0, cur, model.insert_merger_ds_down_b);

        inpL = ggml_add(ctx0, cur, mean_res);
    }

    // ViT layers (insert_layer_id+1)..n_layer-1
    {
        const int64_t n_pos_ds = n_ds;
        for (int il = insert_lid + 1; il < n_layer; il++) {
            auto & layer = model.layers[il];
            ggml_tensor * cur = inpL;
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
            {
                ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos_ds);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos_ds);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos_ds);
                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            }
            if (layer.ls_1_w) cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b,
                            layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
            if (layer.ls_2_w) cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            cur = ggml_add(ctx0, inpL, cur);
            inpL = cur;
        }
    }

    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    }

    // Final Merger (DownsampleMLP): another 2x2 spatial merge
    {
        ggml_tensor * p0 = ggml_get_rows(ctx0, inpL, merger_ds_idx_0);
        ggml_tensor * p1 = ggml_get_rows(ctx0, inpL, merger_ds_idx_1);
        ggml_tensor * p2 = ggml_get_rows(ctx0, inpL, merger_ds_idx_2);
        ggml_tensor * p3 = ggml_get_rows(ctx0, inpL, merger_ds_idx_3);

        ggml_tensor * cat = ggml_concat(ctx0, p0, p1, 0);
        cat = ggml_concat(ctx0, cat, p2, 0);
        cat = ggml_concat(ctx0, cat, p3, 0);

        ggml_tensor * cur = build_norm(cat,
            model.merger_pre_norm_w, model.merger_pre_norm_b,
            NORM_TYPE_NORMAL, eps, -1);
        cur = ggml_mul_mat(ctx0, model.merger_mlp_up_w, cur);
        if (model.merger_mlp_up_b) cur = ggml_add(ctx0, cur, model.merger_mlp_up_b);
        // MiniCPMV4_6DownsampleMLP uses nn.GELU() (erf-based), not the tanh approximation
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_mul_mat(ctx0, model.merger_mlp_down_w, cur);
        if (model.merger_mlp_down_b) cur = ggml_add(ctx0, cur, model.merger_mlp_down_b);

        inpL = cur;
    }

    ggml_build_forward_expand(gf, inpL);
    return gf;
}
