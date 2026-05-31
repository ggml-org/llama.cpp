#include "models.h"

ggml_cgraph * clip_graph_mimo_v2_asr::build() {
    ggml_tensor * cur = build_inp_raw(1); 

    // --- Audio Encoder Frontend ---
    cur = ggml_conv_1d(ctx0, model.conv1d_1_w, cur, 1, 1, 1);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_add(ctx0, cur, model.conv1d_1_b); 
    cur = ggml_gelu_erf(ctx0, cur);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_conv_1d(ctx0, model.conv1d_2_w, cur, 2, 1, 1);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_add(ctx0, cur, model.conv1d_2_b);
    cur = ggml_gelu_erf(ctx0, cur);

    // --- Audio Encoder Transformer ---
    const int d_model  = cur->ne[0];
    const int T_enc    = cur->ne[1];
    const int n_head   = hparams.n_head;
    const int head_dim = d_model / n_head;

    ggml_tensor * enc_pos = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float)T_enc, 1.0f), GGML_TYPE_I32);
    ggml_tensor * skip_connect = nullptr;

    for (int i = 0; i < hparams.n_layer; ++i) {
        const auto & layer = model.layers[i];
        ggml_tensor * inpL = cur;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, i);

        ggml_tensor * Q = ggml_add(ctx0, build_mm(layer.q_w, cur), layer.q_b); 
        ggml_tensor * K = build_mm(layer.k_w, cur); 
        ggml_tensor * V = ggml_add(ctx0, build_mm(layer.v_w, cur), layer.v_b);

        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_head, T_enc);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_head, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_head, T_enc);

        Q = ggml_rope_ext(ctx0, Q, enc_pos, nullptr, head_dim, LLAMA_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, enc_pos, nullptr, head_dim, LLAMA_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        cur = build_attn(layer.o_w, layer.o_b, Q, K, V, nullptr, 1.0f / sqrtf(head_dim), i);
        cur = ggml_add(ctx0, cur, inpL);
        
        ggml_tensor * inpL2 = cur;
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, i);
        cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, nullptr, nullptr, layer.ff_down_w, layer.ff_down_b, FFN_GELU_ERF, i);
        cur = ggml_add(ctx0, cur, inpL2);

        if (i == 2) skip_connect = ggml_dup(ctx0, cur);
    }

    if (skip_connect) cur = ggml_add(ctx0, cur, skip_connect);
    cur = build_norm(cur, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);

    // --- Downsample Layer ---
    if (model.mm_input_proj_w) {
        if (T_enc % 2 != 0) cur = ggml_pad(ctx0, cur, 0, 1, 0, 0); 
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_conv_1d(ctx0, model.mm_input_proj_w, cur, 2, 0, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_gelu_erf(ctx0, cur);
        cur = build_norm(cur, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
    }

    // --- RVQ Stage ---
    const int T_rvq = cur->ne[1]; 
    ggml_tensor * audio_ids_cont = nullptr;
    ggml_tensor * residual = cur;

    for (size_t i = 0; i < model.rvq_codebooks.size(); ++i) {
        ggml_tensor * C = model.rvq_codebooks[i]; 
        
        ggml_tensor * xC = ggml_scale(ctx0, ggml_mul_mat(ctx0, C, residual), 2.0f);
        ggml_tensor * C_sq_rep = ggml_repeat(ctx0, ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, C))), xC);
        
        ggml_tensor * ids_i32 = ggml_reshape_1d(ctx0, ggml_cast(ctx0, ggml_argmax(ctx0, ggml_sub(ctx0, xC, C_sq_rep)), GGML_TYPE_I32), T_rvq);
        audio_ids_cont = (audio_ids_cont == nullptr) ? ids_i32 : ggml_concat(ctx0, audio_ids_cont, ids_i32, 0); 
        residual = ggml_sub(ctx0, residual, ggml_get_rows(ctx0, C, ids_i32));
    }

    // --- Local Transformer ---
    if (audio_ids_cont) {
        const int d_local = model.enc_embd_1_w[0]->ne[0]; 
        const int n_head_audio_local = 64;
        const int head_dim_local = d_local / n_head_audio_local; 
        const int group_size = model.mm_0_w->ne[0] / d_local;
        const float rope_freq_local = hparams.rope_theta != 0.0f ? hparams.rope_theta : 640000.0f;

        ggml_tensor * audio_pos = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float)group_size, 1.0f), GGML_TYPE_I32);
        ggml_tensor * audio_embds = nullptr;
        
        for (size_t i = 0; i < model.rvq_codebooks.size(); ++i) {
            ggml_tensor * cur_ids_1d = ggml_view_1d(ctx0, audio_ids_cont, T_rvq, i * T_rvq * ggml_element_size(audio_ids_cont));
            ggml_tensor * cur_embed  = ggml_get_rows(ctx0, model.enc_embd_1_w[i], cur_ids_1d);
            
            ggml_tensor * mask_2d = ggml_reshape_2d(ctx0, ggml_step(ctx0, ggml_abs(ctx0, ggml_scale_bias(ctx0, ggml_cast(ctx0, cur_ids_1d, GGML_TYPE_F32), 1.0f, -(float)model.speech_zeroemb_idx[i]))), 1, T_rvq); 
            cur_embed = ggml_mul(ctx0, cur_embed, mask_2d); 
            audio_embds = (audio_embds == nullptr) ? cur_embed : ggml_add(ctx0, audio_embds, cur_embed);
        }

        int pad = (group_size - (T_rvq % group_size)) % group_size;
        if (pad > 0) {
            ggml_tensor * last_step = ggml_view_2d(ctx0, audio_embds, d_local, 1, audio_embds->nb[1], (T_rvq - 1) * audio_embds->nb[1]);
            audio_embds = ggml_concat(ctx0, audio_embds, ggml_repeat(ctx0, last_step, ggml_new_tensor_2d(ctx0, audio_embds->type, d_local, pad)), 1);
        }

        const int T_groups = (T_rvq + pad) / group_size;
        cur = ggml_reshape_3d(ctx0, audio_embds, d_local, group_size, T_groups);

        for (size_t i = 0; i < model.local_layers.size(); ++i) {
            const auto & layer = model.local_layers[i];
            ggml_tensor * inpL1 = cur;

            cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), layer.attn_norm_1_w);

            // Qwen2 local attention
            ggml_tensor * Q_cur = ggml_add(ctx0, build_mm(layer.attn_q_1_w, cur), layer.attn_q_1_b);
            ggml_tensor * K_cur = ggml_add(ctx0, build_mm(layer.attn_k_1_w, cur), layer.attn_k_1_b);
            ggml_tensor * V_cur = ggml_add(ctx0, build_mm(layer.attn_v_1_w, cur), layer.attn_v_1_b);

            Q_cur = ggml_rope_ext(ctx0, ggml_reshape_4d(ctx0, Q_cur, head_dim_local, n_head_audio_local, group_size, T_groups), audio_pos, nullptr, head_dim_local, LLAMA_ROPE_TYPE_NEOX, 0, rope_freq_local, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            K_cur = ggml_rope_ext(ctx0, ggml_reshape_4d(ctx0, K_cur, head_dim_local, n_head_audio_local, group_size, T_groups), audio_pos, nullptr, head_dim_local, LLAMA_ROPE_TYPE_NEOX, 0, rope_freq_local, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

            Q_cur = ggml_cont(ctx0, ggml_permute(ctx0, Q_cur, 0, 2, 1, 3));
            K_cur = ggml_cont(ctx0, ggml_permute(ctx0, K_cur, 0, 2, 1, 3));
            V_cur = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, V_cur, head_dim_local, n_head_audio_local, group_size, T_groups), 0, 2, 1, 3));

            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx0, Q_cur, K_cur, V_cur, nullptr, 1.0f / sqrtf(head_dim_local), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
            
            cur = build_mm(layer.attn_out_1_w, ggml_reshape_3d(ctx0, attn_out, d_local, group_size, T_groups));
            cur = ggml_add(ctx0, cur, inpL1);

            ggml_tensor * inpL2 = cur;
            cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), layer.ff_norm_1_w);
            cur = build_mm(layer.ff_down_1_w, ggml_mul(ctx0, ggml_silu(ctx0, build_mm(layer.ff_gate_1_w, cur)), build_mm(layer.ff_up_1_w, cur)));
            cur = ggml_add(ctx0, cur, inpL2);
        }

        cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), model.enc_out_norm_1_w);
        cur = build_mm(model.mm_0_w, ggml_reshape_2d(ctx0, cur, d_local * group_size, T_groups));   
    }

    if (cur) ggml_build_forward_expand(gf, cur);

    return gf;
}