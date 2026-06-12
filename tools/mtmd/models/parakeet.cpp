#include "models.h"

ggml_cgraph * clip_graph_parakeet::build() {
    // Build convolution graph
    ggml_tensor * cur = parakeet_build_graph_conv();
    ggml_build_forward_expand(gf, cur);

    // Build encoder graph
    cur = parakeet_build_graph_encoder(cur);

    cur = ggml_rms_norm(ctx0, cur, 1e-6);
    cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
    cb(cur, "sound_projection.norm", -1);

    cur = build_ffn(cur, model.mm_0_w, model.mm_0_b, nullptr, nullptr, model.mm_1_w, model.mm_1_b, FFN_RELU_SQR, -1);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_tensor * clip_graph_parakeet::parakeet_build_graph_conv() {
    ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, img.ny(), img.nx(), 1);
    ggml_set_name(inp, "inp_raw");
    ggml_set_input(inp);

    // [freq, time, channels, batch]
    ggml_tensor * cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], inp, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[0]);
    cb(cur, "pre_conv_0", -1);
    ggml_set_output(cur);

    cur = ggml_relu(ctx0, cur);
    cb(cur, "pre_conv_0_relu", -1);

    // [freq, time, channels, batch]
    cur = ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[2], cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[2]);
    cb(cur, "pre_conv_2", -1);

    // [freq, time, channels, batch]
    cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
    cb(cur, "pre_conv_3", -1);

    cur = ggml_relu(ctx0, cur);
    cb(cur, "pre_conv_3_relu", -1);

    // [freq, time, channels, batch]
    cur = ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[5], cur, 2, 2, 1, 1, 1, 1);
    cb(cur, "pre_conv_5_direct", -1);
    cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[5]);
    cb(cur, "pre_conv_5", -1);

    // [freq, time, channels, batch]
    cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
    cb(cur, "pre_conv_6", -1);

    cur = ggml_relu(ctx0, cur);
    cb(cur, "pre_conv_6_relu", -1);

    // [freq, time, chan]
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
    // [freq, chan, time]
    cur = ggml_cont(ctx0, cur);

    const int n_freq   = cur->ne[0];
    const int n_chan   = cur->ne[1];
    const int n_frames = cur->ne[2];

    // [freq, time, chan, batch] -> [(freq * chan), time]
    cur = ggml_reshape_2d(ctx0, cur, n_freq * n_chan, n_frames);

    cur = ggml_mul_mat(ctx0, model.pre_encode_out_w, cur);
    cur = ggml_add(ctx0, cur, model.pre_encode_out_b);

    ggml_set_name(cur, "pre_enc_out");
    ggml_set_output(cur);

    return cur;
}

ggml_tensor * clip_graph_parakeet::parakeet_build_graph_encoder(ggml_tensor * cur) {
    const auto & hparams  = model.hparams;
    const int n_layer     = hparams.n_layer;
    const int n_state     = hparams.n_embd;
    const float fc_factor = 0.5f;

    // [time_frames, time_frames, 1, 1]]
    struct ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[1]);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    const int n_time      = cur->ne[1];
    const int window_size = 2 * n_time - 1;
    const int d_half      = n_state / 2;

    struct ggml_tensor * pos_freqs = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_half);
    ggml_set_name(pos_freqs, "pos_freqs");
    ggml_set_input(pos_freqs);

    struct ggml_tensor * rel_positions = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, window_size);
    ggml_set_name(rel_positions, "rel_positions");
    ggml_set_input(rel_positions);

    struct ggml_tensor * freqs = ggml_repeat_4d(ctx0, pos_freqs, d_half, window_size, 1, 1);
    struct ggml_tensor * theta = ggml_mul(ctx0, freqs, rel_positions);

    struct ggml_tensor * sin = ggml_reshape_3d(ctx0, ggml_sin(ctx0, theta), 1, d_half, window_size);
    struct ggml_tensor * cos = ggml_reshape_3d(ctx0, ggml_cos(ctx0, theta), 1, d_half, window_size);
    struct ggml_tensor * pos_emb = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_concat(ctx0, sin, cos, 0)), n_state, window_size);
    ggml_set_name(pos_emb, "pos_emb");

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        // FFN1
        {
            struct ggml_tensor * residual = cur;
            //ggml_format_name(cur, "enc_%d_res", il);

            // norm
            cur = ggml_norm(ctx0, cur, 1e-5);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ff_norm_w), layer.ff_norm_b);
            ggml_format_name(cur, "enc_%d_ffn_norm_1", il);

            // ffn_1
            cur = ggml_mul_mat(ctx0, layer.ff_up_w, cur);
            cur = ggml_silu(ctx0, cur);
            ggml_format_name(cur, "enc_%d_silu", il);

            cur = ggml_mul_mat(ctx0, layer.ff_down_w, cur);
            ggml_format_name(cur, "enc_%d_ffn_1", il);

            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, fc_factor));
            ggml_format_name(cur, "enc_%d_res_ffn", il);
        }

        // self attention block using relative positional encoding from model.position_embedding.
        {
            // [feat, time_frames, 1, 1]
            struct ggml_tensor * residual = cur;

            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ln_1_w), layer.ln_1_b);
            ggml_format_name(cur, "enc_%d_attn_norm", il);

            const int n_head = hparams.n_head;
            const int d_head = n_state / n_head;
            const int n_time = cur->ne[1];

            // [feat, time_frames, 1, 1]
            struct ggml_tensor * Q_cur = ggml_mul_mat(ctx0, layer.q_w, cur);
            struct ggml_tensor * K_cur = ggml_mul_mat(ctx0, layer.k_w, cur);
            struct ggml_tensor * V_cur = ggml_mul_mat(ctx0, layer.v_w, cur);

            // [d_head, n_heads, time_frames, 1]
            Q_cur = ggml_reshape_3d(ctx0, Q_cur, d_head, n_head, n_time);
            K_cur = ggml_reshape_3d(ctx0, K_cur, d_head, n_head, n_time);
            V_cur = ggml_reshape_3d(ctx0, V_cur, d_head, n_head, n_time);

            // [n_state, window_size]
            struct ggml_tensor * pos = ggml_mul_mat(ctx0, layer.linear_pos_w, pos_emb);
            ggml_format_name(pos, "enc_%d_attn_pos", il);

            // Add the content bias to Q.
            // [feat, head, time_frames, batch]
            struct ggml_tensor * Q_u = ggml_add(ctx0, Q_cur, layer.pos_bias_u);
            ggml_format_name(Q_u, "enc_%d_attn_q_u", il);

            // [feat, time_frames, head, 1]
            struct ggml_tensor * K_prep = ggml_permute(ctx0, K_cur, 0, 2, 1, 3);
            // [feat, time_frames, head, 1]
            struct ggml_tensor * Q_prep = ggml_permute(ctx0, Q_u,   0, 2, 1, 3);
            // [feat, feat, head, 1]
            struct ggml_tensor * content_scores = ggml_mul_mat(ctx0, K_prep, Q_prep);
            ggml_format_name(content_scores, "enc_%d_attn_content_scores", il);

            // Add the position bias to Q.
            // [feat, head, time_frames, batch]
            struct ggml_tensor * Q_v = ggml_add(ctx0, Q_cur, layer.pos_bias_v);
            ggml_format_name(Q_v, "enc_%d_attn_q_v", il);

            // [feat, window_size, 1, 1] and we are doing multi-head attention so
            // we need to split this into heads.
            // [feat, head, window_size, 1]
            pos = ggml_reshape_3d(ctx0, pos, d_head, n_head, pos_emb->ne[1]);

            // [feat, window_size, head, 1]
            pos = ggml_permute(ctx0, pos, 0, 2, 1, 3);
            pos = ggml_cont(ctx0, pos);
            ggml_format_name(pos, "enc_%d_attn_pos_perm", il);
            // [feat, time, head, 1]
            Q_v = ggml_permute(ctx0, Q_v, 0, 2, 1, 3);
            Q_v = ggml_cont(ctx0, Q_v);
            ggml_format_name(Q_v, "enc_%d_attn_q_v_perm", il);

            // [window_size, time_frames, head, 1]
            struct ggml_tensor * rel_pos_scores = ggml_mul_mat(ctx0, pos, Q_v);
            ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos", il);

            // Relative positional shift
            {

                const auto pos_window = rel_pos_scores->ne[0];
                const auto n_frame    = rel_pos_scores->ne[1];
                const auto n_head     = rel_pos_scores->ne[2];

                // [feat_padded, window_size, head, 1]
                rel_pos_scores = ggml_pad(ctx0, rel_pos_scores, 1, 0, 0, 0);
                rel_pos_scores = ggml_roll(ctx0, rel_pos_scores, 1, 0, 0, 0);

                rel_pos_scores = ggml_reshape_3d(ctx0, rel_pos_scores, n_frame, pos_window + 1, n_head);
                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);
                ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_reshaped", il);

                int center = pos_window / 2;
                size_t offset = rel_pos_scores->nb[0] * (center+1);

                rel_pos_scores = ggml_view_3d(ctx0, rel_pos_scores,
                                              n_frame, pos_window, n_head,
                                              (pos_window) * 4,
                                              rel_pos_scores->nb[2],
                                              offset);

                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);
                ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_shifted", il);

                rel_pos_scores = ggml_view_3d(ctx0, rel_pos_scores,
                                              content_scores->ne[0],
                                              content_scores->ne[1],
                                              rel_pos_scores->ne[2],
                                              rel_pos_scores->nb[1],
                                              rel_pos_scores->nb[2],
                                              0);
                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);
                ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_shifted_view", il);
            }

            struct ggml_tensor * attn_scores = ggml_add(ctx0, content_scores, rel_pos_scores);
            attn_scores = ggml_cont(ctx0, attn_scores);
            ggml_format_name(attn_scores, "enc_%d_attn_scores", il);
            attn_scores = ggml_scale(ctx0, attn_scores, 1.0f / std::sqrt(d_head));
            attn_scores = ggml_add(ctx0, attn_scores, attn_mask);
            ggml_format_name(attn_scores, "enc_%d_attn_scores_scaled", il);

            struct ggml_tensor * probs = ggml_soft_max(ctx0, attn_scores);
            ggml_format_name(probs, "enc_%d_attn_probs", il);

            V_cur = ggml_cont(ctx0, ggml_permute(ctx0, V_cur, 1, 2, 0, 3));
            ggml_format_name(V_cur, "enc_%d_attn_v_cur", il);
            cur = ggml_mul_mat(ctx0, probs, V_cur);
            ggml_format_name(cur, "enc_%d_attn_inp", il);

            cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, n_state, n_time);
            cur = ggml_mul_mat(ctx0, layer.o_w, cur);
            ggml_format_name(cur, "enc_%d_attn_out", il);

            cur = ggml_add(ctx0, residual, cur);
            ggml_format_name(cur, "enc_%d_attn_res", il);
        }

        // Convolution
        {
            struct ggml_tensor * residual = cur;
            ggml_format_name(cur, "enc_%d_residual_conv", il);

            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.norm_conv_w), layer.norm_conv_b);
            ggml_format_name(cur, "enc_%d_norm_conv", il);

            // pointwise 1d convolution:
            cur = ggml_mul_mat(ctx0, layer.conv_pw1_w, cur);
            ggml_format_name(cur, "enc_%d_conv_pw1", il);

            {
                int64_t d = cur->ne[0] / 2;
                struct ggml_tensor * signal = ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], 0);
                struct ggml_tensor * gate   = ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], d * cur->nb[0]);

                cur = ggml_mul(ctx0, signal, ggml_sigmoid(ctx0, gate));
                ggml_format_name(cur, "enc_%d_conv_glu", il);
            }

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            // use ggml_ssm_conv for f32 precision
            cur = ggml_pad(ctx0, cur, 4, 0, 0, 0);
            cur = ggml_roll(ctx0, cur, 4, 0, 0, 0);
            cur = ggml_pad(ctx0, cur, 4, 0, 0, 0);
            ggml_format_name(cur, "enc_%d_conv_dw_pad", il);

            cur = ggml_ssm_conv(ctx0, cur, layer.conv_dw_w);
            ggml_format_name(cur, "enc_%d_conv_1d_dw", il);

            cur = ggml_sub(ctx0, cur, layer.conv_norm_mean);
            struct ggml_tensor * std = ggml_sqrt(ctx0, layer.conv_norm_var);
            cur = ggml_div(ctx0, cur, std);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.conv_norm_w), layer.conv_norm_b);
            ggml_format_name(cur, "enc_%d_conv_bn", il);

            cur = ggml_silu(ctx0, cur);
            ggml_format_name(cur, "enc_%d_conv_silu", il);

            cur = ggml_mul_mat(ctx0, layer.conv_pw2_w, cur);
            ggml_format_name(cur, "enc_%d_conv_pw2", il);

            cur = ggml_add(ctx0, residual, cur);
            ggml_format_name(cur, "enc_%d_conv_res", il);
        }

        // FFN2
        {
            struct ggml_tensor * residual = cur;
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ff_norm_1_w), layer.ff_norm_1_b);
            ggml_format_name(cur, "enc_%d_ffn_norm_2", il);

            cur = ggml_mul_mat(ctx0, layer.ff_up_1_w, cur);
            cur = ggml_silu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, layer.ff_down_1_w, cur);
            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, 0.5));
            ggml_format_name(cur, "enc_%d_ffn_res", il);
        }

        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ln_2_w), layer.ln_2_b);
    }

    cb(cur, "encoder_out", -1);

    ggml_build_forward_expand(gf, cur);

    return cur;
}
