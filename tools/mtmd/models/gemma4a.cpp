/**
 * Gemma 4 Audio Conformer Encoder (clip_graph_gemma4a)
 *
 * Architecture: Conformer with dual half-step FFN, full self-attention
 * with sinusoidal RPE, depthwise light conv, and output projection.
 */

#include "models.h"
#include <cmath>

ggml_cgraph * clip_graph_gemma4a::build() {
    const float res_weight = 0.5f;
    const float norm_eps   = 1e-6f;

    // 1. Input
    ggml_tensor * inp = build_inp_raw(1);
    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    // 2. Subsampling Conv2D (symmetric padding=1, matching PyTorch)
    {
        for (int i = 0; i < 2; i++) {
            cur = ggml_conv_2d(ctx0, model.sscp_conv_w[i], cur, 2, 2, 1, 1, 1, 1);
            if (model.sscp_conv_b[i]) {
                cur = ggml_add(ctx0, cur, model.sscp_conv_b[i]);
            }
            // nn.LayerNorm(channels): permute ch to ne[0], normalize, permute back
            if (model.sscp_norm_w[i]) {
                cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
                cur = ggml_norm(ctx0, cur, norm_eps);
                cur = ggml_mul(ctx0, cur, model.sscp_norm_w[i]);
                cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
            }
            cur = ggml_relu(ctx0, cur);
        }
        // Flatten [freq, time, ch, 1] -> [ch*freq, time]
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
        if (model.sscp_inp_proj_w) {
            cur = build_mm(model.sscp_inp_proj_w, cur);
            if (model.sscp_inp_proj_b) {
                cur = ggml_add(ctx0, cur, model.sscp_inp_proj_b);
            }
        }
    }

    const int64_t n_pos = cur->ne[1];

    // Chunked local attention parameters
    const int64_t C  = 12;                              // chunk_size
    const int64_t P  = 12;                              // max_past_horizon (context_left - 1)
    const int64_t S  = C + P;                           // context_size = 24
    const int64_t R  = P + 1;                           // RPE positions = 13
    const int64_t B  = (n_pos + C - 1) / C;            // num_blocks
    const int64_t Np = B * C;                           // padded sequence length
    const int64_t pad_seq = Np - n_pos;

    // Input tensors: blocked RPE and blocked attention mask
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_head * d_head, R);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);

    ggml_tensor * kq_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, S, C, B);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // 3. Conformer Blocks
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];
        auto * residual = cur;

        // FFN 1 (half-step)
        if (layer.ff_norm_w && layer.ff_up_w && layer.ff_down_w) {
            cur = build_norm(cur, layer.ff_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_w, nullptr, nullptr, nullptr,
                layer.ff_down_w, nullptr, FFN_SILU, il);
            if (layer.ff_post_norm_w) {
                cur = build_norm(cur, layer.ff_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // Chunked local self-attention with RPE
        if (layer.q_w && layer.k_w && layer.v_w && layer.o_w) {
            const float q_scale = (1.0f / sqrtf((float)d_head)) / logf(2.0f);
            const float k_scale = logf(1.0f + expf(1.0f)) / logf(2.0f);
            const float softcap = 50.0f;

            ggml_tensor * attn_norm_w = layer.attn_pre_norm_w ? layer.attn_pre_norm_w : layer.ln_1_w;
            cur = attn_norm_w
                ? build_norm(residual, attn_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il)
                : residual;

            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            ggml_tensor * Vcur = build_mm(layer.v_w, cur);

            // [n_embd, n_pos] -> [D, H, N]
            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            // Q/K scaling
            Qcur = ggml_scale(ctx0, Qcur, q_scale);
            if (layer.per_dim_scale_w) {
                Qcur = ggml_mul(ctx0, Qcur, ggml_reshape_3d(ctx0, layer.per_dim_scale_w, d_head, 1, 1));
            }
            Kcur = ggml_scale(ctx0, Kcur, k_scale);
            if (layer.per_dim_k_scale_w) {
                Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_3d(ctx0, layer.per_dim_k_scale_w, d_head, 1, 1));
            }

            // Q blocking: [D, H, N] -> pad to Np -> reshape [D, H, C, B]
            // ggml permute: ne[ax_i] = src->ne[i], so (0,3,1,2) sends H->3, C->1, B->2
            Qcur = ggml_pad(ctx0, Qcur, 0, 0, pad_seq, 0);          // [D, H, Np]
            Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, C, B); // [D, H, C, B]
            Qcur = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 3, 1, 2)); // [D, C, B, H]

            // K/V block context extraction via overlapping view:
            // Pad to S*B elements, roll right by P to create left-padding,
            // then view with stride C in the block dimension (overlapping windows).
            auto extract_blocks = [&](ggml_tensor * t) -> ggml_tensor * {
                // [D, H, N] -> pad to S*B -> roll right by P -> cont (materialize)
                const int64_t pad_kv = S * B - n_pos;
                t = ggml_pad(ctx0, t, 0, 0, pad_kv, 0);     // [D, H, S*B]
                t = ggml_roll(ctx0, t, 0, 0, P, 0);          // left-pad by P
                t = ggml_cont(ctx0, t);                       // materialize roll (removes view offset)
                // Overlapping view: stride for B dim is C positions, not S
                // ne = [D, H, S, B], data_size = D*H*S*B*sizeof = source_nbytes (exact fit)
                // nb1=D*sizeof, nb2=D*H*sizeof, nb3=C*D*H*sizeof (overlap: C < S)
                t = ggml_view_4d(ctx0, t, d_head, n_head, S, B,
                    t->nb[1], t->nb[2], C * t->nb[2], 0);
                t = ggml_cont(ctx0, t);                       // materialize overlapping windows
                return t;
            };

            ggml_tensor * Kblk = extract_blocks(Kcur);
            // [D, H, S, B] -> [D, S, B, H] via permute(0,3,1,2)
            Kblk = ggml_cont(ctx0, ggml_permute(ctx0, Kblk, 0, 3, 1, 2));

            ggml_tensor * Vblk = extract_blocks(Vcur);
            // [D, H, S, B] -> [S, D, B, H] via permute(1,3,0,2)
            Vblk = ggml_cont(ctx0, ggml_permute(ctx0, Vblk, 1, 3, 0, 2));

            // Content attention: Q @ K^T
            // Kblk=[D,S,B,H], Qcur=[D,C,B,H] -> mul_mat contracts on D -> [S,C,B,H]
            ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, Kblk, Qcur);

            // Relative position attention
            if (layer.attn_k_rel_w) {
                // RPE: [n_embd, R] -> project -> [D, H, R] -> [D, R, H]
                auto * p = ggml_mul_mat(ctx0, layer.attn_k_rel_w, pos_emb);
                p = ggml_reshape_3d(ctx0, p, d_head, n_head, R);
                p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3)); // [D, R, H]

                // Q_flat @ RPE^T: [D, C*B, H] @ [D, R, H] -> [R, C*B, H]
                auto * Q_flat = ggml_reshape_3d(ctx0, Qcur, d_head, C * B, n_head);
                auto * matrix_bd = ggml_mul_mat(ctx0, p, Q_flat);       // [R, C*B, H]
                matrix_bd = ggml_reshape_4d(ctx0, matrix_bd, R, C, B, n_head); // [R, C, B, H]

                // Blocked relative shift (appendix B of Transformer-XL)
                {
                    matrix_bd = ggml_pad(ctx0, matrix_bd, S + 1 - R, 0, 0, 0); // [S+1, C, B, H]
                    matrix_bd = ggml_reshape_3d(ctx0, matrix_bd, (S + 1) * C, B, n_head);
                    matrix_bd = ggml_view_3d(ctx0, matrix_bd,
                        C * S, B, n_head,
                        matrix_bd->nb[1], matrix_bd->nb[2], 0);
                    matrix_bd = ggml_cont(ctx0, matrix_bd);              // [C*S, B, H]
                    matrix_bd = ggml_reshape_4d(ctx0, matrix_bd, S, C, B, n_head); // [S, C, B, H]
                }

                matrix_ac = ggml_add(ctx0, matrix_ac, matrix_bd);
            }

            auto * scores = matrix_ac; // [S, C, B, H]

            // Softcap
            scores = ggml_scale(ctx0, scores, 1.0f / softcap);
            scores = ggml_tanh(ctx0, scores);
            scores = ggml_scale(ctx0, scores, softcap);

            // Blocked attention mask: [S, C, B] broadcasts over H
            scores = ggml_add(ctx0, scores, kq_mask);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);

            // attn @ V: [S,C,B,H] @ [S,D,B,H] -> [D,C,B,H]
            ggml_tensor * x = ggml_mul_mat(ctx0, Vblk, attn);

            // [D,C,B,H] -> [D,H,C,B] via permute(0,2,3,1) -> flatten -> trim
            x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 3, 1));
            x = ggml_cont_2d(ctx0, x, d_head * n_head, C * B);
            if (pad_seq > 0) {
                x = ggml_view_2d(ctx0, x, d_head * n_head, n_pos, x->nb[1], 0);
                x = ggml_cont(ctx0, x);
            }

            x = build_mm(layer.o_w, x);
            if (layer.o_b) { x = ggml_add(ctx0, x, layer.o_b); }

            if (layer.attn_post_norm_w) {
                x = build_norm(x, layer.attn_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, x);
        }

        // Convolution Module
        if (layer.norm_conv_w && layer.conv_pw1_w && layer.conv_dw_w && layer.conv_pw2_w) {
            cur = build_norm(residual, layer.norm_conv_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            auto * x = build_mm(layer.conv_pw1_w, cur);

            // GLU
            {
                int64_t d = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0,
                    ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0]));
                x = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }

            // Causal depthwise Conv1D via ggml_ssm_conv (pad+roll for left-only padding).
            // NOTE: ggml_ssm_conv on CUDA only supports kernel sizes 3, 4, 9.
            // Gemma 4 uses kernel_size=5. This works on CPU and Vulkan backends.
            // TODO: fix ggml-cuda ssm_conv to support kernel_size=5, or use ggml_conv_1d_dw
            x = ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = ggml_roll(ctx0, x, 4, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            if (layer.conv_dw_b) {
                x = ggml_add(ctx0, x, layer.conv_dw_b);
            }

            if (layer.conv_norm_w) {
                x = ggml_rms_norm(ctx0, x, norm_eps);
                x = ggml_mul(ctx0, x, layer.conv_norm_w);
            }
            x = ggml_silu(ctx0, x);
            x = build_mm(layer.conv_pw2_w, x);
            residual = ggml_add(ctx0, residual, x);
        }

        // FFN 2 (half-step)
        if (layer.ff_norm_1_w && layer.ff_up_1_w && layer.ff_down_1_w) {
            cur = build_norm(residual, layer.ff_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_1_w, nullptr, nullptr, nullptr,
                layer.ff_down_1_w, nullptr, FFN_SILU, il);
            if (layer.ff_post_norm_1_w) {
                cur = build_norm(cur, layer.ff_post_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // Layer output norm
        cur = layer.ln_2_w
            ? build_norm(residual, layer.ln_2_w, nullptr, NORM_TYPE_RMS, norm_eps, il)
            : residual;

    }

    // 4. Output Projection
    if (model.audio_out_proj_w) {
        cur = build_mm(model.audio_out_proj_w, cur);
        if (model.audio_out_proj_b) {
            cur = ggml_add(ctx0, cur, model.audio_out_proj_b);
        }
    }

    // 5. Audio Multimodal Embedder
    cur = ggml_rms_norm(ctx0, cur, norm_eps);
    if (model.mm_soft_emb_norm_w) {
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
    }
    if (model.mm_input_proj_w) {
        cur = build_mm(model.mm_input_proj_w, cur);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_tensor * clip_graph_gemma4a::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    auto it = model.clamp_info_map.find(w->name);
    if (it == model.clamp_info_map.end()) {
        return ggml_mul_mat(ctx0, w, x);
    }
    const auto & ci = it->second;
    ggml_tensor * clamped = ggml_clamp(ctx0, x, ci.inp_min, ci.inp_max);
    ggml_tensor * out = ggml_mul_mat(ctx0, w, clamped);
    return ggml_clamp(ctx0, out, ci.out_min, ci.out_max);
}
