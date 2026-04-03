/**
 * Gemma 4 Audio Conformer Encoder — clip_graph_gemma4a
 *
 * Architecture: Conformer with dual half-step FFN, self-attention,
 * depthwise light conv, and output projection.
 */

#include "models.h"
#include <cmath>

ggml_cgraph * clip_graph_gemma4a::build() {
    const float res_weight = 0.5f;
    const float norm_eps   = 1e-6f;

    // ── 1. Input: mel spectrogram ────────────────────────────
    // build_inp_raw(1) gives us [mel_bins, n_frames]
    ggml_tensor * inp = build_inp_raw(1);

    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    // cur: [n_frames, mel_bins]

    // ── 2. Subsampling Conv2D ────────────────────────────────
    // Two Conv2D: 1→128→32 channels, kernel 3x3, stride 2x2, pad 1
    //
    // build_inp_raw(1) gives ne[]=[n_frames, mel_bins, 1] (img.nx=frames)
    // After transpose+cont: ne[]=[mel_bins=128, n_frames, 1]
    // ggml_conv_2d treats this as [IW=128, IH=n_frames, IC=1, N=1]
    // No reshape needed — matches conformer.cpp pattern exactly.
    //
    // Conv formula: O = floor((I + 2*p - d*(K-1) - 1) / s) + 1
    // With K=3, s=2, p=1: O = floor((I - 1) / 2) + 1
    //
    // Conv1: [128, N, 1, 1] → [64, N/2, 128, 1]  (OW=64 freq, OH=N/2 time, OC=128)
    // Conv2: [64, N/2, 128, 1] → [32, N/4, 32, 1] (OW=32 freq, OH=N/4 time, OC=32)
    {
        for (int i = 0; i < 2; i++) {
            if (!model.audio_conv2d_w[i]) break;
            // Gemma4: Conv2d with symmetric padding=1 on all sides
            // ggml_conv_2d(w, x, s0, s1, p0, p1, d0, d1)
            // p0=1 (freq), p1=1 (time) — symmetric padding
            cur = ggml_conv_2d(ctx0, model.audio_conv2d_w[i], cur, 2, 2, 1, 1, 1, 1);
            // Conv2d output: [OW=freq, OH=time, OC=channels, N=1]
            // HF Gemma4: norm(x.permute(0,2,3,1)).permute(0,3,1,2) — LayerNorm over channels
            // In ggml: permute channels (ne[2]) to ne[0], LayerNorm, permute back
            if (model.audio_conv2d_norm[i]) {
                cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
                cur = ggml_norm(ctx0, cur, 1e-6f);
                cur = ggml_mul(ctx0, cur, model.audio_conv2d_norm[i]);
                cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
            }
            cur = ggml_relu(ctx0, cur);
            cb(cur, "audio_conv2d", i);
        }

        // cur ne[] = [OW=32, OH=time_out, OC=32, N=1]
        //             freq    time        channels
        //
        // Flatten: [OW, OH, OC, 1] → [OW*OC, OH, 1, 1] (merge ne[0]*ne[2] via permute+reshape)
        // HF does [B,C,T,F]→permute(0,2,3,1)→[B,T,F,C]→reshape(B,T,F*C)
        // C must be fastest-varying. In ggml column-major, that's ne[0].
        // cur is [OW=freq, OH=time, OC=ch, 1]
        // permute(1,2,0,3): ne[1]=freq, ne[2]=time, ne[0]=ch → [ch, freq, time, 1]
        // reshape merges ne[0]*ne[1] → [ch*freq, time]
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
        cb(cur, "audio_subsample_flat", -1);

        // Linear projection: [1024, 1024] x [1024, time_out] → [1024, time_out]
        if (model.audio_inp_proj_w) {
            cur = build_mm(model.audio_inp_proj_w, cur);
            cb(cur, "audio_inp_proj", -1);
        }
    }

    // cur: [hidden_size=1024, n_pos]

    // ── Precompute sinusoidal relative position embeddings ──
    // Shared across all layers; only the projection weight (k_rel_w) varies per layer.
    // positions: [max_past, max_past-1, ..., 0] → F_span = max_past + 1 = 13
    // inv_timescales: exp(i * -log(10000) / max(channels/2 - 1, 1)) for i = 0..channels/2-1
    const int max_past_rpe = 12;  // conf_attention_context_left - 1
    const int F_span = max_past_rpe + 1;  // 13
    const int channels = n_head * d_head;  // 1024
    const int num_timescales = channels / 2;  // 512

    ggml_tensor * sin_pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, channels, F_span);
    ggml_set_name(sin_pos_emb, "sin_pos_emb");
    ggml_set_input(sin_pos_emb);

    // ── 3. Conformer Blocks ──────────────────────────────────
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];
        auto * residual = cur;

        // ── 3a. Feed-Forward 1 (half-step) ───────────────────
        if (layer.ff_norm_w && layer.ff_up_w && layer.ff_down_w) {
            cur = build_norm(cur, layer.ff_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_w, nullptr,
                nullptr, nullptr,
                layer.ff_down_w, nullptr,
                FFN_SILU, il);
            if (layer.ff_post_norm_w) {
                cur = build_norm(cur, layer.ff_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // ── 3b. Self-Attention (Gemma4 chunked local + RPE + softcap) ──
        if (layer.q_w && layer.k_w && layer.v_w && layer.o_w) {
            // Gemma4: q_scale = (1/sqrt(d_head)) / ln(2), k_scale = ln(1+e) / ln(2)
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

            // Reshape [hidden, time] → [d_head, n_head, time]
            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, Qcur->ne[1]);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, Kcur->ne[1]);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, Vcur->ne[1]);

            // K scaling: K = K * k_scale
            Kcur = ggml_scale(ctx0, Kcur, k_scale);

            // Q scaling: Q = Q * q_scale * softplus(per_dim_scale)
            Qcur = ggml_scale(ctx0, Qcur, q_scale);
            if (layer.per_dim_scale_w) {
                // softplus(x) = log(1 + exp(x))
                ggml_tensor * ones = ggml_fill(ctx0, layer.per_dim_scale_w, 1.0f);
                ggml_tensor * sp = ggml_log(ctx0, ggml_add(ctx0, ggml_exp(ctx0, layer.per_dim_scale_w), ones));
                ggml_tensor * scale = ggml_reshape_3d(ctx0, sp, d_head, 1, 1);
                Qcur = ggml_mul(ctx0, Qcur, scale);
            }

            // ── Chunked Local Attention ──
            // Config: chunk_size=12, context_left=13, context_right=0
            const int chunk_sz  = 12;
            const int max_past  = 12;  // context_left - 1
            const int ctx_sz    = chunk_sz + max_past; // 24

            // Q,K,V: [d_head, n_head, time]
            // Permute to [d_head, time, n_head] for easier blocking
            Qcur = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 2, 1, 3));
            Kcur = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));
            Vcur = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 0, 2, 1, 3));
            // All now: [d_head, time, n_head]

            const int64_t T = Qcur->ne[1];
            const int64_t T_padded = ((T + chunk_sz - 1) / chunk_sz) * chunk_sz;
            const int64_t n_blocks = T_padded / chunk_sz;

            // Pad Q to T_padded on time dim (ne[1])
            if (T_padded > T) {
                Qcur = ggml_pad(ctx0, Qcur, 0, (int)(T_padded - T), 0, 0);
            }

            // Reshape Q into blocks: [d_head, T_padded, n_head] → [d_head, chunk_sz, n_blocks, n_head]
            ggml_tensor * Q_blocks = ggml_reshape_4d(ctx0, Qcur, d_head, chunk_sz, n_blocks, n_head);

            // Pad K,V: need max_past zeros at the front + pad to T_padded
            // Strategy: pad end by (max_past + T_padded - T), then roll right by max_past
            {
                int64_t kv_extra = max_past + (T_padded - T);
                if (kv_extra > 0) {
                    Kcur = ggml_pad(ctx0, Kcur, 0, (int)kv_extra, 0, 0);
                    Vcur = ggml_pad(ctx0, Vcur, 0, (int)kv_extra, 0, 0);
                }
                Kcur = ggml_roll(ctx0, Kcur, 0, max_past, 0, 0);
                Vcur = ggml_roll(ctx0, Vcur, 0, max_past, 0, 0);
            }
            // K,V now: [d_head, max_past + T_padded, n_head] with max_past leading zeros

            // Extract overlapping context windows for K and V
            ggml_tensor * K_blocks = nullptr;
            ggml_tensor * V_blocks = nullptr;
            for (int64_t b = 0; b < n_blocks; b++) {
                size_t offset = (size_t)(b * chunk_sz) * Kcur->nb[1];
                auto * kb = ggml_cont(ctx0, ggml_view_3d(ctx0, Kcur, d_head, ctx_sz, n_head,
                                         Kcur->nb[1], Kcur->nb[2], offset));
                auto * vb = ggml_cont(ctx0, ggml_view_3d(ctx0, Vcur, d_head, ctx_sz, n_head,
                                         Vcur->nb[1], Vcur->nb[2], offset));
                kb = ggml_reshape_4d(ctx0, kb, d_head, ctx_sz, 1, n_head);
                vb = ggml_reshape_4d(ctx0, vb, d_head, ctx_sz, 1, n_head);
                if (b == 0) {
                    K_blocks = kb;
                    V_blocks = vb;
                } else {
                    K_blocks = ggml_concat(ctx0, K_blocks, kb, 2);
                    V_blocks = ggml_concat(ctx0, V_blocks, vb, 2);
                }
            }

            // Q_blocks: [d_head, chunk_sz, n_blocks, n_head]
            // K_blocks: [d_head, ctx_sz,   n_blocks, n_head]
            // V_blocks: [d_head, ctx_sz,   n_blocks, n_head]

            // ── term_ac: content attention (Q @ K^T) ──
            // ggml_mul_mat contracts over ne[0]=d_head
            // Result: [ctx_sz, chunk_sz, n_blocks, n_head]
            ggml_tensor * term_ac = ggml_mul_mat(ctx0, K_blocks, Q_blocks);

            // ── term_bd: relative position attention (Q @ sin_emb_proj^T) ──
            ggml_tensor * term_bd_shifted = nullptr;
            if (layer.k_rel_w) {
                // Project sinusoidal embeddings through per-layer k_rel_w
                // sin_pos_emb: [channels=1024, F_span=13]
                // k_rel_w: [1024, 1024]
                // Result: [1024, F_span=13]
                ggml_tensor * sin_proj = ggml_mul_mat(ctx0, layer.k_rel_w, sin_pos_emb);

                // Reshape to [d_head, n_head, F_span, 1]
                sin_proj = ggml_reshape_4d(ctx0, sin_proj, d_head, n_head, F_span, 1);
                // Permute to [d_head, F_span, 1, n_head]
                // permute(0,3,1,2): src[0]→res[0], src[1]→res[3], src[2]→res[1], src[3]→res[2]
                sin_proj = ggml_cont(ctx0, ggml_permute(ctx0, sin_proj, 0, 3, 1, 2));
                // sin_proj: [d_head, F_span, 1, n_head] — broadcasts over n_blocks

                // term_bd = mul_mat(sin_proj, Q_blocks)
                // Contracts over ne[0]=d_head
                // Result: [F_span, chunk_sz, n_blocks, n_head]
                ggml_tensor * term_bd = ggml_mul_mat(ctx0, sin_proj, Q_blocks);

                // ── Relative shift (Transformer-XL diagonal skew) ──
                // HF operates on [..., W, F_span] with W outer, F_span inner (contiguous).
                // Our term_bd is [F_span(ne0), chunk_sz(ne1), n_blocks, n_head].
                // The skew trick requires F_span to be the INNER (ne[0]) dimension
                // and W (chunk_sz) to be the OUTER (ne[1]). Our layout already has
                // F_span at ne[0], but ggml's reshape merges ne[0]*ne[1] in
                // column-major order (ne[0] varies fastest), so elements interleave
                // by F_span stride — which IS the correct skew direction.
                //
                // HOWEVER, HF's reshape is row-major where W is outer and F_span is
                // inner. For the skew to work identically, we need chunk_sz at ne[0]
                // and F_span at ne[1] before the reshape, so the memory order matches.
                //
                // Step 1: transpose ne[0] and ne[1] → [chunk_sz, F_span, n_blocks, n_head]
                term_bd = ggml_cont(ctx0, ggml_permute(ctx0, term_bd, 1, 0, 2, 3));
                // Now: [chunk_sz=ne0, F_span=ne1, n_blocks, n_head]

                // Step 2: pad ne[1] (F_span) to ctx_sz+1
                const int pad_amount = (ctx_sz + 1) - F_span;
                term_bd = ggml_pad(ctx0, term_bd, 0, pad_amount, 0, 0);
                // [chunk_sz, ctx_sz+1, n_blocks, n_head]

                // Step 3: reshape to merge ne[0]*ne[1] = chunk_sz*(ctx_sz+1)
                term_bd = ggml_reshape_3d(ctx0, term_bd,
                    chunk_sz * (ctx_sz + 1), n_blocks, n_head);

                // Step 4: slice to chunk_sz * ctx_sz
                term_bd = ggml_view_3d(ctx0, term_bd,
                    (int64_t)chunk_sz * ctx_sz, n_blocks, n_head,
                    term_bd->nb[1], term_bd->nb[2], 0);
                term_bd = ggml_cont(ctx0, term_bd);

                // Step 5: reshape to [chunk_sz, ctx_sz, n_blocks, n_head]
                term_bd = ggml_reshape_4d(ctx0, term_bd,
                    chunk_sz, ctx_sz, n_blocks, n_head);

                // Step 6: transpose back → [ctx_sz, chunk_sz, n_blocks, n_head]
                // to match scores/term_ac layout
                term_bd_shifted = ggml_cont(ctx0, ggml_permute(ctx0, term_bd, 1, 0, 2, 3));
            }

            // ── Combined logits = term_ac + term_bd ──
            ggml_tensor * scores = term_bd_shifted
                ? ggml_add(ctx0, term_ac, term_bd_shifted)
                : term_ac;

            // Softcap: tanh(scores / 50) * 50
            scores = ggml_scale(ctx0, scores, 1.0f / softcap);
            scores = ggml_tanh(ctx0, scores);
            scores = ggml_scale(ctx0, scores, softcap);

            // Causal + validity mask: filled at encode time via set_input_f32("attn_mask")
            // Shape [ctx_sz, chunk_sz, n_blocks] — per-block to handle invalid left context
            // Broadcasts over [n_head] only
            {
                ggml_tensor * mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ctx_sz, chunk_sz, n_blocks);
                ggml_set_name(mask, "attn_mask");
                ggml_set_input(mask);
                scores = ggml_add(ctx0, scores, mask);
            }

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);
            // attn: [ctx_sz, chunk_sz, n_blocks, n_head]

            // Context vectors: V_blocks^T @ attn
            ggml_tensor * V_perm = ggml_cont(ctx0, ggml_permute(ctx0, V_blocks, 1, 0, 2, 3));
            ggml_tensor * x = ggml_mul_mat(ctx0, V_perm, attn);

            // Reshape back: [d_head, chunk_sz, n_blocks, n_head] → [d_head, T_padded, n_head]
            x = ggml_reshape_3d(ctx0, x, d_head, T_padded, n_head);

            // Trim to original length T
            if (T_padded > T) {
                x = ggml_view_3d(ctx0, x, d_head, T, n_head, x->nb[1], x->nb[2], 0);
            }

            // x: [d_head, T, n_head] → flatten to [d_head*n_head, T]
            x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
            x = ggml_reshape_2d(ctx0, x, d_head * n_head, x->ne[2]);

            cur = build_mm(layer.o_w, x);
            if (layer.attn_post_norm_w) {
                cur = build_norm(cur, layer.attn_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, cur);
        }

        // ── 3c. Convolution Module ───────────────────────────
        if (layer.norm_conv_w && layer.conv_pw1_w && layer.conv_dw_w && layer.conv_pw2_w) {
            cur = build_norm(residual, layer.norm_conv_w, nullptr, NORM_TYPE_RMS, norm_eps, il);

            auto * x = build_mm(layer.conv_pw1_w, cur);

            // GLU gating
            {
                int64_t d = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0,
                    ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0]));
                x = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }

            // Depthwise Conv1D (causal pad)
            // Kernel size=5 → left_pad = kernel_size-1 = 4
            // pad(4) adds 4 zeros at end, roll(4) shifts them to the start
            // ssm_conv removes d_conv-1=4 → output time == input time
            x = ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = ggml_roll(ctx0, x, 4, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);

            if (layer.conv_norm_w) {
                x = ggml_rms_norm(ctx0, x, norm_eps);
                x = ggml_mul(ctx0, x, layer.conv_norm_w);
            }
            x = ggml_silu(ctx0, x);
            x = build_mm(layer.conv_pw2_w, x);

            residual = ggml_add(ctx0, residual, x);
        }

        // ── 3d. Feed-Forward 2 (half-step) ───────────────────
        if (layer.ff_norm_1_w && layer.ff_up_1_w && layer.ff_down_1_w) {
            cur = build_norm(residual, layer.ff_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_1_w, nullptr,
                nullptr, nullptr,
                layer.ff_down_1_w, nullptr,
                FFN_SILU, il);
            if (layer.ff_post_norm_1_w) {
                cur = build_norm(cur, layer.ff_post_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // ── 3e. Final layer norm ─────────────────────────────
        if (layer.ln_2_w) {
            cur = build_norm(residual, layer.ln_2_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
        } else {
            cur = residual;
        }
    }

    // ── 4. Output Projection (1024 → 1536) ───────────────────
    cur = build_mm(model.pre_encode_out_w, cur);
    if (model.pre_encode_out_b) {
        cur = ggml_add(ctx0, cur, model.pre_encode_out_b);
    }

    // ── 6. Audio Multimodal Embedder (embed_audio) ───────────
    // Gemma4: embedding_pre_projection_norm (RMSNorm, no scale) → embedding_projection
    // Note: Gemma3n had 3 steps with soft_emb_norm + post_proj_norm; Gemma4 has only 2
    {
        // embedding_pre_projection_norm: RMSNorm with_scale=False
        cur = ggml_rms_norm(ctx0, cur, norm_eps);

        // embedding_projection: mm.a.input_projection.weight [1536, 1536]
        if (model.mm_audio_input_proj_w) {
            cur = build_mm(model.mm_audio_input_proj_w, cur);
        }
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
