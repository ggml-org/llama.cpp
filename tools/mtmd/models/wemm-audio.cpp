// WeMM-Video audio encoder graph (arch = wemm_audio)
// pipeline (matches tencent/WeMM-Video-35BA3B h5_modeling_whisper.py):
//   128-mel frames [3000] -> conv1(s1,p1)+gelu_erf -> conv2(s2,p1)+gelu_erf
//   -> +position_embd(1500) -> 32 transformer layers -> ln_post
//   -> 5-frame non-overlapping windows -> BLIP-style qformer (2 BERT layers,
//      1 learnable query, cross-attn to the 5 whisper frames of the window)
//   -> audio_proj Linear(4104 -> 2048) -> 1 LLM embedding per 0.1s of audio
#include "models.h"

ggml_cgraph * mtmd_graph_wemm_audio::build() {
    const int n_mel_frames = img.nx();            // 3000 per 30s chunk
    const int n_whisper    = (n_mel_frames + 1) / 2;

    GGML_ASSERT(model.wemm_q_tokens     != nullptr);
    GGML_ASSERT(model.wemm_audio_proj_w != nullptr);

    // ---- whisper encoder ----
    ggml_tensor * inp = build_inp_raw(1);
    {
        ggml_tensor * cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_1_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_2_b);
        cur = ggml_gelu_erf(ctx0, cur);
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));   // [1280, n_whisper]
        cb(inp, "wemm_after_conv", -1);
    }

    GGML_ASSERT(model.position_embeddings->ne[0] == 1280);
    GGML_ASSERT(model.position_embeddings->ne[1] >= n_whisper);
    // official export stores position embeddings as F16; element-wise add needs F32
    ggml_tensor * pos_f32 = model.position_embeddings->type == GGML_TYPE_F32
        ? model.position_embeddings
        : ggml_cast(ctx0, model.position_embeddings, GGML_TYPE_F32);
    ggml_tensor * pos = ggml_view_2d(ctx0, pos_f32,
        pos_f32->ne[0], n_whisper,
        pos_f32->nb[1], 0);

    ggml_tensor * enc = build_vit(inp, n_whisper,
                            NORM_TYPE_NORMAL, FFN_GELU_ERF, pos, nullptr);
    cb(enc, "wemm_encoder_out", -1);

    // final encoder layer_norm (a.layer_norm)
    enc = build_norm(enc, model.norm_embd_w, model.norm_embd_b,
                     NORM_TYPE_NORMAL, 1e-5f, -1);
    cb(enc, "wemm_ln_post", -1);

    // ---- 5-frame windows: [1280, 1500] -> [1280, 5, n_win] ----
    const int win        = 5;
    const int n_win      = n_whisper / win;                   // 300 per 30s chunk
    const int C_enc      = 1280;
    const int C_q        = 4104;
    const int n_heads_q  = 12;
    const int d_head_q   = C_q / n_heads_q;
    const float kq_scale = 1.0f / sqrtf((float) d_head_q);
    const float ln_eps   = 1e-12f;                            // BERT layer_norm_eps

    ggml_tensor * enc_win = ggml_reshape_3d(ctx0, enc, C_enc, win, n_win);
    cb(enc_win, "wemm_windows", -1);

    // ---- qformer ----
    // learnable query [1, 4104] -> expand to [4104, 1, n_win]
    // official export stores q_tokens as F16; repeat needs matching F32.
    // reference BertEmbeddings applies LayerNorm to the query embeddings
    // (h5_modeling_bert.py: embeddings = query_embeds; embeddings = self.LayerNorm(...));
    // q_tokens are tiny-magnitude (std 0.02), so skipping this LN is a real numeric error.
    ggml_tensor * qf_query = model.wemm_q_tokens->type == GGML_TYPE_F32
        ? model.wemm_q_tokens
        : ggml_cast(ctx0, model.wemm_q_tokens, GGML_TYPE_F32);
    ggml_tensor * q = ggml_reshape_3d(ctx0, qf_query, C_q, 1, 1);
    if (model.wemm_qf_ln_w) {
        q = build_norm(q, model.wemm_qf_ln_w, model.wemm_qf_ln_b, NORM_TYPE_NORMAL, ln_eps, -1);
    }
    {
        ggml_tensor * q3 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, C_q, 1, n_win);
        q = ggml_repeat(ctx0, q, q3);
        cb(q, "wemm_q_tokens_expanded", -1);
    }

    for (int il = 0; il < (int) model.wemm_qf_layers.size(); ++il) {
        const auto & L = model.wemm_qf_layers[il];

        // self-attention (query attends to itself)
        {
            ggml_tensor * Q = ggml_add(ctx0, build_mm(L.q_w, q), L.q_b);
            ggml_tensor * K = ggml_add(ctx0, build_mm(L.k_w, q), L.k_b);
            ggml_tensor * V = ggml_add(ctx0, build_mm(L.v_w, q), L.v_b);
            Q = ggml_reshape_4d(ctx0, Q, d_head_q, n_heads_q, 1, n_win);
            K = ggml_reshape_4d(ctx0, K, d_head_q, n_heads_q, 1, n_win);
            V = ggml_reshape_4d(ctx0, V, d_head_q, n_heads_q, 1, n_win);
            ggml_tensor * attn_out = build_attn(L.o_w, L.o_b,
                                                Q, K, V, nullptr, kq_scale, il);
            attn_out = ggml_reshape_3d(ctx0, attn_out, C_q, 1, n_win);
            q = build_norm(ggml_add(ctx0, attn_out, q), L.ln_1_w, L.ln_1_b,
                           NORM_TYPE_NORMAL, ln_eps, il);
            cb(q, "wemm_qf_self", il);
        }

        // cross-attention (kv = the whisper frames of this window)
        {
            ggml_tensor * Q = ggml_add(ctx0, build_mm(L.cross_attn_q_w, q), L.cross_attn_q_b);
            ggml_tensor * K = ggml_add(ctx0, build_mm(L.cross_attn_k_w, enc_win), L.cross_attn_k_b);
            ggml_tensor * V = ggml_add(ctx0, build_mm(L.cross_attn_v_w, enc_win), L.cross_attn_v_b);
            Q = ggml_reshape_4d(ctx0, Q, d_head_q, n_heads_q, 1, n_win);
            K = ggml_reshape_4d(ctx0, K, d_head_q, n_heads_q, win, n_win);
            V = ggml_reshape_4d(ctx0, V, d_head_q, n_heads_q, win, n_win);
            ggml_tensor * attn_out = build_attn(L.cross_attn_o_w, L.cross_attn_o_b,
                                                Q, K, V, nullptr, kq_scale, il);
            attn_out = ggml_reshape_3d(ctx0, attn_out, C_q, 1, n_win);
            q = build_norm(ggml_add(ctx0, attn_out, q), L.cross_attn_norm_w, L.cross_attn_norm_b,
                           NORM_TYPE_NORMAL, ln_eps, il);
            cb(q, "wemm_qf_cross", il);
        }

        // ffn (intermediate 12288, gelu)
        {
            // reference ACT2FN['gelu'] = erf gelu (not the tanh approximation)
            ggml_tensor * ffn_out = build_ffn(q,
                L.ff_up_w, L.ff_up_b,
                nullptr, nullptr,
                L.ff_down_w, L.ff_down_b,
                FFN_GELU_ERF, il);
            q = build_norm(ggml_add(ctx0, ffn_out, q), L.ln_2_w, L.ln_2_b,
                           NORM_TYPE_NORMAL, ln_eps, il);
            cb(q, "wemm_qf_ffn", il);
        }
    }

    // ---- audio projection: [4104, 1, n_win] -> [2048, 1, n_win] ----
    ggml_tensor * cur = build_mm(model.wemm_audio_proj_w, q);
    if (model.wemm_audio_proj_b) {
        cur = ggml_add(ctx0, cur, model.wemm_audio_proj_b);
    }
    cb(cur, "wemm_projected", -1);

    // mtmd eval reads the LAST graph node and expects the token count in ne[1]
    // (see the "Invalid number of output tokens" check); our qformer keeps the
    // window count in ne[2], so move it: [2048, 1, n_win] -> [2048, n_win, 1]
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));

    ggml_build_forward_expand(gf, cur);
    return gf;
}
