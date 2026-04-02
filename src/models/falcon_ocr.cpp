#include "models.h"

#include <vector>

// Graph input for 2D spatial positions used by the golden RoPE.
// With rope_3d, ubatch carries 3 position dimensions per token:
//   pos[i]              = temporal (used by standard 1D RoPE)
//   pos[i + n_tokens]   = h spatial pos (fixed-point, scale 1e6)
//   pos[i + 2*n_tokens] = w spatial pos (fixed-point, scale 1e6)
// For text tokens, hw_data stays zero -> identity rotation.
class llm_graph_input_pos_hw : public llm_graph_input_i {
public:
    virtual ~llm_graph_input_pos_hw() = default;

    static constexpr float POS_SCALE_INV = 1.0f / 1000000.0f;

    void set_input(const llama_ubatch * ubatch) override {
        if (!pos_hw || !ubatch->pos) {
            return;
        }

        const int n_tokens = ubatch->n_tokens;
        std::vector<float> hw_data(2 * n_tokens, 0.0f);

        // Only decode spatial positions for embedding batches (image patches).
        // For text batches, M-RoPE broadcasts temporal to all dims, so spatial
        // positions must stay zero for golden RoPE identity.
        if (!ubatch->token && ubatch->n_pos >= 3) {
            for (int i = 0; i < n_tokens; i++) {
                hw_data[2*i]     = (float)ubatch->pos[1 * n_tokens + i] * POS_SCALE_INV;
                hw_data[2*i + 1] = (float)ubatch->pos[2 * n_tokens + i] * POS_SCALE_INV;
            }
        }

        ggml_backend_tensor_set(pos_hw, hw_data.data(), 0, 2 * n_tokens * sizeof(float));
    }

    bool can_reuse(const llm_graph_params & /*params*/) override {
        return true;
    }

    ggml_tensor * pos_hw = nullptr; // F32 [P=2, n_tokens]
};

// Apply learned golden-ratio 2D RoPE to the spatial half of Q or K.
// theta = pos_h * freqs[h,f,0] + pos_w * freqs[h,f,1]
// For text tokens (pos_hw = 0): theta=0 -> identity.
static ggml_tensor * apply_golden_rope_2d(
        ggml_context * ctx0,
        ggml_tensor  * x,          // [head_dim, n_head, n_tokens]
        ggml_tensor  * freqs,      // [P=2, F, H]
        ggml_tensor  * pos_hw,     // [P=2, n_tokens]
        int64_t n_head,
        int64_t n_tokens,
        int64_t n_embd_head) {

    const int64_t half_dim = n_embd_head / 2;
    const int64_t F = half_dim / 2;

    // Split into temporal half (already 1D-rotated) and spatial half
    ggml_tensor * x_temporal = ggml_view_3d(ctx0, x, half_dim, n_head, n_tokens,
                                             x->nb[1], x->nb[2], 0);
    ggml_tensor * x_spatial  = ggml_view_3d(ctx0, x, half_dim, n_head, n_tokens,
                                             x->nb[1], x->nb[2], half_dim * ggml_type_size(x->type));
    x_temporal = ggml_cont(ctx0, x_temporal);
    x_spatial  = ggml_cont(ctx0, x_spatial);

    // theta[f, h, s] = sum_p(freqs[p, f, h] * pos[p, s])
    ggml_tensor * freqs_flat = ggml_reshape_2d(ctx0, freqs, 2, F * n_head);
    ggml_tensor * theta = ggml_mul_mat(ctx0, freqs_flat, pos_hw);
    theta = ggml_reshape_3d(ctx0, theta, F, n_head, n_tokens);

    ggml_tensor * cos_t = ggml_cos(ctx0, theta);
    ggml_tensor * sin_t = ggml_sin(ctx0, theta);

    // Reshape spatial [2F, H, S] -> [2, F, H, S] to separate adjacent pairs
    ggml_tensor * x_pairs = ggml_reshape_4d(ctx0, x_spatial, 2, F, n_head, n_tokens);

    ggml_tensor * x_real = ggml_view_4d(ctx0, x_pairs, 1, F, n_head, n_tokens,
                                         x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], 0);
    ggml_tensor * x_imag = ggml_view_4d(ctx0, x_pairs, 1, F, n_head, n_tokens,
                                         x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3],
                                         ggml_type_size(x->type));
    x_real = ggml_cont(ctx0, x_real);
    x_imag = ggml_cont(ctx0, x_imag);

    cos_t = ggml_reshape_4d(ctx0, cos_t, 1, F, n_head, n_tokens);
    sin_t = ggml_reshape_4d(ctx0, sin_t, 1, F, n_head, n_tokens);

    // out_r = x_r*cos - x_i*sin, out_i = x_r*sin + x_i*cos
    ggml_tensor * out_r = ggml_sub(ctx0,
            ggml_mul(ctx0, x_real, cos_t),
            ggml_mul(ctx0, x_imag, sin_t));
    ggml_tensor * out_i = ggml_add(ctx0,
            ggml_mul(ctx0, x_real, sin_t),
            ggml_mul(ctx0, x_imag, cos_t));

    // Interleave and concatenate temporal + rotated spatial
    ggml_tensor * rotated = ggml_concat(ctx0, out_r, out_i, 0);
    rotated = ggml_reshape_3d(ctx0, rotated, half_dim, n_head, n_tokens);

    return ggml_concat(ctx0, x_temporal, rotated, 0);
}

llm_build_falcon_ocr::llm_build_falcon_ocr(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // With rope_3d, pos tensor is [n_tokens * 3] — extract temporal slice
    ggml_tensor * inp_pos_full = build_inp_pos();
    ggml_tensor * inp_pos = (hparams.n_pos_per_embd() > 1)
        ? ggml_view_1d(ctx0, inp_pos_full, n_tokens, 0)
        : inp_pos_full;

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // 2D spatial position input for golden RoPE [P=2, n_tokens]
    ggml_tensor * inp_pos_hw = nullptr;
    if (model.rope_freqs_golden) {
        auto inp = std::make_unique<llm_graph_input_pos_hw>();
        auto * pos_hw = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, n_tokens);
        ggml_set_input(pos_hw);
        inp->pos_hw = pos_hw;
        res->add_input(std::move(inp));
        inp_pos_hw = pos_hw;
    }

    for (int il = 0; il < n_layer; ++il) {
        // Parameterless RMSNorm (no learned weight)
        cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
        cb(cur, "attn_norm", il);

        {
            // K/V weights use original n_kv_heads (before GQA expansion)
            const int64_t n_kv_orig = hparams.n_head_kv_orig;
            const int64_t n_rep     = n_head / n_kv_orig;

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_kv_orig, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_kv_orig, n_tokens);

            // Parameterless QK-norm (before RoPE)
            Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
            cb(Qcur, "Qcur_normed", il);

            Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
            cb(Kcur, "Kcur_normed", il);

            // 1D temporal RoPE (first half of head_dim; n_rot = head_dim/2)
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            // GQA expansion: K/V from n_kv_orig to n_head (before golden RoPE,
            // because golden frequencies are per-Q-head)
            if (n_rep > 1) {
                Kcur = ggml_reshape_4d(ctx0, Kcur, n_embd_head, 1, n_kv_orig, n_tokens);
                Vcur = ggml_reshape_4d(ctx0, Vcur, n_embd_head, 1, n_kv_orig, n_tokens);

                ggml_tensor * K_tgt = ggml_new_tensor_4d(ctx0, Kcur->type,
                        n_embd_head, n_rep, n_kv_orig, n_tokens);
                ggml_tensor * V_tgt = ggml_new_tensor_4d(ctx0, Vcur->type,
                        n_embd_head, n_rep, n_kv_orig, n_tokens);

                Kcur = ggml_repeat(ctx0, Kcur, K_tgt);
                Vcur = ggml_repeat(ctx0, Vcur, V_tgt);

                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head, n_tokens);
            }

            // 2D golden spatial RoPE (identity for text tokens)
            if (inp_pos_hw) {
                Qcur = apply_golden_rope_2d(ctx0, Qcur, model.rope_freqs_golden,
                        inp_pos_hw, n_head, n_tokens, n_embd_head);
                Kcur = apply_golden_rope_2d(ctx0, Kcur, model.rope_freqs_golden,
                        inp_pos_hw, n_head, n_tokens, n_embd_head);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        // Parameterless pre-FFN RMSNorm
        cur = ggml_rms_norm(ctx0, sa_out, hparams.f_norm_rms_eps);
        cb(cur, "ffn_norm", il);

        // Squared ReLU gating FFN
        {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    // Final RMSNorm (with learned weight)
    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
