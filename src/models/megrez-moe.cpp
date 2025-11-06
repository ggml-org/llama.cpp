#include "models.h"



llm_build_megrez_moe::llm_build_megrez_moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params){
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * pre_gate_hidden;
    // Layer 0
    {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[0].attn_norm, NULL,
                LLM_NORM_RMS, 0);
        cb(cur, "attn_norm", 0);

        // compute Q and K and RoPE them
        ggml_tensor * Qcur = build_lora_mm(model.layers[0].wq, cur);
        cb(Qcur, "Qcur", 0);

        ggml_tensor * Kcur = build_lora_mm(model.layers[0].wk, cur);
        cb(Kcur, "Kcur", 0);

        ggml_tensor * Vcur = build_lora_mm(model.layers[0].wv, cur);
        cb(Vcur, "Vcur", 0);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur", 0);
        cb(Kcur, "Kcur", 0);
        cb(Vcur, "Vcur", 0);

        cur = build_attn(inp_attn,
                model.layers[0].wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, 0);

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", 0);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[0].ffn_norm, NULL,
                LLM_NORM_RMS, 0);
        cb(cur, "ffn_norm", 0);

        pre_gate_hidden = cur;

        cur = build_ffn(cur,
                model.layers[0].ffn_up,   NULL, NULL,
                model.layers[0].ffn_gate, NULL, NULL,
                model.layers[0].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, 0);

        cb(cur, "ffn_out", 0);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out_add", 0);

    }
    inpL = cur;
    for (int il = 1; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(cur,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            pre_gate_hidden = ggml_get_rows(ctx0, pre_gate_hidden, inp_out_ids);
        }


        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            // Note: Megrez-MoE uses pre_gate_hidden (from previous layer's FFN norm) for gating
            // This is different from standard MoE which uses current layer's input
            // Compute gate logits from pre_gate_hidden instead of cur
            ggml_tensor * gate_logits = build_lora_mm(model.layers[il].ffn_gate_inp, pre_gate_hidden);
            cb(gate_logits, "ffn_moe_logits", il);

            // Use standard build_moe_ffn but with pre-computed gate logits
            ggml_tensor * moe_out = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[((il - 1) / (3) * (3)) + 1].ffn_up_exps,
                        model.layers[((il - 1) / (3) * (3)) + 1].ffn_gate_exps,
                        model.layers[((il - 1) / (3) * (3)) + 1].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU,
                        true,  // norm_w
                        false, // scale_w
                        1.0f,  // w_scale
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                        il,
                        gate_logits); // Use pre-computed logits from pre_gate_hidden
            cb(moe_out, "ffn_moe_out", il);

            pre_gate_hidden = cur;

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
