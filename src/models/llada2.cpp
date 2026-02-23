#include "models.h"

llm_build_llada2::llm_build_llada2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    // LLaDA2 uses partial rotary embeddings (n_rot=64, n_embd_head=128)
    // So we don't assert n_embd_head == n_rot
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // LLaDA2 uses combined QKV tensor
            // wqkv has shape {n_embd, n_embd + 2*n_embd_k_gqa}
            // Q: {n_embd, n_embd}, K: {n_embd, n_embd_k_gqa}, V: {n_embd, n_embd_k_gqa}
            ggml_tensor * QKVcur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(QKVcur, "QKVcur", il);

            // Split QKV into Q, K, V using 3D views with byte offsets (same as phi3)
            // Q: offset 0, K: offset n_embd, V: offset n_embd + n_embd_gqa
            ggml_tensor * Qcur = ggml_view_3d(ctx0, QKVcur, n_embd_head, n_head,    n_tokens, n_embd_head * sizeof(float), QKVcur->nb[1], 0 * sizeof(float) * n_embd);
            ggml_tensor * Kcur = ggml_view_3d(ctx0, QKVcur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float), QKVcur->nb[1], 1 * sizeof(float) * n_embd);
            ggml_tensor * Vcur = ggml_view_3d(ctx0, QKVcur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float), QKVcur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_k_gqa));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // Apply RoPE with n_rot (partial rotary embeddings)
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
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FFN branch
        ggml_tensor * ffn_norm_out = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(ffn_norm_out, "ffn_norm", il);

        // Layer 0 is dense, layers 1+ are MoE with shared experts
        if (il == 0) {
            // Dense layer
            // build_ffn(cur, up, up_b, up_s, gate, gate_b, gate_s, down, down_b, down_s, act_scales, type_op, type_gate, il)
            cur = build_ffn(ffn_norm_out,
                    model.layers[il].ffn_up,   NULL, NULL,  // up, up_b, up_s
                    model.layers[il].ffn_gate, NULL, NULL,  // gate, gate_b, gate_s
                    model.layers[il].ffn_down, NULL, NULL,  // down, down_b, down_s
                    NULL,                                     // act_scales
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_dense_out", il);
        } else {
            // MoE layer with shared experts
            // Routed experts use normalized input
            ggml_tensor * moe_out = build_moe_ffn(ffn_norm_out,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,  // Expert probability bias
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // Shared experts also use the same normalized input (in parallel with routed experts)
            // build_ffn(cur, up, up_b, up_s, gate, gate_b, gate_s, down, down_b, down_s, act_scales, type_op, type_gate, il)
            ggml_tensor * shexp_out = build_ffn(ffn_norm_out,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,  // up, up_b, up_s
                    model.layers[il].ffn_gate_shexp, NULL, NULL,  // gate, gate_b, gate_s
                    model.layers[il].ffn_down_shexp, NULL, NULL,  // down, down_b, down_s
                    NULL,                                         // act_scales
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(shexp_out, "ffn_shexp_out", il);

            // Add routed experts + shared experts
            cur = ggml_add(ctx0, moe_out, shexp_out);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
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
