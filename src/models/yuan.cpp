#include "models.h"

llm_build_yuan::llm_build_yuan(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // V is projected from pre-LF hidden states (before localized filtering)
        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);

        // localized filtering (causal conv2d with kernel_size=2)
        // output[t] = W0 @ input[t] + W1 @ input[t-1] + bias  (input[-1] = 0)
        //
        // Weight stored as [in_ch, 2*out_ch]: W0||W1 concatenated in output dim.
        // One matmul produces both parts, then we split and shift W1's part.
        {
            const int64_t lf_mid = n_embd / 2;
            const int64_t n_seq = cur->ne[1];  // actual sequence length

            // conv1: cur [n_embd, n_seq] @ weight [n_embd, 2*lf_mid] -> [2*lf_mid, n_seq]
            ggml_tensor * conv1_full = ggml_mul_mat(ctx0, model.layers[il].attn_lf_conv1, cur);
            cb(conv1_full, "lf_conv1_full", il);

            // split: part0 = W0@cur [lf_mid, n_seq], part1 = W1@cur [lf_mid, n_seq]
            ggml_tensor * c1_part0 = ggml_cont(ctx0, ggml_view_2d(ctx0, conv1_full,
                    lf_mid, n_seq, conv1_full->nb[1], 0));
            ggml_tensor * c1_part1 = ggml_cont(ctx0, ggml_view_2d(ctx0, conv1_full,
                    lf_mid, n_seq, conv1_full->nb[1],
                    lf_mid * ggml_element_size(conv1_full)));

            ggml_tensor * conv1_out;
            if (n_seq > 1) {
                // causal shift: shifted[0] = 0, shifted[t] = part1[t-1]
                ggml_tensor * c1_zero = ggml_scale(ctx0, ggml_view_2d(ctx0, c1_part0,
                        lf_mid, 1, c1_part0->nb[1], 0), 0.0f);
                ggml_tensor * c1_prev = ggml_view_2d(ctx0, c1_part1,
                        lf_mid, n_seq - 1, c1_part1->nb[1], 0);
                ggml_tensor * c1_shifted = ggml_concat(ctx0, c1_zero, c1_prev, 1);
                conv1_out = ggml_add(ctx0, c1_part0, c1_shifted);
            } else {
                // single token: W1 contribution is zero (no previous token)
                conv1_out = c1_part0;
            }
            conv1_out = ggml_add(ctx0, conv1_out, model.layers[il].attn_lf_conv1_b);
            cb(conv1_out, "lf_conv1_out", il);

            // conv2: conv1_out [lf_mid, n_seq] @ weight [lf_mid, 2*n_embd] -> [2*n_embd, n_seq]
            ggml_tensor * conv2_full = ggml_mul_mat(ctx0, model.layers[il].attn_lf_conv2, conv1_out);
            cb(conv2_full, "lf_conv2_full", il);

            ggml_tensor * c2_part0 = ggml_cont(ctx0, ggml_view_2d(ctx0, conv2_full,
                    n_embd, n_seq, conv2_full->nb[1], 0));
            ggml_tensor * c2_part1 = ggml_cont(ctx0, ggml_view_2d(ctx0, conv2_full,
                    n_embd, n_seq, conv2_full->nb[1],
                    n_embd * ggml_element_size(conv2_full)));

            ggml_tensor * conv2_out;
            if (n_seq > 1) {
                ggml_tensor * c2_zero = ggml_scale(ctx0, ggml_view_2d(ctx0, c2_part0,
                        n_embd, 1, c2_part0->nb[1], 0), 0.0f);
                ggml_tensor * c2_prev = ggml_view_2d(ctx0, c2_part1,
                        n_embd, n_seq - 1, c2_part1->nb[1], 0);
                ggml_tensor * c2_shifted = ggml_concat(ctx0, c2_zero, c2_prev, 1);
                conv2_out = ggml_add(ctx0, c2_part0, c2_shifted);
            } else {
                conv2_out = c2_part0;
            }
            conv2_out = ggml_add(ctx0, conv2_out, model.layers[il].attn_lf_conv2_b);
            cb(conv2_out, "lf_conv2_out", il);

            // residual + RMSNorm
            cur = ggml_add(ctx0, conv2_out, cur);
            cur = build_norm(cur,
                    model.layers[il].attn_lf_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "lf_output", il);
        }

        // self-attention (Q, K from post-LF cur; V from pre-LF, already computed above)
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

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
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE with attention-based routing
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // attention-based router
        ggml_tensor * router_qkv = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_inp_qkv, cur);
        cb(router_qkv, "router_qkv", il);
        router_qkv = ggml_cast(ctx0, router_qkv, GGML_TYPE_F32);

        const int64_t n_cur = router_qkv->ne[1];

        ggml_tensor * router_q = ggml_cont(ctx0, ggml_view_2d(ctx0, router_qkv,
                n_expert, n_cur, router_qkv->nb[1], 0));
        ggml_tensor * router_k = ggml_cont(ctx0, ggml_view_2d(ctx0, router_qkv,
                n_expert, n_cur, router_qkv->nb[1],
                n_expert * ggml_element_size(router_qkv)));
        ggml_tensor * router_v = ggml_cont(ctx0, ggml_view_2d(ctx0, router_qkv,
                n_expert, n_cur, router_qkv->nb[1],
                2 * n_expert * ggml_element_size(router_qkv)));

        // per-token outer product attention
        router_q = ggml_reshape_3d(ctx0, router_q, 1, n_expert, n_cur);
        router_k = ggml_reshape_3d(ctx0, router_k, 1, n_expert, n_cur);
        router_v = ggml_reshape_3d(ctx0, router_v, n_expert, 1, n_cur);

        // outer product: mul_mat(K, Q) -> [z, z, n_cur], softmax along ne[0]
        ggml_tensor * router_attn = ggml_mul_mat(ctx0, router_k, router_q);
        router_attn = ggml_soft_max(ctx0, router_attn);
        cb(router_attn, "router_attn", il);

        // attn @ V -> [z, 1, n_cur] -> [z, n_cur]
        ggml_tensor * router_out = ggml_mul_mat(ctx0, router_attn, router_v);
        router_out = ggml_reshape_2d(ctx0, router_out, n_expert, n_cur);
        cb(router_out, "router_logits", il);

        cur = build_moe_ffn(cur,
                /*gate_inp*/ nullptr,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                il,
                router_out);
        cb(cur, "ffn_moe_out", il);

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
