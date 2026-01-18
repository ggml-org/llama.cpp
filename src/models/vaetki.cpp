#include "models.h"

llm_build_vaetki::llm_build_vaetki(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla;
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla;

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_qk_nope + n_embd_head_qk_rope));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(q, "q_a", il);

            q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(q, "q_a_norm", il);

            q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
            cb(q, "q", il);

            // q is now [rope | nope] after weight reordering in conversion
            // reshape to {n_embd_head_k_mla, n_head, n_tokens}
            q = ggml_reshape_3d(ctx0, q, n_embd_head_k_mla, n_head, n_tokens);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe,
                kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe,
                n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            // apply rope - rotates first n_rot dims, copies rest unchanged
            ggml_tensor * Qcur = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe_rope", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr_norm", il);

            ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
            cb(kv, "kv", il);

            // {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * k_nope = ggml_view_3d(ctx0, kv,
                n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla),
                ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla) * n_head, 0);
            cb(k_nope, "k_nope", il);

            // {n_embd_head_v_mla, n_head, n_tokens}
            ggml_tensor * Vcur = ggml_view_3d(ctx0, kv,
                n_embd_head_v_mla, n_head, n_tokens,
                ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla),
                ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla) * n_head,
                ggml_row_size(kv->type, n_embd_head_qk_nope));
            cb(Vcur, "Vcur", il);

            Vcur = ggml_cont(ctx0, Vcur);
            cb(Vcur, "Vcur_cont", il);

            ggml_tensor * q_pe_ref = ggml_view_3d(ctx0, Qcur,
                n_embd_head_qk_rope, n_head, n_tokens,
                Qcur->nb[1], Qcur->nb[2], 0);
            ggml_tensor * Kcur = ggml_concat(ctx0, ggml_repeat(ctx0, k_pe, q_pe_ref), k_nope, 0);
            cb(Kcur, "Kcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
