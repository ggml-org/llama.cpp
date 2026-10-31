#include "models.h"

void llama_model_muse_spark::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_POST_NORM_EPS,               hparams.f_post_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,             hparams.moe_latent_size);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    switch (hparams.n_layer()) {
        case 62: type = LLM_TYPE_816B_A42B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_muse_spark::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_moe = hparams.moe_latent_size;
    const int64_t n_ff_exp   = hparams.n_ff_exp();

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // Pre-attention norm. All block norms hold a delta folded at conversion time.
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // Q/K/V/O projections.
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // QK-norm. Weights are synthesized at conversion to absorb the metap scale.
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        // Attention output gate: sigmoid(gate) * attn_out before o_proj.
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);

        // Norm preserving residual: alpha scales the stream, beta the branch.
        layer.attn_resid_alpha = create_tensor(tn(LLM_TENSOR_ATTN_RESID_ALPHA, "weight", i), {n_embd}, 0);
        layer.attn_resid_beta  = create_tensor(tn(LLM_TENSOR_ATTN_RESID_BETA,  "weight", i), {n_embd}, 0);

        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, 0);
        layer.ffn_resid_alpha = create_tensor(tn(LLM_TENSOR_FFN_RESID_ALPHA, "weight", i), {n_embd}, 0);
        layer.ffn_resid_beta  = create_tensor(tn(LLM_TENSOR_FFN_RESID_BETA,  "weight", i), {n_embd}, 0);

        // Every layer routes. The router reads the full stream, the experts run
        // in a latent space entered by ffn_latent_down and left by ffn_latent_up.
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,   "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);

        layer.ffn_latent_down = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i), {n_embd, n_embd_moe}, 0);
        layer.ffn_latent_up   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP,   "weight", i), {n_embd_moe, n_embd}, 0);

        layer.ffn_pre_exp_norm  = create_tensor(tn(LLM_TENSOR_FFN_PRE_EXP_NORM,  "weight", i), {n_embd_moe}, 0);
        layer.ffn_post_exp_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_EXP_NORM, "weight", i), {n_embd_moe}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd_moe, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd_moe, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd_moe, n_expert}, 0);
    }
}

llama_model_muse_spark::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    inpL = build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1);
    cb(inpL, "embd_norm", -1);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * inpSA = inpL;

        // RoPE runs on the SWA layers, NoPE on the global ones.
        const bool use_rope = hparams.is_swa(il);

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur);
            cb(gate, "attn_gate_proj", il);

            // attn_q_norm carries the metap scale, attn_k_norm is identity.
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            if (use_rope) {
                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur_rope", il);
            }

            cur = build_attn(inp_attn,
                    NULL, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // Weightless norm on the attention output, applied per head.
            cur = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens);
            cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
            cur = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head, n_tokens);
            cb(cur, "attn_o_norm", il);

            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate_sig", il);
            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);

            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_o_proj", il);
        }

        // Post-attention norm holds no weight.
        cur = ggml_rms_norm(ctx0, cur, hparams.f_post_norm_eps);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0,
                ggml_mul(ctx0, inpSA, model.layers[il].attn_resid_alpha),
                ggml_mul(ctx0, cur,   model.layers[il].attn_resid_beta));
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // Enter the latent space, run the experts, leave it.
        ggml_tensor * moe_inp = build_lora_mm(model.layers[il].ffn_latent_down, cur);
        cb(moe_inp, "ffn_latent_down", il);

        moe_inp = build_norm(moe_inp, model.layers[il].ffn_pre_exp_norm, NULL, LLM_NORM_RMS, il);
        cb(moe_inp, "ffn_pre_exp_norm", il);

        // The router reads the full stream, not the latent one.
        ggml_tensor * router_logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur);
        cb(router_logits, "ffn_moe_logits", il);

        ggml_tensor * moe_out = build_moe_ffn(moe_inp,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                router_logits,
                nullptr, nullptr,
                model.layers[il].ffn_post_exp_norm, hparams.f_post_norm_eps);
        cb(moe_out, "ffn_moe_out", il);

        cur = build_lora_mm(model.layers[il].ffn_latent_up, moe_out);
        cb(cur, "ffn_latent_up", il);

        // Post-FFN norm scales the input, then normalizes.
        cur = ggml_mul(ctx0, cur, model.layers[il].ffn_post_norm);
        cur = ggml_rms_norm(ctx0, cur, hparams.f_post_norm_eps);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0,
                ggml_mul(ctx0, ffn_inp, model.layers[il].ffn_resid_alpha),
                ggml_mul(ctx0, cur,     model.layers[il].ffn_resid_beta));
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::unique_ptr<llm_graph_context> llama_model_muse_spark::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
