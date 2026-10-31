#include "models.h"

void llama_model_kolibri1::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,          hparams.n_swa);

    if (hparams.n_swa == 0 || hparams.n_expert_shared != 1 || hparams.n_ff_shexp == 0) {
        throw std::runtime_error("Kolibri-1: invalid sliding window or shared expert configuration");
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    switch (hparams.n_layer()) {
        case 50: type = LLM_TYPE_78B_A3_5B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_kolibri1::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp   = hparams.n_ff_exp();
    const int64_t n_ff_shexp = hparams.n_ff_shexp;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // Routing is sigmoid, the selection bias is added to the logits before top-k.
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
    }
}

llama_model_kolibri1::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // QK-norm, then RoPE on the sliding layers only: the global layers are NoPE.
        auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        cb(Kcur, "Kcur_normed", il);

        if (hparams.is_swa(il)) {
            const float freq_base_l  = model.get_rope_freq_base (cparams, il);
            const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

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
                layer.wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, residual);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // The selection adds the bias to the logits, the weights are the sigmoid of the unbiased logits, so the same logits feed both.
        ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * selection = ggml_add(ctx0, logits, layer.ffn_exp_probs_b);
        cb(selection, "ffn_moe_logits_biased", il);

        ggml_tensor * selected = ggml_argsort_top_k(ctx0, selection, n_expert_used);
        cb(selected, "ffn_moe_topk", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm, 1.0f,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                logits,
                nullptr, selected);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * shared = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr,
                layer.ffn_gate_shexp, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(shared, "ffn_shexp_out", il);

        cur = ggml_add(ctx0, moe_out, shared);
        cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cur = ggml_add(ctx0, cur, ffn_inp);

        inpL = build_cvec(cur, il);
        cb(inpL, "l_out", il);
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::unique_ptr<llm_graph_context> llama_model_kolibri1::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
