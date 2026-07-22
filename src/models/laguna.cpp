// Laguna (poolside): sigmoid-routed MoE with a score-correction bias, one shared
// expert, a softplus attention output gate, QK-norm, and per-layer-type RoPE
// (YaRN on full-attention layers, plain RoPE on sliding-window layers). XS.2 is
// hybrid full/SWA with a per-head gate; M.1 is full-attention with a per-element
// gate. Shares the MoE/gate structure with afmoe.

#include "models.h"

void llama_model_laguna::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // Laguna ships one shared expert and stores its size directly (routed and
    // shared experts may differ), so read the size from expert_shared_feed_forward_length.
    // The count is not in the config; default to 1 but read the key if present.
    hparams.n_expert_shared = 1;
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    if (hparams.n_ff_shexp == 0) {
        // Weightless fixtures (test-llama-archs) omit this key; derive a nonzero
        // size so the shared expert is still built. Real GGUFs always carry the
        // exact value (routed and shared FF lengths may differ).
        hparams.n_ff_shexp = hparams.n_ff_exp * hparams.n_expert_shared;
    }

    // Sliding-window attention is OPTIONAL. XS.2 is hybrid (full / SWA / SWA /
    // SWA repeating, period 4 starting with full); M.1 has no sliding window
    // (all layers full attention). When sliding_window is absent or zero we
    // leave swa_type = NONE and skip the SWA-specific per-layer-type RoPE.
    hparams.n_swa = 0;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

        uint32_t swa_period = 4;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
        hparams.set_swa_pattern(swa_period, /*dense_first=*/true);  // XS.2: FULL at il%4==0

        // Per-layer-type RoPE: full layers use YaRN θ=500000 over 64 dims;
        // SWA layers use default RoPE θ=10000 over 128 dims. Base load_hparams
        // already reads ROPE_FREQ_BASE and ROPE_DIMENSION_COUNT into the
        // non-SWA fields; we explicitly pull the SWA mirrors here.
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = 1.0f;  // SWA uses plain RoPE (no YaRN scaling); do NOT inherit full layers 1/factor
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // Default the expert gating function to SIGMOID when the key is absent
    // (matches the HF reference).
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_30B_A3B;   break;  // Laguna-XS.2
        case 70: type = LLM_TYPE_230B_A10B; break;  // Laguna-M.1
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_laguna::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        // tied embeddings fallback
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int64_t n_ff_exp   = hparams.n_ff_exp;
    const int64_t n_ff_shexp = hparams.n_ff_shexp;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // Per-layer head count — Laguna varies n_head between full and SWA
        // layers (48 vs 64 in XS.2). KV head count is uniform.
        const int64_t n_head_il    = hparams.n_head(i);
        const int64_t n_head_kv_il = hparams.n_head_kv(i);
        const int64_t n_embd_q_il  = n_embd_head_k * n_head_il;
        const int64_t n_embd_k_il  = n_embd_head_k * n_head_kv_il;
        const int64_t n_embd_v_il  = n_embd_head_v * n_head_kv_il;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_q_il, n_embd_k_il, n_embd_v_il, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q_il, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        // Attention output gate. XS.2 is per-head (g_proj -> n_head, one scalar
        // per head broadcast over head_dim at multiply time); M.1 is per-element
        // (g_proj -> n_head*head_dim, like afmoe). Detect from the stored tensor
        // shape so a single arch handles both; the graph mirrors this check.
        // Gate width selects per-head vs per-element. Real GGUFs always carry the
        // gate tensor, so read the width from it and require EXACTLY one of the two
        // valid widths -- never guess between them. Weightless fixtures
        // (test-llama-archs) have no gate tensor; fall back to the per-head layout so
        // the per-head reshape path is still exercised.
        const int64_t n_gate_per_head = n_head_il;
        const int64_t n_gate_per_elem = n_embd_head_k * n_head_il;
        const ggml_tensor * gate_meta = ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_GATE, "weight", i).str().c_str());
        int64_t n_gate_out;
        if (gate_meta != nullptr) {
            n_gate_out = gate_meta->ne[1];
            if (n_gate_out != n_gate_per_head && n_gate_out != n_gate_per_elem) {
                GGML_ABORT("Laguna: unexpected attention gate width %lld at layer %d "
                           "(expected %lld per-head or %lld per-element)",
                           (long long) n_gate_out, i, (long long) n_gate_per_head, (long long) n_gate_per_elem);
            }
        } else {
            n_gate_out = n_gate_per_head;
        }
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_gate_out}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if ((uint32_t)i >= hparams.n_layer_dense_lead) {
            // MoE layer
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);

            // Always-on shared expert.
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,    n_ff_shexp}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,    n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd},    0);
        } else {
            // Dense layer (the leading n_layer_dense_lead layers)
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff,   n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_laguna::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_laguna::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    // No MuP embedding scale (laguna omits this; afmoe scales by sqrt(hidden)).

    ggml_tensor * inp_pos = build_inp_pos();
    // XS.2 is hybrid SWA -> interleaved-SWA KV input; M.1 is all-full -> plain
    // KV input. Pick the matching input (and build_attn overload) per swa_type.
    const bool has_swa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    llm_graph_input_attn_kv      * inp_attn_kv   = has_swa ? nullptr : build_attn_inp_kv();
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = has_swa ? build_attn_inp_kv_iswa() : nullptr;
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const bool    is_swa_il   = hparams.is_swa(il);
        const int64_t n_head_il   = hparams.n_head(il);
        const int64_t n_head_kv_il = hparams.n_head_kv(il);

        // Per-layer-type RoPE config. SWA layers run plain rope (no YaRN),
        // achieved by zeroing the YaRN ext/beta params for those layers.
        const int   n_rot_l       = is_swa_il ? hparams.n_rot_swa : n_rot;
        const float freq_base_l   = is_swa_il ? hparams.rope_freq_base_train_swa : freq_base;
        const float freq_scale_l  = is_swa_il ? hparams.rope_freq_scale_train_swa : freq_scale;
        const float ext_factor_l  = is_swa_il ? 0.0f : ext_factor;
        // YaRN magnitude scaling (mscale) is already handled by the framework:
        // llama_context pre-divides cparams.yarn_attn_factor by (1 + 0.1*ln(factor))
        // to cancel ggml rope_yarn's internal mscale *= 1 + 0.1*ln(1/freq_scale).
        // Pass attn_factor straight through (like every other arch); SWA layers run
        // plain RoPE (ext_factor 0, no mscale) so force 1.0 there.
        const float attn_factor_l = is_swa_il ? 1.0f : attn_factor;
        const float beta_fast_l   = is_swa_il ? 0.0f : beta_fast;
        const float beta_slow_l   = is_swa_il ? 0.0f : beta_slow;
        const int   n_ctx_orig_l  = is_swa_il ? hparams.n_ctx_train : n_ctx_orig;

        ggml_tensor * inpSA = inpL;

        // Pre-norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Self-attention
        {
            ggml_tensor * attn_inp = cur;  // saved for the gate projection

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head_il, n_head_kv_il, il);

            // g_proj on the *pre-attention* hidden state (matches HF
            // reference: gate is computed from the same `hidden_states`
            // input as q/k/v, not from the attn output).
            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, attn_inp);
            cb(gate, "attn_gate_proj", il);

            // QK RMSNorm at head_dim level (Qwen3 style)
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot_l, rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l,
                    ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot_l, rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l,
                    ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = has_swa
                ? build_attn(inp_attn_iswa,
                        NULL, NULL, NULL,    // o_proj deferred until after gating
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
                : build_attn(inp_attn_kv,
                        NULL, NULL, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // Softplus output gate (the unary kernel computes softplus in fp32
            // and casts back). Two shapes, distinguished by the g_proj output
            // dim (matching the load-time detection):
            //   XS.2 per-head     : gate [n_head_il, n_tokens] -> reshape to
            //                       [1, n_head_il, n_tokens] and broadcast over
            //                       head_dim against cur [head_dim, n_head, T].
            //   M.1  per-element  : gate [n_head_il*head_dim, n_tokens] spans the
            //                       full attention output -> direct ggml_mul.
            gate = ggml_softplus(ctx0, gate);
            cb(gate, "attn_gate_softplus", il);

            const int64_t n_tokens = cur->ne[1];
            if (model.layers[il].wqkv_gate->ne[1] == n_head_il) {
                cur  = ggml_reshape_3d(ctx0, cur,  n_embd_head, n_head_il, n_tokens);
                gate = ggml_reshape_3d(ctx0, gate, 1,           n_head_il, n_tokens);
                cur  = ggml_mul(ctx0, cur, gate);
                cur  = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head_il, n_tokens);
            } else {
                cur = ggml_mul(ctx0, cur, gate);
            }
            cb(cur, "attn_gated", il);

            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            cb(cur, "attn_o_proj", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Pre-norm only (no post-attn norm)
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t)il >= hparams.n_layer_dense_lead) {
            // MoE: sigmoid routing + score-correction bias + sum-norm +
            // routed_scaling_factor (all handled by build_moe_ffn).
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // Always-on shared expert, summed in parallel.
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            // Dense FFN for the leading n_layer_dense_lead layers (XS.2: 1, M.1: 3)
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        // No post-ffn norm
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
