#include "models.h"

#include "llama-moe-cache.h"

static bool has_qat_experts(const llama_model & model) {
    const auto it = model.gguf_kv.find(LLM_KV(model.arch)(LLM_KV_TENSOR_DATA_LAYOUT));
    return it != model.gguf_kv.end() && it->second == "qat_output_grouped_g32";
}

void llama_model_mobilemoe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    type = LLM_TYPE_UNKNOWN;

    std::string layout;
    ml.get_key(LLM_KV_TENSOR_DATA_LAYOUT, layout, false);
    gguf_kv[ml.llm_kv(LLM_KV_TENSOR_DATA_LAYOUT)] = layout;
    if (layout == "qat_output_grouped_g32") {
        // Packed expert decoding requires plain CPU buffers.
        params.use_extra_bufts = false;
    }
}

void llama_model_mobilemoe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const bool qat_experts = has_qat_experts(*this);
    const int64_t n_ff_exp   = hparams.n_ff_exp();
    const int64_t n_ff_shexp = hparams.n_ff_shexp > 0 ? hparams.n_ff_shexp : n_ff_exp;
    const int64_t n_qo_dim   = n_head * hparams.n_embd_head_k();
    const int64_t n_kv_dim   = n_head_kv * hparams.n_embd_head_k();

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        // Tied embeddings use token weights for output; the loader reuses them when buffer placement allows.
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        create_tensor_qkv(layer, i, n_embd, n_qo_dim, n_kv_dim, n_kv_dim, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_qo_dim, n_embd}, 0);
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm    = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        const int64_t n_in  = qat_experts ? n_ff_exp : n_embd;
        const int64_t n_out = qat_experts ? n_embd : n_ff_exp;
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_in, n_out, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_out, n_in, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_in, n_out, n_expert}, 0);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mobilemoe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Based on EXAONE MoE, with full attention and QK normalization after RoPE.
llama_model_mobilemoe::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const bool qat_experts = has_qat_experts(model);
    const int64_t n_embd_head = hparams.n_embd_head_k();
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base,
                            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base,
                            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(inp_attn, layer.wo, nullptr, layer.wo_s,
                         Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        cb(cur, "attn_out", il);
        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        ggml_tensor * moe = qat_experts ? build_qat_moe(cur, layer, il) :
            build_moe_ffn(cur, layer.ffn_gate_inp,
                layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_exp_probs_b,
                n_expert, n_expert_used, LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale, LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il);
        ggml_tensor * shared = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr, layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, moe, shared);
        cb(cur, "ffn_out", il);
        inpL = build_cvec(ggml_add(ctx0, cur, ffn_inp), il);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

ggml_tensor * llama_model_mobilemoe::graph::build_qat_mm(
        ggml_tensor * w, ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * slots) const {
    const int64_t n_experts = w->ne[2];
    if (slots != nullptr) {
        w = moe_cache->get_experts(w);
        ids = slots;
    }
    const int64_t n_out = w->ne[0];
    const int64_t n_in = w->ne[1];
    const int64_t n_used = ids->ne[0];
    const int64_t n_tok = ids->ne[1];
    GGML_ASSERT(cur->ne[0] == n_in);
    if (n_tok > 0 && n_used * n_tok < n_experts) {
        // Decode only selected experts for small batches; weights stay packed.
        auto * flat = ggml_reshape_2d(ctx0, w, n_out * n_in, w->ne[2]);
        auto * chosen = ggml_get_rows(ctx0, flat, ggml_reshape_1d(ctx0, ggml_cont(ctx0, ids), n_used * n_tok));
        // Match BF16 rounding of dequantized expert weights in Meta's MobileMoE implementation.
        chosen = ggml_cast(ctx0, ggml_cast(ctx0, chosen, GGML_TYPE_BF16), GGML_TYPE_F32);
        chosen = ggml_reshape_4d(ctx0, chosen, n_out, n_in, n_used, n_tok);
        chosen = ggml_cont(ctx0, ggml_permute(ctx0, chosen, 1, 0, 2, 3));
        auto * input = ggml_reshape_4d(ctx0, cur, n_in, 1, cur->ne[1], n_tok);
        input = ggml_repeat_4d(ctx0, input, n_in, 1, n_used, n_tok);
        auto * result = ggml_mul_mat(ctx0, chosen, input);
        if (prec_policy) {
            prec_policy->apply(result);
        }
        return ggml_reshape_3d(ctx0, result, n_out, n_used, n_tok);
    }
    // Large batches reuse each decoded expert across its selected tokens.
    auto * dense = ggml_cast(ctx0, w, GGML_TYPE_F32);
    dense = ggml_cast(ctx0, ggml_cast(ctx0, dense, GGML_TYPE_BF16), GGML_TYPE_F32);
    dense = ggml_cont(ctx0, ggml_permute(ctx0, dense, 1, 0, 2, 3));
    auto * result = ggml_mul_mat_id(ctx0, dense, cur, ids);
    if (prec_policy) {
        prec_policy->apply(result);
    }
    return result;
}

// Routing follows build_moe_ffn; packed QAT experts need output-grouped weight decoding.
ggml_tensor * llama_model_mobilemoe::graph::build_qat_moe(ggml_tensor * cur, const llama_layer & layer, int il) const {
    const int64_t n_tokens = cur->ne[1];
    auto * logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);
    auto * probs = ggml_sigmoid(ctx0, logits);
    cb(probs, "ffn_moe_probs", il);
    auto * selection_probs = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
    cb(selection_probs, "ffn_moe_probs_biased", il);
    auto * ids = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
    cb(ids->src[0], "ffn_moe_argsort", il);
    cb(ids, "ffn_moe_topk", il);
    auto * weights = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), ids);
    cb(weights, "ffn_moe_weights", il);
    if (hparams.expert_weights_norm) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        auto * weights_sum = ggml_sum_rows(ctx0, weights);
        cb(weights_sum, "ffn_moe_weights_sum", il);
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);
        weights = ggml_div(ctx0, weights, weights_sum);
        cb(weights, "ffn_moe_weights_norm", il);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }
    ggml_build_forward_expand(gf, weights);
    auto * slots = build_moe_cache_slots(ids, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, nullptr, il);
    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    auto * up = build_qat_mm(layer.ffn_up_exps, cur, ids, slots);
    cb(up, "ffn_moe_up", il);
    cur = build_qat_mm(layer.ffn_gate_exps, cur, ids, slots);
    cb(cur, "ffn_moe_gate", il);
    cur = ggml_swiglu_split(ctx0, cur, up);
    cb(cur, "ffn_moe_swiglu", il);
    auto * experts = build_qat_mm(layer.ffn_down_exps, cur, ids, slots);
    cb(experts, "ffn_moe_down", il);
    experts = ggml_mul(ctx0, experts, weights);
    cb(experts, "ffn_moe_weighted", il);
    ggml_build_forward_expand(gf, experts);

    // Order the views before the adds, as in build_moe_ffn.
    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
    const uint32_t n_used = hparams.n_expert_used(il);
    for (uint32_t i = 0; i < n_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(gf, cur_experts[i]);
    }
    auto * result = cur_experts[0];
    for (uint32_t i = 1; i < n_used; ++i) {
        result = ggml_add(ctx0, result, cur_experts[i]);
        ggml_build_forward_expand(gf, result);
    }
    if (n_used == 1) {
        result = ggml_cont(ctx0, result);
    }
    cb(result, "ffn_moe_out", il);
    return result;
}
