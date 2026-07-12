#include "models.h"

// poolside/Laguna-XS-2.1 (33B MoE, agentic coding).
//
// Architecturally a near-clone of Step 3.5 (see models/step35.cpp): head-wise
// attention output gate, QK-norm, sigmoid MoE + correction bias + shared expert,
// mixed sliding/full attention with a per-layer-type RoPE. The deltas vs Step35:
//   - per-layer variable query heads (48 full / 64 sliding, KV=8 constant) — read
//     straight from the per-layer n_head array (handled by hparams.n_head(il)),
//   - the GGUF already carries the *partial* rope dim for full-attention layers
//     (rope.dimension_count = 64) and the full rope dim for sliding layers
//     (..._swa = 128), so — unlike Step35 — n_rot_full is NOT halved here,
//   - no NextN/MTP block (Laguna has none), so this arch has a single graph.
// See LAGUNA-PORT.md for the full spec.

void llama_model_laguna::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // NOTE: unlike Step35, do NOT halve n_rot_full — Laguna's converter writes the
    // already-partial rope dim (head_dim * partial_rotary_factor = 64) directly.

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);

    // Laguna uses sigmoid (aux-loss-free) routing.
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // Mixed sliding/full attention + per-layer type mask + local (sliding) rope base.
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,  hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,        hparams.rope_freq_base_train_swa, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_33B_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_laguna::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // Per-layer partial RoPE dims; the (optional) rope factors are stored once and
    // ggml uses the first n_rot_l/2 entries per layer.
    uint32_t n_rot_max = 0;
    for (int i = 0; i < n_layer; ++i) {
        n_rot_max = std::max(n_rot_max, hparams.n_rot(i));
    }
    if (n_rot_max == 0) {
        n_rot_max = n_rot;
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const uint32_t n_head_l     = hparams.n_head(i);
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i);
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i);

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd},        0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);

        // optional rope factors (llama3) / longrope tensors
        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        } else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head_l, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v * n_head_l, n_embd}, 0);

        // head-wise attention output gate (Laguna self_attn.g_proj)
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head_l}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        // dense MLP (leading dense block: layer 0 only)
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);

        // MoE routed experts + selection bias
        const int64_t n_ff_exp = hparams.n_ff_exp;
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,   "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,  "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,  "weight", i), {n_ff_exp,   n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,    "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",  i), {n_expert}, TENSOR_NOT_REQUIRED);

        // shared expert MLP (added on every MoE layer)
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_laguna::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_laguna::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos     = build_inp_pos();
    auto        * inp_attn    = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const uint32_t n_head_l    = hparams.n_head(il);
        const uint32_t n_head_kv_l = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        cur = inpL;

        // self-attention
        {
            cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head_l,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv_l, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv_l, n_tokens);

            // Q/K per-head RMSNorm (before RoPE)
            if (model.layers[il].attn_q_norm) {
                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);
            }
            if (model.layers[il].attn_k_norm) {
                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);
            }

            // per-layer partial dual RoPE: full-attn layers use partial (n_rot=64) YaRN,
            // sliding layers use full-rotary (n_rot=128) plain rope. YaRN is a no-op on
            // sliding layers because their freq_scale is 1.0.
            const bool is_swa = hparams.is_swa(il);
            ggml_tensor * rope_factors = is_swa ? nullptr : model.get_rope_factors(cparams, il);
            const int64_t n_rot_l = hparams.n_rot(il);
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
            );
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
            );
            cb(Qcur, "Qcur_pos", il);
            cb(Kcur, "Kcur_pos", il);

            const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));
            ggml_tensor * attn_out = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(attn_out, "attn_out", il);

            // head-wise attention output gate: softplus(g_proj(x)) per head, before o_proj.
            // NOTE: Laguna uses softplus (config "gating": "per-head"), NOT the sigmoid
            // that the otherwise-identical Step35 gate uses — see modeling_laguna.py.
            if (model.layers[il].wqkv_gate) {
                ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur); // [n_head_l, n_tokens]
                cb(gate, "attn_gate", il);

                gate = ggml_softplus(ctx0, gate);
                cb(gate, "attn_gate_softplus", il);

                ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v, n_head_l, n_tokens);
                ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,     1,             n_head_l, n_tokens);
                cb(gate_3d, "attn_gate_3d", il);

                attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
                cb(attn_3d, "attn_gated_3d", il);

                attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head_v * n_head_l, n_tokens);
                cb(attn_out, "attn_gated", il);
            }

            cur = build_lora_mm(model.layers[il].wo, attn_out, model.layers[il].wo_s);
            cb(cur, "attn_proj", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense MLP (leading dense block)
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, nullptr,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE routed experts (sigmoid gating + correction bias + routed scaling)
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert MLP (always added on MoE layers)
            ggml_tensor * sh_out = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(sh_out, "ffn_shared_out", il);

            cur = ggml_add(ctx0, moe_out, sh_out);
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
