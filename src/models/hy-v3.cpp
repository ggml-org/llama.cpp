#include "models.h"

// Hy3 (tencent/Hy3, model_type hy_v3): hybrid MoE with GQA, per-head QK-norm,
// NEOX RoPE, sigmoid-gated routing with exp_probs_b bias, 1 shared expert per MoE layer,
// and 1 inert NextN/MTP layer appended after the 80 real layers.
//
// Architecture reference: src/transformers/models/hy_v3/modeling_hy_v3.py
// Forward pass per layer:
//   h = x + Attn(RMSNorm(x))         -- input_layernorm
//   h = h + FFN(RMSNorm(h))          -- post_attention_layernorm
// Attention: GQA, per-head RMSNorm on Q and K (mandatory), then NEOX RoPE.
// FFN layer 0: dense SiLU gate-up-down (intermediate=13312).
// FFN layers 1+: MoE sigmoid router with exp_probs_b correction, top-8 of 192,
//   norm_w=true, scale_w=true (2.826), plus 1 shared expert added plain (no fp32 cast).

void llama_model_hy_v3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_COUNT,              hparams.n_expert);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,         hparams.n_expert_used);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,       hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,      hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,       hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,        hparams.expert_gating_func, false);

    // Hy3 always uses sigmoid gating
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // NextN/MTP parameters
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_all");

    // n_layer() = n_layer_all - n_layer_nextn = 80 for the real model
    switch (hparams.n_layer()) {
        case 80: type = LLM_TYPE_295B_A21B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_hy_v3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    GGML_ASSERT(hparams.n_expert > 0      && "n_expert must be > 0 for HY_V3 MoE layers");
    GGML_ASSERT(hparams.n_expert_used > 0 && "n_expert_used must be > 0 for HY_V3 MoE layers");

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // Load ALL layers including the NextN layer so the total tensor count is satisfied.
    // Tensors for i >= n_layer are tagged TENSOR_SKIP so they are stored but never
    // referenced in the forward graph.
    for (int i = 0; i < n_layer_all; ++i) {
        int flags = 0;
        if (i >= n_layer) {
            // NextN tensors are never referenced in the forward graph, so also
            // tolerate GGUFs where the NextN block has been pruned
            flags |= TENSOR_SKIP | TENSOR_NOT_REQUIRED;
        }

        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM,  "weight", i), { n_embd }, flags);

        create_tensor_qkv(layer, i, n_embd,
                          n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i),
                                 { n_embd_head_k * n_head, n_embd }, flags);

        // Per-head QK norms - mandatory for Hy3
        layer.attn_q_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, flags);
        layer.attn_k_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, flags);

        const bool use_moe = (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead);

        if (use_moe) {
            layer.ffn_gate_inp =
                create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, flags);
            layer.ffn_exp_probs_b =
                create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, i), { n_expert }, flags);

            const int64_t n_ff_exp = hparams.n_ff_exp;

            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, flags);
            layer.ffn_down_exps = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, flags);

            // Shared expert (num_shared_experts=1, intermediate=n_ff_exp)
            if (n_expert_shared > 0) {
                const int64_t n_ff_shexp = n_ff_exp * n_expert_shared;
                layer.ffn_gate_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
                layer.ffn_down_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, flags);
                layer.ffn_up_shexp   = create_tensor(
                    tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), { n_embd, n_ff_shexp }, flags);
            }
        } else {
            // Dense layer 0
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, flags);
        }

        // NextN-specific tensors (layer index >= n_layer)
        // final_layernorm maps to NEXTN_SHARED_HEAD_NORM per SPEC.
        if (i >= n_layer) {
            layer.nextn.eh_proj = create_tensor(
                tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm   = create_tensor(
                tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm   = create_tensor(
                tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);
            layer.nextn.shared_head_norm = create_tensor(
                tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_hy_v3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_hy_v3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Forward pass over real layers only; NextN layer tensors are loaded but
    // never processed - the loop bound n_layer = n_layer_all - n_layer_nextn.
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Pre-attention norm (input_layernorm)
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Self-attention with per-head QK-norm then NEOX RoPE
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // Per-head RMSNorm on Q, then RoPE
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            // Per-head RMSNorm on K, then RoPE
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Pre-FFN norm (post_attention_layernorm)
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // FFN: dense for layer 0, MoE+shared for layers 1+
        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            // Dense SiLU FFN
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // Routed MoE: sigmoid gating, exp_probs_b selection bias,
            // norm_w=true, scale_w=true, scale=2.826
            ggml_tensor * routed_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    model.layers[il].ffn_gate_up_exps);
            cb(routed_out, "ffn_moe_out", il);

            // Shared expert - plain add (enable_moe_fp32_combine=false in hy3-config)
            ggml_tensor * shared_out = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(shared_out, "ffn_shexp_out", il);

            cur = ggml_add(ctx0, routed_out, shared_out);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head (untied; tie_word_embeddings=false)
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
