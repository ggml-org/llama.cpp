#include "models.h"

void llama_model_glm4_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // Expert gating function (GLM-4.5 uses sigmoid)
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func =  LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // NextN/MTP parameters
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers < hparams.n_layer && "nextn_predict_layers must be < n_layer");

    // TODO: when MTP is implemented, this should probably be updated if needed
    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

    switch (hparams.n_layer) {
        case 47: type = LLM_TYPE_106B_A12B; break; // GLM-4.5-Air (46 layers + 1 NextN layer)
        case 48: type = LLM_TYPE_102B_A12B; break; // Solar Open
        case 93: type = LLM_TYPE_355B_A32B; break; // GLM-4.5 (92 layers + 1 NextN layer)
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm4_moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;


    GGML_ASSERT(hparams.n_expert > 0 && "n_expert must be > 0 for GLM4_MOE MoE layers");
    GGML_ASSERT(hparams.n_expert_used > 0 && "n_expert_used must be > 0 for GLM4_MOE MoE layers");

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // Load ALL tensors including NextN layer to satisfy total tensor count
    // but only PROCESS up to last layer (skipping final NextN layer) in forward pass
    for (int i = 0; i < n_layer; ++i) {
        int flags = 0;
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            // skip all tensors in the NextN layers
            flags |= TENSOR_SKIP;
        }

        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, flags);

        // GLM-style attention with bias terms
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, flags);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

        // K/Q norm tensors (optional for GLM-4.5 355B variant)
        layer.attn_q_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, TENSOR_NOT_REQUIRED | flags);
        layer.attn_k_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, TENSOR_NOT_REQUIRED | flags);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, flags);

        // Check if this layer uses MoE or dense FFN based on n_layer_dense_lead
        // GLM 4.5 uses hybrid architecture: layer 0 is dense, layers 1+ are MoE
        const bool use_moe = (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead);

        if (use_moe) {
            // MoE layers
            layer.ffn_gate_inp =
                create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), { n_expert }, flags);

            // MoE branch
            const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

            layer.ffn_gate_exps = create_tensor(
                tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, flags);
            layer.ffn_down_exps = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, flags);
            layer.ffn_up_exps = create_tensor(
                tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, flags);

            // Shared expert
            if (n_expert_shared > 0) {
                const int64_t n_ff_shexp = n_ff_exp * n_expert_shared;
                layer.ffn_gate_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
                layer.ffn_down_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, flags);
                layer.ffn_up_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
            }
        } else {
            // Dense layers (first k layers) - GLM uses separate gate/up projections
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, flags);
        }

        // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);

            // Optional tensors
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_glm4_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_glm4_moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    bool use_mrope = hparams.use_mrope();
    if (ubatch.embd && !use_mrope) {
        // unfortunately, we need to forcefully stop here, to avoid users complaining about wrong results
        GGML_ABORT("This GGUF does not support multimodal. Please reconvert it.");
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Only process up to last layer (skip final NextN layer)
    // Final layer tensors are loaded but not processed in forward pass
    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_transformer_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // Apply Q/K norm if available (GLM-4.5 355B variant)
            if (model.layers[il].attn_q_norm) {
                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);
            }
            if (model.layers[il].attn_k_norm) {
                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);
            }

            if (use_mrope) {
                Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
            } else {
                // Normal RoPE
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot,
                                    rope_type, n_ctx_orig, freq_base, freq_scale,
                                    ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot,
                                    rope_type, n_ctx_orig, freq_base, freq_scale,
                                    ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Post-attention norm
        cur = build_norm(ffn_inp, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        // Check if this is a dense layer (n_layer_dense_lead=1, so layer 0 is dense)
        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            // Dense FFN layer
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // Process routed experts using existing MoE infrastructure
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
                    il);
            cb(routed_out, "ffn_moe_out", il);

            // Process shared expert on original input
            ggml_tensor * shared_out = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(shared_out, "ffn_shexp_out", il);

            // Final output: routed_output + shared_output
            cur = ggml_add(ctx0, routed_out, shared_out);
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
