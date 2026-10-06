#include "models.h"

void llama_model_k2_horizon::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, hparams.yarn_beta_fast, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW, hparams.yarn_beta_slow, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS,  hparams.n_norm_groups, false);
    if (hparams.n_norm_groups == 0) {
        hparams.n_norm_groups = 1;
    }

    if (hparams.n_expert > 0) {
        ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
        ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead,   false);
        ml.get_key(LLM_KV_MOE_EVERY_N_LAYERS,                hparams.moe_every_n_layers,   false);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared,      false);
        ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp,           false);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm,  false);
        ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func,   false);
        if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
        }
    }

    // MoVA
    ml.get_key(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      hparams.n_value_expert,      false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, hparams.n_value_expert_used, false);
    if (hparams.n_value_expert > 0) {
        GGML_ASSERT(hparams.n_value_expert <= LLAMA_MAX_EXPERTS);
        GGML_ASSERT(hparams.n_value_expert_used > 0);
        GGML_ASSERT(hparams.n_value_expert_used <= hparams.n_value_expert);
    } else {
        GGML_ASSERT(hparams.n_value_expert_used == 0);
    }

    switch (hparams.n_layer()) {
        case 28: type = LLM_TYPE_1B; break;
        case 36:
            switch (hparams.n_embd) {
                case 2560: type = LLM_TYPE_4B; break;
                case 4096: type = LLM_TYPE_7B; break;
                default:   type = LLM_TYPE_UNKNOWN;
            } break;
        case 48: type = LLM_TYPE_36B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_k2_horizon::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const bool is_moe_layer  = n_expert > 0 && (uint32_t) i >= hparams.n_layer_dense_lead;
        const bool is_mova_layer = is_moe_layer && hparams.n_value_expert > 0;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq          = create_tensor(tn(LLM_TENSOR_ATTN_Q,      "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk          = create_tensor(tn(LLM_TENSOR_ATTN_K,      "weight", i), {n_embd, n_embd_k_gqa}, 0);
        // one norm weight per head, stored flat; viewed as {head_dim, n_head} so it splits by head like Q/K
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head},    TENSOR_NOT_REQUIRED | TENSOR_ALLOW_RESHAPE);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, TENSOR_NOT_REQUIRED | TENSOR_ALLOW_RESHAPE);

        if (is_mova_layer) {
            layer.attn_v_gate   = create_tensor(tn(LLM_TENSOR_ATTN_V_GATE, "weight", i), {n_embd, hparams.n_value_expert}, 0);
            layer.attn_v_gate_b = create_tensor(tn(LLM_TENSOR_ATTN_V_GATE, "bias",   i), {hparams.n_value_expert}, TENSOR_NOT_REQUIRED);
            layer.attn_v_exps   = create_tensor(tn(LLM_TENSOR_ATTN_V_EXPS, "weight", i), {n_embd, n_embd_v_gqa, hparams.n_value_expert}, 0);
        } else {
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_v_gqa}, 0);
        }

        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head_v * n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_v * n_head}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (is_moe_layer) {
            const int64_t n_ff_exp = hparams.n_ff_exp(i);
            if (n_ff_exp == 0) {
                throw std::runtime_error("K2 Horizon MoE layer requires expert_feed_forward_length");
            }

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);

            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);

            if (hparams.n_expert_shared > 0) {
                const int64_t n_ff_shexp = hparams.n_ff_shexp > 0 ? hparams.n_ff_shexp : n_ff_exp * hparams.n_expert_shared;

                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,     n_ff_shexp}, 0);
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,     n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd},     0);
            }
        } else {
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff,   n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_k2_horizon::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// RMS norm over n_groups equal slices of ne[0], then one full-width weight
static ggml_tensor * k2_horizon_group_rms_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight, int64_t n_groups, float eps) {
    GGML_ASSERT(n_groups > 0 && cur->ne[0] % n_groups == 0);

    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    cur = ggml_reshape_3d(ctx, cur, n_embd / n_groups, n_groups, n_tokens);
    cur = ggml_rms_norm(ctx, cur, eps);
    cur = ggml_reshape_2d(ctx, cur, n_embd, n_tokens);

    return weight ? ggml_mul(ctx, cur, weight) : cur;
}

// MoVA: route each token to n_value_expert_used value experts, V = sum_k w_k * silu(W_k x)
ggml_tensor * llama_model_k2_horizon::graph::build_routed_value(const llama_layer & layer, ggml_tensor * cur, int il) const {
    const int64_t n_embd     = cur->ne[0];
    const int64_t n_tokens   = cur->ne[1];
    const int64_t n_embd_gqa = hparams.n_embd_v_gqa(il);
    const int64_t n_values   = hparams.n_value_expert;
    const int64_t n_used     = hparams.n_value_expert_used;

    ggml_tensor * logits = build_lora_mm(layer.attn_v_gate, cur);
    ggml_tensor * probs  = nullptr;

    switch ((llama_expert_gating_func_type) hparams.expert_gating_func) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: probs = ggml_soft_max(ctx0, logits); break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: probs = ggml_sigmoid(ctx0, logits);  break;
        default: GGML_ABORT("unsupported K2 Horizon value-router gating function");
    }

    // the bias only affects which experts are selected, not their weights
    ggml_tensor * selection_probs = probs;
    if (layer.attn_v_gate_b) {
        selection_probs = ggml_add(ctx0, probs, layer.attn_v_gate_b);
        cb(selection_probs, "v_moe_probs_biased", il);
    }

    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_used);

    probs = ggml_reshape_3d(ctx0, probs, 1, n_values, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts);

    if (hparams.expert_weights_norm) {
        weights = ggml_reshape_2d(ctx0, weights, n_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_used, n_tokens);
        cb(weights, "v_moe_weights_norm", il);
    }

    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "v_moe_weights_scaled", il);
    }

    cb(logits, "v_moe_logits", il);
    cb(probs,  "v_moe_probs",  il);
    cb(selected_experts->src[0], "v_moe_argsort", il);
    cb(selected_experts,         "v_moe_topk",    il);
    cb(weights, "v_moe_weights", il);

    ggml_tensor * values = build_lora_mm_id(layer.attn_v_exps, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), selected_experts);
    values = ggml_silu(ctx0, values);
    values = ggml_mul(ctx0, values, weights);
    cb(values, "v_moe_weighted", il);

    // sum the selected experts; 3D views of {n_embd_gqa, 1, n_tokens} keep the strides of values,
    // which lets the tensor-parallel backend follow its split through the views
    // order the views before the adds so backends can fuse the sum
    ggml_tensor * value_views[LLAMA_MAX_EXPERTS] = { nullptr };
    for (int64_t i = 0; i < n_used; ++i) {
        value_views[i] = ggml_view_3d(ctx0, values, n_embd_gqa, 1, n_tokens, values->nb[1], values->nb[2], i * values->nb[1]);
        ggml_build_forward_expand(gf, value_views[i]);
    }

    ggml_tensor * value_out = value_views[0];
    for (int64_t i = 1; i < n_used; ++i) {
        value_out = ggml_add(ctx0, value_out, value_views[i]);
        ggml_build_forward_expand(gf, value_out);
    }
    if (n_used == 1) {
        value_out = ggml_cont(ctx0, value_out);
    }
    cb(value_out, "Vcur_routed", il);

    return value_out;
}

llama_model_k2_horizon::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        const bool is_moe_layer  = n_expert > 0 && (uint32_t) il >= hparams.n_layer_dense_lead;
        const bool is_mova_layer = is_moe_layer && hparams.n_value_expert > 0;

        cur = k2_horizon_group_rms_norm(ctx0, inpL, layer.attn_norm, hparams.n_norm_groups, hparams.f_norm_rms_eps);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * attn_inp = cur; // saved for the output gate

            ggml_tensor * Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
            ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
            ggml_tensor * Vcur = is_mova_layer ? build_routed_value(layer, cur, il) : build_lora_mm(layer.wv, cur, layer.wv_s);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // per-head RMS norm with a separate weight for every head
            if (layer.attn_q_norm) {
                Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            }
            if (layer.attn_k_norm) {
                Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            }

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // with an output gate, o_proj is applied after gating
            const bool gated = layer.wqkv_gate != nullptr;

            cur = build_attn(inp_attn,
                    gated ? nullptr : layer.wo, gated ? nullptr : layer.wo_b, gated ? nullptr : layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

            if (gated) {
                // softplus with beta = ln(2): log2(1 + 2^x)
                constexpr float ln2 = 0.6931471805599453f;
                ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, attn_inp, layer.wqkv_gate_s);
                gate = ggml_scale(ctx0, gate, ln2);
                gate = ggml_softplus(ctx0, gate);
                gate = ggml_scale(ctx0, gate, 1.4426950408889634f); // 1 / ln(2)

                cur = ggml_mul(ctx0, cur, gate);
                cur = build_lora_mm(layer.wo, cur, layer.wo_s);
                if (layer.wo_b) {
                    cur = ggml_add(ctx0, cur, layer.wo_b);
                }
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = k2_horizon_group_rms_norm(ctx0, ffn_inp, layer.ffn_norm, hparams.n_norm_groups, hparams.f_norm_rms_eps);
        cb(cur, "ffn_norm", il);

        if (is_moe_layer) {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);

            if (layer.ffn_gate_shexp) {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                        layer.ffn_up_shexp,   NULL, NULL,
                        layer.ffn_gate_shexp, NULL, NULL,
                        layer.ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cur = ggml_add(ctx0, moe_out, ffn_shexp);
            } else {
                cur = moe_out;
            }
        } else {
            cur = build_ffn(cur,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = k2_horizon_group_rms_norm(ctx0, inpL, model.output_norm, hparams.n_norm_groups, hparams.f_norm_rms_eps);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
