#include "models.h"

void llama_model_beam::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
    hparams.f_final_logit_softcapping = 0.0f; // absent key = no softcap (default is 30)
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,           hparams.f_final_logit_softcapping, false);

    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }
    if (hparams.n_layer_dense_lead > hparams.n_layer()) {
        throw std::runtime_error("beam: leading_dense_block_count exceeds the layer count");
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD; // masked iff p1 - p0 >= n_swa
    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);
    if (hparams.n_swa == 0 || n_pattern != hparams.n_layer_all) {
        throw std::runtime_error("beam: requires a positive sliding window and a sliding window pattern for each layer");
    }
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    if (!hparams.is_swa_any()) {
        throw std::runtime_error("beam: sliding window pattern has no sliding window layer");
    }

    // full layers are NoPE
    hparams.rope_pattern = hparams.is_swa_impl;
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    // NoPE query scale uses pos + 1
    hparams.f_attn_temp_scale = 0.1f;
    hparams.attn_temp_length  = 2048;
    ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_SCALE,  hparams.f_attn_temp_scale, false);
    ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH, hparams.attn_temp_length,  false);
    if (hparams.f_attn_temp_scale == 0.0f) {
        throw std::runtime_error("beam: attention.temperature_scale must be nonzero");
    }
    if (hparams.attn_temp_length == 0) {
        throw std::runtime_error("beam: attention.temperature_length must be positive");
    }
    hparams.n_attn_temp_floor_scale = hparams.attn_temp_length;
    hparams.f_attn_temp_offset      = 1.0f;
    hparams.attn_temp_pow           = true;

    switch (hparams.n_layer()) {
        case 52: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_beam::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.wqkv_gate   = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wqkv_gate_b = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "bias",   i), {n_head}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        if ((uint32_t) i < hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            const int64_t n_ff_exp = hparams.n_ff_exp(i);

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_beam::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

static ggml_tensor * beam_residual_add(ggml_context * ctx, ggml_tensor * h, ggml_tensor * r, int i) {
    const float h_scale = (float) (1.0 / std::sqrt((double) i));
    const float r_scale = (float) std::sqrt((double) (i - 1) / (double) i);
    return ggml_add(ctx, ggml_scale(ctx, h, h_scale), ggml_scale(ctx, r, r_scale));
}

llama_model_beam::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    bool has_nope = false;
    for (int il = 0; il < n_layer; ++il) {
        has_nope = has_nope || !hparams.is_swa(il);
    }
    ggml_tensor * inp_attn_scale = has_nope ? build_inp_attn_scale() : nullptr;

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const bool is_swa = hparams.is_swa(il);

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * attn_inp = cur;

            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            if (is_swa) {
                const float freq_base_l  = model.get_rope_freq_base (cparams, il);
                const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_rope", il);
                cb(Kcur, "Kcur_rope", il);
            } else {
                Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                cb(Qcur, "Qcur_scaled", il);
            }

            cur = build_attn(inp_attn,
                    NULL, NULL, // o_proj is applied after the gate
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, attn_inp);
            cb(gate, "attn_gate_proj", il);
            gate = ggml_reshape_3d(ctx0, gate, n_embd_head, n_head, n_tokens);
            gate = ggml_add(ctx0, gate, ggml_reshape_3d(ctx0, layer.wqkv_gate_b, 1, n_head, 1));
            gate = ggml_sigmoid(ctx0, gate);
            gate = ggml_reshape_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
            cb(gate, "attn_gate_sig", il);

            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);

            cur = build_lora_mm(layer.wo, cur);
            cb(cur, "attn_o_proj", il);
        }

        cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        const int i_attn = 2 * (il + 1);
        ggml_tensor * ffn_inp = beam_residual_add(ctx0, cur, inpSA, i_attn);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   NULL,
                    layer.ffn_gate, NULL,
                    layer.ffn_down, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_moe_ffn(cur,
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
            cb(cur, "ffn_moe_out", il);
        }

        cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = beam_residual_add(ctx0, cur, ffn_inp, i_attn + 1);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
