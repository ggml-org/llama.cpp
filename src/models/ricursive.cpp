#include "models.h"

void llama_model_ricursive::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_SWIGLU_CLAMP,                hparams.f_swiglu_clamp,    false);

    if (hparams.n_embd_head_k() != hparams.n_embd_head_v()) {
        throw std::runtime_error("ricursive: key_length and value_length must be equal");
    }

    hparams.n_swa = 0;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        load_swa_pattern(ml, 5);

        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = 1.0f;
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    switch (hparams.n_layer()) {
        case 80: type = LLM_TYPE_65B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_ricursive::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int64_t n_embd_q = n_embd_head_k * n_head;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_q},     0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd},     0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.wqkv_gate        = create_tensor(tn(LLM_TENSOR_ATTN_GATE,        "weight", i), {n_embd, n_embd_q}, TENSOR_NOT_REQUIRED);
        layer.attn_vres_lambda = create_tensor(tn(LLM_TENSOR_ATTN_VRES_LAMBDA, "weight", i), {1},                TENSOR_NOT_REQUIRED);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd},       0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd},       0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_ricursive::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_ricursive::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_vres = inpL;

    ggml_tensor * inp_pos = build_inp_pos();

    const bool has_swa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    llm_graph_input_attn_kv      * inp_attn_kv   = has_swa ? nullptr : build_attn_inp_kv();
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = has_swa ? build_attn_inp_kv_iswa() : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * attn_in = cur;

        {
            ggml_tensor * Qcur = build_lora_mm(layer.wq, attn_in, layer.wq_s);
            ggml_tensor * Kcur = build_lora_mm(layer.wk, attn_in, layer.wk_s);

            ggml_tensor * v_in = attn_in;
            if (layer.attn_vres_lambda) {
                ggml_tensor * mix = ggml_sigmoid(ctx0, layer.attn_vres_lambda);
                v_in = ggml_add(ctx0, inp_vres, ggml_mul(ctx0, ggml_sub(ctx0, attn_in, inp_vres), mix));
                cb(v_in, "vres_in", il);
            }
            ggml_tensor * Vcur = build_lora_mm(layer.wv, v_in, layer.wv_s);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            if (hparams.is_swa(il)) {
                const float freq_base_l = model.get_rope_freq_base(cparams, il);
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, 1.0f,
                        0.0f, 1.0f, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, 1.0f,
                        0.0f, 1.0f, beta_fast, beta_slow);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = has_swa
                ? build_attn(inp_attn_iswa, NULL, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
                : build_attn(inp_attn_kv,   NULL, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            if (layer.wqkv_gate) {
                ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, attn_in);
                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "attn_gate", il);
                cur = ggml_mul(ctx0, cur, gate);
                cb(cur, "attn_gated", il);
            }

            cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        {
            ggml_tensor * gate = build_lora_mm(layer.ffn_gate, cur, layer.ffn_gate_s);
            ggml_tensor * up   = build_lora_mm(layer.ffn_up,   cur, layer.ffn_up_s);
            cur = hparams.f_swiglu_clamp > 0.0f
                ? ggml_swiglu_clamp(ctx0, gate, up, hparams.f_swiglu_clamp)
                : ggml_swiglu_split(ctx0, gate, up);
            cb(cur, "ffn_swiglu", il);
            cur = build_lora_mm(layer.ffn_down, cur, layer.ffn_down_s);
            cb(cur, "ffn_out", il);
        }

        cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
