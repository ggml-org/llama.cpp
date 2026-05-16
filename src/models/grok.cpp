#include "models.h"

void llama_model_grok::load_arch_hparams(llama_model_loader & ml) {
    // defaults for old GGUFs
    hparams.yarn_beta_fast = 8.0f;
    hparams.f_logit_scale = 0.5773502691896257f;
    hparams.f_embedding_scale = 78.38367176906169f;
    hparams.f_attn_out_scale = 0.08838834764831845f;
    hparams.f_attn_logit_softcapping = 30.0f;
    hparams.f_router_logit_softcapping = 30.0f;
    // no final_logit_softcapping in grok-1
    hparams.f_final_logit_softcapping = 0.0f;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,   hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_LOGIT_SCALE,                  hparams.f_logit_scale, false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,              hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_SCALE,       hparams.f_attn_out_scale, false);
    ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING,       hparams.f_attn_logit_softcapping, false);
    ml.get_key(LLM_KV_ROUTER_LOGIT_SOFTCAPPING,     hparams.f_router_logit_softcapping, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,      hparams.f_final_logit_softcapping, false);

    ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH,  hparams.attn_temp_length, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,  hparams.yarn_ext_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   hparams.yarn_beta_fast, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   hparams.yarn_beta_slow, false);

    switch (hparams.n_layer) {
        case 64: type = LLM_TYPE_314B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_grok::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (n_expert == 0) {
        throw std::runtime_error(arch_name() + " model cannot have zero experts");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff/* / n_expert_used*/; // grok-1 n_ff_exp == n_ff
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff,   n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        if (!layer.ffn_post_norm) {
            layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_grok::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_grok::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_out_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_out_norm", il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // MoE branch
        ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_GELU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
        cb(moe_out, "ffn_moe_out", il);

        if (model.layers[il].ffn_up) {
            ggml_tensor * ffn_out = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(ffn_out, "ffn_out", il);

            cur = ggml_scale(ctx0, ggml_add(ctx0, ffn_out, moe_out), std::sqrt(2) / 2);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);

    // final logit soft-capping
    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
