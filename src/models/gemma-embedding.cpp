#include "models.h"

void llama_model_gemma_embedding::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
    uint32_t swa_period = 6;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
    hparams.set_swa_pattern(swa_period);

    hparams.causal_attn = false; // embeddings do not use causal attention

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    //applied only if model converted with --sentence-transformers-dense-modules
    ml.get_key(LLM_KV_DENSE_2_FEAT_IN, hparams.dense_2_feat_in, false);
    ml.get_key(LLM_KV_DENSE_2_FEAT_OUT, hparams.dense_2_feat_out, false);
    ml.get_key(LLM_KV_DENSE_3_FEAT_IN, hparams.dense_3_feat_in, false);
    ml.get_key(LLM_KV_DENSE_3_FEAT_OUT, hparams.dense_3_feat_out, false);

    GGML_ASSERT((hparams.dense_2_feat_in == 0 || hparams.dense_2_feat_in == hparams.n_embd) && "dense_2_feat_in must be equal to n_embd");
    GGML_ASSERT((hparams.dense_3_feat_out == 0 || hparams.dense_3_feat_out == hparams.n_embd) && "dense_3_feat_out must be equal to n_embd");

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_0_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
    hparams.f_attention_scale = 1.0f / std::sqrt(float(hparams.n_embd_head_k()));

}

void llama_model_gemma_embedding::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,   "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // Dense linear weights
    dense_2_out_layers = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "weight"), {n_embd, hparams.dense_2_feat_out}, TENSOR_NOT_REQUIRED);
    dense_3_out_layers = create_tensor(tn(LLM_TENSOR_DENSE_3_OUT, "weight"), {hparams.dense_3_feat_in, n_embd}, TENSOR_NOT_REQUIRED);


    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma_embedding::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma_embedding::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image embeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
            Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);

            cur =
                build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        cur = build_norm(sa_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", -1);

        cur = ggml_add(ctx0, cur, sa_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
