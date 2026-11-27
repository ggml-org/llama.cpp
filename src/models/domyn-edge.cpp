#include "models.h"

void llama_model_domyn_edge::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    // required: the base loader defaults it to n_rot_full if the key is not found
    ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa);
    // RoPE cannot rotate more dims than the SWA head has: CPU aborts, GPU backends read past the row
    if (hparams.n_rot_swa > hparams.n_embd_head_k_swa) {
        throw std::runtime_error(format("invalid n_rot_swa: %u, max %u", hparams.n_rot_swa, hparams.n_embd_head_k_swa));
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    ml.get_arr(LLM_KV_ATTENTION_ROPE_PATTERN,           hparams.rope_pattern);

    // only for print_info / saver; get_rope_params uses cparams
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    switch (hparams.n_layer()) {
        case 36: type = LLM_TYPE_4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_domyn_edge::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_embd_head_i = hparams.n_embd_head_k(i);
        const int64_t n_embd_k_i    = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v_i    = hparams.n_embd_v_gqa(i);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        const int64_t n_embd_q_i = n_embd_head_i * n_head;
        create_tensor_qkv(layer, i, n_embd, n_embd_q_i, n_embd_k_i, n_embd_v_i, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_i * n_head, n_embd}, 0);

        // K-norm has a learnable weight; Q-norm does not
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_i}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
    }
}

llama_rope_params llama_model_domyn_edge::get_rope_params(const llama_cparams & cparams, int il) const {
    auto res = llama_model::get_rope_params(cparams, il);

    // the SWA layers use the global base and scale
    res.freq_base  = cparams.rope_freq_base;
    res.freq_scale = cparams.rope_freq_scale;

    return res;
}

llama_model_domyn_edge::graph::graph(const llm_graph_params & params)
    : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL    = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const auto &  layer        = model.layers[il];
        const int64_t n_embd_head  = hparams.n_embd_head_k(il);
        const int64_t n_head_il    = hparams.n_head(il);
        const int64_t n_head_kv_il = hparams.n_head_kv(il);
        const float   kq_scale     = 1.0f / sqrtf((float)n_embd_head);

        ggml_tensor * residual = inpL;

        // pre-attention norm
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head_il, n_head_kv_il, il);

        // Q-norm: RMSNorm without learnable weight (elementwise_affine=False)
        Qcur = build_norm(Qcur, NULL, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        // K-norm: RMSNorm with learnable weight
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        // build_rope already applies RoPE only if the current layer is a RoPE layer
        Qcur = build_rope(Qcur, inp_pos, il);
        cb(Qcur, "Qcur_rope", il);

        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Kcur, "Kcur_rope", il);

        cur = build_attn(inp_attn,
                layer.wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        // post-attention norm applied to attn output before residual add
        cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, residual);
        cb(ffn_inp, "ffn_inp", il);

        // pre-FFN norm
        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // SwiGLU FFN
        cur = build_ffn(cur,
                layer.ffn_up,   NULL,
                layer.ffn_gate, NULL,
                layer.ffn_down, NULL, NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // post-FFN norm applied to ffn output before residual add
        cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur  = ggml_add(ctx0, cur, ffn_inp);
        inpL = build_cvec(cur, il);
        cb(inpL, "l_out", il);
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::unique_ptr<llm_graph_context> llama_model_domyn_edge::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}
