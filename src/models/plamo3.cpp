#include "models.h"

void llama_model_plamo3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (found_swa && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
        uint32_t swa_period = 8;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
        hparams.set_swa_pattern(swa_period);
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_2B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_plamo3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t head_dim_q = hparams.n_embd_head_k();
    const int64_t head_dim_v = hparams.n_embd_head_v();

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t num_attention_heads = hparams.n_head(i);
        const int64_t num_key_value_heads = hparams.n_head_kv(i);
        const int64_t q_proj_dim = num_attention_heads * head_dim_q;
        const int64_t k_proj_dim = num_key_value_heads * head_dim_q;
        const int64_t v_proj_dim = num_key_value_heads * head_dim_v;
        const int64_t n_ff_cur   = hparams.n_ff(i);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i),
                {n_embd,q_proj_dim + k_proj_dim + v_proj_dim}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {head_dim_q}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {head_dim_q}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {num_attention_heads * head_dim_v, n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, i), {n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, i), {n_embd}, 0);

        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff_cur * 2}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_plamo3::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return std::make_unique<graph<true>> (*this, params);
    } else {
        return std::make_unique<graph<false>>(*this, params);
    }
}

template <bool iswa>
llama_model_plamo3::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t head_dim_q = hparams.n_embd_head_k();
    const int64_t head_dim_v = hparams.n_embd_head_v();

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn = nullptr;

    if constexpr (iswa) {
        inp_attn = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;

        float freq_base_l  = 0.0f;
        float freq_scale_l = 0.0f;
        if constexpr (iswa) {
            freq_base_l  = model.get_rope_freq_base (cparams, il);
            freq_scale_l = model.get_rope_freq_scale(cparams, il);
        } else {
            freq_base_l  = freq_base;
            freq_scale_l = freq_scale;
        }

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);
        cb(cur, "wqkv", il);

        const int32_t n_head    = hparams.n_head(il);
        const int32_t n_head_kv = hparams.n_head_kv(il);

        const int64_t q_offset = 0;
        const int64_t k_offset = head_dim_q * n_head;
        const int64_t v_offset = k_offset + head_dim_q * n_head_kv;

        ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, head_dim_q, n_head, n_tokens,
                head_dim_q * sizeof(float), qkv->nb[1], q_offset * ggml_element_size(qkv));
        ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, head_dim_q, n_head_kv, n_tokens,
                head_dim_q * sizeof(float), qkv->nb[1], k_offset * ggml_element_size(qkv));
        ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, head_dim_v, n_head_kv, n_tokens,
                head_dim_v * sizeof(float), qkv->nb[1], v_offset * ggml_element_size(qkv));

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "attn_q_norm", il);
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Kcur, "attn_k_norm", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);

        const float attn_scale = 1.0f / sqrtf(float(head_dim_q));

        cur = build_attn(inp_attn,
                model.layers[il].wo, NULL, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, attn_scale, il);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "attn_residual", il);

        residual = cur;

        cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                NULL,                      NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "ffn_residual", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Explicit template instantiations
template struct llama_model_plamo3::graph<false>;
template struct llama_model_plamo3::graph<true>;
