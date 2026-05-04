#include "models.h"

void llama_model_phi3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_3B; break;
        case 40: type = LLM_TYPE_14B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);

    if (found_swa && hparams.n_swa > 0) {
        LLAMA_LOG_WARN("%s: Phi SWA is currently disabled - results might be suboptimal for some models (see %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13676");

        // TODO: fix conversion scripts to correctly populate `n_swa` and `n_swa_pattern`
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;

        hparams.n_swa         = 0;
        hparams.set_swa_pattern(1);
    }
}

void llama_model_phi3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, TENSOR_NOT_REQUIRED);
        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, 2 * n_ff }, 0);

        layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
    }
}

std::unique_ptr<llm_graph_context> llama_model_phi3::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return std::make_unique<graph<true>> (*this, params);
    } else {
        return std::make_unique<graph<false>>(*this, params);
    }
}

template<bool iswa>
llama_model_phi3::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
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
        auto * residual = inpL;

        // self-attention
        {
            // rope freq factors for 128k context
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            ggml_tensor* attn_norm_output = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM_RMS, il);
            cb(attn_norm_output, "attn_norm", il);

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], attn_norm_output,
                    n_embd_head, n_head, n_head_kv, il);
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
            cb(Qcur, "Qcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }
        cur = ggml_add(ctx0, cur, residual);
        residual = cur;

        cur = build_norm(cur,
                model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, residual, cur);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = build_norm(inpL,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    if (model.output_b != nullptr) {
        cb(cur, "result_output_no_bias", -1);
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Explicit template instantiations
template struct llama_model_phi3::graph<false>;
template struct llama_model_phi3::graph<true>;
