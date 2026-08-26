#include "models.h"

void llama_model_lumma::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (hparams.n_embd_out_impl == 0) {
        throw std::runtime_error("Lumma requires embedding_length_out (factorized embedding rank)");
    }

    switch (hparams.n_layer()) {
        case 30: type = LLM_TYPE_0_6B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_lumma::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_embd_rank = hparams.n_embd_out();

    tok_embd      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), { n_embd_rank, n_vocab }, 0);
    tok_embd_proj = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_PROJ, "weight"), { n_embd_rank, n_embd     }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output_proj = create_tensor(tn(LLM_TENSOR_OUTPUT_PROJ, "weight"), { n_embd, n_embd_rank }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd_rank, n_vocab }, TENSOR_NOT_REQUIRED);

    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd_rank, n_vocab }, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa             }, 0);
        layer.wv = nullptr; // shared KV: value states are derived from the key projection
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_lumma::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_lumma::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // factorized token embedding: rank -> hidden
    ggml_tensor * inp_tokens;
    {
        const int64_t n_embd_rank = hparams.n_embd_out();

        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_rank);

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        inp_tokens = inp->tokens;
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_rank, ubatch.n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);

        res->add_input(std::move(inp));
    }

    cur = ggml_get_rows(ctx0, model.tok_embd, inp_tokens);
    cb(cur, "tok_embd", -1);

    cur = build_lora_mm(model.tok_embd_proj, cur);
    cb(cur, "tok_embd_proj", -1);
    inpL = cur;

    ggml_build_forward_expand(gf, inpL);

    ggml_tensor * inp_pos = build_inp_pos();
    auto *        inp_attn = build_attn_inp_kv();

    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // shared KV: V = raw k_proj (no norm, no RoPE); Q = q_norm + RoPE; K = RoPE only
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = Kcur;

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur, "Vcur", il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);

            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // SwiGLU feed-forward
        {
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out_add", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output_proj, cur);
    cb(cur, "result_proj", -1);

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
