#include "models.h"

void llama_model_openelm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
    case 16: type = LLM_TYPE_270M; break;
    case 20: type = LLM_TYPE_450M; break;
    case 28: type = LLM_TYPE_1B; break;
    case 36: type = LLM_TYPE_3B; break;
    default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_openelm::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // init output from the input tok embed
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    for (int i = 0; i < n_layer; ++i) {
        const int64_t n_head      =   hparams.n_head(i);
        const int64_t n_head_qkv  = 2*hparams.n_head_kv(i) + n_head;
        const int64_t n_ff        =   hparams.n_ff(i);

        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_head_qkv*n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head*n_embd_head_k, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_openelm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_openelm::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_head_qkv = 2*n_head_kv + n_head;

        cur = inpL;
        ggml_tensor * residual = cur;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_reshape_3d(ctx0, cur, n_embd_head_k, n_head_qkv, n_tokens);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, cur->nb[1], cur->nb[2], 0);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*n_head);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*(n_head+n_head_kv));
            cb(Vcur, "Vcur", il);

            Qcur = build_norm(Qcur,
                    model.layers[il].attn_q_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(Qcur, "Qcur", il);

            Kcur = build_norm(Kcur,
                    model.layers[il].attn_k_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(Kcur, "Kcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, NULL,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, NULL,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Qcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, residual, cur);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    // norm
    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
