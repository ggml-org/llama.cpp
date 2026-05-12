#include "models.h"

void llama_model_mpt::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv, false);
    ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias, false);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        case 48: type = LLM_TYPE_30B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_mpt::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, TENSOR_NOT_REQUIRED);

    // output
    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

    output        = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // needs to be on GPU
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, TENSOR_NOT_REQUIRED);

        // FIXME test-llama-archs crashes if q_norm is created
        layer.attn_q_norm   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);

        layer.attn_k_norm   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

        // AWQ ScaleActivation layer
        layer.ffn_act = create_tensor(tn(LLM_TENSOR_FFN_ACT, "scales", i), {n_ff}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mpt::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_mpt::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * pos;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp_attn = build_attn_inp_kv();

    if (model.pos_embd) {
        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();
        pos                   = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * attn_norm;

        attn_norm = build_norm(inpL, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, il);
        cb(attn_norm, "attn_norm", il);

        // self-attention
        {
            cur = attn_norm;

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // Q/K Layernorm
            if (model.layers[il].attn_q_norm) {
                Qcur = ggml_reshape_2d(ctx0, Qcur, n_embd_head * n_head, n_tokens);
                Kcur = ggml_reshape_2d(ctx0, Kcur, n_embd_head * n_head_kv, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, model.layers[il].attn_q_norm_b, LLM_NORM, il);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, model.layers[il].attn_k_norm_b, LLM_NORM, il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // Add the input
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // feed forward
        {
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, model.layers[il].ffn_norm_b, LLM_NORM, il);
            cb(cur, "ffn_norm", il);
            cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                model.layers[il].ffn_act, LLM_FFN_GELU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
