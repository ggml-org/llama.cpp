#include "models.h"

void llama_model_cogvlm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_13B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_cogvlm::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd_head_k * n_head * 3}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.visexp_attn_wqkv = create_tensor(tn(LLM_TENSOR_VISEXP_ATTN_QKV, "weight", i), {n_embd, n_embd_head_k * n_head * 3}, 0);
        layer.visexp_attn_wo = create_tensor(tn(LLM_TENSOR_VISEXP_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        layer.visexp_ffn_gate = create_tensor(tn(LLM_TENSOR_VISEXP_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.visexp_ffn_down = create_tensor(tn(LLM_TENSOR_VISEXP_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.visexp_ffn_up   = create_tensor(tn(LLM_TENSOR_VISEXP_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_cogvlm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_cogvlm::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const float   kq_scale    = 1.0f / sqrtf(float(n_embd_head));

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * inpL;
    ggml_tensor * cur;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    // check ubatch to see if we have input tokens (text)
    // or an input embedding vector (image)
    bool is_text;
    if (ubatch.token) {
        is_text = true;
    } else {
        is_text = false;
    }

    for (int il = 0; il < n_layer; ++il) {
        // get either the text or image weight tensors
        ggml_tensor *wqkv, *wo, *wo_s;
        ggml_tensor *ffn_gate, *ffn_down, *ffn_up;

        if (is_text) {
            wqkv     = model.layers[il].wqkv;
            wo       = model.layers[il].wo;
            wo_s     = model.layers[il].wo_s;
            ffn_gate = model.layers[il].ffn_gate;
            ffn_down = model.layers[il].ffn_down;
            ffn_up   = model.layers[il].ffn_up;
        } else {
            wqkv     = model.layers[il].visexp_attn_wqkv;
            wo       = model.layers[il].visexp_attn_wo;
            wo_s     = nullptr;
            ffn_gate = model.layers[il].visexp_ffn_gate;
            ffn_down = model.layers[il].visexp_ffn_down;
            ffn_up   = model.layers[il].visexp_ffn_up;
        }

        ggml_tensor * inpSA = inpL;
        cur = build_norm(inpSA, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

        // build self attention
        {
            ggml_tensor * qkv = build_lora_mm(wqkv, cur);

            // split qkv into Q, K, V along the first dimension
            ggml_tensor * Qcur =
                ggml_view_3d(ctx0, qkv, n_embd_head, n_head, n_tokens, n_embd_head * sizeof(float), qkv->nb[1], 0);
            ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                              qkv->nb[1], n_embd * ggml_element_size(qkv));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                              qkv->nb[1], 2 * n_embd * ggml_element_size(qkv));

            Qcur = ggml_rope(ctx0, Qcur, inp_pos, n_embd_head, rope_type);
            Kcur = ggml_rope(ctx0, Kcur, inp_pos, n_embd_head, rope_type);

            cur = build_attn(inp_attn,
                wo, nullptr, wo_s,
                Qcur, Kcur, Vcur,
                nullptr, nullptr, nullptr,
                kq_scale, il);
            cb(cur, "attn_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                ffn_up, NULL, NULL,
                ffn_gate, NULL, NULL,
                ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
