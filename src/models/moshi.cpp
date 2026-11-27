#include "models.h"

// backbone of the Moshi family (e.g. Hibiki): one position per audio frame, the input rows (text + audio codes)
// are built by mtmd, the head predicts the next text token. Every layer attends to the same sliding window, see create_memory().

void llama_model_moshi::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // the window of every layer, it is applied by the KV cache, see create_memory()
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    switch (hparams.n_layer()) {
        case 16: type = LLM_TYPE_1B; break; // Hibiki-M
        case 24: type = LLM_TYPE_2B; break; // Hibiki
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_moshi::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    // the last vocab entries are input-only (e.g. the initial text token), the head has no logits for them
    // a model built from metadata only has no tensor to read it from
    const ggml_tensor * meta = ml.get_tensor_meta(tn(LLM_TENSOR_OUTPUT, "weight").str().c_str());
    const int64_t n_vocab_out = meta ? meta->ne[1] : n_vocab;
    if (n_vocab_out < 1 || n_vocab_out > n_vocab) {
        throw std::runtime_error(format("tensor 'output.weight' has invalid head size %lld (vocab size %lld)", (long long) n_vocab_out, (long long) n_vocab));
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab_out}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_moshi::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_moshi::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_rope(Qcur, inp_pos, il);
            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL,
                model.layers[il].ffn_gate, NULL,
                model.layers[il].ffn_down, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    // pad the logits of the input-only tokens with -inf
    const int64_t pad = model.tok_embd->ne[1] - model.output->ne[1];
    if (pad > 0) {
        ggml_tensor * neg_inf = ggml_scale_bias(ctx0, ggml_view_1d(ctx0, model.output_norm, 1, 0), 0.0f, -INFINITY);
        neg_inf = ggml_repeat_4d(ctx0, neg_inf, pad, cur->ne[1], 1, 1);
        cur = ggml_concat(ctx0, cur, neg_inf, 0);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
