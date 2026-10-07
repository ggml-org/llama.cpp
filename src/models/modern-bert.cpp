#include "models.h"

// question types of a decision model: choice, score, noul
static const uint32_t N_DECISION_TYPES = 3;

void llama_model_modern_bert::load_arch_hparams(llama_model_loader & ml) {
    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (found_swa && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
        load_swa_pattern(ml, 3, true);
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    // Some ModernBert derivatives (e.g. IBM Granite Embedding 97m R2) use
    // SiLU/SwiGLU in the FFN instead of the default GELU/GeGLU.
    hparams.llm_ffn_op = LLM_FFN_GEGLU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        hparams.llm_ffn_op = llm_ffn_op_type_from_string(hidden_act, LLM_FFN_GEGLU);
    }

    // GGUFs without a classifier pooling type use mean (gte-reranker-modernbert-base)
    if (hparams.pooling_type_cls == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        hparams.pooling_type_cls = LLAMA_POOLING_TYPE_MEAN;
    }

    ml.get_key(LLM_KV_DECISION_BLOCK_COUNT, hparams.n_layer_decision, false);
    if (hparams.n_layer_decision > 0) {
        if (hparams.n_layer_decision >= hparams.n_layer()) {
            throw std::runtime_error("invalid number of decision blocks");
        }
        // the head blocks always use full attention
        for (uint32_t il = hparams.n_layer() - hparams.n_layer_decision; il < hparams.n_layer(); il++) {
            hparams.is_swa_impl[il] = false;
        }
        // the output is one score per question type
        hparams.n_embd_out_impl = N_DECISION_TYPES;
    }

    switch (hparams.n_layer()) {
        case 12:
            type = LLM_TYPE_47M; break; // granite-embedding-small
        case 22:
            type = LLM_TYPE_149M; break; // modern-bert-base
        case 28:
            type = LLM_TYPE_395M; break; // modern-bert-large
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_modern_bert::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    const int n_layer_enc = n_layer - hparams.n_layer_decision;

    for(int i = 0; i < n_layer_enc; ++i) {
        auto& layer = layers[i];

        if ( i != 0 ) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else{
            // layer 0 uses identity
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        }


        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, 3 * n_embd }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT,   "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, 2 * n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
    }

    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT,  "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT,  "bias"),   {hparams.n_cls_out},         TENSOR_NOT_REQUIRED);
    cls       = create_tensor(tn(LLM_TENSOR_CLS,      "weight"), {n_embd, n_embd},            TENSOR_NOT_REQUIRED);
    cls_norm  = create_tensor(tn(LLM_TENSOR_CLS_NORM, "weight"), {n_embd},                    TENSOR_NOT_REQUIRED);

    if (hparams.n_layer_decision == 0) {
        return;
    }

    // decision head: plain pre-norm blocks with biases
    for (int i = n_layer_enc; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_ff_head = hparams.n_ff(i);

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias",   i), {n_embd}, 0);

        layer.wqkv   = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, 3 * n_embd}, 0);
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   i), {3 * n_embd}, 0);
        layer.wo     = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), {n_embd}, 0);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff_head}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {n_ff_head}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_head, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);
    }

    if (n_token_types != N_DECISION_TYPES) {
        throw std::runtime_error("decision model must have one token type per question type");
    }
    type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, 0);

    cls_b      = create_tensor(tn(LLM_TENSOR_CLS,      "bias"), {n_embd}, 0);
    cls_norm_b = create_tensor(tn(LLM_TENSOR_CLS_NORM, "bias"), {n_embd}, 0);

    if (!cls || !cls_norm || !cls_out || !cls_out_b) {
        throw std::runtime_error("decision model is missing the scorer tensors");
    }

}

std::unique_ptr<llm_graph_context> llama_model_modern_bert::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_modern_bert::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * inp_pos = build_inp_pos();

    // construct input embeddings (token, type, position)
    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    // embed layer norm
    inpL = build_norm(inpL, model.tok_norm, nullptr, LLM_NORM, 0);
    cb(inpL, "inp_norm", 0);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_no_cache();

    const int n_layer_dec = hparams.n_layer_decision;
    const int n_layer_enc = n_layer - n_layer_dec;

    for (int il = 0; il < n_layer_enc; ++il) {
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        cur = inpL;

        // attention layer norm
        if (model.layers[il].attn_norm) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);
        }

        // self attention
        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                n_embd_head, n_head, n_head_kv, il);

        // RoPE
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        cb(cur, "kqv_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // re-add the layer input
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // attention layer norm
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                NULL,                      NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                hparams.llm_ffn_op,
                LLM_FFN_SEQ, il);

        // attentions bypass the intermediate layer
        cur = ggml_add(ctx0, cur, ffn_inp);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM, -1);
    cb(cur, "final_norm_out", -1);

    if (n_layer_dec > 0) {
        cur = build_decision_head(model, cur, inp_attn, inp_out_ids);
    }

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}

// returns one score per question type for each output token, the caller reads them at the option markers
ggml_tensor * llama_model_modern_bert::graph::build_decision_head(
        const llama_model & model,
        ggml_tensor * inp,
        llm_graph_input_attn_no_cache * inp_attn,
        ggml_tensor * inp_out_ids) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int     n_layer_enc = n_layer - hparams.n_layer_decision;

    ggml_tensor * scores = nullptr;

    // the question type is not a graph input, so the head is evaluated for each of them
    for (uint32_t it = 0; it < N_DECISION_TYPES; ++it) {
        ggml_tensor * type_row = ggml_view_1d(ctx0, model.type_embd, n_embd, it * model.type_embd->nb[1]);
        ggml_tensor * inpL = ggml_add(ctx0, inp, type_row);

        for (int il = n_layer_enc; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * cur = build_norm(inpL, layer.attn_norm, layer.attn_norm_b, LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // no positional encoding in the head
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

            cur = build_attn(inp_attn,
                        layer.wo, layer.wo_b, layer.wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    layer.ffn_up,   layer.ffn_up_b,   NULL,
                    NULL,           NULL,             NULL,
                    layer.ffn_down, layer.ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_RELU,
                    LLM_FFN_SEQ, il);

            inpL = ggml_add(ctx0, cur, ffn_inp);
        }

        // scorer
        ggml_tensor * cur = build_norm(inpL, model.cls_norm, model.cls_norm_b, LLM_NORM, -1);
        cur = ggml_add(ctx0, build_lora_mm(model.cls, cur), model.cls_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_add(ctx0, build_lora_mm(model.cls_out, cur), model.cls_out_b);

        scores = scores ? ggml_concat(ctx0, scores, cur, 0) : cur;
    }
    cb(scores, "decision_scores", -1);

    return scores;
}
