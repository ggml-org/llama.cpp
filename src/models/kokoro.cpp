#include "models.h"

// backbone of the Kokoro TTS pipeline: the PL-BERT phoneme encoder (an ALBERT model)
// - token and position embeddings have a small width, a projection brings them to n_embd
// - one layer is stored and applied n_layer times
// it has no lm_head, the rest of the model is inside the mmproj

void llama_model_kokoro::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_kokoro::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    // the width of the token and position embeddings comes from the weights
    const ggml_tensor * pos_meta = ml.get_tensor_meta(tn(LLM_TENSOR_POS_EMBD, "weight").str().c_str());
    const int64_t n_embd_tok = pos_meta ? pos_meta->ne[0] : n_embd;

    // the converter pads the rows with zeros from n_embd_tok to n_embd, token and embedding inputs must have the same width
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    // the token type embedding is folded into the positions by the converter
    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd_tok, n_ctx_train}, 0);

    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd_tok}, 0);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0),   {n_embd_tok}, 0);

    tok_embd_proj   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_PROJ, "weight", 0), {n_embd_tok, n_embd}, 0);
    tok_embd_proj_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_PROJ, "bias",   0),   {n_embd}, 0);

    // no output head, the logits are unused; reuse the embedding table so a sampler can still run
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    // the layers share their weights, only the first one is stored
    {
        auto & layer = layers[0];
        const int i = 0;

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);

        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd}, 0);

        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_kokoro::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_kokoro::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const auto & model = static_cast<const llama_model_kokoro &>(this->model);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_embd_tok  = model.pos_embd->ne[0];

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    ggml_tensor * inp_pos  = build_inp_pos();
    ggml_tensor * embd_pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
    ggml_build_forward_order(gf, embd_pos);

    // token + position (+ type) embeddings, in the small width
    inpL = build_inp_embd(model.tok_embd);
    inpL = ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd_tok, n_tokens, inpL->nb[1], 0));
    inpL = ggml_add(ctx0, inpL, embd_pos);
    cb(inpL, "inp_embd", -1);

    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, 0);
    cb(inpL, "inp_norm", 0);

    inpL = ggml_add(ctx0, ggml_mul_mat(ctx0, model.tok_embd_proj, inpL), model.tok_embd_proj_b);
    cb(inpL, "inp_proj", 0);

    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const auto & layer = model.layers[0];

    for (int il = 0; il < n_layer; ++il) {
        cur = inpL;

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur,
                    n_embd_head, n_head, n_head_kv, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    layer.wo, layer.wo_b, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpL);
        cur = build_norm(cur, layer.attn_out_norm, layer.attn_out_norm_b, LLM_NORM, il);

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        cur = build_ffn(cur,
                layer.ffn_up,   layer.ffn_up_b,   NULL,
                NULL,           NULL,             NULL,
                layer.ffn_down, layer.ffn_down_b, NULL, NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_norm(cur, layer.layer_out_norm, layer.layer_out_norm_b, LLM_NORM, il);

        inpL = cur;
    }

    cur = inpL;

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
