#include "models.h"

llm_build_mbart_dec::llm_build_mbart_dec(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // mBART uses learned positional embeddings
    inpL = build_inp_embd(model.tok_embd);

    // Add positional embeddings
    ggml_tensor * pos_embd = build_inp_pos_embd();
    if (pos_embd) {
        inpL = ggml_add(ctx0, inpL, pos_embd);
        cb(inpL, "pos_embd", -1);
    }

    // Get encoder embeddings for cross-attention
    ggml_tensor * embd_enc = build_inp_cross_embd();
    const int64_t n_outputs_enc = embd_enc->ne[1];

    // Layer normalization before the first layer (mBART characteristic)
    cur = build_norm(inpL,
            model.output_norm, NULL,
            LLM_NORM, -1);
    cb(cur, "input_norm", -1);
    inpL = cur;

    auto * inp_attn_self  = build_attn_inp_kv();
    auto * inp_attn_cross = build_attn_inp_cross();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t dec_n_layer = hparams.dec_n_layer;

    for (int il = 0; il < dec_n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // self-attention
        {
            // norm before attention
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // mBART uses standard scaled dot-product attention
            cur = build_attn(inp_attn_self,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf((float)n_embd_head), il);
            cb(cur, "kqv_out", il);
        }

        // residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "self_attn_out", il);

        ggml_tensor * inpCA = cur;

        // cross-attention
        {
            // norm before cross-attention
            cur = build_norm(cur,
                    model.layers[il].attn_norm_cross, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm_cross", il);

            // Q from decoder, K and V from encoder
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_cross, cur);
            cb(Qcur, "Qcur_cross", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_cross, embd_enc);
            cb(Kcur, "Kcur_cross", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_cross, embd_enc);
            cb(Vcur, "Vcur_cross", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_outputs_enc);

            cur = build_attn(inp_attn_cross,
                    model.layers[il].wo_cross, model.layers[il].bo_cross,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf((float)n_embd_head), il);
            cb(cur, "kqv_cross_out", il);
        }

        if (il == dec_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
        }

        // residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);
        cb(ffn_inp, "cross_attn_out", il);

        // feed-forward network
        {
            // norm before FFN
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            // mBART uses GELU activation
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU,
                    LLM_FFN_SEQ,
                    il);
            cb(cur, "ffn_out", il);
        }

        // residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "layer_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;
    cb(cur, "result_embd", -1);

    // Final layer normalization
    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head for generation
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}