#include "models.h"

llm_build_mbart_enc::llm_build_mbart_enc(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // mBART uses learned positional embeddings
    inpL = build_inp_embd(model.tok_embd);

    // Add positional embeddings for mBART
    ggml_tensor * pos_embd = build_inp_pos_embd();
    if (pos_embd) {
        inpL = ggml_add(ctx0, inpL, pos_embd);
        cb(inpL, "pos_embd", -1);
    }

    // Layer normalization before the first layer (mBART characteristic)
    cur = build_norm(inpL,
            model.output_norm_enc, NULL,
            LLM_NORM, -1);
    cb(cur, "input_norm", -1);
    inpL = cur;

    auto * inp_attn = build_attn_inp_no_cache();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // self-attention (mBART uses pre-norm)
        {
            // norm before attention
            cur = build_norm(inpL,
                    model.layers[il].attn_norm_enc, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_enc, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_enc, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_enc, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // mBART uses standard scaled dot-product attention without relative position bias
            cur = build_attn(inp_attn,
                    model.layers[il].wo_enc, model.layers[il].bo_enc,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf((float)n_embd_head), il);
            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "attn_out", il);

        // feed-forward network
        {
            // norm before FFN
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm_enc, NULL,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            // mBART uses GELU activation
            cur = build_ffn(cur,
                    model.layers[il].ffn_up_enc,   NULL, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down_enc, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU,
                    LLM_FFN_SEQ,
                    il);
            cb(cur, "ffn_out", il);
        }

        // residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;
    cb(cur, "result_embd", -1);

    // Final layer normalization
    cur = build_norm(cur,
            model.output_norm_enc, NULL,
            LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}