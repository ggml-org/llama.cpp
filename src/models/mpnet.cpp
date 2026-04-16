#include "models.h"

llm_build_mpnet::llm_build_mpnet(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // position input for absolute position embeddings
    ggml_tensor * inp_pos = build_inp_pos();

    // construct input embeddings (token + position)
    inpL = build_inp_embd(model.tok_embd);

    // add absolute position embeddings
    inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
    cb(inpL, "inp_embd", -1);

    // embed layer norm
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
    cb(inpL, "inp_norm", -1);

    // relative position buckets for attention bias
    ggml_tensor * pos_bucket_enc = build_inp_pos_bucket_enc();

    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * cur = inpL;

        // self-attention with separate Q, K, V projections
        {
            ggml_tensor * Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, cur), model.layers[il].bq);
            ggml_tensor * Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, cur), model.layers[il].bk);
            ggml_tensor * Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, cur), model.layers[il].bv);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // relative position bias (shared across layers, stored on layer 0)
            ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
            ggml_tensor * kq_b = build_pos_bias(pos_bucket_enc, attn_rel_b);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // residual + attention output layer norm
        cur = ggml_add(ctx0, cur, inpL);
        cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network: up -> GELU -> down
        cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        // residual + output layer norm
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
