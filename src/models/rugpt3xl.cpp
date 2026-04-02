#include "models.h"

llm_build_rugpt3xl::llm_build_rugpt3xl(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const bool has_sparse = hparams.sparse_block_size > 0;

    ggml_tensor * cur;
    ggml_tensor * pos;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_sparse();

    pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
    cb(pos, "pos_embd", -1);

    inpL = ggml_add(ctx0, inpL, pos);
    cb(inpL, "inpL", -1);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        cur = build_norm(inpL,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            const bool is_sparse_layer = has_sparse && (il % 2 == 0);

            if (is_sparse_layer) {
                cur = build_attn_sparse(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, 1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                cur = build_attn_dense(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, 1.0f/sqrtf(float(n_embd_head)), il);
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
