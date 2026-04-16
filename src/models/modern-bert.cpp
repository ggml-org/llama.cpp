#include "models.h"

llm_build_modern_bert::llm_build_modern_bert(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
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

    for (int il = 0; il < n_layer; ++il) {
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
                LLM_FFN_GEGLU, LLM_FFN_SEQ, il);

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

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}
