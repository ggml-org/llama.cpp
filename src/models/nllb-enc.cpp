#include "models.h"

llm_build_nllb_enc::llm_build_nllb_enc(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Token embeddings
    inpL = build_inp_embd(model.tok_embd);
    
    // NLLB uses scaled embeddings: embeddings * sqrt(d_model)
    // This is critical for numerical parity with HuggingFace!
    const float embed_scale = sqrtf((float)hparams.n_embd);
    inpL = ggml_scale(ctx0, inpL, embed_scale);
    cb(inpL, "inp_embd_scaled", -1);
    
    // Add sinusoidal positional embeddings
    // NLLB uses M2M100SinusoidalPositionalEmbedding (pre-computed during conversion)
    // CRITICAL: M2M100 uses an offset of 2 for positions!
    // So actual positions are [2, 3, 4, ...] not [0, 1, 2, ...]
    
    // Get position indices [0, 1, 2, ..., n_tokens-1]
    ggml_tensor * positions = build_inp_pos();
    
    // M2M100 uses an offset of 2, so we need positions [2, 3, 4, ...]
    // We can't easily add a constant in the graph, so instead we'll slice
    // the positional embedding table starting from index 2
    // positions [0,1,2,3,...] will access rows [2,3,4,5,...] of the table
    
    // Get embeddings from rows 2+ of the pre-computed position embedding table
    // model.pos_embd has shape [n_embd, n_ctx_train+2] where the first 2 columns are offset
    // We use a view to skip the first 2 columns
    const int64_t offset_cols = 2;
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_ctx = hparams.n_ctx_train;
    ggml_tensor * pos_embd_offset = ggml_view_2d(ctx0, model.pos_embd,
                                                  n_embd, n_ctx,
                                                  model.pos_embd->nb[1],  // stride (bytes per column)
                                                  offset_cols * model.pos_embd->nb[1]);  // byte offset
    cb(pos_embd_offset, "pos_embd_table_offset", -1);
    
    // Now get rows from the offset table (row 0 of offset table = row 2 of full table)
    ggml_tensor * pos_embd = ggml_get_rows(ctx0, pos_embd_offset, positions);
    cb(pos_embd, "pos_embd", -1);
    
    inpL = ggml_add(ctx0, inpL, pos_embd);
    cb(inpL, "inp_pos", -1);

    // NLLB doesn't use relative position bias like T5, so no pos_bucket needed
    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Encoder layers
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Self-attention layer normalization
        // [AI] Updated API: build_norm now takes (tensor, weight, bias, norm_type, layer)
        cur = build_norm(inpL,
                model.layers[il].attn_norm_enc, model.layers[il].attn_norm_enc_b,
                LLM_NORM, il);
        cb(cur, "attn_norm", il);

        // Self-attention
        {
            // [AI] Note: Biases are now handled by build_lora_mm if tensors exist
            // They should be added via ggml_add if bias tensors are present
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_enc, cur);
            if (model.layers[il].bq_enc) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq_enc);
            }
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_enc, cur);
            if (model.layers[il].bk_enc) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk_enc);
            }
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_enc, cur);
            if (model.layers[il].bv_enc) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv_enc);
            }
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // NLLB encoder uses bidirectional attention without position bias
            // NOTE: kq_scale is the scaling factor for attention scores
            // For NLLB: head_dim = 64, so scale = 1/sqrt(64) = 1/8 = 0.125
            const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
            
            // [AI] Updated API: build_attn takes 9 params
            // (inp, wo, bo, Q, K, V, kq_b, sinks, v_mla, scale, layer)
            cur = build_attn(inp_attn,
                    model.layers[il].wo_enc, model.layers[il].bo_enc,
                    Qcur, Kcur, Vcur, 
                    nullptr,  // kq_b (no position bias for NLLB)
                    nullptr,  // sinks
                    nullptr,  // v_mla
                    kq_scale, il);
            cb(cur, "kqv_out", il);
        }

        // Get rows if needed (for last layer)
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Feed-forward network
        {
            // FFN layer normalization
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm_enc, model.layers[il].ffn_norm_enc_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            // NLLB uses simple feed-forward with ReLU activation (no gating)
            // [AI] Updated API: build_ffn takes 13 params
            // (input, up, up_b, up_s, gate, gate_b, gate_s, down, down_b, down_s, moe, ffn_type, ffn_par, layer)
            cur = build_ffn(cur,
                    model.layers[il].ffn_up_enc,   model.layers[il].ffn_up_enc_b,   nullptr,  // up, up_b, up_s
                    nullptr,                       nullptr,                          nullptr,  // gate, gate_b, gate_s (no gate)
                    model.layers[il].ffn_down_enc, model.layers[il].ffn_down_enc_b, nullptr,  // down, down_b, down_s
                    nullptr,                                                                   // moe
                    LLM_FFN_RELU,  // NLLB uses ReLU
                    LLM_FFN_SEQ,   // Sequential (not parallel)
                    il);
            cb(cur, "ffn_out", il);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        // Control vector
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }

    cur = inpL;
    cb(cur, "result_embd", -1);

    // Final encoder normalization
    cur = build_norm(cur,
            model.output_norm_enc, model.output_norm_enc_b,
            LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
