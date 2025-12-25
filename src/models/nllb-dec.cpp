#include "models.h"

llm_build_nllb_dec::llm_build_nllb_dec(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Token embeddings
    inpL = build_inp_embd(model.tok_embd);
    
    // NLLB decoder uses same embedding scaling as encoder: embeddings * sqrt(d_model)
    const float embed_scale = sqrtf((float)hparams.n_embd);
    inpL = ggml_scale(ctx0, inpL, embed_scale);
    cb(inpL, "inp_embd_scaled", -1);

    // Add sinusoidal positional embeddings with M2M100 offset
    // Decoder uses the SAME positional embedding table as encoder
    {
        const int64_t offset = 2;
        const int64_t n_embd = model.pos_embd->ne[0];
        const int64_t n_positions_total = model.pos_embd->ne[1];
        const int64_t n_positions_usable = n_positions_total - offset;
        
        // Create view starting at column 'offset' (skip first 2 columns)
        ggml_tensor * pos_embd_table = ggml_view_2d(
            ctx0, 
            model.pos_embd,
            n_embd,
            n_positions_usable,
            model.pos_embd->nb[1],
            offset * model.pos_embd->nb[1]
        );
        
        ggml_tensor * positions = build_inp_pos();
        ggml_tensor * pos_embd = ggml_get_rows(ctx0, pos_embd_table, positions);
        cb(pos_embd, "pos_embd", -1);
        
        inpL = ggml_add(ctx0, inpL, pos_embd);
        cb(inpL, "inp_pos", -1);
    }

    // Encoder embeddings for cross-attention
    ggml_tensor * embd_enc = build_inp_cross_embd();

    // NLLB doesn't use relative position bias like T5
    const int64_t n_outputs_enc = embd_enc->ne[1];
    
    // Attention scaling factor (same as encoder)
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    // Attention inputs
    auto * inp_attn_self  = build_attn_inp_kv();
    auto * inp_attn_cross = build_attn_inp_cross();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t dec_n_layer = hparams.dec_n_layer;

    // Decoder layers
    for (int il = 0; il < dec_n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Self-attention layer normalization
        // [AI] Updated API: build_norm now takes (tensor, weight, bias, norm_type, layer)
        cur = build_norm(inpL,
                model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                LLM_NORM, il);
        cb(cur, "attn_norm", il);

        // Self-attention (causal/masked)
        {
            // [AI] Note: Biases are handled separately with ggml_add
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
            }
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
            }
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
            }
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // NLLB decoder uses causal attention without position bias
            // [AI] Updated API: build_attn takes 9 params
            cur = build_attn(inp_attn_self,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, 
                    nullptr,  // kq_b (no position bias for NLLB)
                    nullptr,  // sinks
                    nullptr,  // v_mla
                    kq_scale, il);
            cb(cur, "kqv_out", il);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "cross_inp", il);

        ggml_tensor * inpCA = cur;

        // Cross-attention layer normalization
        cur = build_norm(cur,
                model.layers[il].attn_norm_cross, model.layers[il].attn_norm_cross_b,
                LLM_NORM, il);
        cb(cur, "attn_norm_cross", il);

        // Cross-attention (decoder attends to encoder output)
        {
            // Query from decoder
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_cross, cur);
            if (model.layers[il].bq_cross) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq_cross);
            }
            cb(Qcur, "Qcur", il);

            // Key and Value from encoder output
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_cross, embd_enc);
            if (model.layers[il].bk_cross) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk_cross);
            }
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_cross, embd_enc);
            if (model.layers[il].bv_cross) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv_cross);
            }
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_outputs_enc);

            // [AI] Updated API
            cur = build_attn(inp_attn_cross,
                    model.layers[il].wo_cross, model.layers[il].bo_cross,
                    Qcur, Kcur, Vcur, 
                    nullptr,  // kq_b (no position bias for NLLB)
                    nullptr,  // sinks
                    nullptr,  // v_mla
                    1.0f, il);
            cb(cur, "kqv_out", il);
        }

        // Get rows if needed (for last layer)
        if (il == dec_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);
        cb(ffn_inp, "ffn_inp", il);

        // Feed-forward network
        {
            // FFN layer normalization
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            // NLLB uses simple feed-forward with ReLU activation (no gating)
            // [AI] Updated API: build_ffn takes 13 params
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,  // up, up_b, up_s
                    nullptr,                   nullptr,                      nullptr,  // gate, gate_b, gate_s (no gate)
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,  // down, down_b, down_s
                    nullptr,                                                           // moe
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

    // Final decoder normalization
    cur = build_norm(cur,
            model.output_norm, model.output_norm_b,
            LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head (output projection)
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
