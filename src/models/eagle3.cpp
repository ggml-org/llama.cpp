#include "models.h"

ggml_tensor * llm_build_eagle3_encode::build_inp_embd() const {
    const int64_t n_embd_target_features = 3 * hparams.eagle3_target_hidden_size;

    ggml_tensor * cur = nullptr;

    // Input: Target model features (3 layers concatenated: low, mid, high)
    // Data will be provided via ubatch->embd in encode_eagle3_features()
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_target_features);
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_target_features, n_tokens);
    ggml_set_input(inp_target->embd);

    cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// EAGLE3 Encoder: processes target model features through feature fusion layer
// Input: target_features e.g. [12288, n_tokens] from target model layers low, middle, high
// Output: g_embeddings e.g. [4096, n_tokens] stored in context
llm_build_eagle3_encode::llm_build_eagle3_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = nullptr;

    cur = build_inp_embd();

    // Feature fusion layer
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    // Output: g_embeddings e.g. [4096, n_tokens]
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

// EAGLE3 Decoder: processes draft tokens using g_embeddings from encoder
// Input: draft tokens + g_embeddings from encoder
// Output: draft logits
llm_build_eagle3_decode::llm_build_eagle3_decode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_layer == 1);  // EAGLE-3 has only one decoder layer

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // EAGLE3 Decoder receives:
    // 1. Token embeddings (e.g.from EAGLE3's own tok_embd for Llama 3.3 70B, or target model for Llama 3.1 8B)
    // 2. g_embeddings from encoder
    // Choose token_embd_eagle3: prefer EAGLE3's own if available (Llama 3.3 70B), else use target's (Llama 3.1 8B)
    ggml_tensor * token_embd_eagle3 = (model.tok_embd != nullptr) ? model.tok_embd : model.target_tok_embd;
    GGML_ASSERT(token_embd_eagle3 != nullptr && "EAGLE3 decoder requires token embeddings (own or from target model)");
    ggml_tensor * inp_embd = build_inp_embd(token_embd_eagle3);
    cb(inp_embd, "inp_embd", -1);

    // TODO: refactor into llm_graph_input
    ggml_tensor * inp_g = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp_g);
    cb(inp_g, "inp_g_embeddings", -1); // TODO: do not change the name! refactor into llm_graph_input

    inpL = inp_g;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Single decoder layer (il = 0)
    const int il = 0;
    {
        // Apply input_layernorm to the token embeddings
        ggml_tensor * embd_norm = build_norm(inp_embd,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(embd_norm, "embd_norm", il);

        // Apply hidden_norm to inp_g
        ggml_tensor * g_norm = build_norm(inp_g,
                model.layers[il].eagle3_hidden_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(g_norm, "g_norm", il);

        // norm_before_residual: determines what goes into the residual connection (compatible with Readhat eagle3 speculator model)
        // - false (default): use raw inp_g for residual
        // - true: use normalized g_norm for residual
        // inpL is the concatenated input (normalized inp_embd + normalized inp_g)
        ggml_tensor * inpSA = hparams.eagle3_norm_before_residual ? g_norm : inpL;

        // Concatenate normalized inp_embd and normalized inp_g
        cur = ggml_concat(ctx0, embd_norm, g_norm, il);
        cb(cur, "concat_embd", il);

        // Self-attention with concatenated input
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // rope freq factors, returns nullptr if not available
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // RoPE
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        cur = build_attn(inp_attn,
                model.layers[il].wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        if (inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Add residual and update it
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Apply FFN norm to the sum
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // Output norm with residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "eagle3_prenorm", il);

        inpL = cur;
    }

    cur = inpL;

    // Output prenorm state (for next token's g_embeddings in autoregressive generation)
    ggml_set_output(cur);
    res->t_embd = cur;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head - projects to draft vocabulary
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
