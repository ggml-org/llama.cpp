#include "models.h"

// EAGLE3 Encoder: processes target model features through feature fusion layer
// Input: target_features e.g. [12288, n_tokens] from target model layers low, middle, high
// Output: g_embeddings e.g. [4096, n_tokens] stored in context
llm_build_eagle3_encode::llm_build_eagle3_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {

        const int64_t n_embd_target_features = 3 * hparams.eagle3_target_hidden_size;  

        ggml_tensor * cur;

        // Input: Target model features (3 layers concatenated: low, mid, high)
        // Data will be provided via ubatch->embd in encode_eagle3_features()
        auto inp_target = std::make_unique<llm_graph_input_embd>();
        inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_target_features, n_tokens);
        ggml_set_input(inp_target->embd);
        ggml_tensor * target_features = inp_target->embd;
        res->add_input(std::move(inp_target));
        cb(target_features, "inp_target_features", -1);

        // Feature fusion layer
        ggml_tensor * fused_target = build_lora_mm(model.fc, target_features);
        cb(fused_target, "fc_out", -1);

        // Output: g_embeddings e.g. [4096, n_tokens]
        cur = fused_target;
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
        ggml_tensor * input_embeds = build_inp_embd(token_embd_eagle3);
        cb(input_embeds, "token_embd_eagle3", -1);
        ggml_tensor * g_embeddings = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(g_embeddings);
        ggml_set_name(g_embeddings, "inp_g_embeddings");
        cb(g_embeddings, "inp_g_embeddings", -1);

        // Store raw g_embeddings as residual
        ggml_tensor * residual = g_embeddings;

        // Apply input_layernorm to the token embeddings
        ggml_tensor * input_embeds_normed = build_norm(input_embeds,
                model.layers[0].attn_norm, NULL,
                LLM_NORM_RMS, 0);
        cb(input_embeds_normed, "input_layernorm", -1);

        // Force a sync point between the two parallel RMS_NORM paths
        // This prevents buffer reuse issues on GPU (EAGLE3 GPU fix)
        ggml_set_sync(input_embeds_normed);

        // Apply hidden_norm to g_embeddings
        ggml_tensor * g_embeddings_normed = build_norm(g_embeddings,
                model.layers[0].eagle3_hidden_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(g_embeddings_normed, "g_embeddings_normed", -1);

        // Concatenate normalized input_embeds and normalized g_embeddings
        cur = ggml_concat(ctx0, input_embeds_normed, g_embeddings_normed, 0);
        cb(cur, "concat_embeds_g", -1);
        
        inpL = cur;

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

        // Single decoder layer (il = 0)
        const int il = 0;
        {
        // inpL is the concatenated input (normalized input_embeds + normalized g_embeddings)
        ggml_tensor * inpSA = inpL;

        // Self-attention with concatenated input
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, inpL);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, inpL);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, inpL);
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
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        // Add residual and update it
        ggml_tensor * attn_with_residual = ggml_add(ctx0, cur, residual);
        cb(attn_with_residual, "attn_with_residual", il);
        
        // Update residual
        residual = attn_with_residual;
        
        // Apply FFN norm to the sum
        ggml_tensor * ffn_inp = build_norm(attn_with_residual,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(ffn_inp, "post_attn_norm", il);

        cur = ffn_inp;

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
        
        inpL = cur;
        }

        cur = inpL;

        // Output norm with residual
        ggml_tensor * final_with_residual = ggml_add(ctx0, cur, residual);
        cb(final_with_residual, "eagle3_prenorm", -1);
        
        // Output prenorm state (for next token's g_embeddings in autoregressive generation)
        ggml_set_output(final_with_residual);
        res->t_embd = final_with_residual;
        
        cur = build_norm(final_with_residual,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head - projects to draft vocabulary
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
}