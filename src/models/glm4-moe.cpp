#include "models.h"

llm_build_glm4_moe::llm_build_glm4_moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;

    if (params.mtp_params.op_type != MTP_OP_NONE) {
        ggml_tensor* hidden_states_from_main_model;

        if (params.mtp_params.op_type == MTP_OP_WARMUP || params.mtp_params.op_type == MTP_OP_UPDATE_ACCEPTED) {
            hidden_states_from_main_model = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
            ggml_set_name(hidden_states_from_main_model, "result_embd_pooled");
            ggml_set_input(hidden_states_from_main_model);

            auto inp_mtp = std::make_unique<llm_graph_input_mtp_states>();
            inp_mtp->states = hidden_states_from_main_model;
            res->add_input(std::move(inp_mtp));
        } else {
                hidden_states_from_main_model = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_embd);
                ggml_set_name(hidden_states_from_main_model, "result_embd_pooled");
                ggml_set_input(hidden_states_from_main_model);

                auto inp_mtp = std::make_unique<llm_graph_input_mtp_states>();
                inp_mtp->states = hidden_states_from_main_model;
                res->add_input(std::move(inp_mtp));
        }

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];
        res->t_logits = build_mtp_tail(mtp_layer, hidden_states_from_main_model, n_embd_head, model);

    } else {
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        bool use_mrope = hparams.use_mrope();
        if (ubatch.embd && !use_mrope) {
            // unfortunately, we need to forcefully stop here, to avoid users complaining about wrong results
            GGML_ABORT("This GGUF does not support multimodal. Please reconvert it.");
        }

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        // Only process up to last layer (skip final NextN layer)
        // Final layer tensors are loaded but not processed in forward pass
        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {
            ggml_tensor * inpSA = inpL;

            // Pre-attention norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
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

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                // Apply Q/K norm if available (GLM-4.5 355B variant)
                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                    cb(Qcur, "Qcur_normed", il);
                }
                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                    cb(Kcur, "Kcur_normed", il);
                }

                if (use_mrope) {
                    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow);

                    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                } else {
                    // Normal RoPE
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot,
                                        rope_type, n_ctx_orig, freq_base, freq_scale,
                                        ext_factor, attn_factor, beta_fast, beta_slow);

                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot,
                                        rope_type, n_ctx_orig, freq_base, freq_scale,
                                        ext_factor, attn_factor, beta_fast, beta_slow);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }
            if (il == n_transformer_layers - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // Post-attention norm
            cur = build_norm(ffn_inp, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "post_attn_norm", il);

            // Check if this is a dense layer (n_layer_dense_lead=1, so layer 0 is dense)
            if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
                // Dense FFN layer
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // Process routed experts using existing MoE infrastructure
                ggml_tensor * routed_out = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (llama_expert_gating_func_type) hparams.expert_gating_func,
                        il);
                cb(routed_out, "ffn_moe_out", il);

                // Process shared expert on original input
                ggml_tensor * shared_out = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(shared_out, "ffn_shexp_out", il);

                // Final output: routed_output + shared_output
                cur = ggml_add(ctx0, routed_out, shared_out);
                cb(cur, "ffn_out", il);
            }
            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }
        cur = inpL;
        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, res->t_logits);
}


ggml_tensor * llm_build_glm4_moe::build_mtp_tail(const llama_layer & mtp_layer, ggml_tensor * prev_embeddings, 
    int64_t n_embd_head, const llama_model & model) {
    ggml_tensor * embd_copy = ggml_dup(ctx0, prev_embeddings);
    cb(embd_copy, "mtp_embd_copy", -1);

    const int il = hparams.n_layer - 1;

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();

    // If nextn.embed_tokens is missing (GLM-4.6), use model.tok_embd
    ggml_tensor * mtp_embd_weights = mtp_layer.nextn.embed_tokens;
    if (mtp_embd_weights == nullptr) {
        mtp_embd_weights = model.tok_embd;
    }
    ggml_tensor * token_emb = build_inp_embd_mtp(mtp_embd_weights);

    ggml_tensor * token_emb_norm = build_norm(token_emb, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, il);
    ggml_tensor * hidden_state_norm = build_norm(embd_copy, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, il);
    
    ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
    cb(combined, "mtp_concat", il);
    ggml_tensor* cur = build_lora_mm(mtp_layer.nextn.eh_proj, combined);

    // now proceed through last layer (skipped in main model)
    ggml_tensor * inpSA = cur;
    // Pre-attention norm for the MTP block
    cur = build_norm(cur, mtp_layer.attn_norm, NULL, LLM_NORM_RMS, il);

    // self-attention
    {
        ggml_tensor * Qcur = build_lora_mm(mtp_layer.wq, cur);
        if (mtp_layer.bq) Qcur = ggml_add(ctx0, Qcur, mtp_layer.bq);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(mtp_layer.wk, cur);
        if (mtp_layer.bk) Kcur = ggml_add(ctx0, Kcur, mtp_layer.bk);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(mtp_layer.wv, cur);
        if (mtp_layer.bv) Vcur = ggml_add(ctx0, Vcur, mtp_layer.bv);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // Apply Q/K norm if available (GLM-4.5 355B variant)
        if (mtp_layer.attn_q_norm) {
            Qcur = build_norm(Qcur, mtp_layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
        }
        if (mtp_layer.attn_k_norm) {
            Kcur = build_norm(Kcur, mtp_layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);
        }

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(inp_attn,
                mtp_layer.wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, mtp_layer.attn_post_norm, NULL, LLM_NORM_RMS, il);

    // moe ffn for nextn block
    {
        // Process routed experts using existing MoE infrastructure
        ggml_tensor * routed_out = build_moe_ffn(cur,
                mtp_layer.ffn_gate_inp,
                mtp_layer.ffn_up_exps,
                mtp_layer.ffn_gate_exps,
                mtp_layer.ffn_down_exps,
                mtp_layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                true, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(routed_out, "ffn_moe_out", il);

        // Process shared expert on original input
        ggml_tensor * shared_out = build_ffn(cur,
                mtp_layer.ffn_up_shexp,   NULL, NULL,
                mtp_layer.ffn_gate_shexp, NULL, NULL,
                mtp_layer.ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(shared_out, "ffn_shexp_out", il);

        // Final output: routed_output + shared_output
        cur = ggml_add(ctx0, routed_out, shared_out);
        cb(cur, "ffn_out", il);
    }
    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_ffn_out_resid", il);
    cur = build_norm(cur, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, il);

    // If nextn.shared_head_head is missing (GLM-4.6), use model.output (Main LM Head)
    ggml_tensor * mtp_head_weights = mtp_layer.nextn.shared_head_head;
    if (mtp_head_weights == nullptr) {
        mtp_head_weights = model.output;
    }
    cur = build_lora_mm(mtp_head_weights, cur);

    return cur;
}
