#include "models.h"

template <bool embed>
llm_build_llama<embed>::llm_build_llama(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // IQuestLoopCoder loop attention support
    // ref: https://github.com/ggerganov/llama.cpp/issues/18517
    // Note: Loop attention is unrolled statically in the graph
    // For loop_num=2, we process all 80 layers twice (total 160 layer operations)
    const uint32_t loop_num = hparams.loop_num > 0 ? hparams.loop_num : 1;
    const bool is_loop_model = hparams.loop_num > 0;

    // Global K/V storage for loop attention (from first loop iteration)
    // Each layer stores its K/V from loop 0 for use in later loops
    std::vector<ggml_tensor *> global_K_store(n_layer, nullptr);
    std::vector<ggml_tensor *> global_V_store(n_layer, nullptr);

    // Process layers with loop unrolling
    for (uint32_t loop_idx = 0; loop_idx < loop_num; ++loop_idx) {
        for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

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

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }

            // IQuestLoopCoder loop attention mechanism
            if (!is_loop_model || loop_idx == 0) {
                // Loop 0 (or non-loop models): Standard attention
                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);

                // Save K/V from loop 0 for later loops
                if (is_loop_model && loop_idx == 0) {
                    global_K_store[il] = Kcur;
                    global_V_store[il] = Vcur;
                }
            } else {
                // Loop 1+: Dual attention with gate mixing
                // Compute local attention (current loop's K/V) WITH output projection
                ggml_tensor * local_attn = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
                cb(local_attn, "local_attn", il);

                // Compute global attention (K/V from loop 0) WITH output projection
                ggml_tensor * global_attn = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, global_K_store[il], global_V_store[il], nullptr, nullptr, nullptr, kq_scale, il);
                cb(global_attn, "global_attn", il);

                // Compute per-head gates: gate = sigmoid(Q @ gate_weight.T + gate_bias)
                // gate_weight stored as [n_embd_head, n_head]
                // Qcur shape: [n_embd_head, n_head, n_tokens]
                // For each head h: sum over d of Q[:,h,:] * gate_weight[:,h]

                // Cast gate_weight to f32 for compatibility with Q (which is f32)
                ggml_tensor * gate_weight_f32 = ggml_cast(ctx0, model.layers[il].loop_gate_weight, GGML_TYPE_F32);
                cb(gate_weight_f32, "gate_weight_f32", il);

                // Simplified approach: element-wise multiply and sum
                // gate_weight: [n_embd_head, n_head] -> reshape to [n_embd_head, n_head, 1]
                ggml_tensor * gate_weight_3d = ggml_view_3d(ctx0, gate_weight_f32,
                        n_embd_head, n_head, 1,
                        gate_weight_f32->nb[1],
                        gate_weight_f32->nb[1] * n_head,
                        0);
                cb(gate_weight_3d, "gate_weight_3d", il);

                // Multiply Q [n_embd_head, n_head, n_tokens] * gate_weight [n_embd_head, n_head, 1]
                // Result: [n_embd_head, n_head, n_tokens]
                ggml_tensor * gate_prod = ggml_mul(ctx0, Qcur, gate_weight_3d);
                cb(gate_prod, "gate_prod", il);

                // Sum over embedding dimension: [n_embd_head, n_head, n_tokens] -> [n_head, n_tokens]
                // Reshape to [n_embd_head, n_head * n_tokens] then sum_rows
                gate_prod = ggml_reshape_2d(ctx0, gate_prod, n_embd_head, n_head * n_tokens);
                ggml_tensor * gate_logits = ggml_sum_rows(ctx0, gate_prod);  // [1, n_head * n_tokens]
                gate_logits = ggml_reshape_2d(ctx0, gate_logits, n_head, n_tokens);
                cb(gate_logits, "gate_logits", il);

                // Add gate bias [n_head] -> broadcast to [n_head, n_tokens]
                // Cast bias to f32 for compatibility
                ggml_tensor * gate_bias_f32 = ggml_cast(ctx0, model.layers[il].loop_gate_bias, GGML_TYPE_F32);
                ggml_tensor * gate_bias_2d = ggml_view_2d(ctx0, gate_bias_f32,
                        n_head, 1,
                        gate_bias_f32->nb[0],
                        0);
                gate_logits = ggml_add(ctx0, gate_logits, gate_bias_2d);
                cb(gate_logits, "gate_logits_biased", il);

                // Apply sigmoid
                ggml_tensor * gates = ggml_sigmoid(ctx0, gate_logits);  // [n_head, n_tokens]
                cb(gates, "gates", il);

                // Mix local and global attention: output = local + gate * (global - local)
                // Use actual tensor dimensions from both tensors
                const int64_t n_embd_out = local_attn->ne[0];  // Actual output embedding dimension
                const int64_t n_tokens_attn = local_attn->ne[1];  // Tokens from attention output

                const int64_t n_head_gates = gates->ne[0];  // Actual n_head from gates
                const int64_t n_tokens_gates = gates->ne[1];  // Actual tokens from gates

                // Compute embedding dimension per head from actual tensor
                const int64_t n_embd_per_head = n_embd_out / n_head_gates;

                // Broadcast gates to match attention output dimensions
                // Step 1: Reshape gates from [n_head_gates, n_tokens_gates] to [n_head_gates, 1, n_tokens_gates]
                ggml_tensor * gates_3d = ggml_reshape_3d(ctx0, gates, n_head_gates, 1, n_tokens_gates);
                cb(gates_3d, "gates_3d", il);

                // Step 2: Create template by reshaping local_attn using actual dimensions
                ggml_tensor * template_3d = ggml_reshape_3d(ctx0, local_attn, n_head_gates, n_embd_per_head, n_tokens_attn);
                cb(template_3d, "template_3d", il);

                // Step 3: Repeat gates along middle dimension
                GGML_ASSERT(n_tokens_gates == n_tokens_attn && "Token count mismatch between gates and attention");

                ggml_tensor * gates_repeated = ggml_repeat(ctx0, gates_3d, template_3d);
                cb(gates_repeated, "gates_repeated", il);

                // Step 4: Make contiguous, then reshape back to 2D
                gates_repeated = ggml_cont(ctx0, gates_repeated);
                ggml_tensor * gates_expanded = ggml_reshape_2d(ctx0, gates_repeated, n_embd_out, n_tokens_attn);
                cb(gates_expanded, "gates_expanded", il);

                // Compute (global - local)
                ggml_tensor * attn_diff = ggml_sub(ctx0, global_attn, local_attn);
                cb(attn_diff, "attn_diff", il);

                // Compute gate * (global - local)
                ggml_tensor * gated_diff = ggml_mul(ctx0, gates_expanded, attn_diff);
                cb(gated_diff, "gated_diff", il);

                // Compute local + gate * (global - local)
                ggml_tensor * mixed_attn = ggml_add(ctx0, local_attn, gated_diff);
                cb(mixed_attn, "mixed_attn", il);

                // Output projection already applied in build_attn calls above
                cur = mixed_attn;
                cb(cur, "attn_out", il);
            }
        }
        // IQuestLoopCoder: Only apply ggml_get_rows at the very end (after all loops)
        // to avoid breaking subsequent loop iterations
        if (il == n_layer - 1 && inp_out_ids && loop_idx == loop_num - 1) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    } // end loop iteration (IQuestLoopCoder)

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

template struct llm_build_llama<false>;
template struct llm_build_llama<true>;
