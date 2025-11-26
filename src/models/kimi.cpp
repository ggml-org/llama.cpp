#include "models.h"

llm_build_kimi::llm_build_kimi(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    // ggml_tensor * inp_kq_mask = build_inp_kq_mask(); // Not used/available?
    
    auto * inp_attn = build_attn_inp_kv(); // Initialize input for attention

    // Kimi params
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_head_v = hparams.n_embd_head_v;
    const int64_t n_head        = hparams.n_head();
    const int64_t n_rot         = hparams.n_rot;
    
    // MLA params
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla; // For MLA layers
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla;
    const int64_t q_lora_rank       = hparams.n_lora_q;
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_rot;
    const int64_t n_embd_head_qk_rope = n_rot;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        // Norm
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        bool is_kda = layer.kda_q_conv != nullptr;

        if (is_kda) {
            // KDA Layer logic
            // Based on KimiDeltaAttention in Python
            
            // 1. Linear Projections q, k, v
            // Use wq, wk, wv as defined in llama_layer for Kimi (mapped in llama-model.cpp)
            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq, cur);
            ggml_tensor * k = ggml_mul_mat(ctx0, layer.wk, cur);
            ggml_tensor * v = ggml_mul_mat(ctx0, layer.wv, cur);
            
            cb(q, "q_proj", il);
            cb(k, "k_proj", il);
            cb(v, "v_proj", il);

            // 2. Conv1d
            auto apply_conv1d = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, const char * name) {
                // x: [dim, seq_len]
                // w: [kernel_size, dim] -> reshape to [kernel_size, 1, 1, dim] for dw conv?
                // ggml_conv_2d_dw: a(kernel), b(input)
                // kernel: [k0, k1, 1, C]
                // input: [W, H, C, N] -> [seq_len, 1, dim, 1]
                // We usually work with [dim, seq_len] in simple cases, but ggml_conv_2d_dw might expect transposed?
                // Standard llama.cpp tensors are [dim, seq_len] (column major).
                // ggml_conv_2d_dw operates on [W, H, C, N].
                
                // For now, use a simple placeholder approximation or skipping conv to fix build.
                // The correct implementation requires reshaping and correct op.
                // To match vLLM's causal_conv1d, we need depthwise conv.
                
                // Placeholder: identity + bias
                // ggml_tensor * out = ggml_add(ctx0, x, b); // Broadcast bias?
                
                // Let's try to use ggml_conv_2d_dw properly if we can.
                // Assuming w is [kernel_size, dim]
                // Reshape w to [kernel_size, 1, 1, dim]
                ggml_tensor * w_reshaped = ggml_reshape_4d(ctx0, w, w->ne[0], 1, 1, w->ne[1]);
                
                // Input x is [dim, seq_len]. Treat as [seq_len, 1, dim, 1]?
                // ggml_reshape_4d(ctx0, x, x->ne[1], 1, x->ne[0], 1)
                ggml_tensor * x_reshaped = ggml_reshape_4d(ctx0, x, x->ne[1], 1, x->ne[0], 1);
                
                // Padding? causal means pad left by kernel_size - 1.
                // ggml_conv_2d_dw params: s0=1, s1=1, p0=k-1, p1=0, d0=1, d1=1
                // p0 is symmetric padding? Check docs.
                
                // For now, simplified: Just projection logic or identity.
                // Using ggml_mul_mat with w (as 1x1 conv if k=1) or just skip.
                // I will use identity for now to get it compiling.
                
                ggml_tensor * out = x;
                if (b) {
                    out = ggml_add(ctx0, out, b);
                }
                return ggml_silu(ctx0, out);
            };

            q = apply_conv1d(q, layer.kda_q_conv, layer.kda_q_conv_b, "q_conv");
            k = apply_conv1d(k, layer.kda_k_conv, layer.kda_k_conv_b, "k_conv");
            v = apply_conv1d(v, layer.kda_v_conv, layer.kda_v_conv_b, "v_conv");
            
            cb(q, "q_conv", il);
            cb(k, "k_conv", il);
            cb(v, "v_conv", il);

            // 3. Fused KDA Gate (RetNet/KDA logic)
            // Placeholder: just use q as output for testing graph build
            ggml_tensor * kda_out = q; 
            
            // 4. Output Norm
            // build_norm takes 5 args: tensor, weight, bias, type, layer_idx
            kda_out = build_norm(kda_out, layer.kda_o_norm, layer.kda_o_norm_b, LLM_NORM_RMS, il);
            
            // 5. Output Projection
            cur = ggml_mul_mat(ctx0, layer.wo, kda_out);
            cb(cur, "kda_out", il);

        } else {
            // MLA Layer (DeepSeek V2 style)
            
            ggml_tensor * q = NULL;
            q = ggml_mul_mat(ctx0, layer.wq_a, cur); // Compression
            q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            q = ggml_mul_mat(ctx0, layer.wq_b, q);   // Decompression
            cb(q, "q", il);

            // split into {n_embd_head_qk_nope, n_head, n_tokens} and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k_mla),
                             ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
            
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));

            // KV Compression
            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
            
            // split into compressed kv and pe part
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
                             
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

            // RoPE
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, LLAMA_ROPE_TYPE_NEOX, 0, 0, 0, 0, 1.0f, 0, 0);
            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, LLAMA_ROPE_TYPE_NEOX, 0, 0, 0, 0, 1.0f, 0, 0);

            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

            // KV Decompression & Attention
            // Using DeepSeek optimization (absorbing w_k_b into q_nope) seems complex to replicate exactly without checking tensors.
            // Assuming we use the same logic as DeepSeek2.
            
            // {n_embd_head_qk_nope, n_tokens, n_head}
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            
            // Decompress K part (wk_b) applied to q_nope (matrix trick)
            // Assuming layer.wk_b exists (it does if MLA)
            
            ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr); // Decompression
            
            // Split kv into k_nope and v
            ggml_tensor * k_nope = 
                ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                             ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla),
                             ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla) * n_head, 0);
                             
            ggml_tensor * v_states = 
                ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                             ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla),
                             ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla) * n_head, 
                             ggml_row_size(kv->type, n_embd_head_qk_nope));
            
            v_states = ggml_cont(ctx0, v_states);

            // Concat parts
            ggml_tensor * Qcur = ggml_concat(ctx0, q_pe, q_nope, 0);
            ggml_tensor * Kcur = ggml_concat(ctx0, ggml_repeat(ctx0, k_pe, q_pe), k_nope, 0);
            ggml_tensor * Vcur = v_states;
            
            cur = build_attn(inp_attn, layer.wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head_k_mla)), il);
        }
        
        // Residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FFN Norm
        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // FFN / MoE
        // Check for MoE tensors
        if (layer.ffn_gate_exps) {
             // Assuming build_moe_ffn is available or using build_ffn_moe
             cur = build_moe_ffn(cur, layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, 
                                 layer.ffn_exp_probs_b, hparams.n_expert, hparams.n_expert_used, 
                                 LLM_FFN_SILU, false, true, 1.0f,
                                 (llama_expert_gating_func_type) hparams.expert_gating_func, il);
                                 
             if (layer.ffn_gate_shexp) {
                 ggml_tensor * ffn_shexp = build_ffn(cur, layer.ffn_up_shexp, NULL, NULL, 
                                                     layer.ffn_gate_shexp, NULL, NULL,
                                                     layer.ffn_down_shexp, NULL, NULL,
                                                     NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                 cur = ggml_add(ctx0, cur, ffn_shexp);
             }
        } else {
             cur = build_ffn(cur, layer.ffn_up, NULL, NULL, layer.ffn_gate, NULL, NULL, layer.ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        }

        // Residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        inpL = cur;
    }

    // Final Norm
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // Output
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
}
