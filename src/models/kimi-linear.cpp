#include "models.h"

llm_build_kimi_linear::llm_build_kimi_linear(const llama_model & model, const llm_graph_params & params) : llm_graph_context_mamba(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    
    // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
    // So we don't need inp_pos
    
    // Only use recurrent state input for KDA layers
    // MLA layers use direct softmax attention without KV cache
    auto * inp_rs = build_rs_inp();
    
    // Input for MLA layers (no KV cache)
    auto * inp_no_cache = build_attn_inp_no_cache();

    // Output ids for selecting which tokens to output
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Kimi dimension constants
    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.kda_head_dim > 0 ? hparams.kda_head_dim : 128;
    const int64_t d_conv = hparams.kda_d_conv > 0 ? hparams.kda_d_conv : 4;
    const int64_t d_inner = n_head * head_dim;  // 32 * 128 = 4096
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    
    // Verify batch consistency for recurrent layers
    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    
    // MLA params
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla > 0 ? hparams.n_embd_head_k_mla : 192;
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla > 0 ? hparams.n_embd_head_v_mla : 128;
    const int64_t kv_lora_rank = hparams.n_lora_kv > 0 ? hparams.n_lora_kv : 512;
    // qk_rope_head_dim = 64 (from Kimi config), NOT hparams.n_rot (which is 72)
    // Confirmed from tensor shape: wkv_a_mqa [2304, 576] = [n_embd, kv_lora_rank + qk_rope_head_dim]
    const int64_t n_embd_head_qk_rope = 64;  // config.qk_rope_head_dim
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;  // 192 - 64 = 128
    
    // Attention scale for KDA (1/sqrt(head_dim))
    const float kq_scale_kda = 1.0f / sqrtf((float)head_dim);
    
    // Attention scale for MLA
    const float kq_scale_mla = 1.0f / sqrtf((float)n_embd_head_k_mla);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        // Attention Norm
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Check layer type by checking which tensors exist
        // KDA layers have ssm_a_log tensor, MLA layers have wkv_a_mqa tensor
        bool is_kda = (layer.ssm_a_log != nullptr);
        bool is_mla = (layer.wkv_a_mqa != nullptr);
        
        if (is_kda) {
            // === KDA Layer (Kimi Delta Attention) with Recurrent State ===
            // Reference: vLLM kda.py
            
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();
            
            // Get conv states from r_l tensor (Q, K, V each have separate state)
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            const int64_t conv_state_size = (d_conv - 1) * d_inner;
            const int64_t n_embd_r_total = 3 * conv_state_size;  // Q + K + V
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            // conv_state_all is [n_embd_r_total, n_seqs], split into Q, K, V
            // Each conv state is [(d_conv-1) * d_inner] per sequence, need to reshape to [d_conv-1, d_inner, n_seqs]
            // Memory layout: for each seq, Q state is first conv_state_size elements, then K, then V
            // conv_state_all has stride: nb[0] = element_size, nb[1] = n_embd_r_total * element_size
            
            // View Q conv state: offset 0, size conv_state_size per seq
            // conv_state_all is [n_embd_r_total, n_seqs] with memory layout:
            //   state[i + seq * n_embd_r_total] where i = conv_step + channel * (d_conv-1) + {0, conv_state_size, 2*conv_state_size} for Q/K/V
            // We want [d_conv-1, d_inner, n_seqs] view:
            //   nb1 = (d_conv-1) * element_size (stride between channels)
            //   nb2 = n_embd_r_total * element_size (stride between seqs)
            ggml_tensor * conv_state_q = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_state_all),  // nb1: stride between channels
                n_embd_r_total * ggml_element_size(conv_state_all),  // nb2: stride between seqs
                0);  // offset for Q
            ggml_tensor * conv_state_k = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_state_all),
                n_embd_r_total * ggml_element_size(conv_state_all),
                conv_state_size * ggml_element_size(conv_state_all));  // offset for K
            ggml_tensor * conv_state_v = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_state_all),
                n_embd_r_total * ggml_element_size(conv_state_all),
                2 * conv_state_size * ggml_element_size(conv_state_all));  // offset for V
            
            // Step 1: Q, K, V projections -> [d_inner, n_tokens]
            ggml_tensor * q_proj = ggml_mul_mat(ctx0, layer.wq, cur);
            ggml_tensor * k_proj = ggml_mul_mat(ctx0, layer.wk, cur);
            ggml_tensor * v_proj = ggml_mul_mat(ctx0, layer.wv, cur);
            cb(q_proj, "kda_q_proj", il);
            cb(k_proj, "kda_k_proj", il);
            cb(v_proj, "kda_v_proj", il);
            
            // Step 2: Causal Conv1d for Q
            // Reshape input: {d_inner, n_tokens} -> {d_inner, n_seq_tokens, n_seqs}
            ggml_tensor * q_3d = ggml_reshape_3d(ctx0, q_proj, d_inner, n_seq_tokens, n_seqs);
            
            // Concat Q conv state and current input: {d_conv-1 + n_seq_tokens, d_inner, n_seqs}
            ggml_tensor * conv_q = ggml_concat(ctx0, conv_state_q, ggml_transpose(ctx0, q_3d), 0);
            
            // Save last (d_conv-1) columns back to Q conv state
            ggml_tensor * last_conv_q = ggml_view_3d(ctx0, conv_q, d_conv - 1, d_inner, n_seqs,
                conv_q->nb[1], conv_q->nb[2], n_seq_tokens * conv_q->nb[0]);
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, last_conv_q,
                    ggml_view_1d(ctx0, conv_states_all, conv_state_size * n_seqs,
                        kv_head * n_embd_r_total * ggml_element_size(conv_states_all))));
            
            // Reshape conv weight: GGUF [d_conv, 1, d_inner, 1] -> ggml_ssm_conv expects [d_conv, d_inner]
            // GGUF stores as [d_conv, 1, d_inner, 1] with memory layout w[conv_step + channel * d_conv]
            // vLLM stores as [d_inner, d_conv] with memory layout w[channel * d_conv + conv_step]
            // ggml_ssm_conv computes: c[conv_step + channel * d_conv]
            // GGUF layout: [d_conv, 1, d_inner] or [d_conv, 1, d_inner, 1] -> reshape to [d_conv, d_inner]
            ggml_tensor * conv_weight = nullptr;
            if (layer.ssm_q_conv) {
                // Reshape conv weight from [d_conv, 1, d_inner, 1] to [d_conv, d_inner] for ggml_ssm_conv
                // Cast to F32 if quantized (ggml_ssm_conv requires float weights)
                ggml_tensor * q_conv_f32 = layer.ssm_q_conv;
                if (q_conv_f32->type != GGML_TYPE_F32) {
                    q_conv_f32 = ggml_cast(ctx0, q_conv_f32, GGML_TYPE_F32);
                }
                conv_weight = ggml_reshape_2d(ctx0, q_conv_f32, d_conv, d_inner);
            }
            
            // Apply conv1d
            ggml_tensor * Qcur;
            if (conv_weight) {
                // Make conv_q contiguous for ggml_ssm_conv
                conv_q = ggml_cont(ctx0, conv_q);
                
                // ggml_ssm_conv output: {d_inner, n_seq_tokens, n_seqs}
                Qcur = ggml_ssm_conv(ctx0, conv_q, conv_weight);
                // Reshape to 2D for bias add: {d_inner, n_tokens}
                Qcur = ggml_reshape_2d(ctx0, Qcur, d_inner, n_tokens);
                if (layer.ssm_q_conv_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.ssm_q_conv_b);
                }
                Qcur = ggml_silu(ctx0, Qcur);
            } else {
                GGML_ABORT("KDA layer missing Q conv weight");
            }
            
            // K conv1d (with separate K conv state)
            ggml_tensor * Kcur;
            if (layer.ssm_k_conv) {
                ggml_tensor * k_3d = ggml_reshape_3d(ctx0, k_proj, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * conv_k = ggml_cont(ctx0, ggml_concat(ctx0, conv_state_k, ggml_transpose(ctx0, k_3d), 0));
                
                // Save K conv state
                ggml_tensor * last_conv_k = ggml_view_3d(ctx0, conv_k, d_conv - 1, d_inner, n_seqs,
                    conv_k->nb[1], conv_k->nb[2], n_seq_tokens * conv_k->nb[0]);
                ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0, last_conv_k,
                        ggml_view_1d(ctx0, conv_states_all, conv_state_size * n_seqs,
                            (kv_head * n_embd_r_total + conv_state_size) * ggml_element_size(conv_states_all))));
                
                ggml_tensor * k_conv_f32 = layer.ssm_k_conv;
                if (k_conv_f32->type != GGML_TYPE_F32) {
                    k_conv_f32 = ggml_cast(ctx0, k_conv_f32, GGML_TYPE_F32);
                }
                ggml_tensor * k_conv_weight = ggml_reshape_2d(ctx0, k_conv_f32, d_conv, d_inner);
                Kcur = ggml_ssm_conv(ctx0, conv_k, k_conv_weight);
                Kcur = ggml_reshape_2d(ctx0, Kcur, d_inner, n_tokens);
                if (layer.ssm_k_conv_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.ssm_k_conv_b);
                }
                Kcur = ggml_silu(ctx0, Kcur);
            } else {
                GGML_ABORT("KDA layer missing K conv weight");
            }
            
            // V conv1d (with separate V conv state)
            ggml_tensor * Vcur;
            if (layer.ssm_v_conv) {
                ggml_tensor * v_3d = ggml_reshape_3d(ctx0, v_proj, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * conv_v = ggml_cont(ctx0, ggml_concat(ctx0, conv_state_v, ggml_transpose(ctx0, v_3d), 0));
                
                // Save V conv state
                ggml_tensor * last_conv_v = ggml_view_3d(ctx0, conv_v, d_conv - 1, d_inner, n_seqs,
                    conv_v->nb[1], conv_v->nb[2], n_seq_tokens * conv_v->nb[0]);
                ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0, last_conv_v,
                        ggml_view_1d(ctx0, conv_states_all, conv_state_size * n_seqs,
                            (kv_head * n_embd_r_total + 2 * conv_state_size) * ggml_element_size(conv_states_all))));
                
                ggml_tensor * v_conv_f32 = layer.ssm_v_conv;
                if (v_conv_f32->type != GGML_TYPE_F32) {
                    v_conv_f32 = ggml_cast(ctx0, v_conv_f32, GGML_TYPE_F32);
                }
                ggml_tensor * v_conv_weight = ggml_reshape_2d(ctx0, v_conv_f32, d_conv, d_inner);
                Vcur = ggml_ssm_conv(ctx0, conv_v, v_conv_weight);
                Vcur = ggml_reshape_2d(ctx0, Vcur, d_inner, n_tokens);
                if (layer.ssm_v_conv_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.ssm_v_conv_b);
                }
                Vcur = ggml_silu(ctx0, Vcur);
            } else {
                GGML_ABORT("KDA layer missing V conv weight");
            }
            
            // Step 3: Compute g1 (forget gate)
            // g1 = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_softplus(ctx0, g1);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);
            
            // A_log shape is [1, n_head] or [1, n_head, 1, 1], need to broadcast to [head_dim, n_head, n_tokens]
            // First compute -exp(A_log), then reshape for broadcasting
            ggml_tensor * A_neg_exp = ggml_neg(ctx0, ggml_exp(ctx0, layer.ssm_a_log));
            // Reshape to [1, n_head, 1] for broadcasting with g1 [head_dim, n_head, n_tokens]
            A_neg_exp = ggml_reshape_3d(ctx0, A_neg_exp, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A_neg_exp);
            cb(g1, "kda_g1", il);
            
            // Step 4: Compute beta (mixing coefficient)
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_sigmoid(ctx0, beta);
            cb(beta, "kda_beta", il);
            
            // Step 5: Reshape for KDA recurrence
            // {n_embd, n_tokens} -> {n_embd, n_seq_tokens, n_seqs}
            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);
            
            Qcur = ggml_cont(ctx0, ggml_reshape_4d(ctx0, Qcur, head_dim, n_head, n_seq_tokens, n_seqs));
            Kcur = ggml_cont(ctx0, ggml_reshape_4d(ctx0, Kcur, head_dim, n_head, n_seq_tokens, n_seqs));
            Vcur = ggml_cont(ctx0, ggml_reshape_4d(ctx0, Vcur, head_dim, n_head, n_seq_tokens, n_seqs));
            g1 = ggml_cont(ctx0, ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs));
            beta = ggml_cont(ctx0, ggml_reshape_3d(ctx0, beta, n_head, n_seq_tokens, n_seqs));
            
            cb(Qcur, "kda_Q", il);
            cb(Kcur, "kda_K", il);
            cb(Vcur, "kda_V", il);
            
            // Step 6: Get SSM state and compute KDA recurrence using ggml_kda_scan
            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            
            // Use build_rs with lambda pattern (like Mamba SSM scan)
            auto get_kda_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
                ggml_tensor * h_state = ggml_reshape_4d(ctx, states, head_dim, head_dim, n_head, mctx_cur->get_size());
                // Call ggml_kda_scan which implements the correct KDA recurrence
                return ggml_kda_scan(ctx, h_state, Qcur, Kcur, Vcur, g1, beta, ids);
            };
            
            ggml_tensor * y_kda = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs, get_kda_rows);
            cb(y_kda, "kda_scan_out", il);
            
            // Store updated state back
            // y_kda contains: [attention_output (head_dim * n_head * n_seq_tokens * n_seqs), new_state (head_dim * head_dim * n_head * n_seqs)]
            const int64_t attn_out_size = head_dim * n_head * n_seq_tokens * n_seqs;
            const int64_t state_size = head_dim * head_dim * n_head;
            ggml_build_forward_expand(gf, 
                ggml_cpy(ctx0, 
                    ggml_view_1d(ctx0, y_kda, state_size * n_seqs, attn_out_size * ggml_element_size(y_kda)),
                    ggml_view_1d(ctx0, ssm_states_all, state_size * n_seqs, kv_head * state_size * ggml_element_size(ssm_states_all))));
            
            // Extract attention output
            ggml_tensor * attn_out = ggml_view_1d(ctx0, y_kda, attn_out_size, 0);
            attn_out = ggml_reshape_3d(ctx0, attn_out, head_dim, n_head, n_seq_tokens * n_seqs);
            cb(attn_out, "kda_attn_out", il);
            
            // Step 7: Output gating g2 = g_b(g_a(x))
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
            ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);
            
            // Step 8: Apply o_norm with sigmoid gating
            // Note: Kimi model uses sigmoid gating, not SiLU (despite FusedRMSNormGated default being swish)
            // Formula: output = RMSNorm(x) * sigmoid(g)
            ggml_tensor * normed = build_norm(attn_out, layer.ssm_o_norm, layer.ssm_o_norm_b, LLM_NORM_RMS, il);
            ggml_tensor * gate = ggml_sigmoid(ctx0, g2);
            ggml_tensor * gated = ggml_mul(ctx0, normed, gate);
            
            // Step 9: Output projection
            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);
            
            
            GGML_UNUSED(d_conv);
            GGML_UNUSED(kq_scale_kda);
            
        } else if (is_mla) {
            // === MLA Layer (Multi-head Latent Attention) without KV Cache ===
            // Reference: vLLM mla.py
            // TODO: Implement proper KV caching for MLA (requires custom cache format)
            
            // Step 1: Q projection and reshape
            // vLLM Kimi: q = q_proj(hidden_states), then view as [n_tokens, n_head, qk_head_dim]
            // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
            cb(Qcur, "mla_Q", il);
            
            // Step 2: KV compression
            // kv_lora = kv_a_proj_with_mqa(hidden_states) -> [kv_lora_rank + qk_rope_head_dim, n_tokens]
            ggml_tensor * kv_lora = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
            
            // Split: kv_c = kv_lora[:kv_lora_rank], k_pe = kv_lora[kv_lora_rank:]
            ggml_tensor * kv_c = ggml_view_2d(ctx0, kv_lora, kv_lora_rank, n_tokens,
                ggml_row_size(kv_lora->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_lora, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_lora->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_lora->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_lora->type, kv_lora_rank));
            
            // Note: Kimi MLA does NOT apply RoPE (rotary_emb=None in vLLM)
            // k_pe is used directly without RoPE
            
            // Normalize kv_c
            kv_c = build_norm(kv_c, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            
            // KV decompression: kv = kv_b_proj(kv_c_normed)
            ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_c);
            const int64_t kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;
            
            // Split kv into k_nope and v
            ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(kv->type, kv_per_head),
                ggml_row_size(kv->type, kv_per_head * n_head), 0);
            ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                ggml_row_size(kv->type, kv_per_head),
                ggml_row_size(kv->type, kv_per_head * n_head),
                ggml_row_size(kv->type, n_embd_head_qk_nope));
            k_nope = ggml_cont(ctx0, k_nope);
            Vcur = ggml_cont(ctx0, Vcur);
            
            // Concatenate k_nope + k_pe (broadcast k_pe to all heads)
            // K = [k_nope, k_pe] where k_nope is [qk_nope_head_dim, n_head, n_tokens]
            // and k_pe is [qk_rope_head_dim, 1, n_tokens] broadcast to all heads
            k_pe = ggml_cont(ctx0, k_pe);
            // Need to broadcast k_pe from [qk_rope, 1, n_tokens] to [qk_rope, n_head, n_tokens]
            ggml_tensor * k_pe_target = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
            ggml_tensor * k_pe_repeated = ggml_repeat(ctx0, k_pe, k_pe_target);
            ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, k_pe_repeated, 0);
            cb(Kcur, "mla_K", il);
            cb(Vcur, "mla_V", il);
            
            // Direct softmax attention (without KV cache)
            // Use build_attn with inp_no_cache for proper mask handling
            cur = build_attn(inp_no_cache, layer.wo, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale_mla, il);
            cb(cur, "mla_out", il);
            
        } else {
            // Unknown layer type - this should not happen
            GGML_ABORT("Kimi layer is neither KDA nor MLA - missing required tensors");
        }
        
        // On last layer, select only the output tokens
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        
        // Residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FFN Norm
        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // FFN / MoE
        if (layer.ffn_gate_inp) {
            // MoE layer
            // Kimi uses moe_renormalize=True and routed_scaling_factor (stored as expert_weights_scale) = 2.446
            ggml_tensor * moe_out = build_moe_ffn(cur, layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, 
                                layer.ffn_exp_probs_b, hparams.n_expert, hparams.n_expert_used, 
                                LLM_FFN_SILU, true, true, hparams.expert_weights_scale,
                                (llama_expert_gating_func_type) hparams.expert_gating_func, il);
            cb(moe_out, "ffn_moe_out", il);
            
            // Shared expert (if present)
            if (layer.ffn_gate_shexp) {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                        layer.ffn_up_shexp, NULL, NULL,
                        layer.ffn_gate_shexp, NULL, NULL,
                        layer.ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);
                
                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            } else {
                cur = moe_out;
            }
        } else if (layer.ffn_gate) {
            // Dense FFN layer
            cur = build_ffn(cur, layer.ffn_up, NULL, NULL, layer.ffn_gate, NULL, NULL, 
                           layer.ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // No FFN - this should not happen in Kimi
            GGML_ABORT("Kimi layer missing FFN tensors");
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
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
    
    GGML_UNUSED(n_embd_head_qk_nope);
}
