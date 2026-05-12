#include "models.h"
#include "llama-memory-recurrent.h"

void llama_model_kimi_linear::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);

    // MLA qk_rope_head_dim (for reference)
    // qk_rope_head_dim = 64, qk_nope_head_dim = 128, qk_head_dim = 192

    // Mark KDA layers as recurrent using n_head_kv pattern (like Jamba)
    // Set n_head_kv = 0 for KDA layers (recurrent), n_head_kv = n_head for MLA layers (attention)
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;  // KDA layers are recurrent
    }

    // MoE parameters - Kimi uses moe_intermediate_size = 1024
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);

    switch (hparams.n_layer) {
        case 27: type = LLM_TYPE_48B_A3B; break; // Kimi-Linear-48B-A3B
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_kimi_linear::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // Check for KDA specific tensors to determine layer type or if it's a mixed model
        // Assuming KDA layer if KDA tensors are present

        // KDA uses head_dim = 128 (from linear_attn_config.head_dim)
        const int64_t n_embd_head_k_kda = hparams.n_embd_head_kda;
        const int64_t n_embd_head_v_kda = hparams.n_embd_head_kda;
        const int64_t ssm_d_conv = hparams.ssm_d_conv;

        if (hparams.is_recurrent(i)) {
            // Conv1d weights: try 4D first, then 3D (quantization may remove trailing 1)
            // 4D: [d_conv, 1, d_inner, 1], 3D: [d_conv, 1, d_inner]
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, n_embd_head_k_kda * n_head, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_q_conv) {
                layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, n_embd_head_k_kda * n_head}, 0);
            }

             // KDA Layer - Conv1d weights may be 3D or 4D
             layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, n_embd_head_k_kda * n_head, 1}, TENSOR_NOT_REQUIRED);
             if (!layer.ssm_k_conv) {
                 layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, n_embd_head_k_kda * n_head}, 0);
             }
             layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, n_embd_head_v_kda * n_head, 1}, TENSOR_NOT_REQUIRED);
             if (!layer.ssm_v_conv) {
                 layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, n_embd_head_v_kda * n_head}, 0);
             }

             // q, k, v projections
             // Python: q_proj, k_proj, v_proj
             create_tensor_qkv(layer, i, n_embd, n_embd_head_k_kda * n_head, n_embd_head_k_kda * n_head, n_embd_head_v_kda * n_head, 0);

             // KDA specific projections
             // f_a_proj, f_b_proj
             layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, n_embd_head_k_kda}, 0); // head_dim
             layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", i), {n_embd_head_k_kda, n_embd_head_k_kda * n_head}, 0); // projection_size

             // b_proj (beta mixing coefficient)
             layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

             // A_log - Shape in GGUF: [1, num_heads, 1, 1] (4D) or [1, num_heads] (2D after quantization) Note: -exp(A_log) is applied in convert_hf_to_gguf.py
             layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head, 1, 1}, TENSOR_NOT_REQUIRED);
             if (!layer.ssm_a) {
                 layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
             }

             // dt_bias - shape [n_embd_head_k_kda * n_head] = [4096]
             layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_embd_head_k_kda * n_head}, 0);

             // g_a_proj, g_b_proj (output gate)
             layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", i), {n_embd, n_embd_head_k_kda}, 0);
             layer.ssm_g_b = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", i), {n_embd_head_k_kda, n_embd_head_k_kda * n_head}, 0);

             // o_norm (reusing SSM_NORM)
             layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {n_embd_head_k_kda}, 0); // FusedRMSNormGated

             // o_proj
             layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v_kda * n_head, n_embd}, 0);

        } else {
             // MLA Layer - use MLA-specific head dimensions
             const int64_t q_lora_rank  = hparams.n_lora_q;
             const int64_t kv_lora_rank = hparams.n_lora_kv;
             const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
             const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

             layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, TENSOR_NOT_REQUIRED);
             layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

             if (layer.attn_q_a_norm) {
                 layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
                 layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
             } else {
                 // Kimi MLA without Q compression: wq = [n_embd, n_head * n_embd_head_k_mla]
                 layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, 0);
             }

             // Kimi: qk_rope_head_dim = 64 (actual RoPE dimension for MLA)
             // Note: hparams.n_rot may be 72 (from conversion) but actual is 64
             const int64_t qk_rope_head_dim = hparams.n_rot();  // From config: qk_rope_head_dim
             layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, 0);
             // Support Legacy GGUFs that don't split wkv_b (MLA KV cache disabled)
             layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_k_mla - qk_rope_head_dim + n_embd_head_v_mla)}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
             if (!layer.wkv_b) { // MLA KV cache enabled
                 layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_k_mla - qk_rope_head_dim, kv_lora_rank, n_head}, 0);
                 layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
             }
             layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        // MoE intermediate size (different from dense FFN)
        const int64_t n_ff_exp = hparams.n_ff_exp;

        // Kimi uses n_layer_dense_lead to determine which layers use dense FFN vs MoE
        // first_k_dense_replace = 1 means layer 0 uses dense FFN, layers 1+ use MoE
        if (i < (int) hparams.n_layer_dense_lead) {
            // Dense FFN layer - use normal n_ff
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            // MoE layer - use n_ff_exp (1024) instead of n_ff (9216)
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            // Shared experts use moe_intermediate_size * num_shared_experts
            // Kimi: shared_expert_intermediate_size = 1024 * 1 = 1024
            // Tensors are 2D: [n_embd, n_ff_shexp] or [n_ff_shexp, n_embd]
            const int64_t n_ff_shexp_actual = n_ff_exp * (hparams.n_expert_shared > 0 ? hparams.n_expert_shared : 1);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp_actual}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp_actual, n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp_actual}, TENSOR_NOT_REQUIRED);

            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_kimi_linear::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Causal Conv1d function for Q,K,V
// When qkv is 0, it is Q, 1 is K, 2 is V
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w, int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total = 3 * conv_state_size;  // Q + K + V

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
    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),  // nb1: stride between channels
        n_embd_r_total * ggml_element_size(conv_state_all),  // nb2: stride between seqs
        qkv * conv_state_size * ggml_element_size(conv_state_all));

// Causal Conv1d function for Q,K,V
// When qkv is 0, it is Q, 1 is K, 2 is V
    // Step 1: Q, K, V projections -> [d_inner, n_tokens]
    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);

    // Reshape input: {d_inner, n_tokens} -> {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    // Concat Q conv state and current input: {d_conv-1 + n_seq_tokens, d_inner, n_seqs}
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // Save last (d_conv-1) columns back to Q conv state
    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all,
                d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),           // nb1: contiguous within one channel's conv taps
                n_embd_r_total * ggml_element_size(conv_states_all),         // nb2: stride between sequences (skip over K,V states)
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));  // offset to first seq's Q/K/V state
    // Reshape conv weight: GGUF [d_conv, 1, d_inner, 1] -> ggml_ssm_conv expects [d_conv, d_inner]
    // GGUF stores as [d_conv, 1, d_inner, 1] with memory layout w[conv_step + channel * d_conv]
    // vLLM stores as [d_inner, d_conv] with memory layout w[channel * d_conv + conv_step]
    // ggml_ssm_conv computes: c[conv_step + channel * d_conv]
    // GGUF layout: [d_conv, 1, d_inner] or [d_conv, 1, d_inner, 1] -> reshape to [d_conv, d_inner]
    // Reshape conv weight from [d_conv, 1, d_inner, 1] to [d_conv, d_inner] for ggml_ssm_conv
    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    // Apply conv1d
    // ggml_ssm_conv output: {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    // Reshape to 2D for bias add: {d_inner, n_tokens}
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

llama_model_kimi_linear::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
    // So we don't need inp_pos

    auto * inp_kv = !hparams.is_mla() ? build_inp_mem_hybrid() : nullptr;
    auto * inp_k = hparams.is_mla() ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs = hparams.is_mla() ? inp_k->get_recr() : inp_kv->get_recr();
    auto * inp_attn_kv = !hparams.is_mla() ? inp_kv->get_attn() : nullptr;
    auto * inp_attn_k = hparams.is_mla() ? inp_k->get_attn() : nullptr;

    // Output ids for selecting which tokens to output
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Kimi dimension constants
    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t d_inner = n_head * head_dim;  // 32 * 128 = 4096
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // Verify batch consistency for recurrent layers
    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // MLA params
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    // qk_rope_head_dim = 64 (from Kimi config) which is hparams.n_rot
    // Confirmed from tensor shape: wkv_a_mqa [2304, 576] = [n_embd, kv_lora_rank + qk_rope_head_dim]
    const int64_t n_embd_head_qk_rope = hparams.n_rot();  // config.qk_rope_head_dim
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;  // 192 - 64 = 128
    // Attention scale for MLA
    const float kq_scale_mla = 1.0f / sqrtf((float)n_embd_head_k_mla);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        // Attention Norm
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            // === KDA Layer (Kimi Delta Attention) with Recurrent State ===
            // Reference: vLLM kda.py
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            // Get conv states from r_l tensor (Q, K, V each have separate state)
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            cb(conv_states_all, "conv_states_all", il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // g1 = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
            cb(g1, "g1 f_b(f_a(cur))", il);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_softplus(ctx0, g1);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            // A_log shape is [1, n_head] or [1, n_head, 1, 1], need to broadcast to [head_dim, n_head, n_tokens]. No need to -exp(a_log) because it was done in convert_hf_to_gguf.py
            // Reshape to [1, n_head, 1] for broadcasting with g1 [head_dim, n_head, n_tokens]
            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
            cb(g1, "kda_g1", il);

            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            // Compute beta (mixing coefficient)
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
            cb(beta, "kda_beta", il);

            beta = ggml_sigmoid(ctx0, beta);

            // Reshape for KDA recurrence
            // {n_embd, n_tokens} -> {n_embd, n_seq_tokens, n_seqs}
            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

            // Get SSM state and compute KDA recurrence using ggml_kda_scan
            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            const float eps_norm = hparams.f_norm_rms_eps;

            Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
            Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);

            // Choose between build_delta_net_chunking and build_delta_net_recurrent based on n_tokens
            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

            ggml_tensor * output = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;
            cb(output, "attn_output", il);
            cb(new_state, "new_state", il);

            // Update the recurrent states
            ggml_build_forward_expand(gf,
                                     ggml_cpy(ctx0, new_state,
                                              ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                                                           kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // Output gating g2 = g_b(g_a(x))
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
            ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
            cb(g2, "g2 g_b(g_a(cur_2d))", il);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            // Apply o_norm with sigmoid gating
            // Note: Kimi model uses sigmoid gating, not SiLU (despite FusedRMSNormGated default being swish)
            // Formula: output = RMSNorm(x) * sigmoid(g)
            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head,  n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            cb(normed, "kda_normed", il);
            ggml_tensor * gate = ggml_sigmoid(ctx0, g2);
            ggml_tensor * gated = ggml_mul(ctx0, normed, gate);

            // Output projection
            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);

        } else {
            // === MLA Layer (Multi-head Latent Attention) without KV Cache ===
            // Reference: vLLM mla.py
            // Step 1: Q projection and reshape
            // vLLM Kimi: q = q_proj(hidden_states), then view as [n_tokens, n_head, qk_head_dim]
            // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);

            // Step 2: KV compression
            // kv_cmpr_pe = kv_a_proj_with_mqa(hidden_states) -> [kv_lora_rank + qk_rope_head_dim, n_tokens]
            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

            // Split: kv_cmpr = kv_lora[:kv_lora_rank], k_pe = kv_lora[kv_lora_rank:]
            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            // Note: Kimi MLA does NOT apply RoPE (rotary_emb=None in vLLM)
            // k_pe is used directly without RoPE
            // Normalize kv_c
            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

            if (layer.wk_b && layer.wv_b) { // MLA KV cache enabled
                // extract q_nope
                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(Qcur->type, n_embd_head_k_mla),
                                 ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);
                cb(q_nope, "q_nope", il);

                // and {n_embd_head_qk_rope, n_head, n_tokens}
                ggml_tensor * q_pe = ggml_view_3d(
                    ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(Qcur->type, n_embd_head_k_mla),
                    ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, ggml_row_size(Qcur->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
                Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn_k, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
                cb(cur, "mla_out", il);
            } else { // MLA KV cache disabled. Fall back to MHA KV cache.
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
                cb(Qcur, "mla_Q", il);
                // KV decompression: kv = kv_b_proj(kv_c_normed)
                ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
                const int64_t kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;

                // Split kv into k_nope and v
                ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head), 0);
                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head),
                    ggml_row_size(kv->type, n_embd_head_qk_nope));
                Vcur = ggml_cont(ctx0, Vcur);
                cb(Vcur, "mla_V", il);

                // Concatenate k_nope + k_pe (broadcast k_pe to all heads)
                // K = [k_nope, k_pe] where k_nope is [qk_nope_head_dim, n_head, n_tokens]
                // and k_pe is [qk_rope_head_dim, 1, n_tokens] broadcast to all heads
                // Need to broadcast k_pe from [qk_rope, 1, n_tokens] to [qk_rope, n_head, n_tokens]
                ggml_tensor * k_pe_target = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
                ggml_tensor * k_pe_repeated = ggml_repeat(ctx0, k_pe, k_pe_target);
                ggml_tensor * Kcur = ggml_concat(ctx0, k_pe_repeated, k_nope, 0);
                cb(Kcur, "mla_K", il);

                // Direct softmax attention (with MHA KV cache)
                // Use build_attn with inp_attn for proper mask handling
                cur = build_attn(inp_attn_kv, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale_mla, il);
                cb(cur, "mla_out", il);
            }
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

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // Dense FFN layer
            cur = build_ffn(cur,
                layer.ffn_up, NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE layer
            // Kimi uses moe_renormalize=True and routed_scaling_factor (stored as expert_weights_scale) = 2.446
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            // Shared expert
            {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                        layer.ffn_up_shexp, NULL, NULL,
                        layer.ffn_gate_shexp, NULL, NULL,
                        layer.ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        // Residual
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final Norm
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // Output
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
