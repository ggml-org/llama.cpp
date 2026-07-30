#include "models.h"
#include "llama-memory-recurrent.h"

static ggml_tensor * kimi_tile640_dequant(
        ggml_context * ctx0,
        const llama_tile640_tensor * tensor) {
    GGML_ASSERT(tensor && tensor->valid());
    return ggml_tile640_dequant(
            ctx0,
            tensor->packed,
            tensor->page_scales,
            tensor->lane_scales,
            tensor->outlier_row_offsets,
            tensor->outlier_cols,
            tensor->outlier_vals,
            tensor->ne[0],
            tensor->ne[1],
            tensor->ne[2],
            tensor->ne[3]);
}

static ggml_tensor * kimi_tile640_get_rows(
        ggml_context * ctx0,
        const llama_tile640_tensor * tensor,
        ggml_tensor * ids) {
    GGML_ASSERT(tensor && tensor->valid());
    return ggml_tile640_get_rows(
            ctx0,
            tensor->packed,
            tensor->page_scales,
            tensor->lane_scales,
            tensor->outlier_row_offsets,
            tensor->outlier_cols,
            tensor->outlier_vals,
            ids,
            (int32_t) tensor->ne[0]);
}

void llama_model_kimi_linear::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q, false);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);

    // MLA qk_rope_head_dim (for reference)
    // qk_rope_head_dim = 64, qk_nope_head_dim = 128, qk_head_dim = 192

    // Mark KDA layers as recurrent using n_head_kv pattern (like Jamba)
    // Set n_head_kv = 0 for KDA layers (recurrent), n_head_kv = n_head for MLA layers (attention)
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;  // KDA layers are recurrent
    }

    // MoE parameters - Kimi uses moe_intermediate_size = 1024
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,                    hparams.moe_latent_size, false);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,                hparams.attn_res_block_size, false);
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        hparams.llm_ffn_op = llm_ffn_op_type_from_string(hidden_act, LLM_FFN_SILU);
    } else {
        hparams.llm_ffn_op = LLM_FFN_SILU;
    }

    switch (hparams.n_layer()) {
        case 27: type = LLM_TYPE_48B_A3B; break; // Kimi-Linear-48B-A3B
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_kimi_linear::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor_or_tile640(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor_or_tile640(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor_or_tile640(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
    if (hparams.attn_res_block_size > 0) {
        output_attn_res_norm = create_tensor_or_tile640(
                tn(LLM_TENSOR_OUTPUT_ATTN_RES_NORM, "weight"),
                {n_embd}, 0);
        output_attn_res_proj = create_tensor_or_tile640(
                tn(LLM_TENSOR_OUTPUT_ATTN_RES_PROJ, "weight"),
                {n_embd, 1}, 0);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        if (hparams.attn_res_block_size > 0) {
            layer.attn_res_norm = create_tensor_or_tile640(
                    tn(LLM_TENSOR_ATTN_RES_NORM, "weight", i),
                    {n_embd}, 0);
            layer.attn_res_proj = create_tensor_or_tile640(
                    tn(LLM_TENSOR_ATTN_RES_PROJ, "weight", i),
                    {n_embd, 1}, 0);
            layer.ffn_res_norm = create_tensor_or_tile640(
                    tn(LLM_TENSOR_FFN_RES_NORM, "weight", i),
                    {n_embd}, 0);
            layer.ffn_res_proj = create_tensor_or_tile640(
                    tn(LLM_TENSOR_FFN_RES_PROJ, "weight", i),
                    {n_embd, 1}, 0);
        }

        // Check for KDA specific tensors to determine layer type or if it's a mixed model
        // Assuming KDA layer if KDA tensors are present

        // KDA uses head_dim = 128 (from linear_attn_config.head_dim)
        const int64_t n_embd_head_k_kda = hparams.n_embd_head_kda;
        const int64_t n_embd_head_v_kda = hparams.n_embd_head_kda;
        const int64_t ssm_d_conv = hparams.ssm_d_conv;

        if (hparams.is_recr(i)) {
            layer.wqkv_gate = create_tensor_or_tile640(
                    tn(LLM_TENSOR_ATTN_GATE, "weight", i),
                    {n_embd, n_embd_head_v_kda * n_head},
                    TENSOR_NOT_REQUIRED);
            // Conv1d weights: try 4D first, then 3D (quantization may remove trailing 1)
            // 4D: [d_conv, 1, d_inner, 1], 3D: [d_conv, 1, d_inner]
            layer.ssm_q_conv = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, n_embd_head_k_kda * n_head, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_q_conv &&
                    !get_tile640_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i).str())) {
                layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, n_embd_head_k_kda * n_head}, 0);
            }

             // KDA Layer - Conv1d weights may be 3D or 4D
             layer.ssm_k_conv = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, n_embd_head_k_kda * n_head, 1}, TENSOR_NOT_REQUIRED);
             if (!layer.ssm_k_conv &&
                     !get_tile640_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i).str())) {
                 layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, n_embd_head_k_kda * n_head}, 0);
             }
             layer.ssm_v_conv = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, n_embd_head_v_kda * n_head, 1}, TENSOR_NOT_REQUIRED);
             if (!layer.ssm_v_conv &&
                     !get_tile640_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i).str())) {
                 layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, n_embd_head_v_kda * n_head}, 0);
             }

             // q, k, v projections
             // Python: q_proj, k_proj, v_proj
             layer.wq = create_tensor_or_tile640(
                     tn(LLM_TENSOR_ATTN_Q, "weight", i),
                     {n_embd, n_embd_head_k_kda * n_head}, 0);
             layer.wk = create_tensor_or_tile640(
                     tn(LLM_TENSOR_ATTN_K, "weight", i),
                     {n_embd, n_embd_head_k_kda * n_head}, 0);
             layer.wv = create_tensor_or_tile640(
                     tn(LLM_TENSOR_ATTN_V, "weight", i),
                     {n_embd, n_embd_head_v_kda * n_head}, 0);

             // KDA specific projections
             // f_a_proj, f_b_proj
             layer.ssm_f_a = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, n_embd_head_k_kda}, 0); // head_dim
             layer.ssm_f_b = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_F_B, "weight", i), {n_embd_head_k_kda, n_embd_head_k_kda * n_head}, 0); // projection_size

             // b_proj (beta mixing coefficient)
             layer.ssm_beta = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

             // A_log - Shape in GGUF: [1, num_heads, 1, 1] (4D) or [1, num_heads] (2D after quantization) Note: -exp(A_log) is applied in convert_hf_to_gguf.py
             layer.ssm_a = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_A, i), {1, n_head, 1, 1}, TENSOR_NOT_REQUIRED);
             if (!layer.ssm_a &&
                     !get_tile640_tensor(tn(LLM_TENSOR_SSM_A, i).str())) {
                 layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
             }

             // dt_bias - shape [n_embd_head_k_kda * n_head] = [4096]
             layer.ssm_dt_b = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_embd_head_k_kda * n_head}, 0);

             // g_a_proj, g_b_proj (output gate)
             if (!layer.wqkv_gate &&
                     !get_tile640_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i).str())) {
                 layer.ssm_g_a = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_G_A, "weight", i), {n_embd, n_embd_head_k_kda}, 0);
                 layer.ssm_g_b = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_G_B, "weight", i), {n_embd_head_k_kda, n_embd_head_k_kda * n_head}, 0);
             }

             // o_norm (reusing SSM_NORM)
             layer.ssm_o_norm = create_tensor_or_tile640(tn(LLM_TENSOR_SSM_NORM, "weight", i), {n_embd_head_k_kda}, 0); // FusedRMSNormGated

             // o_proj
             layer.wo = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v_kda * n_head, n_embd}, 0);

        } else {
             // MLA Layer - use MLA-specific head dimensions
             const int64_t q_lora_rank  = hparams.n_lora_q;
             const int64_t kv_lora_rank = hparams.n_lora_kv;
             const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
             const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
             layer.wqkv_gate = create_tensor_or_tile640(
                     tn(LLM_TENSOR_ATTN_GATE, "weight", i),
                     {n_embd, n_head * n_embd_head_v_mla},
                     TENSOR_NOT_REQUIRED);

             layer.attn_q_a_norm = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, TENSOR_NOT_REQUIRED);
             layer.attn_kv_a_norm = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

             const bool has_q_lora = layer.attn_q_a_norm ||
                     get_tile640_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i).str());
             if (has_q_lora) {
                 layer.wq_a = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
                 layer.wq_b = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
             } else {
                 // Kimi MLA without Q compression: wq = [n_embd, n_head * n_embd_head_k_mla]
                 layer.wq = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, 0);
             }

             // Kimi: qk_rope_head_dim = 64 (actual RoPE dimension for MLA)
             // Note: hparams.n_rot may be 72 (from conversion) but actual is 64
             const int64_t qk_rope_head_dim = hparams.n_rot();  // From config: qk_rope_head_dim
             layer.wkv_a_mqa = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, 0);
             // Support Legacy GGUFs that don't split wkv_b (MLA KV cache disabled)
             layer.wkv_b = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_k_mla - qk_rope_head_dim + n_embd_head_v_mla)}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
             const bool has_legacy_kv_b = layer.wkv_b ||
                     get_tile640_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i).str());
             if (!has_legacy_kv_b) { // MLA KV cache enabled
                 layer.wk_b = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_k_mla - qk_rope_head_dim, kv_lora_rank, n_head}, 0);
                 layer.wv_b = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
             }
             layer.wo = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);
        }

        layer.ffn_norm = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        // MoE intermediate size (different from dense FFN)
        const int64_t n_ff_exp = hparams.n_ff_exp;
        const int64_t moe_n_embd =
                hparams.moe_latent_size > 0 ? hparams.moe_latent_size : n_embd;

        // Kimi uses n_layer_dense_lead to determine which layers use dense FFN vs MoE
        // first_k_dense_replace = 1 means layer 0 uses dense FFN, layers 1+ use MoE
        if (i < (int) hparams.n_layer_dense_lead) {
            // Dense FFN layer - use normal n_ff
            layer.ffn_gate = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            // MoE layer - use n_ff_exp (1024) instead of n_ff (9216)
            layer.ffn_gate_inp = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_latent_down = create_tensor_or_tile640(
                    tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i),
                    {n_embd, moe_n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_up = create_tensor_or_tile640(
                    tn(LLM_TENSOR_FFN_LATENT_UP, "weight", i),
                    {moe_n_embd, n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_norm_exps = create_tensor_or_tile640(
                    tn(LLM_TENSOR_FFN_NORM_EXPS, "weight", i),
                    {moe_n_embd}, TENSOR_NOT_REQUIRED);
            create_tensor_gate_up_exps(layer, i, moe_n_embd, n_ff_exp, n_expert, 0);
            layer.ffn_down_exps = create_tensor_or_tile640(
                    tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                    {n_ff_exp, moe_n_embd, n_expert}, 0);

            // Shared experts use moe_intermediate_size * num_shared_experts
            // Kimi: shared_expert_intermediate_size = 1024 * 1 = 1024
            // Tensors are 2D: [n_embd, n_ff_shexp] or [n_ff_shexp, n_embd]
            const int64_t n_ff_shexp_actual = n_ff_exp * (hparams.n_expert_shared > 0 ? hparams.n_expert_shared : 1);
            layer.ffn_gate_shexp = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp_actual}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_shexp = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp_actual, n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_shexp   = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp_actual}, TENSOR_NOT_REQUIRED);

            layer.ffn_exp_probs_b = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_kimi_linear::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Causal Conv1d function for Q,K,V
// When qkv is 0, it is Q, 1 is K, 2 is V
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x_proj, ggml_tensor * conv_w, int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
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

    const LLM_TN qtn(model.arch);
    auto tile = [&](llm_tensor tensor, const char * suffix, int il = -1) {
        const std::string name = il >= 0
                ? qtn(tensor, suffix, il).str()
                : qtn(tensor, suffix).str();
        return model.get_tile640_tensor(name);
    };
    auto weight = [&](ggml_tensor * plain, llm_tensor tensor, const char * suffix, int il = -1) {
        if (plain) {
            return plain;
        }
        const auto * quantized = tile(tensor, suffix, il);
        GGML_ASSERT(quantized && quantized->valid());
        return kimi_tile640_dequant(ctx0, quantized);
    };
    auto mm = [&](ggml_tensor * plain, llm_tensor tensor, ggml_tensor * input, int il) {
        const auto * quantized = tile(tensor, "weight", il);
        if (!quantized) {
            GGML_ASSERT(plain);
            return build_lora_mm(plain, input);
        }
        return build_tile640_lora_mm(
                quantized->packed,
                quantized->page_scales,
                quantized->lane_scales,
                quantized->outlier_row_offsets,
                quantized->outlier_cols,
                quantized->outlier_vals,
                quantized->act_scale,
                input);
    };
    auto dense_ffn = [&](
            ggml_tensor * input,
            ggml_tensor * gate_w,
            llm_tensor gate_tensor,
            ggml_tensor * up_w,
            llm_tensor up_tensor,
            ggml_tensor * down_w,
            llm_tensor down_tensor,
            int il) {
        ggml_tensor * gate = mm(gate_w, gate_tensor, input, il);
        ggml_tensor * up = mm(up_w, up_tensor, input, il);
        if (hparams.llm_ffn_op == LLM_FFN_SITU) {
            ggml_tensor * gate_raw = gate;
            gate = ggml_scale(
                    ctx0,
                    ggml_tanh(
                        ctx0,
                        ggml_scale(ctx0, gate, 1.0f / hparams.situ_beta)),
                    hparams.situ_beta);
            gate = ggml_mul(ctx0, gate, ggml_sigmoid(ctx0, gate_raw));
            up = ggml_scale(
                    ctx0,
                    ggml_tanh(
                        ctx0,
                        ggml_scale(ctx0, up, 1.0f / hparams.situ_linear_beta)),
                    hparams.situ_linear_beta);
        } else {
            GGML_ASSERT(hparams.llm_ffn_op == LLM_FFN_SWIGLU ||
                        hparams.llm_ffn_op == LLM_FFN_SILU);
            gate = ggml_silu(ctx0, gate);
        }
        return mm(down_w, down_tensor, ggml_mul(ctx0, gate, up), il);
    };

    if (const auto * token_q = tile(LLM_TENSOR_TOKEN_EMBD, "weight")) {
        auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp());
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;
        inp->embd = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), ubatch.n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);
        std::array<ggml_tensor *, 2> inps = {
            kimi_tile640_get_rows(ctx0, token_q, inp->tokens),
            inp->embd,
        };
        inpL = ggml_build_forward_select(
                gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);
        res->t_inp_embd = inpL;
        res->add_input(std::move(inp));
        ggml_build_forward_expand(gf, inpL);
    } else {
        inpL = build_inp_embd(model.tok_embd);
    }
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
    const bool use_attn_residuals = hparams.attn_res_block_size > 0;
    std::vector<ggml_tensor *> block_residuals;
    ggml_tensor * prefix_sum = inpL;

    auto apply_attn_residual = [&](
            ggml_tensor * prefix,
            ggml_tensor * proj,
            ggml_tensor * norm,
            int il) {
        GGML_ASSERT(prefix != nullptr && proj != nullptr && norm != nullptr);
        if (block_residuals.empty()) {
            return prefix;
        }

        const int64_t n_tok = prefix->ne[1];
        ggml_tensor * values = ggml_reshape_3d(ctx0, prefix, n_embd, n_tok, 1);
        for (ggml_tensor * residual : block_residuals) {
            values = ggml_concat(
                    ctx0,
                    ggml_reshape_3d(ctx0, residual, n_embd, n_tok, 1),
                    values,
                    2);
        }

        ggml_tensor * keys = build_norm(values, norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * score_weight = ggml_reshape_3d(ctx0, proj, n_embd, 1, 1);
        ggml_tensor * scores = ggml_sum_rows(
                ctx0, ggml_mul(ctx0, keys, score_weight));
        scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 2, 1, 0, 3));
        scores = ggml_soft_max(ctx0, scores);
        scores = ggml_permute(ctx0, scores, 2, 1, 0, 3);

        ggml_tensor * weighted = ggml_mul(ctx0, values, scores);
        weighted = ggml_cont(ctx0, ggml_permute(ctx0, weighted, 2, 0, 1, 3));
        weighted = ggml_sum_rows(ctx0, weighted);
        weighted = ggml_reshape_2d(ctx0, weighted, n_embd, n_tok);
        cb(weighted, "attn_residual_mix", il);
        return weighted;
    };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;
        ggml_tensor * attn_input = inpL;
        if (use_attn_residuals) {
            prefix_sum = inpL;
            attn_input = apply_attn_residual(
                    prefix_sum,
                    weight(layer.attn_res_proj, LLM_TENSOR_ATTN_RES_PROJ, "weight", il),
                    weight(layer.attn_res_norm, LLM_TENSOR_ATTN_RES_NORM, "weight", il),
                    il);
            if (il % hparams.attn_res_block_size == 0) {
                block_residuals.push_back(prefix_sum);
                prefix_sum = nullptr;
            }
        }

        // Attention Norm
        cur = build_norm(
                attn_input,
                weight(layer.attn_norm, LLM_TENSOR_ATTN_NORM, "weight", il),
                NULL, LLM_NORM_RMS, il);
        ggml_tensor * attn_hidden = cur;
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            // === KDA Layer (Kimi Delta Attention) with Recurrent State ===
            // Reference: vLLM kda.py
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            // Get conv states from r_l tensor (Q, K, V each have separate state)
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            cb(conv_states_all, "conv_states_all", il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            ggml_tensor * Qcur = causal_conv1d(
                    gf, ctx0, conv_states_all, conv_state_all, 0,
                    mm(layer.wq, LLM_TENSOR_ATTN_Q, cur, il),
                    weight(layer.ssm_q_conv, LLM_TENSOR_SSM_CONV1D_Q, "weight", il),
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = causal_conv1d(
                    gf, ctx0, conv_states_all, conv_state_all, 1,
                    mm(layer.wk, LLM_TENSOR_ATTN_K, cur, il),
                    weight(layer.ssm_k_conv, LLM_TENSOR_SSM_CONV1D_K, "weight", il),
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = causal_conv1d(
                    gf, ctx0, conv_states_all, conv_state_all, 2,
                    mm(layer.wv, LLM_TENSOR_ATTN_V, cur, il),
                    weight(layer.ssm_v_conv, LLM_TENSOR_SSM_CONV1D_V, "weight", il),
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // g1 = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
            ggml_tensor * f_a = mm(layer.ssm_f_a, LLM_TENSOR_SSM_F_A, cur, il);
            ggml_tensor * g1 = mm(layer.ssm_f_b, LLM_TENSOR_SSM_F_B, f_a, il);
            cb(g1, "g1 f_b(f_a(cur))", il);
            g1 = ggml_add(
                    ctx0,
                    g1,
                    weight(layer.ssm_dt_b, LLM_TENSOR_SSM_DT, "bias", il));
            g1 = ggml_softplus(ctx0, g1);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            // A_log shape is [1, n_head] or [1, n_head, 1, 1], need to broadcast to [head_dim, n_head, n_tokens]. No need to -exp(a_log) because it was done in convert_hf_to_gguf.py
            // Reshape to [1, n_head, 1] for broadcasting with g1 [head_dim, n_head, n_tokens]
            ggml_tensor * A = ggml_reshape_3d(
                    ctx0,
                    weight(layer.ssm_a, LLM_TENSOR_SSM_A, nullptr, il),
                    1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
            cb(g1, "kda_g1", il);

            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            // Compute beta (mixing coefficient)
            ggml_tensor * beta = mm(layer.ssm_beta, LLM_TENSOR_SSM_BETA, cur, il);
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
            ggml_tensor * g2;
            if (layer.wqkv_gate || tile(LLM_TENSOR_ATTN_GATE, "weight", il)) {
                g2 = mm(layer.wqkv_gate, LLM_TENSOR_ATTN_GATE, cur_2d, il);
            } else {
                ggml_tensor * g_a = mm(layer.ssm_g_a, LLM_TENSOR_SSM_G_A, cur_2d, il);
                g2 = mm(layer.ssm_g_b, LLM_TENSOR_SSM_G_B, g_a, il);
            }
            cb(g2, "g2 g_b(g_a(cur_2d))", il);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            // Apply o_norm with sigmoid gating
            // Note: Kimi model uses sigmoid gating, not SiLU (despite FusedRMSNormGated default being swish)
            // Formula: output = RMSNorm(x) * sigmoid(g)
            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head,  n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(
                    attn_out_final,
                    weight(layer.ssm_o_norm, LLM_TENSOR_SSM_NORM, "weight", il),
                    nullptr, LLM_NORM_RMS, il);
            cb(normed, "kda_normed", il);
            ggml_tensor * gate = ggml_sigmoid(ctx0, g2);
            ggml_tensor * gated = ggml_mul(ctx0, normed, gate);

            // Output projection
            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = mm(layer.wo, LLM_TENSOR_ATTN_OUT, gated, il);
            cb(cur, "kda_out", il);

        } else {
            // === MLA Layer (Multi-head Latent Attention) without KV Cache ===
            // Reference: vLLM mla.py
            // Step 1: Q projection and reshape
            // vLLM Kimi: q = q_proj(hidden_states), then view as [n_tokens, n_head, qk_head_dim]
            // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
            ggml_tensor * Qcur;
            if (layer.wq || tile(LLM_TENSOR_ATTN_Q, "weight", il)) {
                Qcur = mm(layer.wq, LLM_TENSOR_ATTN_Q, cur, il);
            } else {
                Qcur = mm(layer.wq_a, LLM_TENSOR_ATTN_Q_A, cur, il);
                Qcur = build_norm(
                        Qcur,
                        weight(layer.attn_q_a_norm, LLM_TENSOR_ATTN_Q_A_NORM, "weight", il),
                        nullptr, LLM_NORM_RMS, il);
                Qcur = mm(layer.wq_b, LLM_TENSOR_ATTN_Q_B, Qcur, il);
            }

            // Step 2: KV compression
            // kv_cmpr_pe = kv_a_proj_with_mqa(hidden_states) -> [kv_lora_rank + qk_rope_head_dim, n_tokens]
            ggml_tensor * kv_cmpr_pe = mm(
                    layer.wkv_a_mqa, LLM_TENSOR_ATTN_KV_A_MQA, cur, il);

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
            kv_cmpr = build_norm(
                    kv_cmpr,
                    weight(layer.attn_kv_a_norm, LLM_TENSOR_ATTN_KV_A_NORM, "weight", il),
                    nullptr, LLM_NORM_RMS, il);

            const auto * wk_b_q = tile(LLM_TENSOR_ATTN_K_B, "weight", il);
            const auto * wv_b_q = tile(LLM_TENSOR_ATTN_V_B, "weight", il);
            if ((layer.wk_b || wk_b_q) && (layer.wv_b || wv_b_q)) { // MLA KV cache enabled
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
                ggml_tensor * wk_b = layer.wk_b
                        ? layer.wk_b
                        : kimi_tile640_dequant(ctx0, wk_b_q);
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, wk_b, q_nope);
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

                ggml_tensor * wv_b = layer.wv_b
                        ? layer.wv_b
                        : kimi_tile640_dequant(ctx0, wv_b_q);
                cur = build_attn(
                        inp_attn_k, nullptr, NULL, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, wv_b,
                        kq_scale_mla, il);
                if (layer.wqkv_gate || tile(LLM_TENSOR_ATTN_GATE, "weight", il)) {
                    cur = ggml_mul(
                            ctx0,
                            cur,
                            ggml_sigmoid(ctx0, mm(
                                layer.wqkv_gate,
                                LLM_TENSOR_ATTN_GATE,
                                attn_hidden,
                                il)));
                }
                cur = mm(layer.wo, LLM_TENSOR_ATTN_OUT, cur, il);
                cb(cur, "mla_out", il);
            } else { // MLA KV cache disabled. Fall back to MHA KV cache.
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
                cb(Qcur, "mla_Q", il);
                // KV decompression: kv = kv_b_proj(kv_c_normed)
                ggml_tensor * kv = mm(
                        layer.wkv_b, LLM_TENSOR_ATTN_KV_B, kv_cmpr, il);
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
                cur = build_attn(
                        inp_attn_kv, nullptr, NULL, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        kq_scale_mla, il);
                if (layer.wqkv_gate || tile(LLM_TENSOR_ATTN_GATE, "weight", il)) {
                    cur = ggml_mul(
                            ctx0,
                            cur,
                            ggml_sigmoid(ctx0, mm(
                                layer.wqkv_gate,
                                LLM_TENSOR_ATTN_GATE,
                                attn_hidden,
                                il)));
                }
                cur = mm(layer.wo, LLM_TENSOR_ATTN_OUT, cur, il);
                cb(cur, "mla_out", il);
            }
        }

        // On last layer, select only the output tokens
        if (!use_attn_residuals && il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp;
        if (use_attn_residuals) {
            prefix_sum = prefix_sum ? ggml_add(ctx0, prefix_sum, cur) : cur;
            ffn_inp = apply_attn_residual(
                    prefix_sum,
                    weight(layer.ffn_res_proj, LLM_TENSOR_FFN_RES_PROJ, "weight", il),
                    weight(layer.ffn_res_norm, LLM_TENSOR_FFN_RES_NORM, "weight", il),
                    il);
        } else {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
        }
        cb(ffn_inp, "ffn_inp", il);

        // FFN Norm
        cur = build_norm(
                ffn_inp,
                weight(layer.ffn_norm, LLM_TENSOR_FFN_NORM, "weight", il),
                NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // Dense FFN layer
            cur = dense_ffn(
                    cur,
                    layer.ffn_gate, LLM_TENSOR_FFN_GATE,
                    layer.ffn_up, LLM_TENSOR_FFN_UP,
                    layer.ffn_down, LLM_TENSOR_FFN_DOWN,
                    il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE layer
            // Kimi uses moe_renormalize=True and routed_scaling_factor (stored as expert_weights_scale) = 2.446
            llama_tile640_tensor gate_up_tile;
            const llama_tile640_tensor * gate_up_tile_ptr = nullptr;
            if (layer.ffn_gate_up_exps_packed) {
                gate_up_tile = {
                    layer.ffn_gate_up_exps_packed,
                    layer.ffn_gate_up_exps_page_scales,
                    layer.ffn_gate_up_exps_lane_scales,
                    layer.ffn_gate_up_exps_outlier_row_offsets,
                    layer.ffn_gate_up_exps_outlier_cols,
                    layer.ffn_gate_up_exps_outlier_vals,
                    {
                        hparams.moe_latent_size > 0
                            ? hparams.moe_latent_size
                            : (uint32_t) n_embd,
                        2 * hparams.n_ff_exp,
                        hparams.n_expert,
                        1,
                    },
                    layer.ffn_gate_up_exps_act_scale,
                };
                gate_up_tile_ptr = &gate_up_tile;
            }
            const auto * down_tile_ptr =
                    model.get_tile640_tensor(qtn(
                        LLM_TENSOR_FFN_DOWN_EXPS, "weight", il).str());
            const auto * gate_tile_ptr =
                    model.get_tile640_tensor(qtn(
                        LLM_TENSOR_FFN_GATE_EXPS, "weight", il).str());
            const auto * up_tile_ptr =
                    model.get_tile640_tensor(qtn(
                        LLM_TENSOR_FFN_UP_EXPS, "weight", il).str());

            ggml_tensor * moe_input = cur;
            ggml_tensor * router_logits = mm(
                    layer.ffn_gate_inp, LLM_TENSOR_FFN_GATE_INP, cur, il);
            cb(router_logits, "ffn_moe_logits", il);
            const auto * router_q = tile(LLM_TENSOR_FFN_GATE_INP, "weight", il);
            ggml_tensor * router_storage = layer.ffn_gate_inp
                    ? layer.ffn_gate_inp
                    : router_q->packed;
            const auto * latent_down_q =
                    model.get_tile640_tensor(qtn(
                        LLM_TENSOR_FFN_LATENT_DOWN, "weight", il).str());
            const auto * latent_up_q =
                    model.get_tile640_tensor(qtn(
                        LLM_TENSOR_FFN_LATENT_UP, "weight", il).str());
            if (layer.ffn_latent_down || latent_down_q) {
                moe_input = latent_down_q
                    ? build_tile640_lora_mm(
                        latent_down_q->packed,
                        latent_down_q->page_scales,
                        latent_down_q->lane_scales,
                        latent_down_q->outlier_row_offsets,
                        latent_down_q->outlier_cols,
                        latent_down_q->outlier_vals,
                        latent_down_q->act_scale,
                        cur)
                    : build_lora_mm(layer.ffn_latent_down, cur);
                cb(moe_input, "ffn_latent_down", il);
            }

            ggml_tensor * moe_out = build_moe_ffn(moe_input,
                router_storage,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                weight(layer.ffn_exp_probs_b, LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il),
                hparams.n_expert,
                hparams.n_expert_used,
                hparams.llm_ffn_op, true,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                router_logits,
                layer.ffn_gate_up_exps,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                gate_up_tile_ptr,
                down_tile_ptr,
                gate_tile_ptr,
                up_tile_ptr);
            cb(moe_out, "ffn_moe_out", il);
            if (layer.ffn_norm_exps) {
                moe_out = build_norm(
                        moe_out,
                        weight(layer.ffn_norm_exps, LLM_TENSOR_FFN_NORM_EXPS, "weight", il),
                        nullptr,
                        LLM_NORM_RMS, il);
                cb(moe_out, "ffn_latent_norm", il);
            } else if (tile(LLM_TENSOR_FFN_NORM_EXPS, "weight", il)) {
                moe_out = build_norm(
                        moe_out,
                        weight(nullptr, LLM_TENSOR_FFN_NORM_EXPS, "weight", il),
                        nullptr,
                        LLM_NORM_RMS, il);
                cb(moe_out, "ffn_latent_norm", il);
            }
            if (layer.ffn_latent_up || latent_up_q) {
                moe_out = latent_up_q
                    ? build_tile640_lora_mm(
                        latent_up_q->packed,
                        latent_up_q->page_scales,
                        latent_up_q->lane_scales,
                        latent_up_q->outlier_row_offsets,
                        latent_up_q->outlier_cols,
                        latent_up_q->outlier_vals,
                        latent_up_q->act_scale,
                        moe_out)
                    : build_lora_mm(layer.ffn_latent_up, moe_out);
                cb(moe_out, "ffn_latent_up", il);
            }

            // Shared expert
            {
                ggml_tensor * ffn_shexp = dense_ffn(
                        cur,
                        layer.ffn_gate_shexp, LLM_TENSOR_FFN_GATE_SHEXP,
                        layer.ffn_up_shexp, LLM_TENSOR_FFN_UP_SHEXP,
                        layer.ffn_down_shexp, LLM_TENSOR_FFN_DOWN_SHEXP,
                        il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        if (use_attn_residuals) {
            prefix_sum = ggml_add(ctx0, prefix_sum, cur);
            cur = prefix_sum;
        } else {
            cur = ggml_add(ctx0, cur, ffn_inp);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    if (use_attn_residuals) {
        cur = apply_attn_residual(
                cur,
                weight(model.output_attn_res_proj, LLM_TENSOR_OUTPUT_ATTN_RES_PROJ, "weight"),
                weight(model.output_attn_res_norm, LLM_TENSOR_OUTPUT_ATTN_RES_NORM, "weight"),
                -1);
        if (inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    }

    // Final Norm
    cur = build_norm(
            cur,
            weight(model.output_norm, LLM_TENSOR_OUTPUT_NORM, "weight"),
            NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // Output
    const auto * output_q = tile(LLM_TENSOR_OUTPUT, "weight");
    cur = output_q
            ? build_tile640_lora_mm(
                output_q->packed,
                output_q->page_scales,
                output_q->lane_scales,
                output_q->outlier_row_offsets,
                output_q->outlier_cols,
                output_q->outlier_vals,
                output_q->act_scale,
                cur)
            : build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
