#include "models.h"
#include "llama-memory-recurrent.h"

// ZONOS3 TTS backbone, the 10 audio heads are in the mmproj

void llama_model_zonos3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 0;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        if (full_attn_interval == 0) {
            throw std::runtime_error("zonos3: full_attention_interval must be > 0");
        }
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i + 1) % full_attn_interval != 0;
        }
    }

    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE, hparams.moe_latent_size);

    // the router picks the top-k of logits + bias, then takes a softmax over the picked logits
    hparams.expert_gating_func  = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT;
    hparams.expert_weights_norm = false;

    switch (hparams.n_layer()) {
        case 11: type = LLM_TYPE_0_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_zonos3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_score = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_SCORE, "weight"), {n_embd}, 0);
    // no output head, the logits are unused; reuse the embedding table so a sampler can still run
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    const int64_t d_inner    = hparams.ssm_d_inner;
    const int64_t head_dim   = hparams.ssm_d_state;
    const int64_t n_gdn_head = hparams.ssm_dt_rank;
    const int64_t n_latent   = hparams.moe_latent_size;
    const int64_t n_ff_exp   = hparams.n_ff_exp();

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.attn_res_score = create_tensor(tn(LLM_TENSOR_ATTN_RES_SCORE, "weight", i), {n_embd}, 0);
        layer.ffn_res_score  = create_tensor(tn(LLM_TENSOR_FFN_RES_SCORE,  "weight", i), {n_embd}, 0);

        if (hparams.is_recr(i)) {
            // split from the fused GDN in_proj at conversion, same layout as qwen35
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", i), {n_embd, 3 * d_inner}, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", i), {n_embd, d_inner}, 0); // z gate
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {hparams.ssm_d_conv, 3 * d_inner}, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   i), {n_gdn_head}, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         i), {n_gdn_head}, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", i), {n_embd, n_gdn_head}, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", i), {n_embd, n_gdn_head}, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", i), {head_dim}, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", i), {d_inner, n_embd}, 0);
        } else {
            create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
            layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
            // per-head scale of the normalized Q, the QK norm itself has no weight
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head}, 0);
            // per-head output gate, same name as the GDN z gate
            layer.wqkv_gate   = create_tensor(tn(LLM_TENSOR_ATTN_GATE,   "weight", i), {n_embd, n_head}, 0);
        }

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_latent_down = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i), {n_embd, n_latent}, 0);
            layer.ffn_latent_up   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP,   "weight", i), {n_latent, n_embd}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", i), {n_latent, n_ff_exp, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), {n_latent, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp, n_latent, n_expert}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_zonos3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_zonos3::graph::graph(const llm_graph_params & params) :
    llm_build_delta_net_base(params) {
    const float eps = hparams.f_norm_rms_eps;

    ggml_tensor * cur;

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the summed row embedding goes through a weightless norm
    inpL = ggml_rms_norm(ctx0, inpL, eps);
    cb(inpL, "emb_norm", -1);
    attn_res_push(inpL);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        cur = attn_res_route(layer.attn_res_score, il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recr(il)) {
            cur = build_layer_gdn(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }
        cb(cur, "attn_out", il);
        attn_res_push(cur);

        cur = attn_res_route(layer.ffn_res_score, il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        attn_res_push(cur);
    }

    cur = attn_res_route(model.output_res_score, -1);
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

void llama_model_zonos3::graph::attn_res_push(ggml_tensor * cur) {
    ggml_tensor * src = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    bank = bank ? ggml_concat(ctx0, bank, src, 1) : src;
}

// softmax over the bank sources, scored by query . rms_norm(source); the sum uses the raw sources
ggml_tensor * llama_model_zonos3::graph::attn_res_route(ggml_tensor * query, int il) {
    const int64_t n_src = bank->ne[1];

    ggml_tensor * scores = ggml_rms_norm(ctx0, bank, hparams.f_norm_rms_eps);
    scores = ggml_mul(ctx0, scores, query);
    scores = ggml_sum_rows(ctx0, scores);                  // [1, n_src, n_tokens]
    scores = ggml_reshape_2d(ctx0, scores, n_src, n_tokens);

    ggml_tensor * probs = ggml_soft_max(ctx0, scores);
    cb(probs, "attn_res_probs", il);

    return ggml_dsv4_hc_pre(ctx0, bank, probs);
}

ggml_tensor * llama_model_zonos3::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // the reference QK norm has no weight and a fixed eps
    const float qk_norm_eps = 1e-6f;

    auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

    ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, cur);
    gate = ggml_sigmoid(ctx0, ggml_reshape_3d(ctx0, gate, 1, n_head, n_tokens));
    cb(gate, "attn_gate", il);

    Qcur = ggml_rms_norm(ctx0, Qcur, qk_norm_eps);
    Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
    Kcur = ggml_rms_norm(ctx0, Kcur, qk_norm_eps);
    cb(Qcur, "Qcur_normed", il);
    cb(Kcur, "Kcur_normed", il);

    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    cur = build_attn(inp,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
    cb(cur, "attn_pregate", il);

    cur = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens);
    cur = ggml_mul(ctx0, cur, gate);
    cur = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head, n_tokens);
    cb(cur, "attn_gated", il);

    return build_lora_mm(layer.wo, cur, layer.wo_s);
}

// copied from qwen35 build_layer_attn_linear, with the reference L2 norm eps
ggml_tensor * llama_model_zonos3::graph::build_layer_gdn(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto & layer    = model.layers[il];
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(num_k_heads == num_v_heads);

    ggml_tensor * qkv_mixed = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(layer.wqkv_gate, cur, layer.wqkv_gate_s);
    cb(z, "z", il);

    ggml_tensor * beta = build_lora_mm(layer.ssm_beta, cur, layer.ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(layer.ssm_alpha, cur, layer.ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, ggml_add(ctx0, alpha, layer.ssm_dt));
    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, layer.ssm_a); // -A_log.exp() * softplus
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(gate, "gate", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = layer.ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * num_k_heads * head_k_dim;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
    cb(conv_out, "conv_output_silu", il);

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_out->type, head_k_dim * num_k_heads));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    // the reference L2 norm is x * rsqrt(sum(x^2) + 1e-6)
    q_conv = build_gdn_l2_norm(ctx0, q_conv, 1e-6f);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, 1e-6f);
    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // gated norm: rms_norm(output) * silu(z), per head
    ggml_tensor * z_4d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    output = build_norm(output, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    output = ggml_mul(ctx0, output, ggml_silu(ctx0, z_4d));

    output = ggml_reshape_2d(ctx0, output, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(output, "linear_attn_gated", il);

    return build_lora_mm(layer.ssm_out, output, layer.ssm_out_s);
}

ggml_tensor * llama_model_zonos3::graph::build_layer_ffn(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];

    if (layer.ffn_gate_inp == nullptr) {
        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
        return cur;
    }

    // the router reads the full-width input, the experts the latent one
    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps));
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * latent = build_lora_mm(layer.ffn_latent_down, cur);
    cb(latent, "ffn_latent_in", il);

    cur = build_moe_ffn(latent,
            nullptr,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            logits);
    cb(cur, "ffn_moe_out", il);

    cur = build_lora_mm(layer.ffn_latent_up, cur);
    cb(cur, "ffn_out", il);
    return cur;
}
