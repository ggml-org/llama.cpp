#include "models.h"
#include "llama-memory-recurrent.h"

void llama_model_berrylm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all);
    ml.get_key(LLM_KV_KDA_GATE_RANK, hparams.kda_gate_rank);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE, hparams.attn_res_block_size);

    if (hparams.attn_res_block_size == 0) {
        throw std::runtime_error("berrylm: attn_res block size must be > 0");
    }
    if (hparams.ssm_n_group == 0 || hparams.ssm_dt_rank % hparams.ssm_n_group != 0) {
        throw std::runtime_error("berrylm: KDA value heads must be a multiple of key heads");
    }

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_18B_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_berrylm::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t n_ff_exp   = hparams.n_ff_exp();
    const int64_t n_ff_shexp = hparams.n_ff_shexp;

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t head_v_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t conv_dim   = key_dim * 2 + value_dim;
    const int64_t gate_dim   = head_k_dim * n_v_heads;  // per-channel KDA gate

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        layer.attn_res_score = create_tensor(tn(LLM_TENSOR_ATTN_RES_SCORE, "weight", il), { n_embd }, 0);
        layer.attn_res_gate  = create_tensor(tn(LLM_TENSOR_ATTN_RES_GATE,  "weight", il), { 1 }, 0);

        if (!hparams.is_recr(il)) {
            // q projection carries the per-head output gate (doubled width)
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, 0);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);
        } else {
            const int64_t gate_rank = hparams.kda_gate_rank;

            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, conv_dim }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_f_a    = create_tensor(tn(LLM_TENSOR_SSM_F_A,    "weight", il), { n_embd, gate_rank }, 0);
            layer.ssm_f_b    = create_tensor(tn(LLM_TENSOR_SSM_F_B,    "weight", il), { gate_rank, gate_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { gate_dim }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, 0);
        }

        // routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, 0);

        // shared expert with a sigmoid gate
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, 0);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_berrylm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_berrylm::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const uint32_t block_size = hparams.attn_res_block_size;

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // committed block streams, the embeddings are block 0
    // raw, transposed [n_blocks, n_embd, n_tokens] for the mix and normalized [n_embd, n_blocks, n_tokens] for the scores
    ggml_tensor * blocks_t = ggml_reshape_3d(ctx0, inpL, 1, n_embd, n_tokens);
    ggml_tensor * blocks_n = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens), hparams.f_norm_rms_eps);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = build_attn_res(inpL, blocks_t, blocks_n, il);

        cur = build_norm(inpSA, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_kda(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;

        // commit the residual stream at the block boundary (not needed after the last layer)
        if ((uint32_t) (il + 1) % block_size == 0 && il + 1 < n_layer) {
            blocks_t = ggml_concat(ctx0, blocks_t, ggml_reshape_3d(ctx0, cur, 1, n_embd, n_tokens), 0);
            blocks_n = ggml_concat(ctx0, blocks_n, ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), hparams.f_norm_rms_eps), 1);
            cb(blocks_t, "attn_res_blocks", il);
        }
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_berrylm::graph::build_attn_res(
        ggml_tensor * cur,
        ggml_tensor * blocks_t,
        ggml_tensor * blocks_n,
        int           il) {
    const auto & layer = model.layers[il];

    const int64_t n_blocks = blocks_n->ne[1];

    // scores over [committed blocks..., current stream], the current stream is scored apart so the block stack stays append-only
    ggml_tensor * sc_blk = ggml_sum_rows(ctx0, ggml_mul(ctx0, blocks_n, layer.attn_res_score)); // [1, n_blocks, n_tokens]
    sc_blk = ggml_reshape_2d(ctx0, sc_blk, n_blocks, n_tokens);

    ggml_tensor * sc_cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
    sc_cur = ggml_sum_rows(ctx0, ggml_mul(ctx0, sc_cur, layer.attn_res_score));             // [1, n_tokens]

    ggml_tensor * probs = ggml_soft_max(ctx0, ggml_concat(ctx0, sc_blk, sc_cur, 0));         // [n_blocks + 1, n_tokens]
    cb(probs, "attn_res_probs", il);

    ggml_tensor * p_blk = ggml_cont(ctx0, ggml_view_3d(ctx0, probs, n_blocks, 1, n_tokens, probs->nb[1], probs->nb[1], 0));
    ggml_tensor * p_cur = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, 1, n_tokens, probs->nb[1], n_blocks * probs->nb[0]));

    // the mix uses the raw streams: [n_blocks, n_embd, n_tokens] x [n_blocks, 1, n_tokens] -> [n_embd, 1, n_tokens]
    ggml_tensor * mixed = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, blocks_t, p_blk), n_embd, n_tokens);
    mixed = ggml_add(ctx0, mixed, ggml_mul(ctx0, cur, p_cur));
    cb(mixed, "attn_res_mixed", il);

    // gate toward identity: h + tanh(gate) * (mixed - h), tanh folded at conversion
    cur = ggml_add(ctx0, cur, ggml_mul(ctx0, ggml_sub(ctx0, mixed, cur), layer.attn_res_gate));
    cb(cur, "attn_res_out", il);

    return cur;
}

ggml_tensor * llama_model_berrylm::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head = hparams.n_embd_head_v();

    // q projection interleaves [q_h, gate_h] per head
    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "attn_gate", il);

    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // partial RoPE (first n_rot dims, rotate-half layout)
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    cur = build_attn(inp,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "attn_out", il);

    return cur;
}

ggml_tensor * llama_model_berrylm::graph::build_layer_kda(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto & layer    = model.layers[il];
    const auto * mctx_cur = inp->mctx;

    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv_mixed = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "kda_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(layer.wqkv_gate, cur, layer.wqkv_gate_s);
    cb(z, "kda_z", il);

    ggml_tensor * beta = build_lora_mm(layer.ssm_beta, cur, layer.ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "kda_beta", il);

    // per-channel forget gate: -exp(A_log)[h] * softplus(f_b(f_a(x)) + dt_bias)[h, k]
    ggml_tensor * g = build_lora_mm(layer.ssm_f_a, cur);
    g = build_lora_mm(layer.ssm_f_b, g);
    g = ggml_add(ctx0, g, layer.ssm_dt);
    g = ggml_softplus(ctx0, g);
    g = ggml_reshape_3d(ctx0, g, head_k_dim, num_v_heads, n_seq_tokens * n_seqs);
    g = ggml_mul(ctx0, g, ggml_reshape_3d(ctx0, layer.ssm_a, 1, num_v_heads, 1));  // ssm_a = -exp(A_log)
    g = ggml_reshape_4d(ctx0, g, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    cb(g, "kda_g", il);

    // short causal conv over q|k|v
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = layer.ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
    cb(conv_out, "kda_conv_out", il);

    const int64_t nb1_qkv = ggml_row_size(conv_out->type, conv_channels);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_out->type, head_k_dim * num_k_heads));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    q_conv = build_gdn_l2_norm(ctx0, q_conv, hparams.f_norm_rms_eps);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, hparams.f_norm_rms_eps);

    // v heads are stored tiled (v_head % num_k_heads), so a plain repeat matches the broadcast of the fused op
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "kda_q", il);
    cb(k_conv, "kda_k", il);
    cb(v_conv, "kda_v", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, g, beta, state, il);

    // gated RMSNorm: norm(o) * w * silu(z)
    ggml_tensor * z_4d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    output = build_norm(output, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    output = ggml_mul(ctx0, output, ggml_silu(ctx0, z_4d));
    cb(output, "kda_gated_norm", il);

    output = ggml_reshape_3d(ctx0, output, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

    cur = build_lora_mm(layer.ssm_out, output, layer.ssm_out_s);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
    cb(cur, "kda_out", il);

    return cur;
}

ggml_tensor * llama_model_berrylm::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    const auto & layer = model.layers[il];

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "moe_routed_out", il);

    ggml_tensor * shexp =
        build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
            layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
            layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);

    ggml_tensor * shexp_gate = ggml_sigmoid(ctx0, build_lora_mm(layer.ffn_gate_inp_shexp, cur));
    shexp = ggml_mul(ctx0, shexp, shexp_gate);
    cb(shexp, "moe_shared_out", il);

    return ggml_add(ctx0, moe_out, shexp);
}
