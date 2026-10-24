#include "models.h"

#include "llama-impl.h"

#include "llama-memory-recurrent.h"

void llama_model_raon1::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,        hparams.kda_gate_lower_bound, false);

    uint32_t n_rda = 0;
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
        n_rda += hparams.is_recr_impl[i];
        if (!hparams.is_recr_impl[i] && hparams.n_head_kv(i) != 1) {
            throw std::runtime_error(format("Raon-1 MLA layers must have one KV head, layer %u has %u", i, hparams.n_head_kv(i)));
        }
    }
    // the hybrid memory needs both kinds: RDA layers (head_count_kv 0) and MLA layers (head_count_kv 1)
    if (n_rda == 0 || n_rda == hparams.n_layer()) {
        throw std::runtime_error(format("Raon-1 needs both RDA and MLA layers, got %u RDA of %u layers", n_rda, hparams.n_layer()));
    }

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func);

    ml.get_key(LLM_KV_ATTENTION_LATENT_NORM_RMS_EPS, hparams.f_norm_rms_eps_latent);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK,            hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,        hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,      hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HYPER_CONNECTION_MAGNITUDE,    hparams.hc_magnitude);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,          hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA,   hparams.situ_linear_beta);

    auto require = [](bool ok, const std::string & what) {
        if (!ok) {
            throw std::runtime_error("invalid Raon-1 hyperparameters: " + what);
        }
    };
    // accepted ranges; only 2-4 mHC streams and a conv width of 4 have been run (the released model uses 4 streams)
    require(hparams.dsv4_hc_mult >= 2 && hparams.dsv4_hc_mult <= 8, format("hyper_connection.count %u is not in [2, 8]", hparams.dsv4_hc_mult));
    require(hparams.ssm_d_conv == 4, format("ssm.conv_kernel %u is not 4", hparams.ssm_d_conv));
    require(hparams.ssm_dt_rank > 0, "ssm.time_step_rank is 0");
    require(hparams.n_lora_q > 0 && hparams.n_lora_kv > 0, "q_lora_rank or kv_lora_rank is 0");
    // the absorbed MLA attends over the latent: V is the kv_lora_rank-wide latent
    require(hparams.n_embd_head_v_full == hparams.n_lora_kv, format("the MLA value length %u is not kv_lora_rank %u", hparams.n_embd_head_v_full, hparams.n_lora_kv));
    require(hparams.n_embd_head_k_mla() > hparams.n_rot(), format("the MLA key length %u is not larger than the RoPE dimension %u", hparams.n_embd_head_k_mla(), hparams.n_rot()));
    require(hparams.f_norm_rms_eps_latent > 0 && hparams.dsv4_hc_eps > 0, "a normalization epsilon is not positive");
    require(hparams.hc_magnitude > 0, "hyper_connection.magnitude is not positive");
    require(hparams.situ_beta > 0 && hparams.situ_linear_beta > 0, "a SiTU beta is not positive");
    require(hparams.n_layer_dense_lead <= hparams.n_layer(), format("leading_dense_block_count %u is larger than block_count %u", hparams.n_layer_dense_lead, hparams.n_layer()));
    if (hparams.n_embd_head_kda != 16 && hparams.n_embd_head_kda != 32 &&
        hparams.n_embd_head_kda != 64 && hparams.n_embd_head_kda != 128) {
        throw std::runtime_error(format("unsupported Raon-1 RDA head dimension %u (16, 32, 64 or 128)", hparams.n_embd_head_kda));
    }

    switch (hparams.n_layer()) {
        case 25: type = LLM_TYPE_29B_A4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_raon1::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc = hparams.dsv4_hc_mult;
    // selector width per mHC site: pre (hc) + post (hc) + combination ((hc-1)^2, Helmert coordinates)
    const int64_t hc_mix_dim = 2*hc + (hc - 1)*(hc - 1);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc*n_embd, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        // scale: alpha_pre, alpha_post, alpha_z, eta
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {4}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc*n_embd, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {4}, 0);

        const int64_t head_dim = hparams.n_embd_head_kda;
        const int64_t d_conv   = hparams.ssm_d_conv;
        const int64_t d_inner  = head_dim * n_head;

        if (hparams.is_recr(i)) {
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {d_conv, 1, d_inner, 1}, 0);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {d_conv, 1, d_inner, 1}, 0);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {d_conv, 1, d_inner, 1}, 0);

            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A,  "weight", i), {n_embd, hparams.ssm_dt_rank}, 0);
            layer.ssm_f_b  = create_tensor(tn(LLM_TENSOR_SSM_F_B,  "weight", i), {hparams.ssm_dt_rank, d_inner}, 0);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);
            layer.ssm_rda  = create_tensor(tn(LLM_TENSOR_SSM_RDA,  "weight", i), {n_embd, n_head}, 0);

            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, i), {n_head}, 0);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

            layer.ssm_g      = create_tensor(tn(LLM_TENSOR_SSM_G,    "weight", i), {n_embd, d_inner}, 0);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim}, 0);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            const int64_t q_lora_rank      = hparams.n_lora_q;
            const int64_t kv_lora_rank     = hparams.n_lora_kv;
            const int64_t n_embd_head_k    = hparams.n_embd_head_k_mla();
            const int64_t n_embd_head_v    = hparams.n_embd_head_v_mla();
            const int64_t qk_rope_head_dim = hparams.n_rot();
            const int64_t qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head*n_embd_head_k}, 0);
            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, 0);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {qk_nope_head_dim, kv_lora_rank, n_head}, 0);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v, n_head}, 0);
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head*n_embd_head_v}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v, n_embd}, 0);
        }

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            const int64_t n_ff_exp = hparams.n_ff_exp();

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            if (hparams.n_expert_shared > 0) {
                const int64_t n_ff_shexp = n_ff_exp * hparams.n_expert_shared;
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_raon1::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// causal conv1d over one third (qkv = 0, 1, 2) of the shared Q/K/V conv state, with the rollback snapshots
// x_proj: (d_inner, n_tokens); returns (head_dim, n_head, n_seq_tokens, n_seqs)
ggml_tensor * llama_model_raon1::graph::build_rda_conv(
        llm_graph_input_rs * inp_rs, ggml_tensor * conv_states_seq, ggml_tensor * x_proj,
        ggml_tensor * conv_w, int64_t qkv, int il) {
    const auto * mctx_cur = inp_rs->mctx;

    const int64_t kv_head  = mctx_cur->get_head();
    const int64_t mem_size = mctx_cur->get_size();
    const int64_t K_rs     = (int64_t) cparams.n_rs_seq + 1;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    const int64_t head_dim        = hparams.n_embd_head_kda;
    const int64_t d_conv          = hparams.ssm_d_conv;
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_states_mem = mctx_cur->get_r_l(il);

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_states_seq, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1)   * ggml_element_size(conv_states_seq),
        n_embd_r_total * ggml_element_size(conv_states_seq),
        qkv * conv_state_size * ggml_element_size(conv_states_seq));

    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // group s holds the conv window s tokens back.
    // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K_rs tokens must share one ubatch.
    for (int64_t s = 0; s < K_rs; ++s) {
        const int64_t s_idx = std::max<int64_t>(0, n_seq_tokens - s);
        ggml_tensor * conv_x_s = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
            conv_x->nb[1], conv_x->nb[2], s_idx * conv_x->nb[0]);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, conv_x_s,
                ggml_view_3d(ctx0, conv_states_mem, d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1)   * ggml_element_size(conv_states_mem),
                    n_embd_r_total * ggml_element_size(conv_states_mem),
                    ((s * mem_size + kv_head) * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_mem))));
    }

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

// decay gate from x = f_b(f_a(cur)) + dt_bias; ssm_a holds -exp(A_log)
// with gate_lower_bound: g = lower_bound * sigmoid(exp(A_log) * x), else g = -softplus(exp(A_log) * x)
ggml_tensor * llama_model_raon1::graph::build_rda_gate(ggml_tensor * x, ggml_tensor * ssm_a) {
    const int64_t head_dim = hparams.n_embd_head_kda;

    ggml_tensor * A = ggml_reshape_3d(ctx0, ssm_a, 1, n_head, 1);

    x = ggml_reshape_3d(ctx0, x, head_dim, n_head, n_tokens);
    x = ggml_mul(ctx0, x, A);

    // gate_lower_bound is not a clamp - when set, it swaps the decay gate activation
    if (hparams.kda_gate_lower_bound > -INFINITY) {
        x = ggml_sigmoid(ctx0, ggml_scale(ctx0, x, -1.0f));
        return ggml_scale(ctx0, x, hparams.kda_gate_lower_bound);
    }

    return ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, x)));
}

// Raon delta attention: a gated delta-rule layer whose write gate differs from its erase gate
ggml_tensor * llama_model_raon1::graph::build_rda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs, int il) {
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_inner      = head_dim * n_head;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    ggml_tensor * conv_states_seq = build_rs(inp_rs, inp_rs->mctx->get_r_l(il), hparams.n_embd_r(), n_seqs);

    ggml_tensor * Qcur = build_rda_conv(inp_rs, conv_states_seq, build_lora_mm(layer.wq, cur), layer.ssm_q_conv, 0, il);
    ggml_tensor * Kcur = build_rda_conv(inp_rs, conv_states_seq, build_lora_mm(layer.wk, cur), layer.ssm_k_conv, 1, il);
    ggml_tensor * Vcur = build_rda_conv(inp_rs, conv_states_seq, build_lora_mm(layer.wv, cur), layer.ssm_v_conv, 2, il);
    cb(Qcur, "rda_q_conv", il);
    cb(Kcur, "rda_k_conv", il);
    cb(Vcur, "rda_v_conv", il);

    ggml_tensor * f_a = build_lora_mm(layer.ssm_f_a, cur);
    ggml_tensor * g1  = build_lora_mm(layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
    g1 = build_rda_gate(g1, layer.ssm_a);
    cb(g1, "rda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

    // erase gate sigmoid(beta), write gate sigmoid(beta + delta); the write/erase ratio is folded into V
    ggml_tensor * beta = build_lora_mm(layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
    ggml_tensor * delta = build_lora_mm(layer.ssm_rda, cur);
    delta = ggml_reshape_4d(ctx0, delta, 1, n_head, n_seq_tokens, n_seqs);
    ggml_tensor * erase = ggml_softplus(ctx0, ggml_neg(ctx0, beta));
    ggml_tensor * write = ggml_softplus(ctx0, ggml_neg(ctx0, ggml_add(ctx0, beta, delta)));
    Vcur = ggml_mul(ctx0, Vcur, ggml_exp(ctx0, ggml_sub(ctx0, erase, write)));
    cb(Vcur, "rda_v_scaled", il);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "rda_beta", il);

    // same as HF F.normalize(eps=1e-6): x / max(||x||, eps)
    const float eps_norm = 1e-6f;
    Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
    Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);

    ggml_tensor * g2 = build_lora_mm(layer.ssm_g, cur);

    // scan and state update
    ggml_tensor * ssm_states_all = inp_rs->mctx->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

    ggml_tensor * output = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g1, beta, state, il);
    output = ggml_cont(ctx0, output);
    cb(output, "rda_scan_out", il);

    // output gate, then RMSNorm(o) * sigmoid(g2)
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_tokens);

    ggml_tensor * o      = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_tokens);
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    cb(g2, "rda_g2", il);
    cb(normed, "rda_normed", il);
    ggml_tensor * gated = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur   = build_lora_mm(layer.wo, gated);
    cb(cur, "rda_out", il);

    return cur;
}

ggml_tensor * llama_model_raon1::graph::build_hc_pre(
        ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
        ggml_tensor ** post, ggml_tensor ** comb, int il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nm = hc - 1;
    const int64_t nt = x->ne[2];
    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, n_embd*hc, nt);
    ggml_tensor * rms = ggml_sqrt(ctx0, ggml_scale_bias(ctx0, ggml_mean(ctx0, ggml_sqr(ctx0, flat)), 1.0f, hparams.dsv4_hc_eps));
    ggml_tensor * mixes = ggml_div(ctx0, build_lora_mm(fn, flat), rms);
    cb(mixes, "hc_mixes", il);

    auto affine = [&](int64_t offset, int64_t width, int64_t si) {
        ggml_tensor * z = ggml_view_2d(ctx0, mixes, width, nt, mixes->nb[1], offset*mixes->nb[0]);
        ggml_tensor * a = ggml_view_1d(ctx0, scale, 1, si*scale->nb[0]);
        ggml_tensor * b = ggml_view_1d(ctx0, base, width, offset*base->nb[0]);
        return ggml_add(ctx0, ggml_mul(ctx0, z, a), b);
    };
    ggml_tensor * pre = ggml_sigmoid(ctx0, affine(0, hc, 0));
    *post = ggml_scale(ctx0, ggml_sigmoid(ctx0, affine(hc, hc, 1)), hparams.hc_magnitude);
    // combination: z ((hc-1)^2 Helmert coordinates) -> e = H z H^T, rho = 1 + max(0, max(-e / chart_base)),
    // phi = chart_base + e / rho (doubly stochastic), comb = I + sigmoid(eta) * (phi - I)
    ggml_tensor * z = ggml_cont_3d(ctx0, affine(2*hc, nm*nm, 2), nm, nm, nt);
    ggml_tensor * e = ggml_mul_mat(ctx0, helmert, z);
    e = ggml_mul_mat(ctx0, helmert, ggml_cont(ctx0, ggml_transpose(ctx0, e)));
    ggml_tensor * ratio = ggml_neg(ctx0, ggml_div(ctx0, e, chart_base));
    ratio = ggml_reshape_2d(ctx0, ratio, hc*hc, nt);
    ggml_tensor * rho = ggml_pool_1d(ctx0, ratio, GGML_OP_POOL_MAX, hc*hc, hc*hc, 0);
    rho = ggml_scale_bias(ctx0, ggml_clamp(ctx0, rho, 0.0f, INFINITY), 1.0f, 1.0f);
    rho = ggml_reshape_3d(ctx0, rho, 1, 1, nt);
    ggml_tensor * phi = ggml_add(ctx0, ggml_div(ctx0, e, rho), chart_base);
    ggml_tensor * gamma = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, scale, 1, 3*scale->nb[0]));
    *comb = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, phi, identity), gamma), identity);
    cb(*comb, "hc_comb", il);
    cb(pre, "hc_pre", il);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_pre) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, pre);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t i = 0; i < hc; ++i) {
        ggml_tensor * xi = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], i*x->nb[1]);
        ggml_tensor * wi = ggml_view_2d(ctx0, pre, 1, nt, pre->nb[1], i*pre->nb[0]);
        ggml_tensor * term = ggml_mul(ctx0, xi, wi);
        result = result ? ggml_add(ctx0, result, term) : term;
    }
    return result;
}

ggml_tensor * llama_model_raon1::graph::build_hc_post(
        ggml_tensor * x, ggml_tensor * branch, ggml_tensor * post, ggml_tensor * comb, int il) {
    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, branch, x, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];
    ggml_tensor * result = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * sum = nullptr;
        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * xi = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], src*x->nb[1]);
            ggml_tensor * wi = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst*comb->nb[0] + src*comb->nb[1]);
            ggml_tensor * term = ggml_mul(ctx0, xi, wi);
            sum = sum ? ggml_add(ctx0, sum, term) : term;
        }
        ggml_tensor * wi = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        sum = ggml_add(ctx0, sum, ggml_mul(ctx0, branch, wi));
        sum = ggml_reshape_3d(ctx0, sum, n_embd, 1, nt);
        result = result ? ggml_concat(ctx0, result, sum, 1) : sum;
    }
    return result;
}

ggml_tensor * llama_model_raon1::graph::build_mla_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_attn_k * inp_attn_k, int il) {
    const int64_t nr = hparams.n_rot();
    const int64_t nk = hparams.n_embd_head_k_mla();
    const int64_t nn = nk - nr;
    const int64_t rank = hparams.n_lora_kv;
    const float eps = hparams.f_norm_rms_eps_latent;
    ggml_tensor * q = build_lora_mm(layer.wq_a, cur);
    q = ggml_mul(ctx0, ggml_rms_norm(ctx0, q, eps), layer.attn_q_a_norm);
    q = build_lora_mm(layer.wq_b, q);
    ggml_tensor * kv = build_lora_mm(layer.wkv_a_mqa, cur);
    ggml_tensor * latent = ggml_view_2d(ctx0, kv, rank, n_tokens, kv->nb[1], 0);
    latent = ggml_mul(ctx0, ggml_rms_norm(ctx0, latent, eps), layer.attn_kv_a_norm);
    // NoPE: the "rope" dims of q and k are concatenated as they are, without rotation
    ggml_tensor * extra = ggml_view_3d(ctx0, kv, nr, 1, n_tokens, kv->nb[1], kv->nb[1], rank*kv->nb[0]);
    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, nn, n_head, n_tokens, nk*q->nb[0], q->nb[1], 0);
    ggml_tensor * q_extra = ggml_view_3d(ctx0, q, nr, n_head, n_tokens, nk*q->nb[0], q->nb[1], nn*q->nb[0]);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    q_nope = build_lora_mm(layer.wk_b, q_nope);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    q = ggml_concat(ctx0, q_nope, q_extra, 0);
    latent = ggml_reshape_3d(ctx0, latent, rank, 1, n_tokens);
    ggml_tensor * k = ggml_concat(ctx0, latent, extra, 0);
    ggml_tensor * out = build_attn(inp_attn_k, nullptr, nullptr, nullptr, q, k, latent,
            nullptr, nullptr, layer.wv_b, 1.0f/sqrtf(nk), il);
    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, cur)));
    cb(out, "mla_gated", il);
    return build_lora_mm(layer.wo, out);
}

llama_model_raon1::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_build_delta_net_base(params) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nm = hc - 1;
    helmert = nullptr;
    for (int64_t i = 0; i < hc; ++i) {
        ggml_tensor * row = nullptr;
        for (int64_t j = 0; j < nm; ++j) {
            const float v = (i <= j ? 1.0f : i == j + 1 ? -(float) (j + 1) : 0.0f)/sqrtf((j + 1)*(j + 2));
            ggml_tensor * c = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1), v);
            row = row ? ggml_concat(ctx0, row, c, 0) : c;
        }
        row = ggml_reshape_2d(ctx0, row, nm, 1);
        helmert = helmert ? ggml_concat(ctx0, helmert, row, 1) : row;
    }
    identity = ggml_diag(ctx0, ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hc), 1.0f));
    chart_base = ggml_scale_bias(ctx0, identity, 0.5f, 0.5f/hc);

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    inp = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inp = ggml_repeat_4d(ctx0, inp, n_embd, hc, n_tokens, 1);
    auto * inputs = build_inp_mem_hybrid_k();
    auto * inp_rs = inputs->get_recr();
    auto * inp_attn = inputs->get_attn();
    auto * out_ids = build_inp_out_ids();
    GGML_ASSERT(ubatch.n_seqs != 0 && ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens*ubatch.n_seqs);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * post;
        ggml_tensor * comb;
        ggml_tensor * cur = build_hc_pre(inp, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        if (hparams.is_recr(il)) {
            cur = build_rda_layer(cur, layer, inp_rs, il);
        } else {
            cur = build_mla_layer(cur, layer, inp_attn, il);
        }
        inp = build_hc_post(inp, cur, post, comb, il);
        cb(inp, "hc_attn_out", il);
        cur = build_hc_pre(inp, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        if (il < (int) hparams.n_layer_dense_lead) {
            cur = build_ffn(cur, layer.ffn_up, nullptr, nullptr, layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr, nullptr, LLM_FFN_SITU, LLM_FFN_PAR, il);
        } else {
            ggml_tensor * moe = build_moe_ffn(cur, layer.ffn_gate_inp, layer.ffn_up_exps,
                    layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_exp_probs_b,
                    hparams.n_expert, hparams.n_expert_used(), LLM_FFN_SITU,
                    hparams.expert_weights_norm, hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func, il);
            if (layer.ffn_gate_shexp) {
                ggml_tensor * shared = build_ffn(cur, layer.ffn_up_shexp, nullptr, nullptr,
                        layer.ffn_gate_shexp, nullptr, nullptr, layer.ffn_down_shexp, nullptr,
                        nullptr, nullptr, LLM_FFN_SITU, LLM_FFN_PAR, il);
                moe = ggml_add(ctx0, moe, shared);
            }
            cur = moe;
        }
        inp = build_hc_post(inp, cur, post, comb, il);
        inp = build_cvec(inp, il);
        cb(inp, "l_out", il);
    }
    ggml_tensor * cur = nullptr;
    for (int64_t i = 0; i < hc; ++i) {
        ggml_tensor * x = ggml_view_2d(ctx0, inp, n_embd, n_tokens, inp->nb[2], i*inp->nb[1]);
        cur = cur ? ggml_add(ctx0, cur, x) : x;
    }
    cur = ggml_scale(ctx0, cur, 1.0f/hc);
    if (out_ids) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
