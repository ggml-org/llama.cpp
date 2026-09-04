#include "models.h"

#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache-kpool.h"

// ssm_a holds -exp(A_log) (kimi-k3), not +exp(A_log) (bailingmoe3); converter checks

static uint32_t glm5next_n_select(const llama_hparams & hparams) {
    GGML_ASSERT(hparams.indexer_kpool > 0);
    GGML_ASSERT(hparams.indexer_top_k >= hparams.indexer_kpool);
    GGML_ASSERT(hparams.indexer_top_k % hparams.indexer_kpool == 0);

    const uint32_t n_select = hparams.indexer_top_k + hparams.indexer_kpool - 1;

    GGML_ASSERT(n_select > hparams.indexer_top_k);
    GGML_ASSERT(n_select == (hparams.indexer_top_k/hparams.indexer_kpool + 1)*hparams.indexer_kpool - 1);

    return n_select;
}

void llama_model_glm5next::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // indexer k_norm is a LayerNorm with bias; without this key it runs at eps 0
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps);
    if (hparams.f_norm_eps <= 0.0f || hparams.f_norm_eps > 2e-6f) {
        LLAMA_LOG_WARN("%s: indexer k_norm eps is %g, but the reference hardcodes 1e-6. "
                "this is invisible to every output comparison; check the converter\n",
                __func__, (double) hparams.f_norm_eps);
    }

    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    GGML_ASSERT(hparams.n_lora_q > 0 && "glm5next requires a q LoRA");
    GGML_ASSERT(hparams.n_rot() == 0 && "glm5next MLA is nope-only");

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,      hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,         hparams.n_embd_head_kda);
    GGML_ASSERT(hparams.ssm_d_conv > 1);
    GGML_ASSERT(hparams.n_embd_head_kda > 0);
    // required: absent, kimi-k3 selects the softplus branch, a different function
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND, hparams.kda_gate_lower_bound);
    GGML_ASSERT(hparams.kda_gate_lower_bound < 0.0f);

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL,      hparams.indexer_kpool);
    GGML_ASSERT(hparams.indexer_kpool > 0);
    GGML_ASSERT(hparams.indexer_top_k % hparams.indexer_kpool == 0);

    const uint32_t n_select = glm5next_n_select(hparams);
    LLAMA_LOG_INFO("%s: indexer selection width = %u cells (%u pools of %u, plus a %u-wide tail)\n",
            __func__, n_select, hparams.indexer_top_k/hparams.indexer_kpool,
            hparams.indexer_kpool, hparams.indexer_kpool - 1);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult > 0);

    // n_embd_out stays n_embd: deepseek4's hc_mult*n_embd overreads t_embd
    hparams.n_embd_out_impl = 0;

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false);

    if (hparams.n_ff_shexp == 0) {
        hparams.n_ff_shexp = hparams.n_ff_exp() * std::max(1u, hparams.n_expert_shared);
    }

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all);

    // n_layer_all, not n_layer(): the NextN block is an attention block the filters read
    uint32_t n_recr = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
        n_recr += il < hparams.n_layer() ? hparams.is_recr_impl[il] : 0;
    }
    GGML_ASSERT(n_recr > 0 && n_recr < hparams.n_layer() && "glm5next needs a per-layer attention.head_count_kv array");

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_indexer_full_impl[il] = !hparams.is_recr_impl[il];
    }

    switch (hparams.n_layer()) {
        case 45: type = hparams.n_embd == 4096 && hparams.n_expert == 288 ? LLM_TYPE_313B_A17B : LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm5next::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner  = head_dim * n_head;
    const int64_t d_conv   = hparams.ssm_d_conv;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t qk_head_dim  = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim   = hparams.n_embd_head_v_mla();

    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t kpool          = hparams.indexer_kpool;

    const int64_t hc_dim     = (int64_t) hparams.dsv4_hc_mult * n_embd;
    const int64_t hc_mix_dim = (2 + (int64_t) hparams.dsv4_hc_mult) * hparams.dsv4_hc_mult;

    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    const int trunk_flags = mtp_only   ? TENSOR_NOT_REQUIRED : 0;
    int       mtp_flags   = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int il = 0; il < n_layer_all; ++il) {
        auto & layer = layers[il];
        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", il), {n_embd}, flags);

        if (il < n_layer) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", il), {hc_dim, hc_mix_dim}, flags);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", il), {hc_mix_dim}, flags);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", il), {3}, flags);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", il), {hc_dim, hc_mix_dim}, flags);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", il), {hc_mix_dim}, flags);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", il), {3}, flags);
        }

        if (hparams.is_recr(il)) {
            create_tensor_qkv(layer, il, n_embd, d_inner, d_inner, d_inner, flags);

            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", il), {d_conv, 1, d_inner, 1}, flags);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", il), {d_conv, 1, d_inner, 1}, flags);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", il), {d_conv, 1, d_inner, 1}, flags);

            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", il), {n_embd, head_dim}, flags);
            layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", il), {head_dim, d_inner}, flags);
            layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", il), {n_embd, head_dim}, flags);
            layer.ssm_g_b = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", il), {head_dim, d_inner}, flags);

            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", il), {n_embd, n_head}, flags);
            layer.ssm_a    = create_tensor(tn(LLM_TENSOR_SSM_A,              il), {n_head}, flags);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT,   "bias",   il), {d_inner}, flags);

            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", il), {head_dim}, flags);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), {d_inner, n_embd}, flags);
        } else {
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", il), {n_embd, q_lora_rank}, flags);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), {q_lora_rank}, flags);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", il), {q_lora_rank, n_head * qk_head_dim}, flags);

            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", il), {n_embd, kv_lora_rank}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), {kv_lora_rank}, flags);
            layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", il), {qk_head_dim, kv_lora_rank, n_head}, flags);
            layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", il), {kv_lora_rank, v_head_dim, n_head}, flags);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), {n_head * v_head_dim, n_embd}, flags);

            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", il), {n_embd_indexer}, flags);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   il), {n_embd_indexer}, flags);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", il), {n_embd, hparams.indexer_n_head}, flags);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", il), {n_embd, n_embd_indexer}, flags);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", il), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

            layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", il), {n_embd, n_embd_indexer}, flags);
            layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", il), {n_embd_indexer, kpool}, flags);
        }

        if (il < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd, n_ff}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {n_ff, n_embd}, flags);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", il), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   il), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), {n_embd, hparams.n_ff_exp(), n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", il), {n_embd, hparams.n_ff_exp(), n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), {hparams.n_ff_exp(), n_embd, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), {n_embd, hparams.n_ff_shexp}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", il), {n_embd, hparams.n_ff_shexp}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), {hparams.n_ff_shexp, n_embd}, flags);
        }

        if (il >= n_layer) {
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), {n_embd}, flags);
            layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), {n_embd}, flags);

            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), {n_embd}, flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

// one conv over concatenated q|k|v: keeps the conv state one block for rollback
ggml_tensor * llama_model_glm5next::graph::build_kda_layer(
        const llama_layer & layer,
        llm_graph_input_rs * inp_rs,
        ggml_tensor * cur,
        int il) {
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_inner      = head_dim * n_head;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto * mctx_cur = inp_rs->mctx;

    // f, g and beta read the layer input, NOT the convolved q/k/v
    ggml_tensor * inp = cur;

    ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, inp);
    ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, inp);
    ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, inp);

    ggml_tensor * qkv = ggml_concat(ctx0, ggml_concat(ctx0, Qcur, Kcur, 0), Vcur, 0);
    qkv = ggml_reshape_3d(ctx0, qkv, 3*d_inner, n_seq_tokens, n_seqs);
    cb(qkv, "kda_qkv", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_in = build_conv_state(inp_rs, conv_states_all, qkv, d_conv, 3*d_inner, il);

    ggml_tensor * conv_w = ggml_concat(ctx0,
            ggml_concat(ctx0,
                ggml_reshape_2d(ctx0, layer.ssm_q_conv, d_conv, d_inner),
                ggml_reshape_2d(ctx0, layer.ssm_k_conv, d_conv, d_inner), 1),
            ggml_reshape_2d(ctx0, layer.ssm_v_conv, d_conv, d_inner), 1);

    // SiLU is applied to the conv output, not to the projections
    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_in, conv_w));
    cb(conv_out, "kda_conv", il);

    const size_t nb_qkv  = ggml_row_size(conv_out->type, 3*d_inner);
    const size_t nb_head = ggml_row_size(conv_out->type, head_dim);

    Qcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, 0);
    Kcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, ggml_row_size(conv_out->type, d_inner));
    Vcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, ggml_row_size(conv_out->type, 2*d_inner));

    // 1e-6 is the reference's own constant, not the model's norm eps
    Qcur = ggml_l2_norm(ctx0, Qcur, 1e-6f);
    Kcur = ggml_l2_norm(ctx0, Kcur, 1e-6f);
    cb(Qcur, "kda_q_norm", il);
    cb(Kcur, "kda_k_norm", il);


    // g = lower_bound * sigmoid(exp(A_log)*(f_b(f_a(x)) + dt_bias)); it scales, not clamps
    ggml_tensor * g = ggml_mul_mat(ctx0, layer.ssm_f_b, ggml_mul_mat(ctx0, layer.ssm_f_a, inp));
    g = ggml_add(ctx0, g, layer.ssm_dt_b);
    g = ggml_reshape_3d(ctx0, g, head_dim, n_head, n_tokens);
    g = ggml_mul(ctx0, g, ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1));
    g = ggml_sigmoid(ctx0, ggml_scale(ctx0, g, -1.0f));
    g = ggml_scale(ctx0, g, hparams.kda_gate_lower_bound);
    g = ggml_reshape_4d(ctx0, g, head_dim, n_head, n_seq_tokens, n_seqs);
    cb(g, "kda_gate", il);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, inp);
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs));
    cb(beta, "kda_beta", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

    ggml_tensor * out = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g, beta, state, il);

    ggml_tensor * o = ggml_cont_3d(ctx0, out, head_dim, n_head, n_tokens);
    cb(o, "kda_scan_out", il);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ssm_g_b, ggml_mul_mat(ctx0, layer.ssm_g_a, inp));
    gate = ggml_reshape_3d(ctx0, gate, head_dim, n_head, n_tokens);

    // plain sigmoid gate, not the SiLU that FusedRMSNormGated defaults to
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, gate));
    cb(gated, "kda_normed", il);

    cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, gated, d_inner, n_tokens));
    cb(cur, "kda_out", il);

    return cur;
}

// the store is NOT gated on the sparse path, the scoring is: gating both leaves cells
// below n_select with no indexer state
ggml_tensor * llama_model_glm5next::graph::build_indexer(
        const llama_layer & layer,
        llm_graph_input_kpool * inp_kp,
        ggml_tensor * cur,
        ggml_tensor * qr,
        bool scoring,
        int il) const {
    const int64_t d_idx   = hparams.indexer_head_size;
    const int64_t n_ihead = hparams.indexer_n_head;
    const int64_t r       = hparams.indexer_kpool;

    const auto * mctx_idx = inp_kp->mctx_idx;

    GGML_ASSERT(layer.indexer_k_norm_b != nullptr && "the indexer k_norm is a LayerNorm with bias");

    ggml_tensor * ik = build_norm(ggml_mul_mat(ctx0, layer.indexer_attn_k, cur),
            layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    cb(ik, "indexer_k", il);

    // a SECOND, INDEPENDENT projection, not a reuse of the key, and cached beside it
    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.indexer_comp_wgate, cur);
    cb(gate, "indexer_gate", il);

    ggml_tensor * packed = ggml_concat(ctx0,
            ggml_reshape_3d(ctx0, ik,   d_idx, 1, n_tokens),
            ggml_reshape_3d(ctx0, gate, d_idx, 1, n_tokens), 1);
    // the third head is the pooled key, written later from other cells; leave it alone
    ggml_build_forward_expand(gf,
            mctx_idx->cpy_k_part(ctx0, ggml_reshape_2d(ctx0, packed, 2*d_idx, n_tokens),
                inp_kp->k_idxs, il, 2*d_idx, 0));

    if (!scoring) {
        return nullptr;
    }

    ggml_tensor * kbuf = mctx_idx->get_k(ctx0, il);

    const int64_t n_kv     = kbuf->ne[2];
    const int64_t n_stream = kbuf->ne[3];
    const int64_t n_tps    = n_tokens/n_stream;
    const int64_t n_pools  = inp_kp->pool_cells->ne[0]/r;

    GGML_ASSERT(kbuf->ne[0] == d_idx && kbuf->ne[1] == 3 &&
            "the pooled indexer cache needs a key head, a gate head and a pooled head");
    GGML_ASSERT(kbuf->nb[1] == (size_t) d_idx*kbuf->nb[0] && "key, gate and pooled must be adjacent in a cell");
    GGML_ASSERT(n_tokens == n_tps*n_stream);

    ggml_tensor * kg_rows = ggml_view_3d(ctx0, kbuf, 2*d_idx, n_kv, n_stream,
            kbuf->nb[2], kbuf->nb[3], 0);

    const int64_t n_new_max = inp_kp->new_pool_cells->ne[0]/r;

    ggml_tensor * members = ggml_get_rows(ctx0, kg_rows, inp_kp->new_pool_cells);
    cb(members, "indexer_pool_members", il);

    const size_t nb_mem = members->nb[1];

    ggml_tensor * mem_k = ggml_view_4d(ctx0, members, d_idx, r, n_new_max, n_stream,
            nb_mem, nb_mem*r, members->nb[2], 0);
    ggml_tensor * mem_g = ggml_view_4d(ctx0, members, d_idx, r, n_new_max, n_stream,
            nb_mem, nb_mem*r, members->nb[2], d_idx*members->nb[0]);

    // r-way softmaxes over the SLOT axis, so it must be dim 0; ape is added PRE-softmax
    ggml_tensor * keys_t = ggml_cont(ctx0, ggml_permute(ctx0, mem_k, 1, 0, 2, 3));
    ggml_tensor * gate_t = ggml_cont(ctx0, ggml_permute(ctx0, mem_g, 1, 0, 2, 3));

    ggml_tensor * ape = ggml_cont(ctx0, ggml_transpose(ctx0, layer.indexer_comp_ape));
    gate_t = ggml_add(ctx0, gate_t, ggml_reshape_4d(ctx0, ape, r, d_idx, 1, 1));

    ggml_tensor * probs = ggml_soft_max(ctx0, gate_t);
    cb(probs, "indexer_pool_probs", il);

    ggml_tensor * pool_new = ggml_sum_rows(ctx0, ggml_mul(ctx0, keys_t, probs));
    pool_new = ggml_reshape_2d(ctx0, pool_new, d_idx, n_new_max*n_stream);
    cb(pool_new, "indexer_pool_new", il);

    // the write is by GLOBAL row while pool_reps is stream-local, so the read must not chain
    // off the write tensor; expand first and let build order sequence them
    ggml_build_forward_expand(gf,
            mctx_idx->cpy_k_part(ctx0, pool_new, inp_kp->new_pool_reps, il, d_idx, 2*d_idx));

    ggml_tensor * pooled_rd = ggml_view_3d(ctx0, kbuf, d_idx, n_kv, n_stream,
            kbuf->nb[2], kbuf->nb[3], 2*d_idx*kbuf->nb[0]);

    ggml_tensor * pool_k = ggml_get_rows(ctx0, pooled_rd, inp_kp->pool_reps);
    pool_k = ggml_reshape_4d(ctx0, pool_k, d_idx, n_pools, 1, n_stream);
    cb(pool_k, "indexer_pool_k", il);

    // no rope: n_rot() is 0 for the whole text tower
    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_4d(ctx0, iq, d_idx, n_ihead, n_tps, n_stream);
    cb(iq, "indexer_q", il);

    // sign-unconstrained head weights; PREC_F32 is load-bearing, bf16 swaps near-tied pools
    ggml_tensor * w = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
    ggml_mul_mat_set_prec(w, GGML_PREC_F32);
    w = ggml_reshape_4d(ctx0, w, n_ihead, n_tps, 1, n_stream);
    w = ggml_scale(ctx0, w, 1.0f/sqrtf(float(d_idx*n_ihead)));
    cb(w, "indexer_weights", il);

    ggml_tensor * pool_score = nullptr;

    if (cparams.fused_lid) {
        // pool_k stays f32 so the kernel takes its f32 path; f16 wmma would undo the prec
        ggml_tensor * pool_kf = ggml_reshape_4d(ctx0, pool_k, d_idx, 1, n_pools, n_stream);

        pool_score = ggml_lightning_indexer(ctx0, iq, pool_kf, w, inp_kp->pool_bias_f16);
        cb(pool_score, "indexer_pool_score", il);

        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, pool_score, il});

        pool_score = ggml_reshape_3d(ctx0, pool_score, n_pools, n_tps, n_stream);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, pool_k, ggml_permute(ctx0, iq, 0, 2, 1, 3));

        // the ReLU sits BETWEEN the per-head dot and the head weighting; either side differs
        kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
        ggml_tensor * score = ggml_relu(ctx0, kq);
        cb(score, "indexer_score", il);

        pool_score = ggml_sum_rows(ctx0, ggml_mul(ctx0, score, w));
        pool_score = ggml_cont(ctx0, ggml_permute(ctx0, pool_score, 2, 1, 0, 3));
        pool_score = ggml_reshape_3d(ctx0, pool_score, n_pools, n_tps, n_stream);

        pool_score = ggml_add(ctx0, pool_score, inp_kp->pool_bias);
        cb(pool_score, "indexer_pool_score", il);
    }

    // top-k over POOLS then expand: a cell-level top-k is wrong, relu ties span pool bounds
    const int64_t select_k = llama_kpool_select_k(n_pools, hparams.indexer_top_k, r);
    GGML_ASSERT(select_k > 0 && select_k <= n_pools);

    ggml_tensor * sel = ggml_cont(ctx0, ggml_top_k(ctx0, pool_score, (int) select_k));
    cb(sel, "indexer_top_k_pools", il);

    ggml_tensor * pc3      = ggml_reshape_3d(ctx0, inp_kp->pool_cells, r, n_pools, n_stream);
    ggml_tensor * sel_flat = ggml_reshape_2d(ctx0, sel, select_k*n_tps, n_stream);

    ggml_tensor * top_k = ggml_get_rows(ctx0, pc3, sel_flat);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && "pool_cells is I32, so the gather stays I32");
    top_k = ggml_reshape_3d(ctx0, top_k, r*select_k, n_tps, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// absorbed form (deepseek2/glm-dsa): q_nope goes through wk_b; the naive form needs a V cache
ggml_tensor * llama_model_glm5next::graph::build_dsa_layer(
        const llama_layer & layer,
        llm_graph_input_attn_k * inp_attn,
        llm_graph_input_kpool * inp_kp,
        bool scoring,
        ggml_tensor * cur,
        int il) const {
    const int64_t qk_head_dim  = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    GGML_ASSERT(hparams.n_rot() == 0);

    // scale is over the MLA head size, as in the reference, not the absorbed width
    const float kq_scale = 1.0f/sqrtf(float(qk_head_dim));

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "dsa_q_a_norm", il);

    ggml_tensor * top_k = inp_kp ? build_indexer(layer, inp_kp, cur, qr, scoring, il) : nullptr;

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, qk_head_dim, n_head, n_tokens);
    cb(q, "dsa_q_b", il);

    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv, "dsa_kv_a_norm", il);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);

    q = ggml_mul_mat(ctx0, layer.wk_b, q);

    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    cb(q, "dsa_q_absorbed", il);

    // absorbed MLA is MQA: one head of keys, and V is the same latent row as K
    ggml_tensor * k = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
    cb(k, "dsa_kv_latent", il);

    if (top_k) {
        cur = build_attn_sparse(inp_attn,
                layer.wo, nullptr, nullptr,
                q, k, k, nullptr, nullptr, layer.wv_b,
                top_k, inp_kp->sel_mask, inp_kp->cand_mask, kq_scale, il);
    } else {
        cur = build_attn(inp_attn,
                layer.wo, nullptr, nullptr,
                q, k, k, nullptr, nullptr, layer.wv_b, kq_scale, il);
    }
    cb(cur, "dsa_out", il);

    return cur;
}

ggml_tensor * llama_model_glm5next::graph::build_layer_attn(
        const llama_model & model,
        llm_graph_input_mem_hybrid_k * inp_mem,
        llm_graph_input_kpool * inp_kp,
        bool scoring,
        ggml_tensor * cur,
        int il) {
    if (hparams.is_recr(il)) {
        return build_kda_layer(model.layers[il], inp_mem->get_recr(), cur, il);
    }

    return build_dsa_layer(model.layers[il], inp_mem->get_attn(), inp_kp, scoring, cur, il);
}

ggml_tensor * llama_model_glm5next::graph::build_layer_ffn(
        const llama_model & model,
        ggml_tensor * cur,
        int il) const {
    const auto & layer = model.layers[il];

    if (il < (int) hparams.n_layer_dense_lead) {
        return build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }

    // noaux_tc: exp_probs_b biases top-k selection only; the weights stay unbiased
    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used(),
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);

    ggml_tensor * shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(shexp, "ffn_shexp", il);

    // shared expert unscaled: routed_scaling_factor applies to the routed weights only
    return ggml_add(ctx0, moe_out, shexp);
}

llama_model_glm5next::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    ggml_tensor * cur;

    ggml_tensor * inp         = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    llm_graph_input_mem_hybrid_k * inp_mem = build_inp_mem_hybrid_k();

    // gated on n_ctx, not n_kv, which grows and would flip the graph topology mid-run
    llm_graph_input_kpool * inp_kp = nullptr;
    bool indexer_scoring = false;
    {
        const auto * mctx_hyb = static_cast<const llama_memory_hybrid_context *>(mctx);

        if (mctx_hyb->get_idx() != nullptr) {
            indexer_scoring = cparams.n_ctx > glm5next_n_select(hparams);

            inp_kp = build_inp_kpool(mctx_hyb,
                    inp_mem->get_attn()->get_kq_mask(), indexer_scoring);
        }
    }

    GGML_ASSERT(ubatch.n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);

    const int64_t hc = hparams.dsv4_hc_mult;

    // exact copies: no scaling, no one-hot into stream 0
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_layer_attn(model, inp_mem, inp_kp, indexer_scoring, cur, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        // expand before the sublayer so op offload does not pull mHC state to the experts
        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(model, cur, il);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = build_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    // unmasked nextn embeddings need all rows, so early output masking is skipped here
    const bool mask_early = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

    if (inp_out_ids && mask_early) {
        // get_rows needs one token's streams contiguous
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
    }

    // unweighted mean, not DeepSeek-V4's learned gated head
    cur = build_hc_mean(ctx0, inpL);
    cb(cur, "hc_mean", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids && !mask_early) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// NextN draft head: an ordinary DSA layer, but a plain residual and no hc_* tensors
llama_model_glm5next::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "glm5next MTP supports a single NextN block");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;

    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");

    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm &&
            "glm5next MTP block tensors missing; convert without --no-mtp");
    GGML_ASSERT(!layer.hc_attn_fn && "the NextN block has no mHC mixer");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    llm_graph_input_mem_hybrid_k * inp_mem = build_inp_mem_hybrid_k();

    // the NextN block is DSA: nothing consumes the recurrent half, so s_copy never enters the
    // graph while set_input would still read its buffer
    ggml_build_forward_expand(gf, inp_mem->get_recr()->s_copy);

    llm_graph_input_kpool * inp_kp = nullptr;
    bool indexer_scoring = false;
    {
        const auto * mctx_hyb = static_cast<const llama_memory_hybrid_context *>(mctx);

        if (mctx_hyb->get_idx() != nullptr) {
            indexer_scoring = cparams.n_ctx > glm5next_n_select(hparams);

            inp_kp = build_inp_kpool(mctx_hyb,
                    inp_mem->get_attn()->get_kq_mask(), indexer_scoring);
        }
    }

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj,
            ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0), layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * residual = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_layer_attn(model, inp_mem, inp_kp, indexer_scoring, cur, il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, residual);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    cur = build_layer_ffn(model, cur, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm_w && "glm5next MTP: no nextn.shared_head_norm and no output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "glm5next MTP: no nextn.shared_head_head and no output");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_glm5next::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }

    return std::make_unique<graph>(*this, params);
}
