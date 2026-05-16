#include "models.h"

#include "ggml.h"
#include "llama-memory-recurrent.h"

#include <cmath>

void llama_model_zaya::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);

    const uint32_t n_qk = (hparams.n_head() + hparams.n_head_kv()) * hparams.n_embd_head_k();
    hparams.ssm_d_inner = 2*n_qk + hparams.n_embd; // CCA conv state + delayed value stream state
    hparams.ssm_d_state = 1;
    hparams.ssm_n_group = 0;

    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = (i % 2) == 0;
    }

    switch (hparams.n_layer) {
        case 80: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_zaya::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output norm
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // output (tied with tok_embd if not present)
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = tok_embd;
    }

    zaya_res_scale_hs    = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_FINAL,    "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_hs_b  = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_FINAL,    "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_res   = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_FINAL,   "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_FINAL,   "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t d_conv      = hparams.ssm_d_conv;
    // Router MLP hidden size (zaya_mlp_expansion)
    const int64_t n_ff_exp    = hparams.n_ff_exp;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head    = hparams.n_head(i);
        const int64_t n_head_kv = hparams.n_head_kv(i);
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;
        const int64_t n_ff      = hparams.n_ff(i);
        const int64_t n_expert  = hparams.n_expert;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // CCA attention layers (even indices only)
        if (i % 2 == 0) {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k}, 0);

            layer.cca_val_proj1 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ1, "weight", i),
                {n_embd, n_embd_k / 2}, 0);
            layer.cca_val_proj2 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ2, "weight", i),
                {n_embd, n_embd_k / 2}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

            layer.cca_conv_dw   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, n_qk}, 0);
            layer.cca_conv_dw_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {n_qk}, TENSOR_NOT_REQUIRED);

            layer.cca_conv_grp   = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "weight", i),
                {d_conv, n_qk / n_groups, n_qk}, 0);
            layer.cca_conv_grp_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "bias", i), {n_qk}, 0);

            layer.cca_k_scale = create_tensor(tn(LLM_TENSOR_CCA_K_SCALE, "weight", i), {n_head_kv}, 0);
        }

        // Residual scaling
        layer.res_scale_hs   = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS, "weight", i), {n_embd}, 0);
        layer.res_scale_hs_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res  = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        // MoE layers (odd indices)
        if (i % 2 == 1) {
            // Router network
            layer.zaya_router_down   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),
                {n_embd, n_ff_exp}, 0);
            layer.zaya_router_down_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "bias", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.zaya_router_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i),
                {n_ff_exp}, 0);
            layer.zaya_router_mlp0   = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i),
                {n_ff_exp, n_ff_exp}, 0);
            layer.zaya_router_mlp0_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.zaya_router_mlp2   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2, "weight", i),
                {n_ff_exp, n_ff_exp}, 0);
            layer.zaya_router_mlp2_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2, "bias", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.zaya_router_mlp4   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP4, "weight", i),
                {n_ff_exp, n_expert + 1}, 0);
            layer.zaya_router_biases = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_BIASES, "weight", i),
                {n_expert + 1}, TENSOR_NOT_REQUIRED);
            layer.zaya_router_eda_scale = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_EDA_SCALE, "weight", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);

            // MoE experts (fused gate_up and down)
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff, n_expert, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {n_ff, n_embd, n_expert}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_zaya::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_zaya::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_expert    = hparams.n_expert;
    const int64_t n_seqs      = ubatch.n_seqs;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_tokens % n_seqs == 0);

    const int64_t n_seq_tokens = n_tokens / n_seqs;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp = build_inp_mem_hybrid();
    auto * inp_recr = inp->get_recr();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * residual    = nullptr;
    ggml_tensor * prev_router = nullptr;

    const auto apply_res_scale = [&](ggml_tensor * x, ggml_tensor * scale, ggml_tensor * bias, const char * name, int il) {
        if (scale == nullptr) {
            return x;
        }
        if (bias != nullptr) {
            x = ggml_add(ctx0, x, bias);
        }
        x = ggml_mul(ctx0, x, scale);
        cb(x, name, il);
        return x;
    };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;
        const int64_t n_gqa     = n_head / n_head_kv;

        ggml_tensor * hidden_states = apply_res_scale(inpL, layer.res_scale_hs, layer.res_scale_hs_b, "res_scale_hs", il);
        if (residual != nullptr) {
            residual = apply_res_scale(residual, layer.res_scale_res, layer.res_scale_res_b, "res_scale_res", il);
            residual = ggml_add(ctx0, hidden_states, residual);
        } else {
            residual = hidden_states;
        }
        cb(residual, "residual", il);

        // Pre-norm
        cur = build_norm(residual, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "input_norm", il);

        if (il % 2 == 0) {
            // ===== CCA Attention =====
            const int64_t conv_state_size = 2*n_qk;
            const int64_t cca_state_size  = conv_state_size + n_embd;
            GGML_ASSERT((int64_t) hparams.n_embd_s() == cca_state_size);

            ggml_tensor * cca_state_all = inp_recr->mctx->get_s_l(il);
            ggml_tensor * cca_state     = build_rs(inp_recr, cca_state_all, hparams.n_embd_s(), n_seqs);
            cb(cca_state, "cca_state", il);

            ggml_tensor * conv_state = ggml_view_3d(ctx0, cca_state, 2, n_qk, n_seqs,
                    2*ggml_element_size(cca_state),
                    cca_state->nb[1],
                    0);
            cb(conv_state, "cca_conv_state", il);

            ggml_tensor * prev_hs = ggml_view_2d(ctx0, cca_state, n_embd, n_seqs,
                    cca_state->nb[1],
                    conv_state_size*ggml_element_size(cca_state));
            cb(prev_hs, "cca_prev_hs", il);

            // Q, K projections
            ggml_tensor * Qraw = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(Qraw, "Qraw", il);
            ggml_tensor * Kraw = ggml_mul_mat(ctx0, layer.wk, cur);
            cb(Kraw, "Kraw", il);

            // HF uses a delayed hidden-state stream for val_proj2. During decode this
            // comes from the recurrent state; during prefill it is a one-token shift.
            ggml_tensor * cur_state_src = ggml_cont(ctx0, cur);
            ggml_tensor * cur_seq = ggml_reshape_3d(ctx0, cur_state_src, n_embd, n_seq_tokens, n_seqs);

            ggml_tensor * hs_d = ggml_reshape_3d(ctx0, prev_hs, n_embd, 1, n_seqs);
            if (n_seq_tokens > 1) {
                ggml_tensor * cur_shift = ggml_view_3d(ctx0, cur_seq, n_embd, n_seq_tokens - 1, n_seqs,
                        cur_seq->nb[1],
                        cur_seq->nb[2],
                        0);
                hs_d = ggml_concat(ctx0, hs_d, cur_shift, 1);
            }
            hs_d = ggml_reshape_2d(ctx0, ggml_cont(ctx0, hs_d), n_embd, n_tokens);
            cb(hs_d, "cca_hs_d", il);

            // V = concat(val_proj1(x), val_proj2(x delayed)) -> [n_embd_k, n_tokens]
            ggml_tensor * V1 = ggml_mul_mat(ctx0, layer.cca_val_proj1, cur);
            cb(V1, "V1", il);
            ggml_tensor * V2 = ggml_mul_mat(ctx0, layer.cca_val_proj2, hs_d);
            cb(V2, "V2", il);
            ggml_tensor * Vcur = ggml_concat(ctx0, V1, V2, 0);
            cb(Vcur, "Vcur", il);

            // Concat Q+K for conv: [n_qk, n_tokens]
            ggml_tensor * QKraw = ggml_concat(ctx0, Qraw, Kraw, 0);
            cb(QKraw, "QKraw", il);

            ggml_tensor * Qpre = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Qraw), n_embd_head, n_head, n_tokens);
            ggml_tensor * Kpre = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Kraw), n_embd_head, n_head_kv, n_tokens);

            ggml_tensor * Kpre_grouped = ggml_reshape_4d(ctx0, Kpre, n_embd_head, 1, n_head_kv, n_tokens);
            Kpre_grouped = ggml_repeat_4d(ctx0, Kpre_grouped, n_embd_head, n_gqa, n_head_kv, n_tokens);
            ggml_tensor * Kpre_rep = ggml_reshape_3d(ctx0, Kpre_grouped, n_embd_head, n_head, n_tokens);
            ggml_tensor * qk_mean_q = ggml_scale(ctx0, ggml_add(ctx0, Qpre, Kpre_rep), 0.5f);
            cb(qk_mean_q, "qk_mean_q", il);

            ggml_tensor * Qgroup = ggml_reshape_4d(ctx0, Qpre, n_embd_head, n_gqa, n_head_kv, n_tokens);
            Qgroup = ggml_permute(ctx0, Qgroup, 1, 0, 2, 3);
            Qgroup = ggml_cont(ctx0, Qgroup);
            ggml_tensor * Qmean = ggml_mean(ctx0, Qgroup);
            Qmean = ggml_reshape_3d(ctx0, Qmean, n_embd_head, n_head_kv, n_tokens);
            ggml_tensor * qk_mean_k = ggml_scale(ctx0, ggml_add(ctx0, Qmean, Kpre), 0.5f);
            cb(qk_mean_k, "qk_mean_k", il);

            ggml_tensor * QKraw_t = ggml_cont(ctx0, ggml_transpose(ctx0, QKraw));
            QKraw_t = ggml_reshape_3d(ctx0, QKraw_t, n_seq_tokens, n_qk, n_seqs);

            ggml_tensor * conv_input = ggml_concat(ctx0, conv_state, QKraw_t, 0);
            cb(conv_input, "cca_conv_input", il);

            ggml_tensor * last_conv_states = ggml_view_3d(ctx0, conv_input, 2, n_qk, n_seqs,
                    conv_input->nb[1],
                    conv_input->nb[2],
                    n_seq_tokens*conv_input->nb[0]);
            cb(last_conv_states, "cca_last_conv_states", il);

            const auto kv_head = inp_recr->mctx->get_head();
            ggml_tensor * conv_state_update_target = ggml_view_2d(ctx0, cca_state_all, conv_state_size, n_seqs,
                    cca_state_all->nb[1],
                    kv_head*cca_state_size*ggml_element_size(cca_state_all));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, conv_state_update_target));

            ggml_tensor * last_hs = ggml_view_2d(ctx0, cur_seq, n_embd, n_seqs,
                    cur_seq->nb[2],
                    (n_seq_tokens - 1)*cur_seq->nb[1]);
            ggml_tensor * prev_hs_update_target = ggml_view_2d(ctx0, cca_state_all, n_embd, n_seqs,
                    cca_state_all->nb[1],
                    (kv_head*cca_state_size + conv_state_size)*ggml_element_size(cca_state_all));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_hs, prev_hs_update_target));

            ggml_tensor * conv_dw = layer.cca_conv_dw;
            if (conv_dw->type != GGML_TYPE_F16) {
                conv_dw = ggml_cont(ctx0, ggml_cast(ctx0, conv_dw, GGML_TYPE_F16));
            }
            conv_dw = ggml_reshape_3d(ctx0, conv_dw, conv_dw->ne[0], 1, n_qk);
            ggml_tensor * QK = ggml_conv_1d_dw(ctx0, conv_dw, conv_input, 1, 0, 1);
            if (layer.cca_conv_dw_b) {
                QK = ggml_add(ctx0, QK, ggml_reshape_3d(ctx0, layer.cca_conv_dw_b, 1, n_qk, 1));
            }
            cb(QK, "QK_dw", il);

            ggml_tensor * conv_grp = layer.cca_conv_grp;
            if (conv_grp->type != GGML_TYPE_F16) {
                conv_grp = ggml_cont(ctx0, ggml_cast(ctx0, conv_grp, GGML_TYPE_F16));
            }
            QK = ggml_conv_1d_grouped(ctx0, conv_grp, QK, 1, 0, 1, n_groups);
            QK = ggml_add(ctx0, QK, ggml_reshape_3d(ctx0, layer.cca_conv_grp_b, 1, n_qk, 1));
            cb(QK, "QK_grp", il);

            QK = ggml_cont(ctx0, ggml_permute(ctx0, QK, 1, 0, 2, 3));
            QK = ggml_reshape_2d(ctx0, QK, n_qk, n_tokens);

            ggml_tensor * Q_conv = ggml_view_2d(ctx0, QK, n_embd_q, n_tokens, QK->nb[1], 0);
            ggml_tensor * K_conv = ggml_view_2d(ctx0, QK, n_embd_k, n_tokens, QK->nb[1], n_embd_q*ggml_element_size(QK));

            ggml_tensor * Qcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Q_conv), n_embd_head, n_head, n_tokens);
            ggml_tensor * Kcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, K_conv), n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_add(ctx0, Qcur, qk_mean_q);
            Kcur = ggml_add(ctx0, Kcur, qk_mean_k);

            Qcur = ggml_scale(ctx0, ggml_l2_norm(ctx0, Qcur, 1e-12f), sqrtf((float) n_embd_head));
            Kcur = ggml_scale(ctx0, ggml_l2_norm(ctx0, Kcur, 1e-12f), sqrtf((float) n_embd_head));
            Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_3d(ctx0, layer.cca_k_scale, 1, n_head_kv, 1));
            cb(Qcur, "Qcur_pre_rope", il);
            cb(Kcur, "Kcur_pre_rope", il);

            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            Vcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Vcur), n_embd_head, n_head_kv, n_tokens);

            // GQA attention
            cur = build_attn(inp->get_attn(), layer.wo, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                1.0f / sqrtf((float) n_embd_head), il);
            cb(cur, "attn_out", il);

        } else {
            // ===== MoE Layer =====

            // Build Zaya router network:
            // down_proj -> optional EDA -> RMSNorm -> GELU MLP -> 17 logits.

            ggml_tensor * router_h = ggml_mul_mat(ctx0, layer.zaya_router_down, cur);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_down_b);
            cb(router_h, "router_down", il);

            if (prev_router != nullptr && layer.zaya_router_eda_scale != nullptr) {
                router_h = ggml_add(ctx0, router_h, ggml_mul(ctx0, prev_router, layer.zaya_router_eda_scale));
                cb(router_h, "router_eda", il);
            }

            prev_router = router_h;

            router_h = build_norm(router_h, layer.zaya_router_norm, nullptr, LLM_NORM_RMS, il);
            cb(router_h, "router_norm", il);

            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp0, router_h);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_mlp0_b);
            router_h = ggml_gelu(ctx0, router_h);
            cb(router_h, "router_mlp0", il);

            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp2, router_h);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_mlp2_b);
            router_h = ggml_gelu(ctx0, router_h);
            cb(router_h, "router_mlp2", il);

            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp4, router_h);
            cb(router_h, "router_logits", il);

            ggml_tensor * router_probs = ggml_soft_max(ctx0, router_h);
            cb(router_probs, "router_probs", il);

            // Keep the MOD skip expert in the softmax denominator, then route
            // over real experts only. The checkpoint's skip bias keeps MOD unused.
            ggml_tensor * gate_probs = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, router_probs, n_expert, n_tokens, router_probs->nb[1], 0));
            cb(gate_probs, "gate_probs", il);

            ggml_tensor * expert_biases = nullptr;
            if (layer.zaya_router_biases != nullptr) {
                expert_biases = ggml_view_1d(ctx0, layer.zaya_router_biases, n_expert, 0);
            }

            cur = build_moe_ffn(cur,
                /* gate_inp */        nullptr,
                /* up_exps */         nullptr,
                /* gate_exps */       nullptr,
                /* down_exps */       layer.ffn_down_exps,
                /* exp_probs_b */     expert_biases,
                /* n_expert */        n_expert,
                /* n_expert_used */   hparams.n_expert_used,
                /* type_op */         LLM_FFN_SILU,
                /* norm_w */          false,
                /* w_scale */         1.0f,
                /* gating_op */       LLAMA_EXPERT_GATING_FUNC_TYPE_NONE,
                /* il */              il,
                /* probs_in */        gate_probs,
                /* gate_up_exps */    layer.ffn_gate_up_exps);
            cb(cur, "moe_out", il);
        }

        inpL = cur;
    }

    ggml_tensor * final_hidden = apply_res_scale(inpL, model.zaya_res_scale_hs, model.zaya_res_scale_hs_b, "final_res_scale_hs", -1);
    if (residual != nullptr) {
        residual = apply_res_scale(residual, model.zaya_res_scale_res, model.zaya_res_scale_res_b, "final_res_scale_res", -1);
        cur = ggml_add(ctx0, final_hidden, residual);
    } else {
        cur = final_hidden;
    }
    cb(cur, "final_residual", -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    // final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // output
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
