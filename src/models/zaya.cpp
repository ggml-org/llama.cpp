#include "models.h"

#include "ggml.h"

void llama_model_zaya::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);

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

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t d_conv      = hparams.ssm_d_conv;
    // Router MLP hidden size (zaya_mlp_expansion = 256 for ZAYA1-8B)
    const int64_t n_ff_exp    = 256;

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

        // CCA projections (present on all layers)
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k}, 0);

        // CCA: V = concat(val_proj1(x), val_proj2(x)) → {n_embd_k}
        layer.cca_val_proj1 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ1, "weight", i),
            {n_embd, n_embd_head}, 0);
        layer.cca_val_proj2 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ2, "weight", i),
            {n_embd, n_embd_head}, 0);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

        // CCA conv_qk.0 (depthwise, causal)
        layer.cca_conv_dw   = create_tensor(tn(LLM_TENSOR_CCA_CONV_DW, "weight", i), {d_conv, n_qk}, 0);
        layer.cca_conv_dw_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_DW_B, "bias", i), {n_qk}, TENSOR_NOT_REQUIRED);

        // CCA conv_qk.1 (grouped, groups = n_groups)
        layer.cca_conv_grp   = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "weight", i),
            {d_conv, n_qk / n_groups, n_qk}, 0);
        layer.cca_conv_grp_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "bias", i), {n_qk}, 0);

        // CCA per-KV-head temperature
        layer.cca_k_scale = create_tensor(tn(LLM_TENSOR_CCA_K_SCALE, "weight", i), {n_head_kv}, 0);

        // Residual scaling
        layer.res_scale_hs   = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS, "weight", i), {n_embd}, 0);
        layer.res_scale_hs_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_B, "bias", i), {n_embd}, 0);
        layer.res_scale_res  = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_B, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        // MoE layers (odd indices)
        if (i % 2 == 1) {
            // Router network
            layer.zaya_router_down   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_DOWN, "weight", i),
                {n_embd, n_ff_exp}, 0);
            layer.zaya_router_down_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_DOWN_B, "bias", i),
                {n_ff_exp}, 0);
            layer.zaya_router_norm   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_NORM, "weight", i),
                {n_ff_exp}, 0);
            layer.zaya_router_mlp0   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP0, "weight", i),
                {n_ff_exp, n_ff_exp}, 0);
            layer.zaya_router_mlp0_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP0_B, "bias", i),
                {n_ff_exp}, 0);
            layer.zaya_router_mlp2   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2, "weight", i),
                {n_ff_exp, n_ff_exp}, 0);
            layer.zaya_router_mlp2_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2_B, "bias", i),
                {n_ff_exp}, 0);
            layer.zaya_router_mlp4   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP4, "weight", i),
                {n_ff_exp, n_expert + 1}, 0);
            layer.zaya_router_biases = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_BIASES, "weight", i),
                {n_expert + 1}, TENSOR_NOT_REQUIRED);
            layer.zaya_router_eda_scale = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_EDA_SCALE, "weight", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);

            // MoE experts (fused gate_up and down)
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff, n_expert, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i),
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
    const int64_t d_conv      = hparams.ssm_d_conv;
    const int64_t n_expert    = hparams.n_expert;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;

        ggml_tensor * inpSA = inpL;

        // Pre-norm
        cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (il % 2 == 0) {
            // ===== CCA Attention =====

            // Q, K projections
            ggml_tensor * Qraw = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(Qraw, "Qraw", il);
            ggml_tensor * Kraw = ggml_mul_mat(ctx0, layer.wk, cur);
            cb(Kraw, "Kraw", il);

            // V = concat(val_proj1(x), val_proj2(x)) → [n_embd_k, n_tokens]
            ggml_tensor * V1 = ggml_mul_mat(ctx0, layer.cca_val_proj1, cur);
            cb(V1, "V1", il);
            ggml_tensor * V2 = ggml_mul_mat(ctx0, layer.cca_val_proj2, cur);
            cb(V2, "V2", il);
            ggml_tensor * Vcur = ggml_concat(ctx0, V1, V2, 0);
            cb(Vcur, "Vcur", il);

            // Concat Q+K for conv: [n_qk, n_tokens]
            ggml_tensor * QK = ggml_concat(ctx0, Qraw, Kraw, 0);
            cb(QK, "QK_cat", il);

            // conv_qk.0 (depthwise, causal)
            {
                ggml_tensor * QK_t = ggml_cont(ctx0, ggml_transpose(ctx0, QK));
                ggml_tensor * pad = ggml_new_tensor_2d(ctx0, QK_t->type, d_conv - 1, n_qk);
                pad = ggml_scale(ctx0, pad, 0.0f);
                ggml_tensor * QK_padded = ggml_concat(ctx0, pad, QK_t, 0);

                QK = ggml_ssm_conv(ctx0, QK_padded, layer.cca_conv_dw);
                if (layer.cca_conv_dw_b) {
                    QK = ggml_add(ctx0, QK, layer.cca_conv_dw_b);
                }
                cb(QK, "QK_dw", il);
            }

            // conv_qk.1 (grouped, causal)
            {
                ggml_tensor * pad = ggml_new_tensor_2d(ctx0, QK->type, d_conv - 1, n_qk);
                pad = ggml_scale(ctx0, pad, 0.0f);
                ggml_tensor * QK_padded = ggml_concat(ctx0, pad, QK, 0);

                QK = ggml_conv_1d_grouped(ctx0, layer.cca_conv_grp, QK_padded, 1, 0, 1, n_groups);
                QK = ggml_add(ctx0, QK, layer.cca_conv_grp_b);
                cb(QK, "QK_grp", il);
            }

            // Transpose back to [n_qk, n_tokens]
            QK = ggml_cont(ctx0, ggml_transpose(ctx0, QK));

            // Split Q_conv, K_conv
            ggml_tensor * Q_conv = ggml_view_2d(ctx0, QK, n_embd_q, n_tokens,
                QK->nb[1], 0);
            ggml_tensor * K_conv = ggml_view_2d(ctx0, QK, n_embd_k, n_tokens,
                QK->nb[1], n_embd_q * ggml_element_size(QK));

            // QK mean skip connection
            ggml_tensor * Qcur = ggml_scale(ctx0, ggml_add(ctx0, Q_conv, Qraw), 0.5f);
            ggml_tensor * Kcur = ggml_scale(ctx0, ggml_add(ctx0, K_conv, Kraw), 0.5f);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            // RMSNorm on concat(Q, K) — weightless (unit RMSNorm)
            ggml_tensor * QK_for_norm = ggml_concat(ctx0, Qcur, Kcur, 0);
            QK_for_norm = build_norm(QK_for_norm, nullptr, nullptr, LLM_NORM_RMS, il);
            cb(QK_for_norm, "QK_normed", il);

            // Split back
            Qcur = ggml_view_2d(ctx0, QK_for_norm, n_embd_q, n_tokens,
                QK_for_norm->nb[1], 0);
            Kcur = ggml_view_2d(ctx0, QK_for_norm, n_embd_k, n_tokens,
                QK_for_norm->nb[1], n_embd_q * ggml_element_size(QK_for_norm));

            // Per-KV-head temperature scaling on K
            // Kcur: [n_embd_k=256, n_tokens], reshape to [n_embd_head, n_head_kv, n_tokens]
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            // cca_k_scale: [n_head_kv] → broadcast
            Kcur = ggml_mul(ctx0, Kcur, layer.cca_k_scale);
            cb(Kcur, "Kcur_scaled", il);

            // Reshape for attention
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // GQA attention
            cur = build_attn(inp->get_attn(), layer.wo, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                1.0f / sqrtf((float) n_embd_head), il);
            cb(cur, "attn_out", il);

        } else {
            // ===== MoE Layer =====

            // Build Zaya router network:
            // down_proj → RMSNorm → SiLU(MLP0) → MLP2 → MLP4 → 17 logits → take first 16

            ggml_tensor * router_h = ggml_mul_mat(ctx0, layer.zaya_router_down, cur);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_down_b);
            cb(router_h, "router_down", il);

            router_h = build_norm(router_h, layer.zaya_router_norm, nullptr, LLM_NORM_RMS, il);
            cb(router_h, "router_norm", il);

            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp0, router_h);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_mlp0_b);
            router_h = ggml_silu(ctx0, router_h);
            cb(router_h, "router_mlp0", il);

            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp2, router_h);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_mlp2_b);
            cb(router_h, "router_mlp2", il);

            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp4, router_h);
            // router_h now has shape [17, n_tokens] — 16 expert logits + 1 MOD skip
            cb(router_h, "router_logits", il);

            // Take only the first 16 logits (expert routing), ignore MOD skip (index 16)
            ggml_tensor * gate_inp = ggml_view_2d(ctx0, router_h, n_expert, n_tokens,
                router_h->nb[1], 0);
            cb(gate_inp, "gate_inp", il);

            // MoE FFN with topk=1 (pass router logits as probs_in)
            cur = build_moe_ffn(cur,
                /* gate_inp */        nullptr,
                /* up_exps */         nullptr,
                /* gate_exps */       nullptr,
                /* down_exps */       layer.ffn_down_exps,
                /* exp_probs_b */     nullptr,
                /* n_expert */        n_expert,
                /* n_expert_used */   hparams.n_expert_used,
                /* type_op */         LLM_FFN_SILU,
                /* norm_w */          false,
                /* w_scale */         1.0f,
                /* gating_op */       LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                /* il */              il,
                /* probs_in */        gate_inp,
                /* gate_up_exps */    layer.ffn_gate_up_exps);
            cb(cur, "moe_out", il);
        }

        // select output tokens on last layer
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual scaling: cur = hs_scale * cur + hs_bias
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.res_scale_hs), layer.res_scale_hs_b);
        cb(cur, "scaled_out", il);

        // Residual scaling: inpSA = res_scale * inpSA + res_bias (if present)
        if (layer.res_scale_res) {
            inpSA = ggml_add(ctx0, ggml_mul(ctx0, inpSA, layer.res_scale_res), layer.res_scale_res_b);
            cb(inpSA, "scaled_residual", il);
        }

        // Residual add
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

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
