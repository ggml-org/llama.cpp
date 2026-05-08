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

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    const int64_t d_conv = hparams.ssm_d_conv;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_i    = hparams.n_head(i);
        const int64_t n_head_kv_i = hparams.n_head_kv(i);
        const int64_t n_embd_q    = n_head_i    * n_embd_head_k;
        const int64_t n_embd_k    = n_head_kv_i * n_embd_head_k;
        const int64_t n_qk        = n_embd_q + n_embd_k;
        const int64_t n_groups    = n_head_i + n_head_kv_i;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // CCA projections (standard Q, K, V, O)
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_k}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

        // CCA conv_qk.0 (depthwise, groups = n_qk, kernel = d_conv)
        layer.cca_conv_dw = create_tensor(tn(LLM_TENSOR_CCA_CONV_DW, "weight", i), {d_conv, n_qk}, 0);

        // CCA conv_qk.1 (grouped, groups = n_groups, kernel = d_conv)
        layer.cca_conv_grp   = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "weight", i), {d_conv, n_qk / n_groups, n_qk}, 0);
        layer.cca_conv_grp_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "bias",   i), {n_qk}, 0);

        // CCA normalization and scale
        layer.cca_qk_norm = create_tensor(tn(LLM_TENSOR_CCA_QK_NORM, "weight", i), {n_qk}, 0);
        layer.cca_k_scale = create_tensor(tn(LLM_TENSOR_CCA_K_SCALE, "weight", i), {n_embd_k}, 0);

        // FFN (dense SwiGLU for now; MoE can be added later)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_zaya::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_zaya::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t d_conv      = hparams.ssm_d_conv;

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

        // norm
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // --- CCA: Q, K, V projections ---
        ggml_tensor * Qraw = ggml_mul_mat(ctx0, layer.wq, cur);
        cb(Qraw, "Qraw", il);
        ggml_tensor * Kraw = ggml_mul_mat(ctx0, layer.wk, cur);
        cb(Kraw, "Kraw", il);
        ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, cur);
        cb(Vcur, "Vcur", il);

        // --- CCA: concat Q+K for conv ---
        // QK: [n_qk, n_tokens]
        ggml_tensor * QK = ggml_concat(ctx0, Qraw, Kraw, 0);
        cb(QK, "QK_cat", il);

        // --- CCA: conv_qk.0 (depthwise, causal) ---
        // Reshape for ssm_conv: [n_tokens, n_qk] -> [n_tokens, n_qk, 1]
        // ssm_conv expects [seq_len, channels, batch] with state already concatenated
        // For prompt processing, we left-pad with (d_conv-1) zeros for causality
        {
            // Left-pad QK with zeros for causal convolution
            ggml_tensor * QK_t = ggml_cont(ctx0, ggml_transpose(ctx0, QK));  // [n_tokens, n_qk]
            ggml_tensor * pad = ggml_new_tensor_2d(ctx0, QK_t->type, d_conv - 1, n_qk);
            pad = ggml_scale(ctx0, pad, 0.0f);
            ggml_tensor * QK_padded = ggml_concat(ctx0, pad, QK_t, 0);  // [d_conv-1 + n_tokens, n_qk]

            QK = ggml_ssm_conv(ctx0, QK_padded, layer.cca_conv_dw);
            // ssm_conv output: [n_tokens, n_qk]
            cb(QK, "QK_dw", il);
        }

        // --- CCA: conv_qk.1 (grouped, causal) ---
        {
            // Left-pad for second causal conv
            ggml_tensor * pad = ggml_new_tensor_2d(ctx0, QK->type, d_conv - 1, n_qk);
            pad = ggml_scale(ctx0, pad, 0.0f);
            ggml_tensor * QK_padded = ggml_concat(ctx0, pad, QK, 0);  // [d_conv-1 + n_tokens, n_qk]

            // ggml_conv_1d_grouped expects kernel [K, IC/G, OC] and input [L, IC]
            // QK_padded is [d_conv-1 + n_tokens, n_qk] which matches [L, IC]
            QK = ggml_conv_1d_grouped(ctx0, layer.cca_conv_grp, QK_padded, 1, 0, 1, n_groups);
            QK = ggml_add(ctx0, QK, layer.cca_conv_grp_b);
            cb(QK, "QK_grp", il);
        }

        // QK is now [n_tokens, n_qk] from conv output, transpose back to [n_qk, n_tokens]
        QK = ggml_cont(ctx0, ggml_transpose(ctx0, QK));

        // --- CCA: split Q_conv, K_conv ---
        ggml_tensor * Q_conv = ggml_view_2d(ctx0, QK, n_embd_q, n_tokens,
            QK->nb[1], 0);
        ggml_tensor * K_conv = ggml_view_2d(ctx0, QK, n_embd_k, n_tokens,
            QK->nb[1], n_embd_q * ggml_element_size(QK));

        // --- CCA: QK mean (skip connection) ---
        ggml_tensor * Qcur = ggml_scale(ctx0, ggml_add(ctx0, Q_conv, Qraw), 0.5f);
        ggml_tensor * Kcur = ggml_scale(ctx0, ggml_add(ctx0, K_conv, Kraw), 0.5f);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);

        // --- CCA: RMSNorm on concat(Q, K) ---
        ggml_tensor * QK_for_norm = ggml_concat(ctx0, Qcur, Kcur, 0);  // [n_qk, n_tokens]
        QK_for_norm = build_norm(QK_for_norm, layer.cca_qk_norm, NULL, LLM_NORM_RMS, il);
        cb(QK_for_norm, "QK_normed", il);

        // Split back
        Qcur = ggml_view_2d(ctx0, QK_for_norm, n_embd_q, n_tokens,
            QK_for_norm->nb[1], 0);
        Kcur = ggml_view_2d(ctx0, QK_for_norm, n_embd_k, n_tokens,
            QK_for_norm->nb[1], n_embd_q * ggml_element_size(QK_for_norm));

        // --- CCA: K temperature scaling ---
        Kcur = ggml_mul(ctx0, Kcur, layer.cca_k_scale);
        cb(Kcur, "Kcur_scaled", il);

        // Reshape for attention: [head_dim, n_heads, n_tokens]
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // --- GQA attention ---
        cur = build_attn(inp->get_attn(), layer.wo, NULL, NULL,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
            1.0f / sqrtf((float) n_embd_head), il);
        cb(cur, "attn_out", il);

        // select output tokens on last layer
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // --- FFN (dense SwiGLU) ---
        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
            layer.ffn_up, NULL, NULL,
            layer.ffn_gate, NULL, NULL,
            layer.ffn_down, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    // final norm
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // output
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
