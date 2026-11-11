#include "models.h"
#include "../llama-memory-hybrid.h"

void llama_model_zaya1::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_ROUTER_HIDDEN_SIZE,          hparams.n_router_hidden);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    hparams.n_cca_qk = (hparams.n_head() + hparams.n_head_kv()) * hparams.n_embd_head_k();

    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), true);

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_8B_A1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_zaya1::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head = n_embd_head_k;
    const int64_t n_group     = n_head + n_head_kv;
    const int64_t n_qk        = hparams.n_cca_qk;
    const int64_t d_conv      = hparams.ssm_d_conv;
    const int64_t n_ff_exp    = hparams.n_ff_exp();

    const int64_t n_router    = hparams.n_router_hidden;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    res_scale_in   = create_tensor(tn(LLM_TENSOR_RES_SCALE_IN, "weight", 0), {n_embd}, 0);
    res_scale_in_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_IN, "bias",   0), {n_embd}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        auto & z     = layer.zaya1;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // the two value heads are v_cur and v_del
        create_tensor_qkv(layer, i, n_embd, n_embd_head * n_head, n_embd_head * n_head_kv, n_embd_v_gqa, 0);

        layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,   "weight", i), {d_conv, n_qk}, 0);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,   "bias",   i), {n_qk}, 0);
        z.cca_conv_grp     = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "weight", i), {n_embd_head, n_embd_head, n_group, d_conv}, 0);
        z.cca_conv_grp_b   = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "bias",   i), {n_qk}, 0);
        z.cca_temp         = create_tensor(tn(LLM_TENSOR_CCA_TEMP,     "weight", i), {n_head_kv}, 0);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

        z.attn_res_hs            = create_tensor(tn(LLM_TENSOR_ATTN_RES_SCALE_HS, "weight", i), {n_embd}, 0);
        layer.attn_resid_scale   = create_tensor(tn(LLM_TENSOR_ATTN_RESID_SCALE,  "weight", i), {n_embd}, 0);
        layer.attn_resid_scale_b = create_tensor(tn(LLM_TENSOR_ATTN_RESID_SCALE,  "bias",   i), {n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_router}, 0);
        layer.ffn_gate_inp_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "bias",   i), {n_router}, 0);
        z.router_eda   = create_tensor(tn(LLM_TENSOR_ROUTER_EDA,  "weight", i), {n_router}, i == 0 ? TENSOR_NOT_REQUIRED : 0);
        z.router_norm  = create_tensor(tn(LLM_TENSOR_ROUTER_NORM, "weight", i), {n_router}, 0);
        z.router_fc1   = create_tensor(tn(LLM_TENSOR_ROUTER_FC1,  "weight", i), {n_router, n_router}, 0);
        z.router_fc1_b = create_tensor(tn(LLM_TENSOR_ROUTER_FC1,  "bias",   i), {n_router}, 0);
        z.router_fc2   = create_tensor(tn(LLM_TENSOR_ROUTER_FC2,  "weight", i), {n_router, n_router}, 0);
        z.router_fc2_b = create_tensor(tn(LLM_TENSOR_ROUTER_FC2,  "bias",   i), {n_router}, 0);
        // one extra class for the skip expert
        z.router_out   = create_tensor(tn(LLM_TENSOR_ROUTER_OUT,  "weight", i), {n_router, n_expert + 1}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert + 1}, 0);

        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2 * n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

        z.ffn_res_hs            = create_tensor(tn(LLM_TENSOR_FFN_RES_SCALE_HS, "weight", i), {n_embd}, 0);
        layer.ffn_resid_scale   = create_tensor(tn(LLM_TENSOR_FFN_RESID_SCALE,  "weight", i), {n_embd}, 0);
        layer.ffn_resid_scale_b = create_tensor(tn(LLM_TENSOR_FFN_RESID_SCALE,  "bias",   i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_zaya1::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

static ggml_tensor * zaya1_residual_scale(ggml_context * ctx, ggml_tensor * hs, ggml_tensor * res,
        ggml_tensor * s_hs, ggml_tensor * s_res, ggml_tensor * b) {
    return ggml_add(ctx, ggml_mul(ctx, hs, s_hs), ggml_add(ctx, ggml_mul(ctx, res, s_res), b));
}

// grouped causal conv1d with kernel 2, as one mul_mat per tap broadcast over the groups
// x: {n_embd_head * n_group, n_t + 1, n_seqs}, w: {n_embd_head (in), n_embd_head (out), n_group, 2}
// returns {n_embd_head * n_group, n_t, n_seqs} with the bias b added
static ggml_tensor * zaya1_conv_grouped(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    const int64_t n_embd_head = w->ne[0];
    const int64_t n_group     = w->ne[2];
    const int64_t n_t         = x->ne[1] - 1;
    const int64_t n_seqs      = x->ne[2];
    GGML_ASSERT(w->ne[3] == 2);

    x = ggml_reshape_4d(ctx, x, n_embd_head, n_group, n_t + 1, n_seqs);
    x = ggml_permute(ctx, x, 0, 2, 1, 3);

    ggml_tensor * x_prev = ggml_view_4d(ctx, x, n_embd_head, n_t, n_group, n_seqs, x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor * x_cur  = ggml_view_4d(ctx, x, n_embd_head, n_t, n_group, n_seqs, x->nb[1], x->nb[2], x->nb[3], x->nb[1]);

    ggml_tensor * w_prev = ggml_view_3d(ctx, w, n_embd_head, n_embd_head, n_group, w->nb[1], w->nb[2], 0);
    ggml_tensor * w_cur  = ggml_view_3d(ctx, w, n_embd_head, n_embd_head, n_group, w->nb[1], w->nb[2], w->nb[3]);

    ggml_tensor * y = ggml_add(ctx, ggml_mul_mat(ctx, w_prev, x_prev), ggml_mul_mat(ctx, w_cur, x_cur));

    y = ggml_add(ctx, ggml_permute(ctx, y, 0, 2, 1, 3), ggml_reshape_2d(ctx, b, n_embd_head, n_group));
    return ggml_reshape_3d(ctx, y, n_embd_head * n_group, n_t, n_seqs);
}

ggml_tensor * llama_model_zaya1::graph::build_cca_block(const llama_model & model, ggml_tensor * cur,
        ggml_tensor * inp_pos, llm_graph_input_mem_hybrid * inp, int il) {
    const auto & layer = model.layers[il];
    const auto & z     = layer.zaya1;

    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx)->get_recr();
    const uint32_t kv_head = mctx_cur->get_head();

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;
    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_head_kv   = hparams.n_head_kv(il);
    const int64_t n_q         = n_embd_head * n_head;
    const int64_t n_qk        = hparams.n_cca_qk;
    const int64_t n_v_half    = hparams.n_embd_v_gqa(il) / 2;
    const int64_t n_hist      = 2 * (hparams.ssm_d_conv - 1);

    auto [q_proj, k_proj, v_raw] = build_qkv(layer, cur,
            n_embd_head, n_head, n_embd_head, n_head_kv, n_embd_head, n_head_kv, il, false);
    ggml_tensor * qk_raw = ggml_concat(ctx0, q_proj, k_proj, 0);
    cb(qk_raw, "cca_qk_raw", il);
    cb(v_raw,  "cca_v_raw",  il);

    const size_t  es    = ggml_element_size(qk_raw);
    const int64_t n_rep = n_head / n_head_kv;

    ggml_tensor * q_raw = ggml_view_4d(ctx0, qk_raw, n_embd_head, n_rep, n_head_kv, n_tokens,
            n_embd_head * es, n_embd_head * n_rep * es, qk_raw->nb[1], 0);
    ggml_tensor * k_raw = ggml_view_4d(ctx0, qk_raw, n_embd_head, 1, n_head_kv, n_tokens,
            n_embd_head * es, n_embd_head * es, qk_raw->nb[1], n_q * es);
    ggml_tensor * q_res = ggml_scale(ctx0, ggml_add(ctx0, q_raw, k_raw), 0.5f); // {n_embd_head, n_rep, n_head_kv, n_tokens}
    ggml_tensor * k_res = nullptr;
    for (int64_t r = 0; r < n_rep; ++r) {
        ggml_tensor * q_r = ggml_view_4d(ctx0, q_res, n_embd_head, 1, n_head_kv, n_tokens,
                q_res->nb[1], q_res->nb[2], q_res->nb[3], r * q_res->nb[1]);
        k_res = k_res ? ggml_add(ctx0, k_res, q_r) : q_r;
    }
    k_res = ggml_scale(ctx0, k_res, 1.0f / n_rep);
    q_res = ggml_reshape_3d(ctx0, q_res, n_embd_head, n_head,    n_tokens);
    k_res = ggml_reshape_3d(ctx0, k_res, n_embd_head, n_head_kv, n_tokens);

    // conv state: the last n_hist raw q/k columns of each sequence
    ggml_tensor * conv_state = mctx_cur->get_r_l(il);
    ggml_tensor * conv_rs    = build_rs(inp->get_recr(), conv_state, hparams.n_embd_r(), n_seqs);
    conv_rs = ggml_reshape_3d(ctx0, conv_rs, n_hist, n_qk, n_seqs);

    ggml_tensor * qk = ggml_view_3d(ctx0, qk_raw, n_qk, n_seq_tokens, n_seqs, qk_raw->nb[1], n_seq_tokens * qk_raw->nb[1], 0);
    qk = ggml_concat(ctx0, conv_rs, ggml_transpose(ctx0, qk), 0); // {n_hist + n_seq_tokens, n_qk, n_seqs}

    // delayed value state: v_del of the previous token
    ggml_tensor * v_state = mctx_cur->get_s_l(il);
    ggml_tensor * v_rs    = build_rs(inp->get_recr(), v_state, hparams.n_embd_s(), n_seqs);
    ggml_tensor * v_del_ext = ggml_concat(ctx0,
            ggml_reshape_3d(ctx0, v_rs, n_v_half, 1, n_seqs),
            ggml_view_3d(ctx0, v_raw, n_v_half, n_seq_tokens, n_seqs, v_raw->nb[1], n_seq_tokens * v_raw->nb[1], n_v_half * es), 1); // {n_v_half, n_seq_tokens + 1, n_seqs}

    {
        const size_t conv_row = ggml_row_size(conv_state->type, n_hist * n_qk);
        const size_t v_row    = ggml_row_size(v_state->type, n_v_half);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0,
                ggml_view_3d(ctx0, qk, n_hist, n_qk, n_seqs, qk->nb[1], qk->nb[2], n_seq_tokens * ggml_element_size(qk)),
                ggml_view_2d(ctx0, conv_state, n_hist * n_qk, n_seqs, conv_state->nb[1], kv_head * conv_row)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0,
                ggml_view_2d(ctx0, v_del_ext, n_v_half, n_seqs, v_del_ext->nb[2], n_seq_tokens * v_del_ext->nb[1]),
                ggml_view_2d(ctx0, v_state, n_v_half, n_seqs, v_state->nb[1], kv_head * v_row)));
    }

    // depthwise conv, then grouped conv
    ggml_tensor * qk_conv = ggml_ssm_conv(ctx0, qk, layer.ssm_conv1d); // {n_qk, n_seq_tokens + 1, n_seqs}
    qk_conv = ggml_add(ctx0, qk_conv, layer.ssm_conv1d_b);
    qk_conv = zaya1_conv_grouped(ctx0, qk_conv, z.cca_conv_grp, z.cca_conv_grp_b); // {n_qk, n_seq_tokens, n_seqs}
    qk_conv = ggml_reshape_2d(ctx0, qk_conv, n_qk, n_tokens);
    cb(qk_conv, "cca_qk_conv", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, qk_conv, n_embd_head, n_head, n_tokens,
            n_embd_head * ggml_element_size(qk_conv), qk_conv->nb[1], 0);
    ggml_tensor * Kcur = ggml_view_3d(ctx0, qk_conv, n_embd_head, n_head_kv, n_tokens,
            n_embd_head * ggml_element_size(qk_conv), qk_conv->nb[1], n_q * ggml_element_size(qk_conv));
    Qcur = ggml_add(ctx0, Qcur, q_res);
    Kcur = ggml_add(ctx0, Kcur, k_res);

    // L2 norm to sqrt(head_dim), then per kv head key temperature
    const float head_scale = sqrtf((float) n_embd_head);
    Qcur = ggml_scale(ctx0, ggml_l2_norm(ctx0, Qcur, 1e-6f), head_scale);
    Kcur = ggml_scale(ctx0, ggml_l2_norm(ctx0, Kcur, 1e-6f), head_scale);
    Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_2d(ctx0, z.cca_temp, 1, n_head_kv));
    cb(Qcur, "cca_q", il);
    cb(Kcur, "cca_k", il);

    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    // value heads: head 0 = current token, head 1 = previous token
    ggml_tensor * v_prev = ggml_view_3d(ctx0, v_del_ext, n_v_half, n_seq_tokens, n_seqs,
            v_del_ext->nb[1], v_del_ext->nb[2], 0);
    ggml_tensor * v_cur = ggml_view_3d(ctx0, v_raw, n_v_half, n_seq_tokens, n_seqs, v_raw->nb[1], n_seq_tokens * v_raw->nb[1], 0);
    ggml_tensor * Vcur  = ggml_concat(ctx0, v_cur, v_prev, 0);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "cca_v", il);

    return build_attn(inp->get_attn(),
            layer.wo, NULL, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf((float) n_embd_head), il);
}

ggml_tensor * llama_model_zaya1::graph::build_router_moe(const llama_model & model, ggml_tensor * cur,
        ggml_tensor *& router_state, int il) {
    const auto & layer = model.layers[il];
    const auto & z     = layer.zaya1;

    ggml_tensor * r = ggml_add(ctx0, build_lora_mm(layer.ffn_gate_inp, cur), layer.ffn_gate_inp_b);
    if (router_state) {
        r = ggml_add(ctx0, r, ggml_mul(ctx0, router_state, z.router_eda));
    }
    router_state = r;

    r = build_norm(r, z.router_norm, NULL, LLM_NORM_RMS, il);
    r = ggml_gelu_erf(ctx0, ggml_add(ctx0, build_lora_mm(z.router_fc1, r), z.router_fc1_b));
    r = ggml_gelu_erf(ctx0, ggml_add(ctx0, build_lora_mm(z.router_fc2, r), z.router_fc2_b));
    ggml_tensor * logits = build_lora_mm(z.router_out, r); // {n_expert + 1, n_tokens}
    ggml_tensor * probs  = ggml_soft_max(ctx0, logits);
    cb(probs, "router_probs", il);

    const int64_t n_rows = cur->ne[1];
    ggml_tensor * scores = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
    ggml_tensor * real   = ggml_cont(ctx0, ggml_view_2d(ctx0, scores, n_expert, n_rows, scores->nb[1], 0));
    ggml_tensor * skip   = ggml_view_2d(ctx0, scores, 1, n_rows, scores->nb[1], n_expert * ggml_element_size(scores));
    ggml_tensor * expert = ggml_argsort_top_k(ctx0, real, 1); // {1, n_rows}
    ggml_tensor * best   = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, real, 1, n_expert, n_rows), expert), 1, n_rows);
    ggml_tensor * keep   = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_sub(ctx0, skip, best)), -1.0f, 1.0f);
    cb(keep, "router_keep", il);

    ggml_tensor * moe = build_moe_ffn(cur,
            nullptr, nullptr, nullptr, layer.ffn_down_exps, nullptr,
            n_expert + 1, n_expert_used,
            LLM_FFN_SILU, false, 0.0f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
            il, logits, layer.ffn_gate_up_exps,
            nullptr, nullptr, nullptr, expert);

    return ggml_mul(ctx0, moe, keep);
}

llama_model_zaya1::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cur = ggml_mul(ctx0, ggml_add(ctx0, cur, model.res_scale_in_b), model.res_scale_in);
    cb(cur, "inp_scaled", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * router_state = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const auto & z     = layer.zaya1;

        res->t_layer_inp[il] = cur;

        ggml_tensor * residual = cur;

        cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cur = build_cca_block(model, cur, inp_pos, inp, il);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur          = ggml_get_rows(ctx0, cur,          inp_out_ids);
            residual     = ggml_get_rows(ctx0, residual,     inp_out_ids);
            if (router_state) {
                router_state = ggml_get_rows(ctx0, router_state, inp_out_ids);
            }
        }

        residual = zaya1_residual_scale(ctx0, cur, residual, z.attn_res_hs, layer.attn_resid_scale, layer.attn_resid_scale_b);

        cur = build_norm(residual, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cur = build_router_moe(model, cur, router_state, il);
        cb(cur, "ffn_moe_out", il);

        cur = zaya1_residual_scale(ctx0, cur, residual, z.ffn_res_hs, layer.ffn_resid_scale, layer.ffn_resid_scale_b);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
