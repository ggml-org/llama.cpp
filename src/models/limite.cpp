#include "models.h"

// Limite: pre-norm transformer with value embeddings, exclusive self attention (XSA),
// per-head output gates, learned residual scales and dense (MUDD) cross-layer connections.
// ref: https://github.com/paradigma-inc/limite-violetto

void llama_model_limite::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale);
    ml.get_key(LLM_KV_ATTENTION_XSA_EPS,           hparams.f_xsa_eps, false);
    ml.get_arr(LLM_KV_FINAL_LOGIT_SIGMOID_CAPPING, hparams.final_logit_sigmoid_capping, false);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    load_swa_pattern(ml, 4);

    std::fill(hparams.mudd_tap_idx.begin(), hparams.mudd_tap_idx.end(), -1);
    ml.get_key(LLM_KV_MUDD_TAP_COUNT, hparams.n_mudd_taps, false);
    if (hparams.n_mudd_taps > 0) {
        ml.get_key(LLM_KV_MUDD_FEED_FORWARD_LENGTH, hparams.n_mudd_ff);
        ml.get_arr(LLM_KV_MUDD_TAP_INDICES,         hparams.mudd_tap_idx);
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_limite::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd,       n_vocab}, 0);
    value_embd = create_tensor(tn(LLM_TENSOR_VALUE_EMBD, "weight"), {n_embd_v_gqa, n_vocab}, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int64_t n_taps = hparams.n_mudd_taps;
    const int64_t n_mudd = hparams.n_mudd_ff;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", i), {n_embd, n_head},    TENSOR_NOT_REQUIRED);
        layer.attn_ve_gate   = create_tensor(tn(LLM_TENSOR_ATTN_VE_GATE,   "weight", i), {n_embd, n_head_kv}, TENSOR_NOT_REQUIRED);
        layer.attn_xsa_alpha = create_tensor(tn(LLM_TENSOR_ATTN_XSA_ALPHA, "weight", i), {n_head},            TENSOR_NOT_REQUIRED);

        layer.attn_resid_scale = create_tensor(tn(LLM_TENSOR_ATTN_RESID_SCALE, "weight", i), {1}, 0);
        layer.ffn_resid_scale  = create_tensor(tn(LLM_TENSOR_FFN_RESID_SCALE,  "weight", i), {1}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);

        if (n_taps > 0 && hparams.mudd_tap_idx[i*n_taps] >= 0) {
            // rows [0, n_taps) mix the attention input, rows [n_taps, 2*n_taps) mix the residual
            layer.mudd_down = create_tensor(tn(LLM_TENSOR_MUDD_DOWN, "weight", i), {n_embd, n_mudd},   0);
            layer.mudd_up   = create_tensor(tn(LLM_TENSOR_MUDD_UP,   "weight", i), {n_mudd, 2*n_taps}, 0);
            layer.mudd_up_b = create_tensor(tn(LLM_TENSOR_MUDD_UP,   "bias",   i), {2*n_taps},         0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_limite::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_limite::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_taps      = hparams.n_mudd_taps;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // value embeddings are looked up by token id, embeddings input is not supported
    ggml_tensor * inp_ve = ggml_get_rows(ctx0, model.value_embd, res->t_inp_tokens);
    inp_ve = ggml_reshape_3d(ctx0, inp_ve, n_embd_head, n_head_kv, n_tokens);
    cb(inp_ve, "inp_ve", -1);

    inpL = build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1);
    cb(inpL, "inp_norm", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // hist[il] is the input of layer il, read by the dense connections
    std::vector<ggml_tensor *> hist(n_layer, nullptr);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        hist[il] = inpL;

        ggml_tensor * resid = inpL;

        if (layer.mudd_up) {
            const int32_t * taps = hparams.mudd_tap_idx.data() + il*n_taps;

            ggml_tensor * w = build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, il);
            w = ggml_gelu_erf(ctx0, build_lora_mm(layer.mudd_down, w));
            w = ggml_add(ctx0, build_lora_mm(layer.mudd_up, w), layer.mudd_up_b);
            cb(w, "mudd_w", il);

            ggml_tensor * stack = nullptr;
            for (int64_t j = 0; j < n_taps; ++j) {
                GGML_ASSERT(taps[j] >= 0 && taps[j] <= il);
                ggml_tensor * h = ggml_reshape_3d(ctx0, hist[taps[j]], n_embd, 1, n_tokens);
                stack = stack ? ggml_concat(ctx0, stack, h, 1) : h;
            }

            ggml_tensor * w_attn  = ggml_cont(ctx0, ggml_view_2d(ctx0, w, n_taps, n_tokens, w->nb[1], 0));
            ggml_tensor * w_resid = ggml_cont(ctx0, ggml_view_2d(ctx0, w, n_taps, n_tokens, w->nb[1], n_taps*w->nb[0]));

            cur   = ggml_dsv4_hc_pre(ctx0, stack, w_attn);
            resid = ggml_dsv4_hc_pre(ctx0, stack, w_resid);
            cb(cur,   "mudd_attn",  il);
            cb(resid, "mudd_resid", il);
        } else {
            cur = inpL;
        }

        ggml_tensor * attn_in = build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, il);
        cb(attn_in, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, attn_in, n_embd_head, n_head, n_head_kv, il);

            if (layer.attn_ve_gate) {
                ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(layer.attn_ve_gate, attn_in));
                gate = ggml_reshape_3d(ctx0, gate, 1, n_head_kv, n_tokens);

                Vcur = ggml_add(ctx0, Vcur, ggml_mul(ctx0, inp_ve, gate));
                cb(Vcur, "Vcur_ve", il);
            }

            Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
            Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            // the full attention layers use no positional encoding
            if (hparams.is_swa(il)) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
            }

            cur = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
            cb(cur, "attn_raw", il);

            if (layer.attn_xsa_alpha) {
                // remove from each head output its projection on the token's own value
                const int64_t n_group = n_head/n_head_kv;

                ggml_tensor * y  = ggml_reshape_4d(ctx0, cur, n_embd_head, n_group, n_head_kv, n_tokens);
                ggml_tensor * vn = ggml_l2_norm(ctx0, Vcur, hparams.f_xsa_eps);
                vn = ggml_reshape_4d(ctx0, vn, n_embd_head, 1, n_head_kv, n_tokens);
                vn = ggml_repeat(ctx0, vn, y);

                ggml_tensor * proj = ggml_sum_rows(ctx0, ggml_mul(ctx0, y, vn));
                proj = ggml_mul(ctx0, proj, ggml_reshape_4d(ctx0, layer.attn_xsa_alpha, 1, n_group, n_head_kv, 1));

                cur = ggml_sub(ctx0, y, ggml_mul(ctx0, vn, proj));
                cb(cur, "attn_xsa", il);
            }

            cur = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens);

            if (layer.wqkv_gate) {
                ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, attn_in));
                cur = ggml_mul(ctx0, cur, ggml_reshape_3d(ctx0, gate, 1, n_head, n_tokens));
                cb(cur, "attn_gated", il);
            }

            cur = ggml_reshape_2d(ctx0, cur, n_embd_head*n_head, n_tokens);
            cur = build_lora_mm(layer.wo, cur, layer.wo_s);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            resid = ggml_get_rows(ctx0, resid, inp_out_ids);
        }

        cur = ggml_add(ctx0, ggml_mul(ctx0, resid, layer.attn_resid_scale), cur);
        cb(cur, "attn_resid", il);

        ggml_tensor * ffn_inp = cur;

        cur = build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, ggml_mul(ctx0, ffn_inp, layer.ffn_resid_scale), cur);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    // scale * sigmoid((x + shift) / temperature)
    const auto & cap = hparams.final_logit_sigmoid_capping;
    if (cap[0] != 0.0f) {
        cur = ggml_scale_bias(ctx0, cur, 1.0f/cap[2], cap[1]/cap[2]);
        cur = ggml_sigmoid(ctx0, cur);
        cur = ggml_scale(ctx0, cur, cap[0]);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
