#include "models.h"

void llama_model_mimo2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,   hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,         hparams.rope_freq_base_train_swa, false);

    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    float value_scale = 0.0f;
    if (ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, value_scale, false) && value_scale != 1.0f) {
        hparams.f_attn_value_scale = value_scale;
    }

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_impl");

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_310B_A15B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_mimo2::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (hparams.n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    int mtp_flags         = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i);
        uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i);
        uint32_t n_head = hparams.n_head(i);

        const bool is_nextn = i >= n_layer;
        const int  flags    = is_nextn ? mtp_flags : 0;

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_v * n_head, n_embd }, flags);

        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_ATTN_NORM,  "weight", i), {n_embd}, flags);
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, TENSOR_NOT_REQUIRED | flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // non-MoE branch
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED | flags);

        // MoE branch
        int64_t n_ff_exp = hparams.n_ff_exp;
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED | flags);

        if (is_nextn) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED | flags);
            layer.layer_out_norm         = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM,         "weight", i), {n_embd}, TENSOR_NOT_REQUIRED | flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_mimo2::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_mimo2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float v_scale = hparams.f_attn_value_scale;
    const bool emit_h_nextn = cparams.embeddings_nextn;
    const bool crop_last_layer = inp_out_ids && (!emit_h_nextn || cparams.embeddings_nextn_masked);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        uint32_t n_head_l    = hparams.n_head(il);
        uint32_t n_head_kv_l = hparams.n_head_kv(il);
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        cur = inpL;

        // self_attention
        {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;

            if (model.layers[il].wqkv) {
                // Fused qkv_proj - Q/K share head_dim_k, V uses head_dim_v
                ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);
                cb(qkv, "wqkv", il);

                const size_t row_k    = ggml_row_size(qkv->type, n_embd_head_k);
                const size_t row_v    = ggml_row_size(qkv->type, n_embd_head_v);
                const size_t row_full = qkv->nb[1];
                const size_t k_off    = row_k * n_head_l;
                const size_t v_off    = k_off + row_k * n_head_kv_l;

                Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_l,    n_tokens, row_k, row_full, 0);
                Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_kv_l, n_tokens, row_k, row_full, k_off);
                Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_kv_l, n_tokens, row_v, row_full, v_off);
            } else {
                // Split path
                Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head_l,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv_l, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv_l, n_tokens);
            }

            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            ggml_tensor * sinks = model.layers[il].attn_sinks;

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, sinks, nullptr, 1.0f/sqrtf(float(n_embd_head_k)), il);
            cb(cur, "attn_out", il);

            if (v_scale) {
                cur = ggml_scale(ctx0, cur, v_scale);
                cb(cur, "attn_out_scaled", il);
            }
        }

        if (il == n_layer - 1 && crop_last_layer) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense branch
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                    il);
            cb(cur, "ffn_moe_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    if (emit_h_nextn) {
        cb(cur, "h_nextn", -1);
        res->t_h_nextn = cur;

        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    }

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Mirrors MiMo's appended NextN block: normalize and fuse token and hidden inputs, run the decoder block,
// expose its pre-head-norm state to the next draft step, then apply the shared output norm and LM head.
// Converted checkpoints may store that shared norm as layer_out_norm, so it remains in the fallback chain.
llama_model_mimo2::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "MIMO2 MTP requires n_layer_nextn > 0");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");

    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.nextn.eh_proj && "MIMO2 MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MIMO2 MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MIMO2 MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.wqkv          && "MIMO2 MTP requires fused attn_qkv");

    const uint32_t n_head_l    = hparams.n_head(il);
    const uint32_t n_head_kv_l = hparams.n_head_kv(il);

    const float freq_base_l  = model.get_rope_freq_base(cparams, il);
    const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
    const float v_scale      = hparams.f_attn_value_scale;

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * h_input    = inp->embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto        * inp_attn    = build_attn_inp_kv_iswa();

    ggml_tensor * h_norm = build_norm(h_input, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
    cb(qkv, "mtp_wqkv", il);

    const size_t row_k    = ggml_row_size(qkv->type, n_embd_head_k);
    const size_t row_v    = ggml_row_size(qkv->type, n_embd_head_v);
    const size_t row_full = qkv->nb[1];
    const size_t k_off    = row_k * n_head_l;
    const size_t v_off    = k_off + row_k * n_head_kv_l;

    ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_l,    n_tokens, row_k, row_full, 0);
    ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_kv_l, n_tokens, row_k, row_full, k_off);
    ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_kv_l, n_tokens, row_v, row_full, v_off);

    Qcur = ggml_rope_ext(
        ctx0, Qcur, inp_pos, nullptr,
        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
        ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(
        ctx0, Kcur, inp_pos, nullptr,
        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
        ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);
    cb(Vcur, "mtp_Vcur", il);

    cur = build_attn(inp_attn,
            layer.wo, nullptr, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr,
            1.0f / sqrtf(float(n_embd_head_k)), il);
    cb(cur, "mtp_attn_out", il);

    if (v_scale) {
        cur = ggml_scale(ctx0, cur, v_scale);
        cb(cur, "mtp_attn_out_scaled", il);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    GGML_ASSERT(layer.ffn_gate && layer.ffn_down && layer.ffn_up && "MIMO2 MTP requires dense FFN tensors");
    cur = build_ffn(cur,
            layer.ffn_up,   layer.ffn_up_b,   nullptr,
            layer.ffn_gate, layer.ffn_gate_b, nullptr,
            layer.ffn_down, layer.ffn_down_b, nullptr,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : (layer.layer_out_norm ? layer.layer_out_norm : model.output_norm);
    GGML_ASSERT(head_norm_w && "MIMO2 MTP missing head norm fallback");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "MIMO2 MTP missing LM head fallback");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
