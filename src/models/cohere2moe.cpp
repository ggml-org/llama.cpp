#include "models.h"

void llama_model_cohere2moe::load_arch_hparams(llama_model_loader & ml) {
    const bool found_norm     = ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps,     false);
    const bool found_norm_rms = ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
    if (!found_norm && !found_norm_rms) {
        throw std::runtime_error("missing Cohere2 MoE norm epsilon");
    }
    if (!found_norm_rms) {
        hparams.f_norm_rms_eps = 0.0f;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer");

    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    uint32_t swa_period = 4;
    if (ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false)) {
        hparams.set_swa_pattern(swa_period, true);
    } else {
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());
    }

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    switch (hparams.n_layer()) {
        case 49: type = LLM_TYPE_30B_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_cohere2moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    // Trunk-only: the GGUF declares MTP layers in metadata but the actual MTP
    // tensors live in a separate file. Mark MTP tensors NOT_REQUIRED so the
    // trunk loads cleanly.
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (hparams.n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    const int trunk_flags = mtp_only  ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    if (n_expert == 0) {
        throw std::runtime_error("n_expert must be > 0 for Cohere2Moe");
    }
    if (n_expert_used == 0) {
        throw std::runtime_error("n_expert_used must be > 0 for Cohere2Moe");
    }

    auto load_block_trunk = [&](int i, int flags) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, flags);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

        if (static_cast<uint32_t>(i) < hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, flags);
        } else {
            const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff;

            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert }, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, flags);
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, flags);

            if (hparams.n_expert_shared > 0) {
                const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp * hparams.n_expert_shared;
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, flags);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), { n_embd, n_ff_shexp }, flags);
            }
        }
    };

    auto load_block_mtp = [&](int i, int flags) {
        auto & layer = layers[i];

        // MTP block looks like a full-attention Cohere2 MoE decoder block.
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, flags);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff;

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, flags);

        if (hparams.n_expert_shared > 0) {
            const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp * hparams.n_expert_shared;

            // Shared experts
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), { n_embd, n_ff_shexp }, flags);
        }

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), { 2 * n_embd, n_embd }, flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), { n_embd },              flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), { n_embd },              flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd },              TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    // MTP/NextN layers are loaded as extra decoder blocks.
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i, mtp_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_cohere2moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_cohere2moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const llm_norm_type cohere2moe_norm_type = hparams.f_norm_rms_eps == 0.0f ? LLM_NORM : LLM_NORM_RMS;
    const float f_logit_scale = hparams.f_logit_scale;
    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.is_swa(il);
        // Dense-prefix full-attention layers use RoPE; later layers follow the SWA pattern.
        const bool force_rope = static_cast<uint32_t>(il) < hparams.n_layer_dense_lead;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, cohere2moe_norm_type, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * ffn_inp = cur;

        {
            const auto & layer = model.layers[il];

            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur,
                    n_embd_head, n_head, n_head_kv, il);

            if (is_swa || force_rope) {
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    layer.wo, layer.wo_b, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur     = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL    = ggml_get_rows(ctx0, inpL, inp_out_ids);
            ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
        }

        ggml_tensor * attn_out = cur;

        const auto & layer = model.layers[il];

        if (layer.ffn_gate_inp == nullptr) {
            cur = build_ffn(ffn_inp,
                    layer.ffn_up,   nullptr, layer.ffn_up_s,
                    layer.ffn_gate, nullptr, layer.ffn_gate_s,
                    layer.ffn_down, nullptr, layer.ffn_down_s,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_moe_ffn(ffn_inp,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr, layer.ffn_gate_up_exps,
                    layer.ffn_up_exps_s,
                    layer.ffn_gate_exps_s,
                    layer.ffn_down_exps_s);
            cb(cur, "ffn_moe_out", il);

            if (layer.ffn_up_shexp) {
                ggml_tensor * ffn_shexp = build_ffn(ffn_inp,
                        layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                        layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                        layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                        nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, cur, ffn_shexp);
                cur = ggml_scale(ctx0, cur, 0.5f);
                cb(cur, "ffn_out", il);
            }
        }

        cur = ggml_add(ctx0, cur, inpL);
        cur = ggml_add(ctx0, cur, attn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, nullptr, cohere2moe_norm_type, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    if (f_logit_scale) {
        cur = ggml_scale(ctx0, cur, f_logit_scale);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

llama_model_cohere2moe::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "COHERE2MOE MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "COHERE2MOE MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp  && "MTP block missing ffn_gate_inp");

    const llm_norm_type cohere2moe_norm_type = hparams.f_norm_rms_eps == 0.0f ? LLM_NORM : LLM_NORM_RMS;

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
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

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, cohere2moe_norm_type, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, cohere2moe_norm_type, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpL = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, cohere2moe_norm_type, il);
    cb(cur, "mtp_attn_norm", il);
    ggml_tensor * ffn_inp = cur;

    auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);
    ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
    Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, rope_factors,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, rope_factors,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);
    cb(Vcur, "mtp_Vcur", il);

    cur = build_attn(inp_attn,
            layer.wo, layer.wo_b, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
            1.0f / sqrtf(float(n_embd_head)), il);
    cb(cur, "mtp_attn_out", il);

    ggml_tensor * attn_out = cur;

    cur = build_moe_ffn(ffn_inp,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(cur, "mtp_ffn_moe_out", il);

    if (layer.ffn_up_shexp) {
        ggml_tensor * ffn_shexp = build_ffn(ffn_inp,
                layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        cur = ggml_add(ctx0, cur, ffn_shexp);
        cur = ggml_scale(ctx0, cur, 0.5f);
        cb(cur, "mtp_ffn_out", il);
    }

    cur = ggml_add(ctx0, cur, inpL);
    cur = ggml_add(ctx0, cur, attn_out);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "COHERE2MOE MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, cohere2moe_norm_type, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "COHERE2MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : nullptr);

    if (hparams.f_logit_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
