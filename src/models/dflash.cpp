#include "models.h"

// DFlash draft model: graph<true> is the feature-fusion encoder, graph<false> is the
// block-diffusion decoder. The decoder injects the target context into every layer's K/V
// via the cross-embedding store instead of keeping a private KV cache.

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers (n=%zu)\n", __func__, target_layer_ids.size());

    uint32_t n_embd_tgt = 0;
    ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE, n_embd_tgt);
    LLAMA_LOG_INFO("%s: DFlash n_embd_tgt = %u (draft n_embd = %u)\n", __func__, n_embd_tgt, hparams.n_embd);

    hparams.n_embd_inp_impl = (uint32_t) target_layer_ids.size() * n_embd_tgt;

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,    hparams.dflash_block_size,    false);
    ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID, hparams.dflash_mask_token_id, false);
    LLAMA_LOG_INFO("%s: DFlash block_size = %u, mask_token_id = %u\n", __func__,
            hparams.dflash_block_size, hparams.dflash_mask_token_id);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp();

    // optional draft<->target vocabulary mapping (draft vocab may be smaller than target)
    int64_t n_draft_vocab = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_draft_vocab = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), {n_draft_vocab}, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_draft_vocab);
    } else {
        d2t = nullptr;
        LLAMA_LOG_INFO("%s: DFlash sharing target vocab (vocab_size = %lld)\n", __func__, (long long) n_draft_vocab);
    }

    // encoder: feature fusion + hidden-state normalization
    fc                 = create_tensor(tn(LLM_TENSOR_FC,                  "weight"), {n_embd_inp, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM,  "weight"), {n_embd}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_draft_vocab}, TENSOR_NOT_REQUIRED);

    // optional own token embeddings (otherwise inherited from the target via ctx_other)
    const struct ggml_tensor * tok_embd_meta = ml->get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
    if (tok_embd_meta) {
        const int64_t n_target_vocab = tok_embd_meta->ne[1];
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_target_vocab}, 0);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // Qwen3-style per-head q/k normalization
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    // Input: concatenated target-model hidden states of target_layer_ids, provided via
    // ubatch->embd (filled host-side from llama_get_embeddings_layer_inp on the target ctx).
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// Encoder: target features -> fused + normalized target_ctx rows (one per committed token).
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "hidden_norm_out", -1);

    ggml_set_output(cur);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

// Decoder: one block-diffusion forward pass over `n_tokens` noise tokens, with the committed
// target context injected into every layer's K/V via the cross-embedding store.
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // noise [MASK] token embeddings (own tok_embd, or the target's via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash decoder requires token embeddings (own or from target)");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * noise_embd = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(noise_embd, "inp_noise_embd", -1);
    res->add_input(std::move(inp));

    // injected target context: [n_embd, n_enc] from llama_cross::v_embd
    ggml_tensor * target_ctx = build_inp_cross_embd();
    const int64_t n_ctx = target_ctx->ne[1];

    const int64_t n_tokens_kv = n_ctx + n_tokens;

    // positions cover target_ctx (0..n_ctx-1) followed by the noise block; filled by name.
    ggml_tensor * inp_pos_full = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens_kv);
    ggml_set_input(inp_pos_full);
    cb(inp_pos_full, "inp_pos_full", -1);

    // query positions = the noise block (last n_tokens entries)
    ggml_tensor * inp_pos_q = ggml_view_1d(ctx0, inp_pos_full, n_tokens, n_ctx * ggml_element_size(inp_pos_full));

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * inpL = noise_embd;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        // Q from noise only; K/V inject target context: [proj(target_ctx) ; proj(noise)]
        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = ggml_concat(ctx0,
                build_lora_mm(layer.wk, target_ctx),
                build_lora_mm(layer.wk, noise_norm), 1);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = ggml_concat(ctx0,
                build_lora_mm(layer.wv, target_ctx),
                build_lora_mm(layer.wv, noise_norm), 1);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens_kv);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens_kv);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        cb(Kcur, "Kcur_normed", il);

        // RoPE: K over the full [target_ctx ; noise] positions, Q over the noise block only
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos_full, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur, "Kcur_rope", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos_q, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_rope", il);

        // single-block baseline: full attention (no causal mask). The block-diffusion mask
        // (causal-to-anchor for context, bidirectional within block) is added with np>1.
        ggml_tensor * cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "kqv_out", il);

        cur = build_lora_mm(layer.wo, cur);
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "attn_res", il);

        ggml_tensor * ffn_inp = cur;
        cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head (own, or the target's via ctx_other)
    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DFlash decoder requires an output projection (own or from target)");
        output = model_other->output;
    }
    cur = build_lora_mm(output, cur);

    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
