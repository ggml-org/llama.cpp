#include "models.h"

#include <cmath>

// Gemma 4 Multi-Token Prediction (MTP) drafter ("assistant") model.
//
// Released: https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
// Reference impl: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4_assistant
// Model card:    https://huggingface.co/google/gemma-4-E2B-it-assistant
//                https://huggingface.co/google/gemma-4-E4B-it-assistant
//
// Architectural notes (vs. the Gemma 4 backbone):
// - 4 transformer layers, all of which read K/V from the target backbone's last
//   layer of the matching attention type (full vs. sliding). The drafter has
//   only Q/Q-norm/O projections — no K/V projections.
// - `pre_projection`: [2*backbone_hidden -> hidden] reduces a concat of
//   (target final hidden state, embedding of next-step token in backbone-dim).
// - `post_projection`: [hidden -> backbone_hidden] feeds the next chained MTP step.
// - Optional centroid masked-embedding head: a top-K of `num_centroids` clusters
//   selects a small slice of the 262144-row LM head; logits outside the slice
//   are masked to a small constant.
//
// IMPORTANT: this assistant cannot run with `llama_decode` alone. It needs the
// target context's last-layer K/V plus the previous target hidden state, so it
// is driven by the attached-MTP speculative-decoding path.

static llm_graph_params graph_params_for_gemma4_mtp(llm_graph_params p, const llama_model & mtp) {
    p.hparams = mtp.hparams;
    p.gtype   = LLM_GRAPH_TYPE_DECODER_MTP;
    return p;
}

static int32_t gemma4_mtp_kv_layer_last_matching(const llama_hparams & hparams, bool want_swa) {
    int32_t best = -1;
    for (int32_t il = 0; il < (int32_t) hparams.n_layer; ++il) {
        if (hparams.is_swa((uint32_t) il) == want_swa) {
            best = il;
        }
    }
    return best;
}

llama_model_gemma4::graph_mtp::graph_mtp(
        const llama_model & target_model,
        const llama_model & mtp_model,
        const llm_graph_params & params) :
        llm_graph_context(graph_params_for_gemma4_mtp(params, mtp_model)),
        target(target_model),
        mtp(mtp_model) {
    const int64_t n_embd_backbone = mtp.hparams.n_embd_backbone;
    GGML_ASSERT(n_embd_backbone > 0);
    GGML_ASSERT(mtp.assist_pre_proj && mtp.assist_post_proj);

    ggml_tensor * inp_tok = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_tok);
    cb(inp_tok, "mtp_inp_last_token", -1);

    ggml_tensor * inp_h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_backbone, 1);
    ggml_set_input(inp_h);
    cb(inp_h, "mtp_inp_h_prev", -1);

    {
        auto inp = std::make_unique<llm_graph_input_mtp>();
        inp->inp_last_token = inp_tok;
        inp->inp_h_prev     = inp_h;
        res->add_input(std::move(inp));
    }

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * tok_e = ggml_get_rows(ctx0, target.tok_embd, inp_tok);
    cb(tok_e, "mtp_tgt_tok_embd", -1);
    tok_e = ggml_scale(ctx0, tok_e, sqrtf((float) target.hparams.n_embd));
    cb(tok_e, "mtp_tgt_tok_embd_scaled", -1);

    ggml_tensor * cur = ggml_concat(ctx0, tok_e, inp_h, 0);
    cb(cur, "mtp_concat", -1);

    cur = build_lora_mm(mtp.assist_pre_proj, cur);
    cb(cur, "mtp_pre_proj_out", -1);

    ggml_build_forward_expand(gf, cur);

    for (int il = 0; il < (int) hparams.n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head      = hparams.n_head(il);

        ggml_tensor * inpL = cur;

        cur = build_norm(inpL, mtp.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(mtp.layers[il].wq, cur, mtp.layers[il].wq_s);
        cb(Qcur, "Qcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, mtp.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        ggml_tensor * freq_factors = hparams.is_swa(il) ? nullptr : mtp.layers[il].rope_freqs;
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, hparams.n_rot(il), rope_type, n_ctx_orig,
                mtp.get_rope_freq_base(cparams, il), mtp.get_rope_freq_scale(cparams, il),
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        const bool read_swa = hparams.is_swa(il);
        const int32_t il_kv = gemma4_mtp_kv_layer_last_matching(target.hparams, read_swa);
        GGML_ASSERT(il_kv >= 0);

        cur = build_attn_mtp(inp_attn, mtp.layers[il].wo, nullptr, mtp.layers[il].wo_s,
                Qcur, nullptr, nullptr, nullptr, hparams.f_attention_scale, il, il_kv, read_swa);

        cur = build_norm(cur, mtp.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        cur = build_norm(attn_out, mtp.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                mtp.layers[il].ffn_up,   nullptr, mtp.layers[il].ffn_up_s,
                mtp.layers[il].ffn_gate, nullptr, mtp.layers[il].ffn_gate_s,
                mtp.layers[il].ffn_down, nullptr, mtp.layers[il].ffn_down_s,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, mtp.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        if (mtp.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, mtp.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, mtp.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * h_inner = cur;
    ggml_tensor * h_backbone = build_lora_mm(mtp.assist_post_proj, h_inner);
    cb(h_backbone, "mtp_post_proj_out", -1);

    const int64_t n_vocab = mtp.tok_embd->ne[1];
    if (mtp.hparams.use_ordered_embeddings) {
        GGML_ASSERT(mtp.assist_embed_centroids && mtp.assist_token_ordering);

        const uint32_t n_c   = mtp.hparams.n_assist_centroids;
        const uint32_t top_k = mtp.hparams.n_assist_centroid_top_k;
        GGML_ASSERT(n_c > 0 && top_k > 0 && n_vocab % (int64_t) n_c == 0);

        const int64_t vsc = n_vocab / (int64_t) n_c;

        ggml_tensor * centroid_logits = build_lora_mm(mtp.assist_embed_centroids, h_inner);
        cb(centroid_logits, "mtp_centroid_logits", -1);

        ggml_tensor * topk_idx = ggml_top_k(ctx0, centroid_logits, (int) top_k);
        cb(topk_idx, "mtp_centroid_topk_idx", -1);

        ggml_tensor * ordering = ggml_view_2d(
                ctx0, mtp.assist_token_ordering, vsc, (int64_t) n_c, ggml_row_size(GGML_TYPE_I32, vsc), 0);
        cb(ordering, "mtp_token_ordering_view", -1);

        ggml_tensor * sel_ids = ggml_get_rows(ctx0, ordering, topk_idx);
        cb(sel_ids, "mtp_selected_token_ids", -1);

        const int64_t n_sel = (int64_t) top_k * vsc;
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx0, sel_ids, n_sel);
        cb(flat_ids, "mtp_selected_token_ids_flat", -1);

        ggml_tensor * sel_emb = ggml_get_rows(ctx0, mtp.tok_embd, flat_ids);
        cb(sel_emb, "mtp_selected_embd", -1);

        ggml_tensor * sel_logits = build_lora_mm(sel_emb, h_inner);
        cb(sel_logits, "mtp_selected_logits", -1);

        ggml_tensor * logits_full = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, 1);
        logits_full = ggml_fill_inplace(ctx0, logits_full, -1e30f);
        cb(logits_full, "mtp_logits_masked_base", -1);

        ggml_tensor * scatter_dst = ggml_cont_2d(ctx0, logits_full, 1, n_vocab);
        ggml_tensor * scatter_src = ggml_cont_2d(ctx0, ggml_cast(ctx0, sel_logits, GGML_TYPE_F32), 1, n_sel);
        cur = ggml_set_rows(ctx0, scatter_dst, scatter_src, flat_ids);
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, 1);
        cb(cur, "mtp_logits_full", -1);
    } else {
        cur = build_lora_mm(mtp.tok_embd, h_inner);
        cb(cur, "result_output_dense", -1);
    }

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);

    ggml_tensor * arg = ggml_argmax(ctx0, cur);
    cb(arg, "result_argmax", -1);

    res->t_logits = cur;
    res->t_embd   = h_backbone;
    res->t_argmax = arg;

    ggml_build_forward_expand(gf, arg);
    ggml_build_forward_expand(gf, h_backbone);
}

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    // Most Gemma 4 hparams apply: SWA pattern, dual head dim (full vs SWA),
    // shared-KV layout, etc. We deliberately reuse the same metadata keys.
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);
    hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t) n_kv_shared_layers;
    hparams.f_attention_scale     = 1.0f;

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // assistant-specific
    uint32_t backbone_hidden = 0;
    ml.get_key(LLM_KV_BACKBONE_HIDDEN_SIZE,             backbone_hidden);
    hparams.n_embd_backbone = backbone_hidden;

    uint32_t num_centroids = 0;
    uint32_t centroid_top_k = 0;
    bool     use_ordered = false;
    ml.get_key(LLM_KV_ASSISTANT_NUM_CENTROIDS,           num_centroids,  false);
    ml.get_key(LLM_KV_ASSISTANT_CENTROID_TOP_K,          centroid_top_k, false);
    ml.get_key(LLM_KV_ASSISTANT_USE_ORDERED_EMBEDDINGS,  use_ordered,    false);
    hparams.n_assist_centroids      = num_centroids;
    hparams.n_assist_centroid_top_k = centroid_top_k;
    hparams.use_ordered_embeddings  = use_ordered;

    // E2B / E4B assistants both have 4 layers; mirror Gemma 4 backbone naming so
    // metadata reads stay consistent.
    switch (hparams.n_layer) {
        case 4:
            // Disambiguate by the backbone hidden size — the only shape that
            // varies between the two officially-released drafters.
            type = (backbone_hidden >= 2560) ? LLM_TYPE_E4B : LLM_TYPE_E2B;
            break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_embd_backbone == 0) {
        throw std::runtime_error("gemma4-assistant: missing backbone_hidden_size in GGUF metadata");
    }
    if (hparams.n_embd_head_k(0) != hparams.n_embd_head_v(0)) {
        throw std::runtime_error("gemma4-assistant requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("gemma4-assistant requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    const int64_t n_embd_backbone = (int64_t) hparams.n_embd_backbone;
    const int64_t n_centroids     = (int64_t) hparams.n_assist_centroids;

    // Token embedding is tied with `lm_head` per the released checkpoints
    // (`tie_word_embeddings: true`); `output` is therefore a duplicate.
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // Drafter <-> backbone projections.
    assist_pre_proj  = create_tensor(tn(LLM_TENSOR_ASSIST_PRE_PROJ,  "weight"), {2 * n_embd_backbone, n_embd}, 0);
    assist_post_proj = create_tensor(tn(LLM_TENSOR_ASSIST_POST_PROJ, "weight"), {n_embd, n_embd_backbone}, 0);

    // Optional centroid head — only present when `use_ordered_embeddings` was
    // baked into the checkpoint (the released E2B/E4B drafters both have it).
    if (hparams.use_ordered_embeddings) {
        if (n_centroids == 0) {
            throw std::runtime_error("gemma4-assistant: use_ordered_embeddings is set but num_centroids is 0");
        }
        assist_embed_centroids = create_tensor(tn(LLM_TENSOR_ASSIST_EMBED_CENTROIDS, "weight"), {n_embd, n_centroids}, 0);
        assist_token_ordering  = create_tensor(tn(LLM_TENSOR_ASSIST_TOKEN_ORDERING,  "weight"), {n_vocab}, 0);
    }

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_l    = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        // Q-only attention: K/V are borrowed from the target backbone, so the
        // drafter checkpoint has no `wk` / `wv` weights at all.
        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), {n_embd, n_embd_head * n_head_l}, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_embd_head * n_head_l, n_embd}, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        if (!hparams.is_swa(i)) {
            // Full-attention layer reuses the Gemma 4 partial-RoPE shim.
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {hparams.n_ff(i), n_embd}, 0);

        // Per-layer scalar gating (Gemma 4 layer_scalar): a single bf16 weight.
        layer.out_scale     = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_assistant::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_backbone(model.hparams.n_embd_backbone),
        n_assist_centroids(model.hparams.n_assist_centroids),
        n_assist_centroid_top_k(model.hparams.n_assist_centroid_top_k),
        use_ordered_embeddings(model.hparams.use_ordered_embeddings) {
    // Standalone assistant decoding cannot work because the graph needs the
    // target backbone's last-layer K/V. The supported runtime path is the
    // attached-MTP graph built from the compatible target model.
    GGML_UNUSED(model);
    GGML_ABORT(
        "gemma4-assistant: this model is the Multi-Token Prediction drafter "
        "and requires the target backbone's last-layer K/V to decode. "
        "Load it as an attached MTP assistant for a compatible target model.");
}

ggml_tensor * llama_model_gemma4_assistant::graph::build_masked_embedding_logits(ggml_tensor * hidden, ggml_tensor * lm_head_w) {
    // Reference: transformers Gemma4AssistantMaskedEmbedder
    // 1. centroid_logits = hidden @ centroids^T                  [B, L, num_centroids]
    // 2. top_k_idx       = topk(centroid_logits, k=top_k)        [B, L, top_k]
    // 3. canonical[c, j] = token_ordering[c * (V/num_centroids) + j]
    //    (token_ordering is a vocab-sized i32 permutation buffer)
    // 4. selected_emb    = lm_head[canonical[top_k_idx]]         [B, L, top_k * V_per, D]
    // 5. selected_logits = hidden @ selected_emb^T               [B, L, top_k * V_per]
    // 6. scatter selected_logits into [B, L, V] positions; everything else
    //    is filled with `min(selected_logits) - 1`.
    //
    // Standalone assistant decoding is intentionally disabled; the attached
    // MTP path implements masked logits in the target-driven graph.
    GGML_UNUSED(hidden);
    GGML_UNUSED(lm_head_w);
    GGML_ABORT("gemma4-assistant: build_masked_embedding_logits not yet implemented");
}
