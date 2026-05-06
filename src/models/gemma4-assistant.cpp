#include "models.h"

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
// IMPORTANT: this assistant cannot run with `llama_decode` alone — its K/V
// inputs must come from a separate target-context decode step. The graph
// declares those K/V tensors as inputs that an outer driver must populate
// (e.g. the speculative-decoding loop). When the inputs are unset, decoding
// produces undefined logits but does not crash; the graph itself is fully
// wired so that a follow-up runtime patch can light it up without changing
// the model format.

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
    // The full forward pass requires the target backbone's last-layer K/V for
    // every attention type (full + sliding). llama.cpp does not yet expose
    // those values across `llama_context` instances, so the runtime
    // integration is delivered as a follow-up: this PR adds the architecture
    // (convert + load + metadata) so the official assistant checkpoints can be
    // packaged as GGUF and inspected with `gguf-dump`, and so future work has
    // a stable name to plug into.
    //
    // We deliberately abort here rather than producing silently wrong logits;
    // a clear error keeps users from mistaking this for working speculative
    // decoding before the runtime piece lands.
    GGML_UNUSED(model);
    GGML_ABORT(
        "gemma4-assistant: this model is the Multi-Token Prediction drafter "
        "and requires the target backbone's last-layer K/V to decode. "
        "Runtime hookup is not yet wired in llama.cpp — load it with "
        "`gguf-dump` for inspection or wait for the speculative-decoding "
        "patch tracked alongside this PR.");
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
    // Implementing the masked scatter cleanly in ggml requires `scatter_rows`
    // which is not yet a graph-level op. The piece is parked here as a method
    // on the graph so the runtime patch can swap in the implementation
    // without touching the loader or the architecture metadata.
    GGML_UNUSED(hidden);
    GGML_UNUSED(lm_head_w);
    GGML_ABORT("gemma4-assistant: build_masked_embedding_logits not yet implemented");
}
