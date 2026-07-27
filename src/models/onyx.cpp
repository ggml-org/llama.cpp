#include "models.h"

void llama_model_onyx::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // ISWA period + NoPE tie: Onyx's [SW, SW, SW, Full] pattern has NoPE on the
    // full-attention layers, so `n_no_rope_layer_step` shares the SWA period.
    // (afmoe.cpp:13-19 sets up the SWA pattern the same way; the NoPE tie is
    //  Onyx-specific — afmoe leaves `n_no_rope_layer_step` at its default.)
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        uint32_t swa_period = 4;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
        hparams.set_swa_pattern(swa_period);
        hparams.n_no_rope_layer_step = swa_period;
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_onyx::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // Pre/post-attention norms (Onyx's `weight + 1` fold applied at conversion time).
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        // Q/K/V/O projections. `create_tensor_qkv` handles the split-vs-merged layout
        // and optional biases (Onyx has no biases; helper skips them cleanly).
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // QK-norm. Weights are synthesized at conversion time to absorb `qk_scale_factor`.
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        // Attention output gate: sigmoid(gate) * attn_out before o_proj (afmoe.cpp:73).
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);

        // Pre/post-FFN norms (FFN_PRE_NORM is aliased to LLM_TENSOR_FFN_NORM).
        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // Dense FFN (unlike afmoe, no MoE branches).
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
    }
}

llama_model_onyx::graph::graph(const llama_model & /*model*/, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ABORT("onyx: build_arch_graph not implemented yet");
}

std::unique_ptr<llm_graph_context> llama_model_onyx::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
