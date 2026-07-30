#pragma once

#include "models.h"

// DFlash drafter for the Gemma4 target model.
//
// Architectural differences from the arch-agnostic `llama_model_dflash`:
//   * per-layer `attn_post_norm` (RMS) applied after attention
//   * per-layer `ffn_post_norm`  (RMS) applied after FFN
//   * per-layer `rope_freqs` (proportional RoPE frequency factors)
//   * per-layer `out_scale` (1-element output multiplier, drafter artifact)
//
// `llama_model_dflash` (the base) is arch-agnostic and ships without these
// tensors.  The selection between base and this subclass is made in
// `llama_model_mapping()` (see src/llama-model.cpp) based on the presence of
// the Gemma4-specific `blk.0.post_attention_norm` tensor in the GGUF.

struct llama_model_dflash_gemma4 : public llama_model_dflash {
    llama_model_dflash_gemma4(const struct llama_model_params & params) : llama_model_dflash(params) {}

    // Adds the Gemma4-specific tensors (attn_post_norm, ffn_post_norm,
    // rope_freqs, out_scale) on top of the base dflash tensor set.
    void load_arch_tensors(llama_model_loader & ml) override;

    // Encoder graph is the same as the base dflash encoder (fc + rmsnorm).
    // The decoder graph is gemma4-aware: it consumes the extras loaded above.
    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
