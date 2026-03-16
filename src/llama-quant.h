#pragma once

#include "llama.h"

#include <memory>

struct llama_model;
struct compiled_tensor_type_patterns;

struct llama_quant {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_k_quantized = 0;
    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight

    // tensor type override patterns (compiled once, used in llama_tensor_get_type)
    std::unique_ptr<compiled_tensor_type_patterns> tensor_type_patterns;

    llama_quant(const llama_model & model, const llama_model_quantize_params * params);
    ~llama_quant();
};
