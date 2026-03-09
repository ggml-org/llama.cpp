#pragma once

#include "llama.h"

#include "ggml.h"

#include "llama-arch.h"

#include <string>
#include <vector>

struct llama_model;

struct quantize_state_impl {
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

    // used to figure out if a model shares tok_embd with the output weight
    bool has_output = false;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};

ggml_type llama_tensor_get_type(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype);
ggml_type llama_ftype_get_default_type(llama_ftype ftype);

// Ftype name <-> enum conversions.
// Returns (llama_ftype)-1 on failure.
llama_ftype  llama_ftype_from_name(const char * name);
const char * llama_ftype_to_name(llama_ftype ftype);

// Initialize quantize_state_impl counters by scanning tensor names.
// tensor_names: all quantizable weight tensor names in the model.
void init_quantize_state_counters(quantize_state_impl & qs, const std::vector<std::string> & tensor_names);

// Returns true if this tensor should be quantized (based on name, dims, params).
bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor);
