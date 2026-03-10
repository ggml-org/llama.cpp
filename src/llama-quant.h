#pragma once

#include "llama.h"

#include "ggml.h"

#include "llama-arch.h"

#include <regex>
#include <string>
#include <vector>

struct llama_model;

// result of parsing --tensor-type option
// (changes to this struct must be reflected in tools/quantize/quantize.cpp)
struct tensor_type_option {
    std::string name;
    ggml_type   type = GGML_TYPE_COUNT;
};

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

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight

    // tensor type override patterns (compiled once, used twice)
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params):
        model(model), params(params)
    {
        // compile regex patterns once - they are expensive
        if (params->tensor_types) {
            const auto & tensor_types = *static_cast<const std::vector<tensor_type_option> *>(params->tensor_types);
            for (const auto & [tname, qtype] : tensor_types) {
                tensor_type_patterns.emplace_back(std::regex(tname), qtype);
            }
        }
    }
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
