#pragma once

#include "llama.h"

#include "ggml.h"

#include "llama-arch.h"

#include <regex>
#include <string>
#include <vector>

struct llama_model;

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    tensor_category category;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
};

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

ggml_type llama_tensor_get_type(quantize_state_impl & qs, const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type default_type, const tensor_metadata & tm);
ggml_type llama_ftype_get_default_type(llama_ftype ftype);

// Ftype name <-> enum conversions.
// Returns (llama_ftype)-1 on failure.
llama_ftype  llama_ftype_from_name(const char * name);
const char * llama_ftype_to_name(llama_ftype ftype);

// Initialize quantize_state_impl counters and populate tensor_metadata categories.
// metadata: vector with name fields already set, will have category field populated.
void init_quantize_state_counters(quantize_state_impl & qs, std::vector<tensor_metadata> & metadata);

// Returns true if this tensor should be quantized (based on name, dims, params).
bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor);
