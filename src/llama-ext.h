#pragma once

#include "llama.h"

// TODO: try to remove this headers
#include "llama-arch.h"
#include "llama-model.h"
#include "llama-quant.h"

#include <cstdint>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

// TODO: use llama_quant_ prefix to name these consistently:

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor);

// TODO: add:
// LLAMA_API llama_quant * llama_quant_init(...);
// LLAMA_API void llama_quant_free(llama_quant * qnt);

// TODO: become member function of llama_quant
LLAMA_API ggml_type llama_tensor_get_type(
        llama_quant & qs,
        const llama_model_quantize_params * params,
        const ggml_tensor * tensor,
        ggml_type default_type,
        const tensor_metadata & tm);

// Initialize llama_quant counters and populate tensor_metadata categories.
// metadata: vector with name fields already set, will have category field populated.
// TODO: become member function of llama_quant
LLAMA_API void init_quantize_state_counters(
        llama_quant & qs,
        std::vector<tensor_metadata> & metadata);
