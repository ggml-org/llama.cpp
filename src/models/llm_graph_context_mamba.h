#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"
#include "../llama-memory-recurrent.h"

#include <cmath>

struct llm_graph_context_mamba : public llm_graph_context {
    llm_graph_context_mamba(const llm_graph_params & params);

    virtual ~llm_graph_context_mamba() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il) const;

};