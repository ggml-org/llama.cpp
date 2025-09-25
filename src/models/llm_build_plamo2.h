#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_plamo2 : public llm_graph_context_mamba {
    llm_build_plamo2(const llama_model & model, const llm_graph_params & params);
    private: 
        ggml_tensor * build_plamo2_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
        ggml_tensor * build_plamo2_attn_layer(llm_graph_input_attn_kv * inp, ggml_tensor * inp_pos, ggml_tensor * cur,
                                                const llama_model & model, int il);
};