#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_nemotron_h : public llm_graph_context_mamba {
    llm_build_nemotron_h(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * build_ffn_layer(ggml_tensor * cur, const llama_model & model, const int il);
    ggml_tensor * build_attention_layer(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn,
        const llama_model & model, const int64_t n_embd_head, const int il);
};
