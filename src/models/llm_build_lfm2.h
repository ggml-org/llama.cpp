#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_lfm2 : public llm_graph_context {
    const llama_model & model;

    llm_build_lfm2(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * build_feed_forward(ggml_tensor * cur, int il) const;
    ggml_tensor * build_attn_block(ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn, int il) const;
    ggml_tensor * build_shortconv_block(ggml_tensor * cur, llm_graph_input_rs * inp_recr, int il);

};
