#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_gemma3n_iswa : public llm_graph_context {
    const llama_model & model;

    const int64_t n_embd_head;
    const int64_t n_embd_altup;
    const int64_t n_altup;
    const int     i_altup_act;
    const int     n_layer_sparsity = 10; // number of layers using activation sparsity
    const float   f_sparsity_std_mul = 1.6448533535003662f; // std_multiplier = normal_dist.icdf(0.95)

    llm_build_gemma3n_iswa(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * calc_magnitude(ggml_tensor * x);
    ggml_tensor * view_2d_slice(ggml_tensor * x, int idx);
    ggml_tensor * get_per_layer_inputs();
    ggml_tensor * project_per_layer_inputs(ggml_tensor * inputs_embeds, ggml_tensor * inp_per_layer);
    ggml_tensor * gaussian_topk(ggml_tensor * x);
    ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il);
    ggml_tensor * altup_predict(ggml_tensor * cur, int il);
    ggml_tensor * laurel(ggml_tensor * cur, int il);
    ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il);
};