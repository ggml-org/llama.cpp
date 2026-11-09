#include "models.h"

llama_rope_params llama_model_hunyuan_dense::get_rope_params(const llama_cparams & cparams, int il) const {
    auto res = llama_model::get_rope_params(cparams, il);

    res.k_norm = layers[il].attn_k_norm;

    return res;
}

std::unique_ptr<llm_graph_context> llama_model_hunyuan_dense::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

