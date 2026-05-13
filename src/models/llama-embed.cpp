#include "models.h"

std::unique_ptr<llm_graph_context> llama_model_llama_embed::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<true>>(*this, params);
}

