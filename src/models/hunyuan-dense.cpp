#include "models.h"

std::unique_ptr<llm_graph_context> llama_model_hunyuan_dense::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

