#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_deci : public llm_graph_context {
    llm_build_deci(const llama_model & model, const llm_graph_params & params);
};
