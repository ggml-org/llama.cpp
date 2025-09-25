#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_exaone : public llm_graph_context {
    llm_build_exaone(const llama_model & model, const llm_graph_params & params);
};
