#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

template <bool iswa>
struct llm_build_smallthinker : public llm_graph_context {
    llm_build_smallthinker(const llama_model & model, const llm_graph_params & params);
};
