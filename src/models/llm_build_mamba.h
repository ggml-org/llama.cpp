#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_mamba : public llm_graph_context_mamba {
    llm_build_mamba(const llama_model & model, const llm_graph_params & params);
};