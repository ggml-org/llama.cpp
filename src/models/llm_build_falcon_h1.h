#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"
#include "llm_graph_context_mamba.h"

#include <cmath>

struct llm_build_falcon_h1 : public llm_graph_context_mamba {
    llm_build_falcon_h1(const llama_model & model, const llm_graph_params & params);
};
