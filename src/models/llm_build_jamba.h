#pragma once

#include "../llama-model.h"
#include "llm_graph_context_mamba.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_jamba : public llm_graph_context_mamba {
    llm_build_jamba(const llama_model & model, const llm_graph_params & params);
};
