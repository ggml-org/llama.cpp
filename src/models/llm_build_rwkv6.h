#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

#include <cmath>

struct llm_build_rwkv6 : public llm_build_rwkv6_base {
    llm_build_rwkv6(const llama_model & model, const llm_graph_params & params);
};