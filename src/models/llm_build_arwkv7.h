#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"
#include "llm_build_rwkv_base.h"

#include <cmath>

struct llm_build_arwkv7 : public llm_build_rwkv7_base {
    llm_build_arwkv7(const llama_model & model, const llm_graph_params & params);
};
