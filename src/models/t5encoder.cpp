#include "models.h"

llm_build_t5encoder::llm_build_t5encoder(const llama_model & model, const llm_graph_params & params) : llm_build_t5<true>(model, params) {}
