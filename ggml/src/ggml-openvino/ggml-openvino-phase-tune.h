#pragma once

#include "ggml.h"

#include <memory>

struct ov_runtime_context;

bool ggml_openvino_phase_tune_in_production();

enum ggml_status ov_graph_compute_phase_tune(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);
