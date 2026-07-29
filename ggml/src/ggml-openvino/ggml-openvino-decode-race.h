#pragma once

#include "ggml.h"

#include <memory>

struct ggml_cgraph;
struct ov_runtime_context;

enum ggml_status ov_graph_compute_decode_race(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);
