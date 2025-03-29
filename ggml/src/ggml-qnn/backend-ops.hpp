#pragma once

#include "backend.hpp"
#include "ggml.h"

namespace qnn {

bool device_supports_op(ggml_backend_qnn_device_context * ctx, const ggml_tensor * op);
bool device_compute_graph(ggml_backend_qnn_device_context * ctx, ggml_cgraph * cgraph);

}  // namespace qnn
