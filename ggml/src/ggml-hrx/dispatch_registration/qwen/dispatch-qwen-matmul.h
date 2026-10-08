#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

bool qwen_accepts_q6k_q8_input(const DispatchMatchContext & context,
                               const GraphNode &            consumer,
                               const Value &                input);

void register_qwen_matmul_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
