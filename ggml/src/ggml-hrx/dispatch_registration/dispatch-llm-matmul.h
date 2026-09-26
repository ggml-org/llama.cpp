#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

void register_llm_matmul_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
