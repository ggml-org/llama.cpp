#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

void register_qwen_matmul_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
