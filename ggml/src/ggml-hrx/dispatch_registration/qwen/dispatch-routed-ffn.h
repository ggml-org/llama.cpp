#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_routed_ffn_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
