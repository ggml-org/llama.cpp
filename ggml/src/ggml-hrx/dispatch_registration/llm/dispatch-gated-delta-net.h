#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_llm_gated_delta_net_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
