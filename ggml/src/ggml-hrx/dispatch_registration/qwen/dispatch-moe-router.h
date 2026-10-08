#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_moe_router_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
