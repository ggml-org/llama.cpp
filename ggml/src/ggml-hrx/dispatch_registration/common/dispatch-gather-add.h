#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_gather_add_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
