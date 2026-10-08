#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_unary_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
