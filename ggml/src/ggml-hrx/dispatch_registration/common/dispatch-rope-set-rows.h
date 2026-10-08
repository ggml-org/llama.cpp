#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_rope_set_rows_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
