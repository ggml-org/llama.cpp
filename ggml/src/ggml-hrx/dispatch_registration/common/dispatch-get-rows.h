#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

void register_get_rows_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
