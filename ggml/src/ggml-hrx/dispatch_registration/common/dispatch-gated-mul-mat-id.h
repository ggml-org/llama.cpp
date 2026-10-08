#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_gated_mul_mat_id_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
