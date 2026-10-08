#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

bool common_match_rmsnorm_gate_dispatch(const DispatchMatchContext & context, DispatchMatch & match);

void register_rmsnorm_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
