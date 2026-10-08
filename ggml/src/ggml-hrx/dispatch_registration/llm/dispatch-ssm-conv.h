#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_llm_ssm_conv_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
