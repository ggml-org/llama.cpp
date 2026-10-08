#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_flash_attention_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
