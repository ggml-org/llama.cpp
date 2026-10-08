#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

void register_llm_attention_qkv_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
