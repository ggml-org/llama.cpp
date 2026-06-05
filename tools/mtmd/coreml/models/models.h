#pragma once

// Forward declarations of all CoreML model adapters. Each adapter is defined
// in its own .cpp file under coreml/models/ and registers itself via the
// extern symbol below. The central registry in mtmd-coreml.cpp lists them.

#define MTMD_INTERNAL_HEADER

#include "../../mtmd-coreml.h"

namespace mtmd_coreml::models::minicpmv {
extern const model_adapter g_adapter;
} // namespace mtmd_coreml::models::minicpmv
