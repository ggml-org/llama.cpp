#pragma once

#include "common.hpp"

void ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
