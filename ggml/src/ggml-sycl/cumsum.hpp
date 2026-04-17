#pragma once

#include "common.hpp"

void ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
