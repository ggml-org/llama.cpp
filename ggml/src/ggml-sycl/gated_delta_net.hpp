#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
