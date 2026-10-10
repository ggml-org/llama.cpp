#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
