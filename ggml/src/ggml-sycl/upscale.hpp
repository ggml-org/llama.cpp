#pragma once

#include <sycl/sycl.hpp>
#include "sycl_core.hpp"
#include "common.hpp"

#define SYCL_UPSCALE_BLOCK_SIZE 256

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
