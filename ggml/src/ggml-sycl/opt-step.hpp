#pragma once

#include "common.hpp"

void ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
