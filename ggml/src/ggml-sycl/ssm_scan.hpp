#pragma once

#include "common.hpp"

void ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
