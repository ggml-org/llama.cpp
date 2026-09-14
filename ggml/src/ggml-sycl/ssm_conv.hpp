#pragma once

#include "common.hpp"

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// dst is the silu node; reads the conv operands from dst->src[0] and writes dst directly
void ggml_sycl_ssm_conv_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
