#pragma once

#include "common.hpp"

void ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
