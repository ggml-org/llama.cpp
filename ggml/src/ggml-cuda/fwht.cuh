#pragma once
#include "common.cuh"

void ggml_cuda_forward_fwht_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
