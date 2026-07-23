#pragma once

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
