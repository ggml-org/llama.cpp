#pragma once

#include "common.cuh"

bool ggml_cuda_spark_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
    void * quantized = nullptr, const float * gate = nullptr);
