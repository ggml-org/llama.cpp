#pragma once

// Fused RMS Norm + Q8_1 Quantization for GFX906 KV/MoE caching

#include "../../common.cuh"
#include "../../mmq.cuh"

// Fused RMS_NORM + MUL -> Q8 output
void ggml_cuda_op_rms_norm_fused_q8_1(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* rms_norm,
    void* q8_output,
    mmq_q8_1_ds_layout ds_layout,
    const float* mul_weights = nullptr,
    const ggml_tensor* mul_src = nullptr
);

// Get required buffer size for Q8_1 MMQ output
size_t ggml_cuda_get_q8_1_buffer_size(int64_t ncols, int64_t nrows, int cc);
