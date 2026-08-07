#pragma once

#include "common.cuh"

enum class ggml_cuda_moe_weight_layout : int {
    cutlass,
};

struct ggml_cuda_moe_weight_view {
    const char *                data;
    const uint8_t *             scales;
    int64_t                     ncols;
    int64_t                     stride_row;
    int64_t                     stride_channel;
    int                         scale_stride;
    ggml_cuda_moe_weight_layout layout;
    int                         rows_padded = 0;
    cudaEvent_t                 ready       = nullptr;
    cudaEvent_t                 last_use    = nullptr;
    ggml_type                   type        = GGML_TYPE_COUNT;
};

bool ggml_cuda_moe_repack_weight(ggml_backend_cuda_context & ctx,
                                 const ggml_tensor *         tensor,
                                 ggml_cuda_moe_weight_layout layout,
                                 ggml_cuda_moe_weight_view & view,
                                 cudaStream_t                stream          = nullptr,
                                 size_t                      cache_entries   = 2,
                                 bool                        wait_ready      = true,
                                 bool                        preserve_source = false);

bool ggml_cuda_moe_repack_weight_pair(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor *         first,
                                      const ggml_tensor *         second,
                                      ggml_cuda_moe_weight_view & view,
                                      cudaStream_t                stream = nullptr);

void ggml_cuda_moe_weight_wait_ready(const ggml_cuda_moe_weight_view & view, cudaStream_t stream);
void ggml_cuda_moe_weight_mark_used(const ggml_cuda_moe_weight_view & view, cudaStream_t stream);
