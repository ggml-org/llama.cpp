#pragma once

#include "common.cuh"

struct ggml_cuda_cutlass_weight {
    const char *    data         = nullptr;
    const uint8_t * scales       = nullptr;
    int64_t         k            = 0;
    int             scale_stride = 0;
    cudaEvent_t     ready        = nullptr;
    ggml_type       type         = GGML_TYPE_COUNT;
};

bool ggml_cuda_cutlass_repack_weight(ggml_backend_cuda_context & ctx,
                                     const ggml_tensor *         tensor,
                                     ggml_cuda_cutlass_weight &  weight,
                                     cudaStream_t                stream     = nullptr,
                                     bool                        wait_ready = true,
                                     bool                        preserve_source = true);

bool ggml_cuda_cutlass_repack_weight_pair(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor *         first,
                                          const ggml_tensor *         second,
                                          ggml_cuda_cutlass_weight &  weight,
                                          cudaStream_t                stream = nullptr);

void ggml_cuda_cutlass_weight_wait_ready(const ggml_cuda_cutlass_weight & weight, cudaStream_t stream);

bool ggml_cuda_cutlass_get_inplace_weight(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor *         tensor,
                                          ggml_cuda_cutlass_weight &  weight);
