#pragma once

#include "common.cuh"

struct alignas(16) block_mxfp8_mmq {
    uint32_t scale;
    uint8_t  qs[32];
};

static_assert(sizeof(block_mxfp8_mmq) == 48, "unexpected MXFP8 block size");

size_t ggml_cuda_moe_mxfp8_size(int64_t n_rows, int64_t ne0);

void ggml_cuda_moe_quantize_scatter_mxfp8(const float *   src,
                                          const int32_t * ids_src1_inv,
                                          void *          dst,
                                          int64_t         ne00,
                                          int64_t         stride_token,
                                          int64_t         ne0,
                                          int64_t         n_tokens,
                                          int64_t         n_rows,
                                          int             n_expert_used,
                                          cudaStream_t    stream);
