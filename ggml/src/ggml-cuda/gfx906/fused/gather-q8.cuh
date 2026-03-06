#pragma once

// Gather selected rows from Q8_1 quantized tensor for MoE caching

#include "../../common.cuh"

void gather_q8_1_rows_cuda(
    const void* src,
    const int* ids,
    void* dst,
    int64_t block_size,
    int64_t n_blocks,
    int64_t n_src_rows_per_slice,
    int64_t n_dst_rows,
    cudaStream_t stream
);
