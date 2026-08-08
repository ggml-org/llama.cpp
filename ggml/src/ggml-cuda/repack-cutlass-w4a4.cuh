#pragma once

#include "common.cuh"

static __host__ __device__ __forceinline__ int64_t ggml_cuda_cutlass_blockscaled_scale_offset(
        int row, int scale_block, int padded_scale_blocks) {
    const int inner_k       = scale_block % 4;
    const int inner_m       = (row % 128) / 32;
    const int outer_m       = row % 32;
    const int k_tile        = scale_block / 4;
    const int m_tile        = row / 128;
    const int k_tile_stride = 512;
    const int m_tile_stride = (padded_scale_blocks / 4) * k_tile_stride;
    return (int64_t) m_tile * m_tile_stride + (int64_t) k_tile * k_tile_stride +
        outer_m * 16 + inner_m * 4 + inner_k;
}

struct ggml_cuda_cutlass_weight {
    const char *    data         = nullptr;
    const uint8_t * scales       = nullptr;
    int64_t         k            = 0;
    int             scale_stride = 0;
    ggml_type       type         = GGML_TYPE_COUNT;
};

struct ggml_cuda_cutlass_weight_layout {
    size_t values_size;
    size_t scales_offset;
    size_t scales_size;
    size_t allocation_size;

    int padded_k;
    int padded_rows;
    int padded_scale_blocks;
    int scale_stride;
    int k_blocks;
    int rows;
    int groups;
};

bool ggml_cuda_cutlass_get_weight_layout(
        const ggml_tensor * tensor, ggml_cuda_cutlass_weight_layout & layout);

bool ggml_cuda_cutlass_weight_from_tensor(
        const ggml_tensor * tensor, ggml_cuda_cutlass_weight & weight);

bool ggml_cuda_cutlass_weight_supported(const ggml_tensor * tensor);

bool ggml_backend_buft_is_cuda_cutlass(ggml_backend_buffer_type_t buft);

bool ggml_cuda_cutlass_pack_weight(
        ggml_tensor * tensor, const void * canonical, cudaStream_t stream);

bool ggml_cuda_cutlass_unpack_weight(
        const ggml_tensor * tensor, void * canonical, cudaStream_t stream);
