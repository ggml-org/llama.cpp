#pragma once

#include "common.cuh"

static __host__ __device__ __forceinline__ int64_t ggml_cuda_cutlass_blockscaled_scale_offset(
        int row, int scale_block, int scale_blocks_padded) {
    const int inner_k       = scale_block % 4;
    const int inner_m       = (row % 128) / 32;
    const int outer_m       = row % 32;
    const int k_tile        = scale_block / 4;
    const int m_tile        = row / 128;
    const int k_tile_stride = 512;
    const int m_tile_stride = (scale_blocks_padded / 4) * k_tile_stride;
    return (int64_t) m_tile * m_tile_stride + (int64_t) k_tile * k_tile_stride +
        outer_m * 16 + inner_m * 4 + inner_k;
}

struct ggml_cuda_cutlass_weight {
    const char *    values        = nullptr;
    const uint8_t * scales        = nullptr;
    const uint8_t * scales_linear = nullptr;
    int64_t         k             = 0;
    ggml_type       type          = GGML_TYPE_COUNT;
};

struct ggml_cuda_cutlass_weight_layout {
    size_t size_values;
    size_t offset_scales;
    size_t size_scales;
    size_t offset_scales_linear;
    size_t size_scales_linear;
    size_t size_allocation;

    int k_padded;
    int rows_padded;
    int scale_blocks;
    int scale_blocks_padded;
    int k_blocks;
    int rows;
};

enum ggml_cuda_repack_type {
    GGML_CUDA_REPACK_TYPE_CUTLASS_BLOCKSCALED,
};

struct ggml_cuda_repack_metadata {
    ggml_cuda_repack_type type;
};

bool ggml_cuda_cutlass_get_weight_layout(
        const ggml_tensor * tensor, ggml_cuda_cutlass_weight_layout & layout);

bool ggml_cuda_cutlass_weight_from_tensor(
        const ggml_tensor * tensor, ggml_cuda_cutlass_weight & weight);

bool ggml_cuda_cutlass_weight_supported(const ggml_tensor * tensor);

bool ggml_backend_buft_is_cuda_repacked(ggml_backend_buffer_type_t buft);

bool ggml_cuda_repack_is_cutlass_blockscaled(const ggml_tensor * tensor);

bool ggml_cuda_cutlass_pack_weight(
        ggml_tensor * tensor, const void * canonical, cudaStream_t stream);

bool ggml_cuda_cutlass_unpack_weight(
        const ggml_tensor * tensor, void * canonical, cudaStream_t stream);
