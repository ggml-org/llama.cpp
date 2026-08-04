#pragma once

#include "common.cuh"

enum class ggml_cuda_moe_weight_layout : int {
    canonical,
    interleaved,
    split,
    tma,
    tma_inplace,
};

static constexpr int ggml_cuda_moe_repack_group_blocks = 16;
static constexpr int ggml_cuda_moe_repack_group_bytes =
    ggml_cuda_moe_repack_group_blocks * (QK_MXFP4 / 2) + ggml_cuda_moe_repack_group_blocks;
static_assert(ggml_cuda_moe_repack_group_bytes % 16 == 0, "repacked MXFP4 groups must be aligned");

static constexpr int ggml_cuda_moe_tma_k            = 512;
static constexpr int ggml_cuda_moe_tma_k_blocks     = ggml_cuda_moe_tma_k / QK_MXFP4;
static constexpr int ggml_cuda_moe_tma_data_bytes   = ggml_cuda_moe_tma_k / 2;
static constexpr int ggml_cuda_moe_tma_scale_bytes  = ggml_cuda_moe_tma_k_blocks;
static constexpr int ggml_cuda_moe_tma_record_bytes = ggml_cuda_moe_tma_data_bytes + ggml_cuda_moe_tma_scale_bytes;
static constexpr int ggml_cuda_moe_tma_modes        = 3;
static constexpr int ggml_cuda_moe_tma_map_qwords   = 16;
static_assert(ggml_cuda_moe_tma_record_bytes % 16 == 0, "TMA records must be 16-byte aligned");

struct ggml_cuda_moe_weight_view {
    const char *                data;
    const uint8_t *             scales;
    int64_t                     ncols;
    int64_t                     stride_row;
    int64_t                     stride_channel;
    int                         scale_stride;
    ggml_cuda_moe_weight_layout layout;
    int                         rows_padded                                              = 0;
    int                         k_tiles                                                  = 0;
    int                         tma_tiles                                                = 0;
    int                         tail_blocks                                              = 0;
    int64_t                     expert_stride                                            = 0;
    int64_t                     tail_offset                                              = 0;
    bool                        tma_valid[ggml_cuda_moe_tma_modes]                       = {};
    alignas(128) uint64_t tma_map[ggml_cuda_moe_tma_modes][ggml_cuda_moe_tma_map_qwords] = {};
    cudaEvent_t ready                                                                    = nullptr;
    cudaEvent_t last_use                                                                 = nullptr;
};

bool ggml_cuda_moe_repack_weight(ggml_backend_cuda_context & ctx,
                                 const ggml_tensor *         tensor,
                                 ggml_cuda_moe_weight_layout layout,
                                 ggml_cuda_moe_weight_view & view,
                                 cudaStream_t                stream        = nullptr,
                                 size_t                      cache_entries = 2,
                                 bool                        wait_ready    = true);

void ggml_cuda_moe_weight_wait_ready(const ggml_cuda_moe_weight_view & view, cudaStream_t stream);

void ggml_cuda_moe_weight_mark_used(const ggml_cuda_moe_weight_view & view, cudaStream_t stream);

bool ggml_cuda_moe_weight_is_inplace_repacked(const ggml_backend_cuda_context & ctx, const ggml_tensor * tensor);
