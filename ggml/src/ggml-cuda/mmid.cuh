#pragma once

#include "common.cuh"

#include <cstddef>
#include <cstdint>

enum class ggml_cuda_mm_ids_src1_map {
    compact_to_source,
    source_to_compact,
};

struct ggml_cuda_mm_ids_plan {
    int n_experts;
    int n_tokens;
    int n_expert_used;
    int nchannels_y;
    int si1;
    int sis1;

    ggml_cuda_mm_ids_src1_map src1_map;
    bool                      populate_row_expert;
    bool                      prefer_prefix;
};

struct ggml_cuda_mm_ids_plan_view {
    int32_t * ids_src1;
    int32_t * ids_dst;
    int32_t * expert_bounds;
    int32_t * row_expert;

    // Null prefix workspace selects the helper fallback.
    int32_t * block_counts;
    int32_t * block_offsets;
};

// Buffer sizes are measured in int32_t elements.
struct ggml_cuda_mm_ids_plan_requirements {
    size_t ids_src1_count;
    size_t ids_dst_count;
    size_t expert_bounds_count;
    size_t row_expert_count;

    size_t block_counts_count;
    size_t block_offsets_count;
};

struct ggml_cuda_mm_ids_plan_storage {
    ggml_cuda_pool_alloc<int32_t> ids_src1;
    ggml_cuda_pool_alloc<int32_t> ids_dst;
    ggml_cuda_pool_alloc<int32_t> expert_bounds;
    ggml_cuda_pool_alloc<int32_t> row_expert;
    ggml_cuda_pool_alloc<int32_t> block_counts;
    ggml_cuda_pool_alloc<int32_t> block_offsets;

    ggml_cuda_mm_ids_plan_storage(
            ggml_cuda_pool & pool, const ggml_cuda_mm_ids_plan_requirements & requirements);

    ggml_cuda_mm_ids_plan_view view();
};

bool ggml_cuda_mm_ids_get_requirements(
        const ggml_cuda_mm_ids_plan & plan, ggml_cuda_mm_ids_plan_requirements & requirements);

bool ggml_cuda_launch_mm_ids_plan(
        const int32_t * ids, const ggml_cuda_mm_ids_plan & plan, const ggml_cuda_mm_ids_plan_view & view,
        cudaStream_t stream);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
