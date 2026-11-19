#pragma once

struct ggml_cuda_mm_id_tile_spec {
    int32_t expert;
    int32_t col_low;
    int32_t col_high;

    int32_t padding;
};
static_assert(sizeof(ggml_cuda_mm_id_tile_spec) == 16, "bad ggml_cuda_mm_id_tile_spec size");

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream);

// Zero the dst rows of the slots with id -1: the matrix multiplication kernels do not write them.
// si0/si1 are the ids strides and s1/s2 the dst strides per slot and per token, all in elements.
void ggml_cuda_mm_ids_zero_skipped(
        const int32_t * ids, float * dst, int n_tokens, int n_expert_used, int nrows,
        int64_t si0, int64_t si1, int64_t s1, int64_t s2, cudaStream_t stream);
