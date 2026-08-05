#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

int ggml_cuda_mm_ids_prefix_block_count(int n_tokens, int n_expert_used);

bool ggml_cuda_launch_mm_ids_prefix(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int32_t * row_expert, int32_t * block_counts, int32_t * block_offsets,
        int n_experts, int n_tokens, int n_expert_used, int si1, cudaStream_t stream);
