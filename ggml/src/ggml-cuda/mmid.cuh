#pragma once

// Whether the mm_ids_helper kernel's per-block shared-memory scratch (n_tokens entries) fits
// within the shared memory available on the current device. Callers that would otherwise launch
// mm_ids_helper must check this and fall back to a path that does not need it, because the kernel
// requires the scratch to be in shared memory and cannot itself degrade gracefully.
bool ggml_cuda_mm_ids_helper_fits(int n_tokens);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
