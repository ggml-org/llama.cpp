#include "common.cuh"
#include "mmid.cuh"

static __global__ void mm_ids_count(
        const int32_t * __restrict__ ids, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used, const int si1) {
    const int64_t index = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t count = (int64_t) n_tokens*n_expert_used;

    for (int64_t i = index; i < count; i += blockDim.x*gridDim.x) {
        const int it  = i / n_expert_used;
        const int iex = i % n_expert_used;
        atomicAdd(&expert_bounds[ids[it*si1 + iex] + 1], 1);
    }
}

static __global__ void mm_ids_prefix_sum(int32_t * __restrict__ expert_bounds, const int n_experts) {
    for (int expert = 0; expert < n_experts; ++expert) {
        expert_bounds[expert + 1] += expert_bounds[expert];
    }
}

static __global__ void mm_ids_scatter(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds, const int n_tokens, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    const int64_t index = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t count = (int64_t) n_tokens*n_expert_used;

    for (int64_t i = index; i < count; i += blockDim.x*gridDim.x) {
        const int it     = i / n_expert_used;
        const int iex    = i % n_expert_used;
        const int expert = ids[it*si1 + iex];
        const int itc    = atomicSub(&expert_bounds[expert + 1], 1) - 1;

        ids_dst[itc] = it*n_expert_used + iex;
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex] = itc;
        } else {
            ids_src1[itc] = it*sis1 + iex % nchannels_y;
        }
    }
}

static __global__ void mm_ids_restore_bounds(
        int32_t * __restrict__ expert_bounds, const int n_experts, const int count) {
    for (int expert = 1; expert < n_experts; ++expert) {
        expert_bounds[expert] = expert_bounds[expert + 1];
    }
    expert_bounds[n_experts] = count;
}

static __global__ void mm_ids_fused(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds, const int n_experts, const int n_tokens, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    const int count = n_tokens*n_expert_used;

    for (int expert = threadIdx.x; expert <= n_experts; expert += blockDim.x) {
        expert_bounds[expert] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int it  = i / n_expert_used;
        const int iex = i % n_expert_used;
        atomicAdd(&expert_bounds[ids[it*si1 + iex] + 1], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int expert = 0; expert < n_experts; ++expert) {
            expert_bounds[expert + 1] += expert_bounds[expert];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int it     = i / n_expert_used;
        const int iex    = i % n_expert_used;
        const int expert = ids[it*si1 + iex];
        const int itc    = atomicSub(&expert_bounds[expert + 1], 1) - 1;

        ids_dst[itc] = it*n_expert_used + iex;
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex] = itc;
        } else {
            ids_src1[itc] = it*sis1 + iex % nchannels_y;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int expert = 1; expert < n_experts; ++expert) {
            expert_bounds[expert] = expert_bounds[expert + 1];
        }
        expert_bounds[n_experts] = count;
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds, const int n_experts, const int n_tokens, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int64_t count = (int64_t) n_tokens*n_expert_used;
    const int block_size = 256;
    GGML_ASSERT(count > 0 && count <= INT_MAX);

    if (count <= 4096) {
        mm_ids_fused<<<1, block_size, 0, stream>>>(
            ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used,
            nchannels_y, si1, sis1, write_inverse);
        return;
    }

    const int num_blocks = std::min<int64_t>((count + block_size - 1) / block_size, 1024);

    CUDA_CHECK(cudaMemsetAsync(expert_bounds, 0, (n_experts + 1)*sizeof(int32_t), stream));
    mm_ids_count<<<num_blocks, block_size, 0, stream>>>(ids, expert_bounds, n_tokens, n_expert_used, si1);
    mm_ids_prefix_sum<<<1, 1, 0, stream>>>(expert_bounds, n_experts);
    mm_ids_scatter<<<num_blocks, block_size, 0, stream>>>(
        ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
    mm_ids_restore_bounds<<<1, 1, 0, stream>>>(expert_bounds, n_experts, static_cast<int>(count));
}
