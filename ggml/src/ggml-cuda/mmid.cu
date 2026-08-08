#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}

static constexpr int MM_IDS_PREFIX_THREADS = 256;
static constexpr int MM_IDS_PREFIX_WARPS   = MM_IDS_PREFIX_THREADS / WARP_SIZE;

static bool mm_ids_prefix_supported(
        int n_experts, int n_tokens, int n_expert_used, int si1) {
    const int device = ggml_cuda_get_device();
    if (n_experts <= 0 || n_experts > MM_IDS_PREFIX_THREADS || n_tokens <= 0 ||
        (n_expert_used != 4 && n_expert_used != 8) || si1 < n_expert_used ||
        ggml_cuda_info().devices[device].warp_size != WARP_SIZE) {
        return false;
    }

    return (int64_t) n_tokens * n_expert_used <= INT_MAX;
}

template <int n_expert_used>
static __global__ void mm_ids_prefix_count(const int32_t * __restrict__ ids,
                                           int32_t * __restrict__ block_counts,
                                           int n_rows,
                                           int n_experts,
                                           int si1) {
    extern __shared__ int32_t counts[];

    for (int expert = threadIdx.x; expert < n_experts; expert += blockDim.x) {
        counts[expert] = 0;
    }
    __syncthreads();

    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route < n_rows) {
        const int token  = route / n_expert_used;
        const int slot   = route - token * n_expert_used;
        const int expert = ids[(int64_t) token * si1 + slot];
        if ((unsigned) expert < (unsigned) n_experts) {
            atomicAdd(&counts[expert], 1);
        }
    }
    __syncthreads();

    for (int expert = threadIdx.x; expert < n_experts; expert += blockDim.x) {
        block_counts[(int64_t) blockIdx.x * n_experts + expert] = counts[expert];
    }
}

static __global__ void mm_ids_prefix_scan(const int32_t * __restrict__ block_counts,
                                          int32_t * __restrict__ block_offsets,
                                          int32_t * __restrict__ expert_bounds,
                                          int n_blocks,
                                          int n_experts) {
    __shared__ int32_t totals[MM_IDS_PREFIX_THREADS];

    const int expert = threadIdx.x;
    int       total  = 0;
    if (expert < n_experts) {
        for (int block = 0; block < n_blocks; ++block) {
            const int64_t index   = (int64_t) block * n_experts + expert;
            block_offsets[index] = total;
            total += block_counts[index];
        }
    }
    totals[expert] = total;
    __syncthreads();

    for (int offset = 1; offset < MM_IDS_PREFIX_THREADS; offset *= 2) {
        const int32_t lower = expert >= offset ? totals[expert - offset] : 0;
        __syncthreads();
        totals[expert] += lower;
        __syncthreads();
    }

    if (expert < n_experts) {
        expert_bounds[expert] = expert == 0 ? 0 : totals[expert - 1];
    }
    if (expert == n_experts - 1) {
        expert_bounds[n_experts] = totals[expert];
    }
}

template <int n_expert_used>
static __global__ void mm_ids_prefix_scatter(const int32_t * __restrict__ ids,
                                             const int32_t * __restrict__ block_offsets,
                                             const int32_t * __restrict__ expert_bounds,
                                             int32_t * __restrict__ ids_src1,
                                             int32_t * __restrict__ ids_dst,
                                             int32_t * __restrict__ row_expert,
                                             int n_rows,
                                             int n_experts,
                                             int si1) {
    extern __shared__ int32_t warp_counts[];

    for (int index = threadIdx.x; index < MM_IDS_PREFIX_WARPS * n_experts; index += blockDim.x) {
        warp_counts[index] = 0;
    }
    __syncthreads();

    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    int       expert = -1;
    if (route < n_rows) {
        const int token = route / n_expert_used;
        const int slot  = route - token * n_expert_used;
        expert          = ids[(int64_t) token * si1 + slot];
        if ((unsigned) expert < (unsigned) n_experts) {
            atomicAdd(&warp_counts[(threadIdx.x / WARP_SIZE) * n_experts + expert], 1);
        }
    }
    __syncthreads();

    const bool     valid  = route < n_rows && (unsigned) expert < (unsigned) n_experts;
    const unsigned active = __ballot_sync(0xFFFFFFFF, valid);
    if (!valid) {
        return;
    }

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const unsigned peers = __match_any_sync(active, expert);
#else
    unsigned peers = 0;
    for (int source = 0; source < WARP_SIZE; ++source) {
        if ((active & (1u << source)) != 0 && __shfl_sync(active, expert, source) == expert) {
            peers |= 1u << source;
        }
    }
#endif
    const unsigned lower_mask = lane == 0 ? 0 : (1u << lane) - 1;
    int            local_rank = __popc(peers & lower_mask);
    for (int previous_warp = 0; previous_warp < warp; ++previous_warp) {
        local_rank += warp_counts[previous_warp * n_experts + expert];
    }

    const int row = expert_bounds[expert] + block_offsets[(int64_t) blockIdx.x * n_experts + expert] + local_rank;
    ids_src1[route] = row;
    ids_dst[row]    = route;
    if (row_expert != nullptr) {
        row_expert[row] = expert;
    }
}

static __global__ void mm_ids_fill_row_expert(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ row_expert) {
    const int expert = blockIdx.x;
    for (int row = expert_bounds[expert] + threadIdx.x; row < expert_bounds[expert + 1]; row += blockDim.x) {
        row_expert[row] = expert;
    }
}

static int ggml_cuda_mm_ids_prefix_block_count(int n_tokens, int n_expert_used) {
    GGML_ASSERT(n_tokens >= 0 && n_expert_used >= 0);
    const int64_t n_rows = (int64_t) n_tokens * n_expert_used;
    GGML_ASSERT(n_rows <= INT_MAX);
    return (int) ((n_rows + MM_IDS_PREFIX_THREADS - 1) / MM_IDS_PREFIX_THREADS);
}

static bool ggml_cuda_launch_mm_ids_prefix(const int32_t * __restrict__ ids,
                                    int32_t * __restrict__ ids_src1,
                                    int32_t * __restrict__ ids_dst,
                                    int32_t * __restrict__ expert_bounds,
                                    int32_t * __restrict__ row_expert,
                                    int32_t * __restrict__ block_counts,
                                    int32_t * __restrict__ block_offsets,
                                    int n_experts,
                                    int n_tokens,
                                    int n_expert_used,
                                    int si1,
                                    cudaStream_t stream) {
    if (!mm_ids_prefix_supported(n_experts, n_tokens, n_expert_used, si1)) {
        return false;
    }

    const int n_rows   = (int) ((int64_t) n_tokens * n_expert_used);
    const int n_blocks = ggml_cuda_mm_ids_prefix_block_count(n_tokens, n_expert_used);
    if (n_expert_used == 4) {
        mm_ids_prefix_count<4><<<n_blocks, MM_IDS_PREFIX_THREADS, n_experts * sizeof(int32_t), stream>>>(
            ids, block_counts, n_rows, n_experts, si1);
    } else {
        mm_ids_prefix_count<8><<<n_blocks, MM_IDS_PREFIX_THREADS, n_experts * sizeof(int32_t), stream>>>(
            ids, block_counts, n_rows, n_experts, si1);
    }
    CUDA_CHECK(cudaGetLastError());
    mm_ids_prefix_scan<<<1, MM_IDS_PREFIX_THREADS, 0, stream>>>(
        block_counts, block_offsets, expert_bounds, n_blocks, n_experts);
    CUDA_CHECK(cudaGetLastError());
    if (n_expert_used == 4) {
        mm_ids_prefix_scatter<4><<<n_blocks, MM_IDS_PREFIX_THREADS,
                                   MM_IDS_PREFIX_WARPS * n_experts * sizeof(int32_t), stream>>>(
            ids, block_offsets, expert_bounds, ids_src1, ids_dst, row_expert, n_rows, n_experts, si1);
    } else {
        mm_ids_prefix_scatter<8><<<n_blocks, MM_IDS_PREFIX_THREADS,
                                   MM_IDS_PREFIX_WARPS * n_experts * sizeof(int32_t), stream>>>(
            ids, block_offsets, expert_bounds, ids_src1, ids_dst, row_expert, n_rows, n_experts, si1);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mm_ids_get_requirements(
        const ggml_cuda_mm_ids_plan & plan, ggml_cuda_mm_ids_plan_requirements & requirements) {
    requirements = {};

    if (plan.n_experts <= 0 || plan.n_tokens <= 0 || plan.n_expert_used <= 0 ||
        plan.si1 < plan.n_expert_used) {
        return false;
    }
    if (plan.src1_map != ggml_cuda_mm_ids_src1_map::compact_to_source &&
        plan.src1_map != ggml_cuda_mm_ids_src1_map::source_to_compact) {
        return false;
    }
    if (plan.src1_map == ggml_cuda_mm_ids_src1_map::compact_to_source &&
        (plan.nchannels_y <= 0 || plan.sis1 <= 0)) {
        return false;
    }

    const int64_t n_rows = (int64_t) plan.n_tokens * plan.n_expert_used;
    if (n_rows > INT_MAX) {
        return false;
    }

    const bool use_prefix =
        plan.prefer_prefix && plan.src1_map == ggml_cuda_mm_ids_src1_map::source_to_compact &&
        mm_ids_prefix_supported(plan.n_experts, plan.n_tokens, plan.n_expert_used, plan.si1);
    if (!use_prefix && (plan.n_tokens >= (1 << 22) || plan.n_expert_used >= (1 << 10))) {
        return false;
    }

    requirements.ids_src1_count      = (size_t) n_rows;
    requirements.ids_dst_count       = (size_t) n_rows;
    requirements.expert_bounds_count = (size_t) plan.n_experts + 1;
    requirements.row_expert_count    = plan.populate_row_expert ? (size_t) n_rows : 0;

    if (use_prefix) {
        const size_t n_blocks = (size_t) ggml_cuda_mm_ids_prefix_block_count(plan.n_tokens, plan.n_expert_used);
        if (n_blocks != 0 && (size_t) plan.n_experts > SIZE_MAX / n_blocks) {
            requirements = {};
            return false;
        }
        requirements.block_counts_count  = n_blocks * (size_t) plan.n_experts;
        requirements.block_offsets_count = requirements.block_counts_count;
    }
    return true;
}

ggml_cuda_mm_ids_plan_storage::ggml_cuda_mm_ids_plan_storage(
        ggml_cuda_pool & pool, const ggml_cuda_mm_ids_plan_requirements & requirements) :
    ids_src1(pool, requirements.ids_src1_count),
    ids_dst(pool, requirements.ids_dst_count),
    expert_bounds(pool, requirements.expert_bounds_count),
    row_expert(pool),
    block_counts(pool),
    block_offsets(pool) {
    if (requirements.row_expert_count != 0) {
        row_expert.alloc(requirements.row_expert_count);
    }
    if (requirements.block_counts_count != 0) {
        block_counts.alloc(requirements.block_counts_count);
        block_offsets.alloc(requirements.block_offsets_count);
    }
}

ggml_cuda_mm_ids_plan_view ggml_cuda_mm_ids_plan_storage::view() {
    return { ids_src1.ptr, ids_dst.ptr, expert_bounds.ptr, row_expert.ptr, block_counts.ptr, block_offsets.ptr };
}

bool ggml_cuda_launch_mm_ids_plan(
        const int32_t * ids, const ggml_cuda_mm_ids_plan & plan, const ggml_cuda_mm_ids_plan_view & view,
        cudaStream_t stream) {
    ggml_cuda_mm_ids_plan_requirements requirements;
    if (!ggml_cuda_mm_ids_get_requirements(plan, requirements) || ids == nullptr || view.ids_src1 == nullptr ||
        view.ids_dst == nullptr || view.expert_bounds == nullptr ||
        (plan.populate_row_expert && view.row_expert == nullptr)) {
        return false;
    }

    const bool use_prefix = requirements.block_counts_count != 0 && view.block_counts != nullptr &&
        view.block_offsets != nullptr;
    if (use_prefix) {
        int32_t * row_expert = plan.populate_row_expert ? view.row_expert : nullptr;
        return ggml_cuda_launch_mm_ids_prefix(
            ids, view.ids_src1, view.ids_dst, view.expert_bounds, row_expert, view.block_counts,
            view.block_offsets, plan.n_experts, plan.n_tokens, plan.n_expert_used, plan.si1, stream);
    }

    if (plan.n_tokens >= (1 << 22) || plan.n_expert_used >= (1 << 10)) {
        return false;
    }

    const int    device        = ggml_cuda_get_device();
    const size_t shared_memory = (size_t) plan.n_tokens * sizeof(mm_ids_helper_store);
    if (shared_memory > ggml_cuda_info().devices[device].smpbo) {
        return false;
    }

    const bool write_inverse = plan.src1_map == ggml_cuda_mm_ids_src1_map::source_to_compact;
    ggml_cuda_launch_mm_ids_helper(
        ids, view.ids_src1, view.ids_dst, view.expert_bounds, plan.n_experts, plan.n_tokens, plan.n_expert_used,
        plan.nchannels_y, plan.si1, plan.sis1, write_inverse, stream);
    CUDA_CHECK(cudaGetLastError());

    if (plan.populate_row_expert) {
        mm_ids_fill_row_expert<<<plan.n_experts, MM_IDS_PREFIX_THREADS, 0, stream>>>(
            view.expert_bounds, view.row_expert);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
