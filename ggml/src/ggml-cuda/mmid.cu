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

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// mm_ids_helper writes the column range of each expert to tile_spec[expert], mm_ids_tiles then splits the ranges into
//   tiles of tile_width columns that start at the first column of an expert.
// There are at most (n_tokens*n_expert_used + tile_width - 1) / tile_width + n_experts tiles, unused elements are empty.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse) {
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
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        constexpr int tokens_per_iter = warp_size/neu_padded;
        // load several iterations ahead, the loop is bound by the global load latency otherwise
        constexpr int nload = 8;
        const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
        for (int it00 = 0; it00 < n_tokens; it00 += nload*tokens_per_iter) {
            int expert_used_buf[nload];
#pragma unroll
            for (int il = 0; il < nload; ++il) {
                const int it = it00 + il*tokens_per_iter + threadIdx.x / neu_padded;
                expert_used_buf[il] = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;
            }
#pragma unroll
            for (int il = 0; il < nload; ++il) {
            const int it = it00 + il*tokens_per_iter + threadIdx.x / neu_padded;
            const int expert_used = expert_used_buf[il];
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            const uint64_t match = ggml_cuda_ballot(iex_used != -1);
            // most experts are not used by any token of the warp
            if (match == 0) {
                continue;
            }

            // one bit per token position in the warp, set if the token uses the expert
            uint64_t token_mask = 0;
            if constexpr (neu_padded == 1) {
                token_mask = match;
            } else {
                constexpr uint64_t group_bits = (uint64_t(1) << neu_padded) - 1;
#pragma unroll
                for (int g = 0; g < tokens_per_iter; ++g) {
                    token_mask |= ((match >> (g*neu_padded)) & group_bits) != 0 ? uint64_t(1) << g : 0;
                }
            }

            // compact index: number of lower token positions in the warp that use the expert
            const int it_compact_add_lower = __popcll(token_mask & ((uint64_t(1) << (threadIdx.x / neu_padded)) - 1));

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            it_compact += __popcll(token_mask);
            }
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

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

    if (threadIdx.x == 0) {
        tile_spec[expert] = {expert, nex_prev, nex_prev + it_compact, /*padding =*/ 0};
    }
}

static __global__ void mm_ids_tiles(
        ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, const int n_experts, const int max_tiles, const int tile_width) {
    extern __shared__ int data_mm_ids_tiles[];
    int * col_low  = data_mm_ids_tiles;              // [n_experts]
    int * col_high = col_low + n_experts;            // [n_experts]
    int * tile_sum = col_high + n_experts;           // [blockDim.x]

    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        const ggml_cuda_mm_id_tile_spec ts = tile_spec[e];
        col_low[e]  = ts.col_low;
        col_high[e] = ts.col_high;
    }
    __syncthreads();

    // each thread handles a contiguous range of experts, scan the tile counts of the ranges
    const int e_per_thread = (n_experts + blockDim.x - 1) / blockDim.x;
    const int e_start      = min(n_experts, int(threadIdx.x)*e_per_thread);
    const int e_stop       = min(n_experts, e_start + e_per_thread);

    int ntiles = 0;
    for (int e = e_start; e < e_stop; ++e) {
        ntiles += (col_high[e] - col_low[e] + tile_width - 1) / tile_width;
    }
    tile_sum[threadIdx.x] = ntiles;
    __syncthreads();
    for (int offset = 1; offset < int(blockDim.x); offset *= 2) {
        const int tmp = threadIdx.x >= offset ? tile_sum[threadIdx.x - offset] : 0;
        __syncthreads();
        tile_sum[threadIdx.x] += tmp;
        __syncthreads();
    }
    const int ntiles_total = tile_sum[blockDim.x - 1];

    int it = tile_sum[threadIdx.x] - ntiles;
    for (int e = e_start; e < e_stop; ++e) {
        for (int c = col_low[e]; c < col_high[e]; c += tile_width) {
            tile_spec[it++] = {e, c, min(c + tile_width, col_high[e]), /*padding =*/ 0};
        }
    }
    for (int i = ntiles_total + threadIdx.x; i < max_tiles; i += blockDim.x) {
        tile_spec[i] = {0, 0, 0, /*padding =*/ 0};
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
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
        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);

    const int max_tiles = (n_tokens*n_expert_used_var + tile_width - 1) / tile_width + n_experts;
    const int nthreads_tiles = 256;
    const size_t nbytes_shared_tiles = (2*n_experts + nthreads_tiles)*sizeof(int);
    GGML_ASSERT(nbytes_shared_tiles <= smpbo);
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_tiles, smpbo);
    mm_ids_tiles<<<1, nthreads_tiles, nbytes_shared_tiles, stream>>>(tile_spec, n_experts, max_tiles, tile_width);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
    }
}
