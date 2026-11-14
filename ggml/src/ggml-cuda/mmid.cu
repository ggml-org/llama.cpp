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

// The last block of mm_ids_helper places the slots with id -1 behind the columns of all experts.
// Their ids stay valid indices for the quantization of src1, and the tiles over them are empty,
// so no matrix multiplication reads or writes them.
template <int warp_size>
static __device__ void mm_ids_helper_skipped(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse) {
    const int n_slots = n_tokens*n_expert_used;
    int n_skipped = 0;

    for (int s0 = 0; s0 < n_slots; s0 += warp_size) {
        const int s   = s0 + threadIdx.x;
        const int it  = s / n_expert_used;
        const int iex = s - it*n_expert_used;
        const int skipped = s < n_slots && ids[it*si1 + iex] < 0;

        int scan = skipped; // inclusive prefix sum over the warp
#pragma unroll
        for (int offset = 1; offset < warp_size; offset *= 2) {
            const int tmp = __shfl_up_sync(0xFFFFFFFF, scan, offset, warp_size);
            if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                scan += tmp;
            }
        }

        if (skipped) {
            const int col = n_slots - 1 - (n_skipped + scan - 1);
            ids_dst[col] = s;
            if (write_inverse) {
                ids_src1[s] = col;
            } else {
                ids_src1[col] = it*sis1 + iex % nchannels_y;
            }
        }
        n_skipped += __shfl_sync(0xFFFFFFFF, scan, warp_size - 1, warp_size);
    }

    const int n_valid = n_slots - n_skipped;
    for (int jt = (n_valid + tile_width - 1)/tile_width + threadIdx.x; jt < (n_slots + tile_width - 1)/tile_width; jt += warp_size) {
        tile_spec[jt] = {0, 0, 0, /*padding =*/ 0};
    }
}

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    if (expert == int(gridDim.x) - 1) {
        mm_ids_helper_skipped<warp_size>(ids, ids_src1, ids_dst, tile_spec,
            n_tokens, n_expert_used, nchannels_y, tile_width.z, si1, sis1, write_inverse);
        return;
    }

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
                nex_prev += uint32_t(expert_used) < uint32_t(expert); // id -1 is no expert
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
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += uint32_t(expert_used) < uint32_t(expert);

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

    if (threadIdx.x != 0) {
        return;
    }

    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), it_compact);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + it_compact);
    }

    // Finally, at most one extra tile_spec between the seams per expert.
    tile_spec[fastdiv(n_tokens*n_expert_used + tile_width.z - 1, tile_width) + expert]
        = {expert, col_low, col_high, /*padding =*/ 0};
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

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 num_blocks(n_experts + 1, 1, 1); // the last block handles the slots with id -1
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse);
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

static __global__ void mm_ids_zero_skipped(
        const int32_t * __restrict__ ids, float * __restrict__ dst, const int nrows,
        const int64_t si0, const int64_t si1, const int64_t s1, const int64_t s2) {
    const int64_t it  = blockIdx.x;
    const int64_t iex = blockIdx.y;

    if (ids[it*si1 + iex*si0] >= 0) {
        return;
    }

    float * dst_row = dst + it*s2 + iex*s1;
    for (int i = threadIdx.x; i < nrows; i += blockDim.x) {
        dst_row[i] = 0.0f;
    }
}

void ggml_cuda_mm_ids_zero_skipped(
        const int32_t * ids, float * dst, const int n_tokens, const int n_expert_used, const int nrows,
        const int64_t si0, const int64_t si1, const int64_t s1, const int64_t s2, cudaStream_t stream) {
    const dim3 num_blocks(n_tokens, n_expert_used, 1);
    mm_ids_zero_skipped<<<num_blocks, 256, 0, stream>>>(ids, dst, nrows, si0, si1, s1, s2);
}
