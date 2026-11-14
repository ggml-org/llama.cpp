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

// Compacts the (token, slot) pairs of tokens [it_start, it_stop) that use expert into store.
// Returns the number of pairs written, if count_nex_prev also adds the number of slots with a lower expert to nex_prev.
template <int n_expert_used_template, bool count_nex_prev>
static __device__ __forceinline__ int mm_ids_helper_compact(
        const int32_t * __restrict__ ids, mm_ids_helper_store * __restrict__ store, const int expert,
        const int it_start, const int it_stop, const int n_expert_used, const int si1, int & nex_prev) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = it_start; it < it_stop; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                if constexpr (count_nex_prev) {
                    nex_prev += expert_used < expert;
                }
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
        for (int it0 = it_start; it0 < it_stop; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < it_stop ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            if constexpr (count_nex_prev) {
                nex_prev += expert_used < expert;
            }

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

    return it_compact;
}

// Writes the it_compact pairs in store to columns [col0, col0 + it_compact).
static __device__ __forceinline__ void mm_ids_helper_flush(
        const mm_ids_helper_store * __restrict__ store, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int col0, const int it_compact, const int n_expert_used, const int nchannels_y, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[col0 + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = col0 + itc;
        } else {
            ids_src1[col0 + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }
}

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
// If the store for all tokens does not fit into shared memory (chunked), the tokens are compacted in chunks of store_cap.
template <int n_expert_used_template, bool chunked>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse, const int store_cap) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev  = 0; // Number of columns for experts with a lower index.
    int n_compact = 0; // Number of columns for this expert.

    if constexpr (!chunked) {
        n_compact = mm_ids_helper_compact<n_expert_used_template, true>(ids, store, expert, 0, n_tokens, n_expert_used, si1, nex_prev);
        nex_prev = warp_reduce_sum<warp_size>(nex_prev);
        ggml_cuda_syncwarp();

        mm_ids_helper_flush(store, ids_src1, ids_dst, nex_prev, n_compact, n_expert_used, nchannels_y, sis1, write_inverse);
    } else {
        // nex_prev is needed before the first chunk can be written, count it upfront:
        if (n_expert_used <= warp_size) {
            const int it_per_iter = warp_size / n_expert_used;
            const int iex         = threadIdx.x % n_expert_used;
            if (threadIdx.x < static_cast<unsigned int>(it_per_iter*n_expert_used)) {
                for (int it = threadIdx.x / n_expert_used; it < n_tokens; it += it_per_iter) {
                    nex_prev += ids[it*si1 + iex] < expert;
                }
            }
        } else {
            for (int it = 0; it < n_tokens; ++it) {
                for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                    nex_prev += ids[it*si1 + iex] < expert;
                }
            }
        }
        nex_prev = warp_reduce_sum<warp_size>(nex_prev);

        for (int it_start = 0; it_start < n_tokens; it_start += store_cap) {
            const int it_stop = min(it_start + store_cap, n_tokens);

            int unused = 0;
            const int it_compact = mm_ids_helper_compact<n_expert_used_template, false>(
                ids, store, expert, it_start, it_stop, n_expert_used, si1, unused);
            ggml_cuda_syncwarp();

            mm_ids_helper_flush(store, ids_src1, ids_dst, nex_prev + n_compact, it_compact, n_expert_used, nchannels_y, sis1, write_inverse);
            n_compact += it_compact;
            ggml_cuda_syncwarp(); // The store is overwritten by the next chunk.
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), n_compact);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + n_compact);
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
    CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper<n_expert_used_template, false>), smpbo);
    CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper<n_expert_used_template, true>),  smpbo);

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);

    // For very large batches the store for all tokens does not fit into shared memory, the kernel then works in chunks.
    const int store_cap_max = smpbo / sizeof(mm_ids_helper_store);
    GGML_ASSERT(store_cap_max > 0);
    const int store_cap = std::min(n_tokens, store_cap_max);
    const size_t nbytes_shared = store_cap*sizeof(mm_ids_helper_store);
    if (n_tokens <= store_cap_max) {
        mm_ids_helper<n_expert_used_template, false><<<num_blocks, block_size, nbytes_shared, stream>>>
            (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse, store_cap);
    } else {
        mm_ids_helper<n_expert_used_template, true><<<num_blocks, block_size, nbytes_shared, stream>>>
            (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse, store_cap);
    }
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
