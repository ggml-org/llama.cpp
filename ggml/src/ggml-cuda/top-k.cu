#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

// also used on CUDA when cub::DeviceTopK is unavailable (the argsort fallback sorts the whole row per query)
#if !defined(CUB_TOP_K_AVAILABLE)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

// deterministic gather. Each block owns a contiguous range of the row, so with per-block prefix counts the
// output holds the elements above the threshold in ascending index order followed by the tied elements with the
// smallest indices, i.e. exactly the set a stable descending sort would select. A final in-block bitonic sort then
// orders the k results by (value desc, index asc), matching the argsort path bit for bit.
template<int BLOCK_SIZE>
static __device__ __forceinline__ int top_k_block_exclusive_scan(int v, int * warp_sums, int & block_total) {
    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o *= 2) {
        const int n = __shfl_up_sync(0xffffffff, incl, o);
        if (lane >= o) incl += n;
    }
    if (lane == 31) warp_sums[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        int w = lane < BLOCK_SIZE/32 ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o *= 2) {
            const int n = __shfl_up_sync(0xffffffff, w, o);
            if (lane >= o) w += n;
        }
        if (lane < BLOCK_SIZE/32) warp_sums[lane] = w; // inclusive warp prefix
    }
    __syncthreads();
    const int excl = incl - v + (wid > 0 ? warp_sums[wid-1] : 0);
    block_total = warp_sums[BLOCK_SIZE/32 - 1];
    __syncthreads();
    return excl;
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_count(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ counts, // [nrows][blocks_per_row][2] : greater, equal
        int ncols,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int chunk = (ncols + blocks_per_row - 1) / blocks_per_row;
    const int begin = row_block * chunk;
    const int end = min(begin + chunk, ncols);
    const float * row_src = src + (size_t) row * ncols;
    const uint32_t thr = states[row].prefix;
    int g = 0, e = 0;
    for (int col = begin + threadIdx.x; col < end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        g += key > thr;
        e += key == thr;
    }
    __shared__ int ws[BLOCK_SIZE/32];
    int tg, te;
    top_k_block_exclusive_scan<BLOCK_SIZE>(g, ws, tg);
    top_k_block_exclusive_scan<BLOCK_SIZE>(e, ws, te);
    if (threadIdx.x == 0) {
        counts[((size_t) row * blocks_per_row + row_block) * 2 + 0] = tg;
        counts[((size_t) row * blocks_per_row + row_block) * 2 + 1] = te;
    }
}

// exclusive prefix over the blocks of a row (blocks_per_row <= 64): one thread, in place
static __global__ void top_k_radix_block_offsets(int * __restrict__ counts, int blocks_per_row) {
    if (threadIdx.x != 0) {
        return;
    }
    int * c = counts + (size_t) blockIdx.x * blocks_per_row * 2;
    int accg = 0, acce = 0;
    for (int b = 0; b < blocks_per_row; ++b) {
        const int g = c[b*2 + 0], e = c[b*2 + 1];
        c[b*2 + 0] = accg; c[b*2 + 1] = acce;
        accg += g; acce += e;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int * __restrict__ offsets, // exclusive per-block offsets [nrows][blocks_per_row][2]
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int chunk = (ncols + blocks_per_row - 1) / blocks_per_row;
    const int begin = row_block * chunk;
    const int end = min(begin + chunk, ncols);
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const top_k_radix_state st = states[row];
    const uint32_t thr = st.prefix;
    const int n_equal = st.rank;          // tied elements to include (smallest indices first)
    const int n_greater = k - n_equal;
    int goff = offsets[((size_t) row * blocks_per_row + row_block) * 2 + 0];
    int eoff = offsets[((size_t) row * blocks_per_row + row_block) * 2 + 1];
    __shared__ int ws[BLOCK_SIZE/32];
    for (int base = begin; base < end; base += BLOCK_SIZE) {
        const int col = base + threadIdx.x;
        int g = 0, e = 0;
        if (col < end) {
            const uint32_t key = top_k_float_to_ordered(row_src[col]);
            g = key > thr; e = key == thr;
        }
        int tg, te;
        const int pg = top_k_block_exclusive_scan<BLOCK_SIZE>(g, ws, tg);
        const int pe = top_k_block_exclusive_scan<BLOCK_SIZE>(e, ws, te);
        if (g) {
            row_dst[goff + pg] = col;
        } else if (e && eoff + pe < n_equal) {
            row_dst[n_greater + eoff + pe] = col;
        }
        goff += tg; eoff += te;
    }
}

// sort the k selected indices of a row by (value desc, index asc); k <= SORT_N
template<int SORT_N, int BLOCK_SIZE>
static __global__ void top_k_sort_rows(const float * __restrict__ src, int * __restrict__ dst, int ncols, int k) {
    __shared__ float sv[SORT_N];
    __shared__ int   si[SORT_N];
    const int row = blockIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    for (int i = threadIdx.x; i < SORT_N; i += BLOCK_SIZE) {
        if (i < k) { const int idx = row_dst[i]; si[i] = idx; sv[i] = row_src[idx]; }
        else       { si[i] = 0x7fffffff; sv[i] = -INFINITY; }
    }
    __syncthreads();
    for (int size = 2; size <= SORT_N; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            for (int i = threadIdx.x; i < SORT_N; i += BLOCK_SIZE) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = ((i & size) == 0); // descending run when up
                    const float a = sv[i], b = sv[j]; const int ia = si[i], ib = si[j];
                    // "a should come before b" in a descending-by-value, ascending-by-index order
                    const bool a_first = (a > b) || (a == b && ia < ib);
                    if (up ? !a_first : a_first) { sv[i] = b; sv[j] = a; si[i] = ib; si[j] = ia; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < k; i += BLOCK_SIZE) row_dst[i] = si[i];
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    ggml_cuda_pool_alloc<int> counts_alloc(pool, (size_t) nrows * blocks_per_row * 2);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();
    int * counts = counts_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_count<BLOCK_SIZE><<<row_grid, BLOCK_SIZE, 0, stream>>>(src, states, counts, ncols, blocks_per_row);
    top_k_radix_block_offsets<<<nrows, 32, 0, stream>>>(counts, blocks_per_row);
    top_k_radix_gather<BLOCK_SIZE><<<row_grid, BLOCK_SIZE, 0, stream>>>(src, dst, states, counts, ncols, k, blocks_per_row);
    if (k <= 1024) {
        top_k_sort_rows<1024, 256><<<nrows, 256, 0, stream>>>(src, dst, ncols, k);
    } else if (k <= 4096) {
        top_k_sort_rows<4096, 1024><<<nrows, 1024, 0, stream>>>(src, dst, ncols, k);
    }
}

#endif // !defined(CUB_TOP_K_AVAILABLE)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // radix select instead of sorting the whole row (e.g. sparse-attention indexers: k=2051 of up to 262144 per query)
    static const bool force_sort = getenv("GGML_CUDA_TOPK_FORCE_ARGSORT") != nullptr;
    if (!force_sort && ncols > 1024 && k <= 4096 && (int64_t) k * 4 <= ncols) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
