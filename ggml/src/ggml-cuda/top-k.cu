#include "argsort.cuh"
#include "top-k.cuh"

#if defined(GGML_USE_HIP)
#include <hip/hip_version.h>
#endif

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

#if defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

template<int BLOCK_SIZE>
static __global__ void top_k_nary_search_cuda(
        const float * __restrict__ src,
        const int2 * __restrict__ src_pairs,
        int * __restrict__ dst,
        int2 * __restrict__ dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int k,
        int nrows,
        bool first_pass,
        bool last_pass) {
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const int warp_count = BLOCK_SIZE / warpSize;

    __shared__ int2 candidates[BLOCK_SIZE];
    __shared__ uint32_t counts[64];
    __shared__ uint32_t selected_bucket;
    __shared__ uint32_t selected_total;
    __shared__ uint32_t warp_offsets[32];
    __shared__ uint32_t warp_equal_offsets[32];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col = blockIdx.x * BLOCK_SIZE + tid;
        const bool valid = col < ncols_input;
        int2 value;
        if (valid) {
            value = first_pass
                ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col]))
                : src_pairs[(size_t) row * ncols_input + col];
        } else {
            value = make_int2(original_ncols, (int) 0xff800000U);
        }
        candidates[tid] = value;
        __syncthreads();

        const int limit = min(k, ncols_input - blockIdx.x * BLOCK_SIZE);
        if (k == 1) {
#pragma unroll
            for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
                if (tid < stride) {
                    const int2 a = candidates[tid];
                    const int2 b = candidates[tid + stride];
                    if (a.x >= original_ncols ||
                        (b.x < original_ncols && __int_as_float(b.y) > __int_as_float(a.y))) {
                        candidates[tid] = b;
                    }
                }
                __syncthreads();
            }
        } else {
            const int radix_bits = warpSize == 64 ? 6 : 5;
            const int radix_size = 1 << radix_bits;
            int shift = 32 - radix_bits;
            uint32_t mask = ((1U << radix_bits) - 1) << shift;
            uint32_t range_min = 0;
            uint32_t range_max = 0xff800000U;
            uint32_t total = 0;

            while (mask != 0) {
                if (tid < radix_size) {
                    counts[tid] = 0;
                }
                __syncthreads();

                const uint32_t key = top_k_float_to_ordered(__int_as_float(value.y));
                if (valid && key >= range_min && key < range_max) {
                    atomicAdd(&counts[(key & mask) >> shift], 1U);
                }
                __syncthreads();

                if (tid < radix_size) {
                    uint32_t partial = counts[radix_size - 1 - tid];
                    for (int offset = 1; offset < radix_size; offset *= 2) {
                        const uint32_t previous = __shfl_up(partial, offset, radix_size);
                        if (tid >= offset) {
                            partial += previous;
                        }
                    }
                    partial += total;
                    const unsigned long long selected = __ballot(partial >= (uint32_t) limit);
                    const int first = __ffsll(selected) - 1;
                    if (tid == first) {
                        selected_bucket = radix_size - 1 - first;
                        selected_total = partial;
                    }
                }
                __syncthreads();

                const uint32_t bucket = selected_bucket;
                total = selected_total;
                range_max = range_min + ((bucket + 1) << shift);
                range_min = range_min + (bucket << shift);
                if (total == (uint32_t) limit) {
                    break;
                }
                total -= counts[bucket];
                mask >>= radix_bits;
                shift -= radix_bits;
                if (shift < 0) {
                    shift = 0;
                }
            }

            const uint32_t key = top_k_float_to_ordered(__int_as_float(value.y));
            const bool above = valid && key > range_min;
            const bool equal = valid && key == range_min;
            const unsigned long long above_mask = __ballot(above);
            const unsigned long long equal_mask = __ballot(equal);
            if (lane == 0) {
                warp_offsets[warp] = __popcll(above_mask);
                warp_equal_offsets[warp] = __popcll(equal_mask);
            }
            __syncthreads();

            uint32_t above_base = 0;
            uint32_t equal_base = 0;
            uint32_t above_total = 0;
            for (int i = 0; i < warp_count; ++i) {
                if (i < warp) {
                    above_base += warp_offsets[i];
                    equal_base += warp_equal_offsets[i];
                }
                above_total += warp_offsets[i];
            }
            equal_base += above_total;

            const unsigned long long lane_mask = lane == 0 ? 0 : (1ULL << lane) - 1;
            if (above) {
                candidates[above_base + __popcll(above_mask & lane_mask)] = value;
            }
            const uint32_t equal_index = equal_base + __popcll(equal_mask & lane_mask);
            if (equal && equal_index < (uint32_t) limit) {
                candidates[equal_index] = value;
            }
            __syncthreads();
        }

        if (tid < k) {
            if (last_pass) {
                dst[(size_t) row * k + tid] = candidates[tid].x;
            } else {
                const int output_col = blockIdx.x * k + tid;
                if (output_col < ncols_output) {
                    dst_pairs[(size_t) row * ncols_output + output_col] = candidates[tid];
                }
            }
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __device__ __forceinline__ int2 top_k_one_reduce(
        int2 * candidates,
        int2 value,
        int original_ncols) {
    const int tid = threadIdx.x;
    candidates[tid] = value;
    __syncthreads();

#pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            const int2 a = candidates[tid];
            const int2 b = candidates[tid + stride];
            if (a.x >= original_ncols ||
                (b.x < original_ncols && __int_as_float(b.y) > __int_as_float(a.y))) {
                candidates[tid] = b;
            }
        }
        __syncthreads();
    }
    return candidates[0];
}

template<int BLOCK_SIZE>
static __global__ void top_k_one_first_cuda(
        const float * __restrict__ src,
        int2 * __restrict__ dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int nrows) {
    const int tid = threadIdx.x;
    __shared__ int2 candidates[BLOCK_SIZE];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col = blockIdx.x * BLOCK_SIZE + tid;
        const int2 value = col < ncols_input
            ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col]))
            : make_int2(original_ncols, (int) 0xff800000U);
        const int2 result = top_k_one_reduce<BLOCK_SIZE>(candidates, value, original_ncols);
        if (tid == 0) {
            dst_pairs[(size_t) row * ncols_output + blockIdx.x] = result;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_one_first_last_cuda(
        const float * __restrict__ src,
        int * __restrict__ dst,
        int original_ncols,
        int ncols_input,
        int nrows) {
    const int tid = threadIdx.x;
    __shared__ int2 candidates[BLOCK_SIZE];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col = blockIdx.x * BLOCK_SIZE + tid;
        const int2 value = col < ncols_input
            ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col]))
            : make_int2(original_ncols, (int) 0xff800000U);
        const int2 result = top_k_one_reduce<BLOCK_SIZE>(candidates, value, original_ncols);
        if (tid == 0) {
            dst[row] = result.x;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_one_middle_cuda(
        const int2 * __restrict__ src_pairs,
        int2 * __restrict__ dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int nrows) {
    const int tid = threadIdx.x;
    __shared__ int2 candidates[BLOCK_SIZE];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col = blockIdx.x * BLOCK_SIZE + tid;
        const int2 value = col < ncols_input
            ? src_pairs[(size_t) row * ncols_input + col]
            : make_int2(original_ncols, (int) 0xff800000U);
        const int2 result = top_k_one_reduce<BLOCK_SIZE>(candidates, value, original_ncols);
        if (tid == 0) {
            dst_pairs[(size_t) row * ncols_output + blockIdx.x] = result;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_one_last_cuda(
        const int2 * __restrict__ src_pairs,
        int * __restrict__ dst,
        int original_ncols,
        int ncols_input,
        int nrows) {
    const int tid = threadIdx.x;
    __shared__ int2 candidates[BLOCK_SIZE];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col = blockIdx.x * BLOCK_SIZE + tid;
        const int2 value = col < ncols_input
            ? src_pairs[(size_t) row * ncols_input + col]
            : make_int2(original_ncols, (int) 0xff800000U);
        const int2 result = top_k_one_reduce<BLOCK_SIZE>(candidates, value, original_ncols);
        if (tid == 0) {
            dst[row] = result.x;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_select_cuda(
        const float * __restrict__ src,
        int * __restrict__ dst,
        int ncols,
        int nrows,
        int k) {
    constexpr int RADIX_BITS = 8;
    constexpr int RADIX_SIZE = 1 << RADIX_BITS;

    const int tid = threadIdx.x;
    __shared__ uint32_t histogram[RADIX_SIZE];
    __shared__ uint32_t selected_bucket;
    __shared__ uint32_t count_above;
    __shared__ uint32_t output_count;

    for (int row = blockIdx.x; row < nrows; row += gridDim.x) {
        const float * row_src = src + (size_t) row * ncols;
        int * row_dst = dst + (size_t) row * k;
        uint32_t prefix = 0;
        uint32_t desired = k;

#pragma unroll
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            for (int bin = tid; bin < RADIX_SIZE; bin += BLOCK_SIZE) {
                histogram[bin] = 0;
            }
            __syncthreads();

            const uint32_t high_mask = shift == 32 - RADIX_BITS ? 0 : 0xffffffffU << (shift + RADIX_BITS);
            const uint32_t prefix_high = prefix & high_mask;
            for (int col = tid; col < ncols; col += BLOCK_SIZE) {
                const uint32_t key = top_k_float_to_ordered(row_src[col]);
                if ((key & high_mask) == prefix_high) {
                    atomicAdd(&histogram[(key >> shift) & (RADIX_SIZE - 1)], 1U);
                }
            }
            __syncthreads();

            if (tid == 0) {
                uint32_t above = 0;
                uint32_t bucket = 0;
                for (int bin = RADIX_SIZE - 1; bin >= 0; --bin) {
                    const uint32_t count = histogram[bin];
                    if (above + count >= desired) {
                        bucket = bin;
                        break;
                    }
                    above += count;
                }
                selected_bucket = bucket;
                count_above = above;
            }
            __syncthreads();

            prefix |= selected_bucket << shift;
            desired -= count_above;
            __syncthreads();
        }

        if (tid == 0) {
            output_count = 0;
        }
        __syncthreads();

        for (int col = tid; col < ncols; col += BLOCK_SIZE) {
            if (top_k_float_to_ordered(row_src[col]) > prefix) {
                row_dst[atomicAdd(&output_count, 1U)] = col;
            }
        }
        __syncthreads();

        for (int col = tid; col < ncols; col += BLOCK_SIZE) {
            if (top_k_float_to_ordered(row_src[col]) == prefix) {
                const uint32_t output = atomicAdd(&output_count, 1U);
                if (output < (uint32_t) k) {
                    row_dst[output] = col;
                }
            }
        }
        __syncthreads();
    }
}

static void top_k_radix_select_cuda(
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 1024;
    const int grid_size = std::min(nrows, 65535);
    top_k_radix_select_cuda<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(src, dst, ncols, nrows, k);
}

static int top_k_floor_log2(int value) {
    int result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

static int top_k_ceil_log2(int value) {
    const int floor = top_k_floor_log2(value);
    return value == (1 << floor) ? floor : floor + 1;
}

static int top_k_nary_block_log2(int ncols, int k) {
    const int min_block = std::max(top_k_floor_log2(k) + 1, 6);
    if (min_block > 10) {
        return -1;
    }
    const int max_block = std::min(std::max(top_k_floor_log2(k) + 2, 8), 10);
    int block = std::min(std::max(top_k_ceil_log2(ncols), min_block), max_block);
    if (ncols > (1 << block)) {
        for (int candidate = block; candidate <= 10; ++candidate) {
            if (ncols <= (1 << candidate)) {
                block = candidate;
                break;
            }
        }
    }
    return block;
}

template<int BLOCK_SIZE>
static void top_k_one_cuda_launch(
        const float * src,
        const int2 * src_pairs,
        int * dst,
        int2 * dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int nrows,
        bool first_pass,
        bool last_pass,
        cudaStream_t stream) {
    const dim3 grid((ncols_input + BLOCK_SIZE - 1) / BLOCK_SIZE, std::min(nrows, 65535), 1);
    if (first_pass && last_pass) {
        top_k_one_first_last_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, original_ncols, ncols_input, nrows);
    } else if (first_pass) {
        top_k_one_first_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(
            src, dst_pairs, original_ncols, ncols_input, ncols_output, nrows);
    } else if (last_pass) {
        top_k_one_last_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(
            src_pairs, dst, original_ncols, ncols_input, nrows);
    } else {
        top_k_one_middle_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(
            src_pairs, dst_pairs, original_ncols, ncols_input, ncols_output, nrows);
    }
}

static void top_k_one_cuda_launch(
        int block_log2,
        const float * src,
        const int2 * src_pairs,
        int * dst,
        int2 * dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int nrows,
        bool first_pass,
        bool last_pass,
        cudaStream_t stream) {
    switch (block_log2) {
        case 6:
            top_k_one_cuda_launch<64>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case 7:
            top_k_one_cuda_launch<128>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case 8:
            top_k_one_cuda_launch<256>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case 9:
            top_k_one_cuda_launch<512>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case 10:
            top_k_one_cuda_launch<1024>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);
            break;
        default:
            GGML_ABORT("invalid HIP TOP_K block size");
    }
}

template<int BLOCK_SIZE>
static void top_k_nary_search_cuda_launch(
        const float * src,
        const int2 * src_pairs,
        int * dst,
        int2 * dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int k,
        int nrows,
        bool first_pass,
        bool last_pass,
        cudaStream_t stream) {
    const dim3 grid((ncols_input + BLOCK_SIZE - 1) / BLOCK_SIZE, std::min(nrows, 65535), 1);
    top_k_nary_search_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(
        src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output,
        k, nrows, first_pass, last_pass);
}

static void top_k_nary_search_cuda_launch(
        int block_log2,
        const float * src,
        const int2 * src_pairs,
        int * dst,
        int2 * dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int k,
        int nrows,
        bool first_pass,
        bool last_pass,
        cudaStream_t stream) {
    switch (block_log2) {
        case 6:
            top_k_nary_search_cuda_launch<64>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        case 7:
            top_k_nary_search_cuda_launch<128>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        case 8:
            top_k_nary_search_cuda_launch<256>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        case 9:
            top_k_nary_search_cuda_launch<512>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        case 10:
            top_k_nary_search_cuda_launch<1024>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        default:
            GGML_ABORT("invalid HIP TOP_K block size");
    }
}

static void top_k_small_cuda(
        ggml_cuda_pool & pool,
        const float * src,
        int * dst,
        int ncols,
        int nrows,
        int k,
        cudaStream_t stream) {
    int block_log2 = top_k_nary_block_log2(ncols, k);
    if (block_log2 < 0) {
        top_k_radix_select_cuda(src, dst, ncols, nrows, k, stream);
        return;
    }

    const int block_size = 1 << block_log2;
    const int first_output = (ncols / block_size) * k + std::min(k, ncols % block_size);
    const size_t scratch_elements = (size_t) first_output * nrows;
    ggml_cuda_pool_alloc<int2> scratch_alloc(pool, 2 * scratch_elements);
    int2 * scratch[2] = {scratch_alloc.get(), scratch_alloc.get() + scratch_elements};

    int ncols_input = ncols;
    int buffer = 0;
    bool first_pass = true;
    while (ncols_input > k || first_pass) {
        block_log2 = top_k_nary_block_log2(ncols_input, k);
        const int current_block_size = 1 << block_log2;
        const int ncols_output = (ncols_input / current_block_size) * k + std::min(k, ncols_input % current_block_size);
        const bool last_pass = ncols_output == k;
        if (k == 1) {
            top_k_one_cuda_launch(
                block_log2, src, first_pass ? nullptr : scratch[buffer], dst,
                last_pass ? nullptr : scratch[buffer ^ 1], ncols, ncols_input,
                ncols_output, nrows, first_pass, last_pass, stream);
        } else {
            top_k_nary_search_cuda_launch(
                block_log2, src, first_pass ? nullptr : scratch[buffer], dst,
                last_pass ? nullptr : scratch[buffer ^ 1], ncols, ncols_input,
                ncols_output, k, nrows, first_pass, last_pass, stream);
        }
        ncols_input = ncols_output;
        first_pass = false;
        buffer ^= 1;
    }
}

struct top_k_parallel_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_parallel_radix_init(top_k_parallel_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_parallel_radix_histogram(
        const float * __restrict__ src,
        const top_k_parallel_radix_state * __restrict__ states,
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

    const top_k_parallel_radix_state state = states[row];
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
static __global__ void top_k_parallel_radix_select(
        const int * __restrict__ block_histograms,
        top_k_parallel_radix_state * __restrict__ states,
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
        top_k_parallel_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_parallel_radix_reset_counters(top_k_parallel_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_parallel_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_parallel_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_parallel_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_parallel_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_parallel_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_parallel_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_parallel_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_parallel_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_parallel_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_parallel_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_parallel_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

static bool top_k_use_small_kernel(int ncols, int nrows, int k) {
    if (k == 1) {
        return true;
    }
#if HIP_VERSION >= 71500000
    if (ncols <= 1024) {
        return true;
    }
    const uint64_t elements = (uint64_t) ncols * nrows;
    if (k <= 32) {
        return elements <= (1U << 20);
    }
    return nrows == 1 && ncols <= (1U << 17);
#else
    GGML_UNUSED(ncols);
    GGML_UNUSED(nrows);
    return false;
#endif
}
#endif

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
#ifdef CUB_TOP_K_AVAILABLE
    ggml_cuda_pool & pool = ctx.pool();
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_USE_HIP)
    ggml_cuda_pool & pool = ctx.pool();
    if (top_k_use_small_kernel(ncols, nrows, k)) {
        top_k_small_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else if (ncols > 1024) {
        top_k_parallel_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int * tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    ggml_cuda_pool & pool = ctx.pool();
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
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
