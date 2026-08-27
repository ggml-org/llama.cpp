#include "argsort.cuh"
#include "top-k.cuh"

#include <algorithm>
#include <cstdlib>

#ifdef GGML_CUDA_USE_CUB
#    if defined(GGML_USE_HIP)
#        include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#        if defined(__has_include)
#            if __has_include(<rocprim/device/device_topk.hpp>)
#                include <rocprim/block/block_radix_sort.hpp>
#                include <rocprim/device/device_topk.hpp>
#                include <rocprim/iterator/counting_iterator.hpp>
#                define ROCPRIM_TOP_K_AVAILABLE
#            endif
#        endif
#    else
#        include <cub/cub.cuh>
#    endif
#    if !defined(GGML_USE_HIP) && (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef ROCPRIM_TOP_K_AVAILABLE

static constexpr int ROCPRIM_TOP_K_SORT_BLOCK = 256;
static constexpr int ROCPRIM_TOP_K_SORT_BLOCK_LARGE = 512;

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

template<int BLOCK>
static __global__ void top_k_rocprim_sort_candidates(
        const float * keys_in,
        const int *   values_in,
        int *         values_out,
        int           k) {
    using block_sort = rocprim::block_radix_sort<float, BLOCK, 1, int>;
    __shared__ typename block_sort::storage_type storage;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float keys[1] = {tid < k ? keys_in[row * k + tid] : -INFINITY};
    int values[1] = {tid < k ? values_in[row * k + tid] : INT_MAX};
    block_sort().sort_desc(keys, values, storage);
    if (tid < k) {
        values_out[row * k + tid] = values[0];
    }
}

static __global__ void top_k_rocprim_init_offsets(int * offsets, int width, int nrows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= nrows) {
        offsets[i] = i * width;
    }
}

static void top_k_rocprim_sort_selected_large(
        ggml_cuda_pool & pool,
        float *          selected_keys,
        int *            selected_vals,
        int *            dst,
        int              k,
        int              nrows,
        cudaStream_t     stream) {
    ggml_cuda_pool_alloc<float> sorted_keys_alloc(pool, (size_t) k * nrows);
    ggml_cuda_pool_alloc<int>   offsets_alloc(pool, nrows + 1);
    float * sorted_keys = sorted_keys_alloc.get();
    int *   offsets     = offsets_alloc.get();

    static constexpr int block_size = 256;
    top_k_rocprim_init_offsets<<<(nrows + 1 + block_size - 1) / block_size, block_size, 0, stream>>>(
            offsets, k, nrows);
    CUDA_CHECK(cudaGetLastError());

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr, temp_storage_bytes,
            selected_keys, sorted_keys, selected_vals, dst,
            k * nrows, nrows, offsets, offsets + 1, 0, sizeof(float) * 8, stream));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_storage_alloc.get(), temp_storage_bytes,
            selected_keys, sorted_keys, selected_vals, dst,
            k * nrows, nrows, offsets, offsets + 1, 0, sizeof(float) * 8, stream));
}

static void top_k_rocprim(
        ggml_cuda_pool & pool,
        const float *    src,
        int *            dst,
        int              ncols,
        int              nrows,
        int              k,
        cudaStream_t     stream) {
    GGML_ASSERT(k > 0 && k <= ncols);

    ggml_cuda_pool_alloc<float> selected_keys_alloc(pool, (size_t) k * nrows);
    ggml_cuda_pool_alloc<int>   selected_vals_alloc(pool, (size_t) k * nrows);
    float * selected_keys = selected_keys_alloc.get();
    int *   selected_vals = selected_vals_alloc.get();

    const rocprim::counting_iterator<int> indices(0);
    size_t temp_storage_bytes = 0;
    const hipError_t query_status = rocprim::topk_pairs<rocprim::default_config, true>(
        nullptr, temp_storage_bytes, src, selected_keys, indices, selected_vals, (uint32_t) ncols, (uint32_t) k,
        rocprim::identity_decomposer{}, stream);
    CUDA_CHECK(query_status);

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void * temp_storage = temp_storage_alloc.get();

    for (int row = 0; row < nrows; ++row) {
        size_t row_storage_bytes = temp_storage_bytes;
        const hipError_t status = rocprim::topk_pairs<rocprim::default_config, true>(
            temp_storage, row_storage_bytes,
            src + (size_t) row * ncols,
            selected_keys + (size_t) row * k,
            indices,
            selected_vals + (size_t) row * k,
            (uint32_t) ncols, (uint32_t) k, rocprim::identity_decomposer{}, stream);
        CUDA_CHECK(status);
    }

    if (k <= ROCPRIM_TOP_K_SORT_BLOCK) {
        top_k_rocprim_sort_candidates<ROCPRIM_TOP_K_SORT_BLOCK><<<nrows, ROCPRIM_TOP_K_SORT_BLOCK, 0, stream>>>(
            selected_keys, selected_vals, dst, k);
    } else if (k <= ROCPRIM_TOP_K_SORT_BLOCK_LARGE) {
        top_k_rocprim_sort_candidates<ROCPRIM_TOP_K_SORT_BLOCK_LARGE><<<nrows, ROCPRIM_TOP_K_SORT_BLOCK_LARGE, 0, stream>>>(
            selected_keys, selected_vals, dst, k);
    } else {
        top_k_rocprim_sort_selected_large(pool, selected_keys, selected_vals, dst, k, nrows, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

#elif defined(CUB_TOP_K_AVAILABLE)

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

#ifdef ROCPRIM_TOP_K_AVAILABLE
    // The HIP CUB full-argsort fallback can return invalid indices on gfx1030
    // when the row width exceeds 4096. DSV4 reaches this with k=512 and
    // Qwen4Exp QSA reaches it with k=2051. Select candidates with rocPRIM;
    // candidate sets above one workgroup are sorted at their reduced width by
    // segmented radix sort, preserving ordered ggml_top_k semantics.
    static constexpr int rocprim_wide_k_max = 4096;
    const bool use_rocprim_wide = ncols > 4096 && k <= rocprim_wide_k_max;
    if (k <= ROCPRIM_TOP_K_SORT_BLOCK || use_rocprim_wide) {
        // Keep temporary candidate storage near the existing 64 MiB argsort
        // chunk budget for generic callers with many rows.
        const size_t bytes_per_row = (size_t) k * (sizeof(float) + sizeof(int));
        const int chunk_nrows = std::max<int>(1, std::min<int64_t>(nrows, (1 << 26) / bytes_per_row));
        for (int64_t i = 0; i < nrows; i += chunk_nrows) {
            const int iter_nrows = std::min<int64_t>(chunk_nrows, nrows - i);
            top_k_rocprim(pool, src0_d + i * ncols, dst_d + i * k, ncols, iter_nrows, k, stream);
        }
        return;
    }
    // Fall through to the full argsort path for unusually large K.
#endif
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
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
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
