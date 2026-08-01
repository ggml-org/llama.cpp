#include "argsort.cuh"
#include "top-k.cuh"

#if defined(GGML_USE_HIP)
#    include <hipcub/hipcub.hpp>
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

static __global__ void init_top_k_indices(int * indices, const int ncols, const int nitems) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nitems) {
        indices[idx] = idx % ncols;
    }
}

static __global__ void init_top_k_offsets(int * offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}

static void top_k_hip(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        nrows,
                      const int        k,
                      cudaStream_t     stream) {
    const int nitems = ncols * nrows;

    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, nitems);
    ggml_cuda_pool_alloc<int>   temp_indices_in_alloc(pool, nitems);
    ggml_cuda_pool_alloc<int>   temp_indices_out_alloc(pool, nitems);
    ggml_cuda_pool_alloc<int>   offsets_alloc(pool, nrows + 1);

    float * temp_keys        = temp_keys_alloc.get();
    int *   temp_indices_in  = temp_indices_in_alloc.get();
    int *   temp_indices_out = temp_indices_out_alloc.get();
    int *   offsets          = offsets_alloc.get();

    constexpr int block_size = 256;
    init_top_k_indices<<<(nitems + block_size - 1) / block_size, block_size, 0, stream>>>(
        temp_indices_in, ncols, nitems);
    init_top_k_offsets<<<(nrows + block_size) / block_size, block_size, 0, stream>>>(offsets, ncols, nrows);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes, src, temp_keys, temp_indices_in, temp_indices_out, nitems, nrows, offsets,
        offsets + 1, 0, sizeof(float) * 8, stream));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    CUDA_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage_alloc.get(), temp_storage_bytes, src, temp_keys, temp_indices_in, temp_indices_out, nitems, nrows,
        offsets, offsets + 1, 0, sizeof(float) * 8, stream));

    CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), temp_indices_out, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
}

#endif  // defined(GGML_USE_HIP)

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
#if defined(GGML_USE_HIP)
    if (ncols <= 1024) {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();

        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const int chunk_bytes = 1 << 26;
    const int64_t bytes_per_row = ncols * (sizeof(float) + 2 * sizeof(int));
    const int chunk_nrows = std::min<int64_t>(std::max<int64_t>(chunk_bytes / bytes_per_row, 1), nrows);

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min<int64_t>(chunk_nrows, nrows - i);
        top_k_hip(pool, src0_d, dst_d, ncols, iter_nrows, k, stream);

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#elif defined(CUB_TOP_K_AVAILABLE)
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
