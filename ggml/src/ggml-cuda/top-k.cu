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

#if defined(GGML_USE_HIP)
#include <hipcub/hipcub.hpp>
// hipCUB has no DeviceTopK, so on ROCm top-k is implemented by sorting (score, index) pairs in
// descending order with DeviceRadixSort (single row) / DeviceSegmentedRadixSort (multi-row) and
// keeping the first k of each row. These radix sorts are stream-capture-safe. The helpers below
// fill the per-element value (original column index) and the per-row segment offsets.
static __global__ void ggml_cuda_topk_iota_rows(int * idx, const int ncols, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        idx[i] = (int) (i % ncols);
    }
}
static __global__ void ggml_cuda_topk_init_offsets(int * off, const int ncols, const int n1) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < n1) {
        off[s] = s * ncols;
    }
}
#endif // GGML_USE_HIP

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
#elif defined(GGML_USE_HIP)  // CUB_TOP_K_AVAILABLE
    // ROCm top-k via capture-safe radix sort-pairs (see doc/PLAN_hip_partial_topk.md).
    {
        const int     block = 256;
        const int64_t n     = ncols * nrows;
        ggml_cuda_pool_alloc<float> keys_alloc(pool, n);
        ggml_cuda_pool_alloc<int>   vals_in_alloc(pool, n);
        ggml_cuda_pool_alloc<int>   vals_out_alloc(pool, n);
        float * keys     = keys_alloc.get();
        int   * vals_in  = vals_in_alloc.get();
        int   * vals_out = vals_out_alloc.get();

        CUDA_CHECK(cudaMemcpyAsync(keys, src0_d, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        ggml_cuda_topk_iota_rows<<<(n + block - 1) / block, block, 0, stream>>>(vals_in, (int) ncols, n);

        size_t tmp_bytes = 0;
        if (nrows == 1) {
            CUDA_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(nullptr, tmp_bytes, keys, keys,
                vals_in, vals_out, (int) ncols, 0, (int) sizeof(float) * 8, stream));
            ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_bytes);
            CUDA_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(tmp_alloc.get(), tmp_bytes, keys, keys,
                vals_in, vals_out, (int) ncols, 0, (int) sizeof(float) * 8, stream));
        } else {
            ggml_cuda_pool_alloc<int> off_alloc(pool, nrows + 1);
            int * off = off_alloc.get();
            ggml_cuda_topk_init_offsets<<<(nrows + 1 + block - 1) / block, block, 0, stream>>>(off, (int) ncols, (int) (nrows + 1));
            CUDA_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, tmp_bytes, keys, keys,
                vals_in, vals_out, (int) n, (int) nrows, off, off + 1, 0, (int) sizeof(float) * 8, stream));
            ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_bytes);
            CUDA_CHECK(hipcub::DeviceSegmentedRadixSort::SortPairsDescending(tmp_alloc.get(), tmp_bytes, keys, keys,
                vals_in, vals_out, (int) n, (int) nrows, off, off + 1, 0, (int) sizeof(float) * 8, stream));
        }
        // keep the first k sorted indices of each row
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), vals_out, ncols * sizeof(int),
                                     k * sizeof(int), nrows, cudaMemcpyDeviceToDevice, stream));
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
