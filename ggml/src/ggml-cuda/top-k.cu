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

// process at most this many rows with sequential per-row DeviceTopK dispatches
#define GGML_CUDA_TOP_K_SEQ_MAX_NROWS 16

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

#endif  // CUB_TOP_K_AVAILABLE

#ifdef GGML_CUDA_USE_CUB

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

// batched top k via full argsort of all rows and strided copy of the first k columns
static void top_k_via_argsort(ggml_cuda_pool & pool,
                              const float *    src0_d,
                              int *            dst_d,
                              const int64_t    ncols,
                              const int64_t    nrows,
                              const int64_t    k,
                              const size_t     nb01,
                              cudaStream_t     stream) {
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(nb01, nrows);

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
}

#endif  // GGML_CUDA_USE_CUB

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
    // DeviceTopK has no batched/segmented variant yet, therefore multi row inputs require one sequential dispatch per row
    // The dispatch overhead is catastrophic for batch workloads, and batched argsort is much faster.
    // TODO: Switch to `DeviceSegmentedTopK` once implemented:
    // https://github.com/NVIDIA/cccl/issues/6391
    bool use_seq_top_k = nrows <= GGML_CUDA_TOP_K_SEQ_MAX_NROWS;
#ifdef USE_CUDA_GRAPH
    if (use_seq_top_k) {
        // DeviceTopK stream capture is not guaranteed. The argsort handles capture explicitly.
        cudaStreamCaptureStatus capture_status;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        use_seq_top_k = capture_status == cudaStreamCaptureStatusNone;
    }
#endif  // USE_CUDA_GRAPH
    if (use_seq_top_k) {
        for (int i = 0; i < nrows; i++) {
            top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
        }
    } else {
        top_k_via_argsort(pool, src0_d, dst_d, ncols, nrows, k, src0->nb[1], stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    top_k_via_argsort(pool, src0_d, dst_d, ncols, nrows, k, src0->nb[1], stream);
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
