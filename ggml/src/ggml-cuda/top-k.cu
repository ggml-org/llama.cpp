#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
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

// The DeepSeek lightning indexer uses top-k = 512, 1024, or 2048. On RDNA3.5,
// sorting its small score rows locally avoids a device-wide hipCUB argsort.
__device__ __forceinline__ static bool top_k_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

template <uint32_t SORT_N, typename index_t>
__global__ static void top_k_strix_bitonic(
        int *         dst,
        const float * src,
        uint32_t      ncols,
        uint32_t      nrows,
        uint32_t      k) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= nrows) {
        return;
    }

    __shared__ float   values[SORT_N];
    __shared__ index_t indices[SORT_N];

    const float * src_row = src + (uint64_t) row * ncols;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < ncols) {
            values[i] = src_row[i];
            indices[i] = (index_t) i;
        } else {
            values[i] = -INFINITY;
            indices[i] = (index_t) -1;
        }
    }
    __syncthreads();

    for (uint32_t width = 2; width <= SORT_N; width <<= 1) {
        for (uint32_t stride = width >> 1; stride > 0; stride >>= 1) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                const uint32_t other = i ^ stride;
                if (other > i && other < SORT_N) {
                    const float av = values[i];
                    const float bv = values[other];
                    const uint32_t ai = indices[i];
                    const uint32_t bi = indices[other];
                    const bool descending_half = (i & width) == 0;
                    const bool swap = descending_half
                        ? top_k_score_better(bv, bi, av, ai)
                        : top_k_score_better(av, ai, bv, bi);
                    if (swap) {
                        values[i] = bv;
                        indices[i] = (index_t) bi;
                        values[other] = av;
                        indices[other] = (index_t) ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    int * dst_row = dst + (uint64_t) row * k;
    for (uint32_t i = tid; i < k; i += blockDim.x) {
        dst_row[i] = indices[i];
    }
}

static bool top_k_strix(
        const float * src,
        int *         dst,
        int64_t       ncols,
        int64_t       nrows,
        int64_t       k,
        cudaStream_t  stream) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc) || ncols <= 0 || ncols > 8192 || nrows <= 0 ||
        (k != 512 && k != 1024 && k != 2048) || k > ncols || nrows > UINT32_MAX) {
        return false;
    }

    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    if (ncols <= 4096) {
        if (sizeof(float) * 4096 + sizeof(uint32_t) * 4096 > max_shared_mem) {
            return false;
        }
        top_k_strix_bitonic<4096, uint32_t><<<nrows, 1024, 0, stream>>>(
            dst, src, (uint32_t) ncols, (uint32_t) nrows, (uint32_t) k);
    } else {
        if (sizeof(float) * 8192 + sizeof(uint16_t) * 8192 > max_shared_mem) {
            return false;
        }
        top_k_strix_bitonic<8192, uint16_t><<<nrows, 1024, 0, stream>>>(
            dst, src, (uint32_t) ncols, (uint32_t) nrows, (uint32_t) k);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
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
    if (top_k_strix(src0_d, dst_d, ncols, nrows, k, stream)) {
        return;
    }
#endif  // defined(GGML_USE_HIP)
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
