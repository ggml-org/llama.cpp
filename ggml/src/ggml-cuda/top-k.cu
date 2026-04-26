#include "argsort.cuh"
#include "top-k.cuh"

#include <cfloat>
#include <climits>
#include <cmath>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
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

template<int ncols>
static __global__ void top_k_warp_f32_i32(const float * src, int * dst, const int k, const int nrows) {
    constexpr int experts_per_thread = (ncols + WARP_SIZE - 1) / WARP_SIZE;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }

    const int lane = threadIdx.x;
    src += row * ncols;
    dst += row * k;

    float vals[experts_per_thread];
    uint32_t active_mask = 0;

#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        const int idx = lane + i * WARP_SIZE;
        const bool active = idx < ncols;
        if (active) {
            active_mask |= 1u << i;
        }
        float val = active ? src[idx] : -INFINITY;
        vals[i] = __isnanf(val) ? -FLT_MAX : val;
    }

    for (int out = 0; out < k; ++out) {
        float max_val = -INFINITY;
        int   max_idx = INT_MAX;

#pragma unroll
        for (int i = 0; i < experts_per_thread; ++i) {
            const int idx = lane + i * WARP_SIZE;
            if (((active_mask >> i) & 1u) && (vals[i] > max_val || (vals[i] == max_val && idx < max_idx))) {
                max_val = vals[i];
                max_idx = idx;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   other_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, mask, WARP_SIZE);
            if (other_val > max_val || (other_val == max_val && other_idx < max_idx)) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        if (lane == out) {
            dst[out] = max_idx;
        }

        if (max_idx < ncols && (max_idx & (WARP_SIZE - 1)) == lane) {
            active_mask &= ~(1u << (max_idx / WARP_SIZE));
        }
    }
}

static bool top_k_warp(const float * src, int * dst, const int ncols, const int nrows, const int k, cudaStream_t stream) {
    if (k <= 0 || k > WARP_SIZE) {
        return false;
    }

    constexpr int rows_per_block = 4;
    const dim3 grid((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block(WARP_SIZE, rows_per_block, 1);

    switch (ncols) {
        case 128:
            top_k_warp_f32_i32<128><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            return true;
        case 256:
            top_k_warp_f32_i32<256><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            return true;
        case 512:
            top_k_warp_f32_i32<512><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            return true;
        case 576:
            top_k_warp_f32_i32<576><<<grid, block, 0, stream>>>(src, dst, k, nrows);
            return true;
        default:
            return false;
    }
}

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
    if (top_k_warp(src0_d, dst_d, ncols, nrows, k, stream)) {
        return;
    }
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

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
