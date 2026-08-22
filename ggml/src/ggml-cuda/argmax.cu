#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __device__ __forceinline__ void argmax_warp_reduce(float & maxval, int & argmax) {
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }
}

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = (int) col;
        }
    }

    argmax_warp_reduce(maxval, argmax);

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
            argmax_warp_reduce(maxval, argmax);
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

// one warp per chunk of a row, writing the chunk-local (val, idx)
static __global__ void argmax_f32_chunk(const float * __restrict__ x, float * __restrict__ part_val, int32_t * __restrict__ part_idx,
        const int64_t ncols, const int64_t nchunks) {
    const int64_t row        = blockIdx.x / nchunks;
    const int64_t chunk      = blockIdx.x % nchunks;
    const int64_t chunk_size = (ncols + nchunks - 1) / nchunks;
    const int64_t beg        = chunk * chunk_size;
    const int64_t end        = beg + chunk_size < ncols ? beg + chunk_size : ncols;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int64_t col = beg + threadIdx.x; col < end; col += WARP_SIZE) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = (int) col;
        }
    }

    argmax_warp_reduce(maxval, argmax);

    if (threadIdx.x == 0) {
        part_val[row*nchunks + chunk] = maxval;
        part_idx[row*nchunks + chunk] = argmax;
    }
}

// one warp per row, reducing the per-chunk partials to the row's column index
static __global__ void argmax_f32_combine(const float * __restrict__ part_val, const int32_t * __restrict__ part_idx,
        int32_t * __restrict__ dst, const int64_t nchunks) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;

    for (int i = threadIdx.x; i < nchunks; i += WARP_SIZE) {
        const float val = part_val[row*nchunks + i];
        if (val > maxval) {
            maxval = val;
            argmax = part_idx[row*nchunks + i];
        }
    }

    argmax_warp_reduce(maxval, argmax);

    if (threadIdx.x == 0) {
        dst[row] = argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    // the result indices are I32
    GGML_ASSERT(ne00 <= INT32_MAX);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    // one block per row leaves most SMs idle for few large rows -> reduce chunks first
    const int64_t nchunks = ne00 < 8192 ? 0 :
        std::min<int64_t>(ne00/128, 8*ggml_cuda_info().devices[ggml_cuda_get_device()].nsm/nrows);

    if (nchunks >= 4) {
        ggml_cuda_pool_alloc<float>   part_val(ctx.pool(), nrows*nchunks);
        ggml_cuda_pool_alloc<int32_t> part_idx(ctx.pool(), nrows*nchunks);

        argmax_f32_chunk<<<dim3(nrows*nchunks, 1, 1), dim3(WARP_SIZE, 1, 1), 0, stream>>>(src0_d, part_val.get(), part_idx.get(), ne00, nchunks);
        argmax_f32_combine<<<dim3(nrows, 1, 1), dim3(WARP_SIZE, 1, 1), 0, stream>>>(part_val.get(), part_idx.get(), dst_d, nchunks);

        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
