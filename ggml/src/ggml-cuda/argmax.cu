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

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
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

// Two-stage reduction for wide rows: stage 1 tiles the row across SMs (ARGMAX_BLOCK
// threads each, smem tree reduce); stage 2 reduces the small partial array with one warp.
#define ARGMAX_TILE  8192
#define ARGMAX_BLOCK  256

static __global__ void argmax_f32_tile(
        const float * __restrict__ x, float * __restrict__ part_val, int32_t * __restrict__ part_idx,
        const int64_t ncols, const int64_t ntiles) {
    __shared__ float smem_val[ARGMAX_BLOCK];
    __shared__ int   smem_idx[ARGMAX_BLOCK];

    const int64_t row  = blockIdx.x / ntiles;
    const int64_t tile = blockIdx.x % ntiles;
    const int64_t beg  = tile * ARGMAX_TILE;
    const int64_t end  = beg + ARGMAX_TILE < ncols ? beg + ARGMAX_TILE : ncols;

    float maxval = -FLT_MAX;
    int   argmax = -1;

    for (int64_t col = beg + threadIdx.x; col < end; col += ARGMAX_BLOCK) {
        const float val = x[row * ncols + col];
        if (val > maxval) { maxval = val; argmax = (int) col; }
    }

    smem_val[threadIdx.x] = maxval;
    smem_idx[threadIdx.x] = argmax;
    __syncthreads();

    for (int s = ARGMAX_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smem_val[threadIdx.x + s] > smem_val[threadIdx.x]) {
            smem_val[threadIdx.x] = smem_val[threadIdx.x + s];
            smem_idx[threadIdx.x] = smem_idx[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        part_val[row * ntiles + tile] = smem_val[0];
        part_idx[row * ntiles + tile] = smem_idx[0];
    }
}

static __global__ void argmax_f32_combine(
        const float * __restrict__ part_val, const int32_t * __restrict__ part_idx,
        int32_t * __restrict__ dst, const int64_t ntiles) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;

    for (int i = threadIdx.x; i < ntiles; i += WARP_SIZE) {
        const float val = part_val[row * ntiles + i];
        if (val > maxval) { maxval = val; argmax = part_idx[row * ntiles + i]; }
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

    GGML_ASSERT(ne00 <= INT32_MAX);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;
    cudaStream_t stream = ctx.stream();

    if (ne00 >= ARGMAX_TILE) {
        const int64_t ntiles = (ne00 + ARGMAX_TILE - 1) / ARGMAX_TILE;

        ggml_cuda_pool_alloc<float>   part_val(ctx.pool(), nrows * ntiles);
        ggml_cuda_pool_alloc<int32_t> part_idx(ctx.pool(), nrows * ntiles);

        argmax_f32_tile   <<<dim3(nrows * ntiles, 1, 1), dim3(ARGMAX_BLOCK, 1, 1), 0, stream>>>(
                src0_d, part_val.get(), part_idx.get(), ne00, ntiles);
        argmax_f32_combine<<<dim3(nrows,          1, 1), dim3(WARP_SIZE,    1, 1), 0, stream>>>(
                part_val.get(), part_idx.get(), dst_d, ntiles);
        return;
    }

    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    argmax_f32<<<dim3(nrows, 1, 1), dim3(num_threads, 1, 1), 0, stream>>>(src0_d, dst_d, ne00);
}
