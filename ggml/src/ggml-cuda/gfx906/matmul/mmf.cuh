#pragma once

// GFX906 custom FP16 GEMM kernel for medium batch sizes (9-2048)

#include "../gfx906-common.cuh"
#include "../gfx906-config.h"
#include "sgemm.cuh"

#ifdef GGML_USE_HIP

#define GFX906_MMF_TILE_M 32
#define GFX906_MMF_TILE_N 64
#define GFX906_MMF_TILE_K 64

template <int tile_m, int tile_n, int tile_k>
__launch_bounds__(256, 4)
static __global__ void gfx906_mul_mat_f16_packed(
        const half * __restrict__ src0,
        const half * __restrict__ src1,
        float * __restrict__ dst,
        const int M, const int N, const int K,
        const int stride_src0_row,
        const int stride_src1_col,
        const int stride_dst_col) {

    const int block_row = blockIdx.x * tile_m;
    const int block_col = blockIdx.y * tile_n;

    if (block_row >= M) return;

    constexpr int lds_a_stride = tile_k + 4;
    constexpr int lds_b_stride = tile_n + 4;

    extern __shared__ char shared_mem[];
    half * tile_A = (half *)shared_mem;
    half * tile_B = tile_A + tile_m * lds_a_stride;

    const int thread_row = threadIdx.x >> 3;
    const int thread_col_group = threadIdx.x & 7;
    const int thread_col_base = thread_col_group << 3;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int out_row = block_row + thread_row;
    const int out_col_base = block_col + thread_col_base;

    for (int k_tile = 0; k_tile < K; k_tile += tile_k) {
        {
            const int elems_A_h2 = (tile_m * tile_k) / 2;

            for (int idx = threadIdx.x; idx < elems_A_h2; idx += blockDim.x) {
                const int flat_idx = idx * 2;
                const int m = flat_idx / tile_k;
                const int k = flat_idx % tile_k;
                const int global_m = block_row + m;
                const int global_k = k_tile + k;

                half2 val;
                if (global_m < M && global_k + 1 < K) {
                    val = *((const half2 *)&src0[global_k + global_m * stride_src0_row]);
                } else if (global_m < M && global_k < K) {
                    val = __halves2half2(src0[global_k + global_m * stride_src0_row], __float2half(0.0f));
                } else {
                    val = __float2half2_rn(0.0f);
                }

                tile_A[m * lds_a_stride + k] = __low2half(val);
                tile_A[m * lds_a_stride + k + 1] = __high2half(val);
            }
        }

        {
            const int elems_B = tile_k * tile_n;
            for (int idx = threadIdx.x; idx < elems_B; idx += blockDim.x) {
                const int k = idx / tile_n;
                const int n = idx % tile_n;
                const int global_k = k_tile + k;
                const int global_n = block_col + n;

                half val = (global_k < K && global_n < N) ?
                    src1[global_k + global_n * stride_src1_col] : __float2half(0.0f);
                tile_B[k * lds_b_stride + n] = val;
            }
        }

        __syncthreads();

        if (thread_row < tile_m) {
            const half * a_row = &tile_A[thread_row * lds_a_stride];

            #pragma unroll 4
            for (int kk = 0; kk < tile_k; kk++) {
                const float a_val = __half2float(a_row[kk]);
                const half * b_row = &tile_B[kk * lds_b_stride + thread_col_base];

                acc0 = __fmaf_rn(a_val, __half2float(b_row[0]), acc0);
                acc1 = __fmaf_rn(a_val, __half2float(b_row[1]), acc1);
                acc2 = __fmaf_rn(a_val, __half2float(b_row[2]), acc2);
                acc3 = __fmaf_rn(a_val, __half2float(b_row[3]), acc3);
                acc4 = __fmaf_rn(a_val, __half2float(b_row[4]), acc4);
                acc5 = __fmaf_rn(a_val, __half2float(b_row[5]), acc5);
                acc6 = __fmaf_rn(a_val, __half2float(b_row[6]), acc6);
                acc7 = __fmaf_rn(a_val, __half2float(b_row[7]), acc7);
            }
        }

        __syncthreads();
    }

    if (out_row < M) {
        if (out_col_base + 0 < N) dst[out_row + (out_col_base + 0) * stride_dst_col] = acc0;
        if (out_col_base + 1 < N) dst[out_row + (out_col_base + 1) * stride_dst_col] = acc1;
        if (out_col_base + 2 < N) dst[out_row + (out_col_base + 2) * stride_dst_col] = acc2;
        if (out_col_base + 3 < N) dst[out_row + (out_col_base + 3) * stride_dst_col] = acc3;
        if (out_col_base + 4 < N) dst[out_row + (out_col_base + 4) * stride_dst_col] = acc4;
        if (out_col_base + 5 < N) dst[out_row + (out_col_base + 5) * stride_dst_col] = acc5;
        if (out_col_base + 6 < N) dst[out_row + (out_col_base + 6) * stride_dst_col] = acc6;
        if (out_col_base + 7 < N) dst[out_row + (out_col_base + 7) * stride_dst_col] = acc7;
    }
}

static bool gfx906_mmf_dispatch(
        const half * src0,
        const half * src1,
        float * dst,
        const int M, const int N, const int K,
        const int stride_src0,
        const int stride_src1,
        const int stride_dst,
        cudaStream_t stream) {

    if (N < 9 || N > 2048) {
        return false;
    }

    if (K < 32 || M < 16) {
        return false;
    }

    constexpr int tile_m = GFX906_MMF_TILE_M;
    constexpr int tile_n = GFX906_MMF_TILE_N;
    constexpr int tile_k = GFX906_MMF_TILE_K;

    const int grid_m = (M + tile_m - 1) / tile_m;
    const int grid_n = (N + tile_n - 1) / tile_n;

    dim3 grid(grid_m, grid_n);
    dim3 block(256);

    const size_t shmem_size = (tile_m * (tile_k + 4) + tile_k * (tile_n + 4)) * sizeof(half);

    gfx906_mul_mat_f16_packed<tile_m, tile_n, tile_k><<<grid, block, shmem_size, stream>>>(
        src0, src1, dst,
        M, N, K,
        stride_src0, stride_src1, stride_dst
    );

    return true;
}

static bool gfx906_sgemm_dispatch(
        const float * src0,
        const float * src1,
        float * dst,
        const int M, const int N, const int K,
        const int stride_src0,
        const int stride_src1,
        const int stride_dst,
        cudaStream_t stream) {

    return gfx906_sgemm_custom_dispatch(src0, src1, dst, M, N, K,
                                        stride_src0, stride_src1, stride_dst, stream);
}

#endif // GGML_USE_HIP
