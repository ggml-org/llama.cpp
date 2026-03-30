// tq3-prefill.cuh — Native TQ3_0 prefill kernel
// Design: shared memory activation tile, WHT once per weight block
// TR=1 warps per output row, TT=token tile size (adaptive)

#pragma once
#include "tq3-native.cuh"

#define TQ3_PREFILL_WARP 32

// Tile: TR output rows per block, TT tokens per block
// Shared memory: smem[TT][QK] — activation slice loaded once, reused by all TR warps
template<int TR, int TT>
__global__ void tq3_prefill_kernel_tiled(
    const block_tq3_0 * __restrict__ W,
    const float       * __restrict__ A,
    float             * __restrict__ D,
    int ne00, int ne01, int ne11)
{
    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;
    const int row0    = blockIdx.x * TR;
    const int tok0    = blockIdx.y * TT;
    const int row     = row0 + warp_id;
    const int nb      = ne00 / QK_TQ3_0;

    __shared__ float smem[TT][QK_TQ3_0];
    float acc[TT] = {};

    for (int blk = 0; blk < nb; blk++) {
        // Load activation tile (all warps cooperate, each loads one token row)
        if (warp_id < TT) {
            const int tok = tok0 + warp_id;
            smem[warp_id][lane] = (tok < ne11) ? A[tok * ne00 + blk * QK_TQ3_0 + lane] : 0.0f;
        }
        __syncthreads();

        if (row < ne01) {
            const block_tq3_0 * bq = W + row * nb + blk;
            const float rms = __half2float(bq->d);

            // Subgroup leader loads packed bytes, broadcasts
            const int g = lane / 8, r = lane % 8, leader = g * 8;
            uint32_t packed = 0;
            if (r == 0) {
                const uint8_t * qp = bq->qs + g * 3;
                packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
            }
            packed = __shfl_sync(0xFFFFFFFF, packed, leader);

            // Centroid + WHT (once per block, reused for all TT tokens)
            float val = ggml_cuda_tq3_centroid(ggml_cuda_tq3_unpack_idx(packed, r));
            #pragma unroll
            for (int step = 1; step < 32; step <<= 1) {
                const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
                val = (lane & step) ? (other - val) : (other + val);
            }
            const float wj = val * ggml_cuda_tq3_sign(lane) * (rms / sqrtf((float)QK_TQ3_0));

            // Dot with all TT tokens from shared memory
            #pragma unroll
            for (int t = 0; t < TT; t++)
                acc[t] += wj * smem[t][lane];
        }
        __syncthreads();
    }

    // Warp reduce and write output
    if (row < ne01) {
        #pragma unroll
        for (int t = 0; t < TT; t++) {
            float s = acc[t];
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1)
                s += __shfl_xor_sync(0xFFFFFFFF, s, m);
            if (lane == 0) {
                const int tok = tok0 + t;
                if (tok < ne11) D[row * ne11 + tok] = s;
            }
        }
    }
}

// Launch: choose TT based on ne11 to maximize occupancy
static void tq3_prefill_launch(
    const block_tq3_0 * weights,
    const float       * act,
    float             * dst,
    int ne00, int ne01, int ne11,
    cudaStream_t stream)
{
    // TR=1 (one warp per output row), TT=min(ne11, 64)
    // TT must be a compile-time constant, so we dispatch to the best fit
    if (ne11 >= 64) {
        const dim3 grid(ne01, (ne11 + 63) / 64);
        tq3_prefill_kernel_tiled<1, 64><<<grid, dim3(32, 1), 0, stream>>>(
            weights, act, dst, ne00, ne01, ne11);
    } else if (ne11 >= 32) {
        const dim3 grid(ne01, (ne11 + 31) / 32);
        tq3_prefill_kernel_tiled<1, 32><<<grid, dim3(32, 1), 0, stream>>>(
            weights, act, dst, ne00, ne01, ne11);
    } else if (ne11 >= 16) {
        const dim3 grid(ne01, (ne11 + 15) / 16);
        tq3_prefill_kernel_tiled<1, 16><<<grid, dim3(32, 1), 0, stream>>>(
            weights, act, dst, ne00, ne01, ne11);
    } else {
        const dim3 grid(ne01, (ne11 + 7) / 8);
        tq3_prefill_kernel_tiled<1, 8><<<grid, dim3(32, 1), 0, stream>>>(
            weights, act, dst, ne00, ne01, ne11);
    }
}

// Minimum token count to use native kernel (below this, cuBLAS is faster)
#define TQ3_PREFILL_MIN_TOKENS 8
