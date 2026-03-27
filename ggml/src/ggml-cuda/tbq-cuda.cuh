#pragma once
#include "common.cuh"
#include "../ggml-common.h"

/*
 * TurboQuant CUDA device functions for ggml integration.
 * Provides SET_ROWS quantize + convert/CPY dequantize on GPU.
 * Rotation matrix accessed via device global pointers.
 */

/* Device global: rotation matrix pointers (set from host) */
extern __device__ float* g_tbq_rotation;
extern __device__ float* g_tbq_rotation_t;

void tbq_cuda_set_rotation(float* d_rotation, float* d_rotation_t);

/* ================================================================
 * Per-block quantize functions for SET_ROWS template
 * Single thread per block — correct but not optimal.
 * ================================================================ */

static __device__ void quantize_f32_tbq2_0_block(const float* src, block_tbq2_0* dst) {
    const float* rot = g_tbq_rotation;
    if (!rot) return;
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    dst->d = __float2half(norm);
    uint8_t* p = dst->qs;
    for (int i = 0; i < QK_TBQ / 4; i++) p[i] = 0;
    for (int r = 0; r < QK_TBQ; r++) {
        float y = 0.0f;
        for (int c = 0; c < QK_TBQ; c++) y += rot[r * QK_TBQ + c] * src[c];
        y *= inv_norm;
        int best = 0; float bd = fabsf(y - turbo_quant_centroids_2bit[0]);
        for (int k = 1; k < 4; k++) { float d = fabsf(y - turbo_quant_centroids_2bit[k]); if (d < bd) { bd = d; best = k; } }
        p[r / 4] |= (uint8_t)(best << ((r % 4) * 2));
    }
}

static __device__ void quantize_f32_tbq3_0_block(const float* src, block_tbq3_0* dst) {
    const float* rot = g_tbq_rotation;
    if (!rot) return;
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    dst->d = __float2half(norm);
    uint8_t* p = dst->qs;
    for (int i = 0; i < QK_TBQ * 3 / 8; i++) p[i] = 0;
    for (int r = 0; r < QK_TBQ; r++) {
        float y = 0.0f;
        for (int c = 0; c < QK_TBQ; c++) y += rot[r * QK_TBQ + c] * src[c];
        y *= inv_norm;
        int best = 0; float bd = fabsf(y - turbo_quant_centroids_3bit[0]);
        for (int k = 1; k < 8; k++) { float d = fabsf(y - turbo_quant_centroids_3bit[k]); if (d < bd) { bd = d; best = k; } }
        int bp = r * 3, bi = bp / 8, bo = bp % 8;
        p[bi] |= (uint8_t)(best << bo);
        if (bo + 3 > 8) p[bi + 1] |= (uint8_t)(best >> (8 - bo));
    }
}

static __device__ void quantize_f32_tbq4_0_block(const float* src, block_tbq4_0* dst) {
    const float* rot = g_tbq_rotation;
    if (!rot) return;
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TBQ; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    dst->d = __float2half(norm);
    uint8_t* p = dst->qs;
    for (int i = 0; i < QK_TBQ / 2; i++) p[i] = 0;
    for (int r = 0; r < QK_TBQ; r++) {
        float y = 0.0f;
        for (int c = 0; c < QK_TBQ; c++) y += rot[r * QK_TBQ + c] * src[c];
        y *= inv_norm;
        int best = 0; float bd = fabsf(y - turbo_quant_centroids_4bit[0]);
        for (int k = 1; k < 16; k++) { float d = fabsf(y - turbo_quant_centroids_4bit[k]); if (d < bd) { bd = d; best = k; } }
        p[r / 2] |= (uint8_t)(best << ((r % 2) * 4));
    }
}

/* ================================================================
 * OPTIMIZED GPU dequantize kernel v2
 *
 * Key optimizations over v1:
 *   1. Rotation matrix loaded to shared memory in 2 tiles (32 KB each)
 *   2. 4 vectors processed per block (amortized rotation load)
 *   3. Coalesced global memory reads
 *
 * Architecture: 128 threads per block, 4 vectors per block
 *   - Thread tid handles coordinate tid for all 4 vectors
 *   - Rotation tile loaded cooperatively by all 128 threads
 *   - Two passes: coords [0,63] and [64,127]
 *
 * Memory: 32 KB shared (64 rows × 128 cols × 4 bytes per tile)
 * ================================================================ */

#define TBQ_VECTORS_PER_BLOCK 4
#define TBQ_TILE_ROWS 64

template <int bits>
static __global__ void k_dequantize_tbq_v2(
    const void * __restrict__ vx,
    float      * __restrict__ y,
    const float* __restrict__ rotation_t,  /* explicit param, not device global */
    int64_t k
) {
    /* Block handles TBQ_VECTORS_PER_BLOCK consecutive vectors */
    const int block_vec_start = blockIdx.x * TBQ_VECTORS_PER_BLOCK;
    const int tid = threadIdx.x;  /* 0..127 */
    const int dim = QK_TBQ;       /* 128 */
    const int n_vectors = k / dim;

    const int packed_bytes = (dim * bits + 7) / 8;
    const int block_bytes = 2 + packed_bytes;

    /* Shared memory: rotation tile (64 rows × 128 cols) + centroid buffer */
    __shared__ float s_rot_tile[TBQ_TILE_ROWS * QK_TBQ];  /* 32 KB */
    __shared__ float s_centroids[TBQ_VECTORS_PER_BLOCK * QK_TBQ]; /* 2 KB */

    /* Select centroid table */
    const float* codebook;
    int n_levels;
    if (bits == 2)      { codebook = turbo_quant_centroids_2bit; n_levels = 4; }
    else if (bits == 3) { codebook = turbo_quant_centroids_3bit; n_levels = 8; }
    else                { codebook = turbo_quant_centroids_4bit; n_levels = 16; }

    /* Unpack all vectors' centroids into shared memory */
    for (int vi = 0; vi < TBQ_VECTORS_PER_BLOCK; vi++) {
        int vec_id = block_vec_start + vi;
        if (vec_id >= n_vectors) {
            /* Zero out unused slots */
            s_centroids[vi * dim + tid] = 0.0f;
            continue;
        }

        const uint8_t* block = (const uint8_t*)vx + (int64_t)vec_id * block_bytes;
        const uint8_t* packed = block + 2;

        int idx = 0;
        if (bits == 2) {
            idx = (packed[tid / 4] >> ((tid % 4) * 2)) & 0x3;
        } else if (bits == 3) {
            int bp = tid * 3, bi = bp / 8, bo = bp % 8;
            int val = packed[bi] >> bo;
            if (bo + 3 > 8 && bi + 1 < packed_bytes)
                val |= ((int)packed[bi + 1]) << (8 - bo);
            idx = val & 0x7;
        } else {
            idx = (packed[tid / 2] >> ((tid % 2) * 4)) & 0xF;
        }
        s_centroids[vi * dim + tid] = codebook[idx];
    }
    __syncthreads();

    /* Partial sums for each vector (accumulated across 2 tiles) */
    float partial[TBQ_VECTORS_PER_BLOCK] = {0.0f, 0.0f, 0.0f, 0.0f};

    /* Process rotation in 2 tiles of 64 rows each */
    for (int tile = 0; tile < 2; tile++) {
        int tile_start = tile * TBQ_TILE_ROWS;

        /* Cooperatively load rotation tile to shared memory */
        /* 128 threads load 64×128 = 8192 floats → 64 floats per thread */
        for (int i = tid; i < TBQ_TILE_ROWS * dim; i += dim) {
            int row = tile_start + i / dim;
            int col = i % dim;
            if (row < dim)
                s_rot_tile[i] = rotation_t[row * dim + col];
        }
        __syncthreads();

        /* For this thread's coordinate (tid), accumulate partial sum */
        /* We need: out[tid] = sum_c rot_t[tid][c] * centroid[c] */
        /* But tid might be in tile range [tile_start, tile_start+64) or not */

        /* Actually: out[r] = sum_{c=0}^{127} rot_t[r][c] * centroid[c] */
        /* With tiles on rows: we process rows [tile_start..tile_start+63] */
        /* But tid is the OUTPUT coordinate. The rotation sum goes over ALL columns. */

        /* Rethink: rotation_t[tid][c] for c in [0..127] */
        /* We tile over COLUMNS of rotation_t to keep data in shared memory */
        /* Actually better: tile over the summation index (columns) */

        /* Let me restructure: load tile of COLUMNS, not rows */
        __syncthreads(); /* ensure tile is loaded */

        /* Compute: for each vector vi, partial contribution from columns [tile_start..tile_start+63] */
        for (int vi = 0; vi < TBQ_VECTORS_PER_BLOCK; vi++) {
            float sum = 0.0f;
            /* Sum over columns [tile_start .. tile_start+TBQ_TILE_ROWS-1] */
            /* s_rot_tile stores rotation_t rows [tile_start..], but we need rotation_t[tid][c] */
            /* rotation_t[tid][tile_start + j] for j in [0..63] */
            /* This is NOT in shared memory as loaded — we loaded rows tile_start..tile_start+63 */
            /* We need column access: rot_t[tid][tile_start+j] = the (tile_start+j)-th row, tid-th column */
            /* = s_rot_tile[j * dim + tid] ← this IS in shared memory! (row j, col tid) */

            for (int j = 0; j < TBQ_TILE_ROWS; j++) {
                sum += s_rot_tile[j * dim + tid] * s_centroids[vi * dim + tile_start + j];
            }
            partial[vi] += sum;
        }
        __syncthreads();
    }

    /* Write results */
    for (int vi = 0; vi < TBQ_VECTORS_PER_BLOCK; vi++) {
        int vec_id = block_vec_start + vi;
        if (vec_id >= n_vectors) continue;

        const uint8_t* block = (const uint8_t*)vx + (int64_t)vec_id * block_bytes;
        float norm = __half2float(*(const __half*)block);

        y[vec_id * dim + tid] = partial[vi] * norm;
    }
}

/* Host wrappers for optimized kernel */
static void dequantize_row_tbq2_0_cuda_v2(const void* vx, float* y, int64_t k,
                                            const float* d_rot_t, cudaStream_t stream) {
    const int n = k / QK_TBQ;
    const int n_blocks = (n + TBQ_VECTORS_PER_BLOCK - 1) / TBQ_VECTORS_PER_BLOCK;
    if (n > 0) k_dequantize_tbq_v2<2><<<n_blocks, QK_TBQ, 0, stream>>>(vx, y, d_rot_t, k);
}

static void dequantize_row_tbq3_0_cuda_v2(const void* vx, float* y, int64_t k,
                                            const float* d_rot_t, cudaStream_t stream) {
    const int n = k / QK_TBQ;
    const int n_blocks = (n + TBQ_VECTORS_PER_BLOCK - 1) / TBQ_VECTORS_PER_BLOCK;
    if (n > 0) k_dequantize_tbq_v2<3><<<n_blocks, QK_TBQ, 0, stream>>>(vx, y, d_rot_t, k);
}

static void dequantize_row_tbq4_0_cuda_v2(const void* vx, float* y, int64_t k,
                                            const float* d_rot_t, cudaStream_t stream) {
    const int n = k / QK_TBQ;
    const int n_blocks = (n + TBQ_VECTORS_PER_BLOCK - 1) / TBQ_VECTORS_PER_BLOCK;
    if (n > 0) k_dequantize_tbq_v2<4><<<n_blocks, QK_TBQ, 0, stream>>>(vx, y, d_rot_t, k);
}



/* ================================================================
 * OPTIMIZED GPU dequantize kernel v3 — FP16 rotation
 *
 * Key improvement over v2:
 *   - Rotation stored as FP16 (32 KB total)
 *   - ENTIRE 128x128 matrix fits in shared memory in ONE pass
 *   - No tiling needed — simpler, faster
 *   - FP16→FP32 conversion is free on GPU hardware
 * ================================================================ */

extern half* tbq_cuda_get_rotation_t_fp16(void);

template <int bits>
static __global__ void k_dequantize_tbq_v3(
    const void * __restrict__ vx,
    float      * __restrict__ y,
    const half * __restrict__ rotation_t_fp16,
    int64_t k
) {
    const int block_vec_start = blockIdx.x * TBQ_VECTORS_PER_BLOCK;
    const int tid = threadIdx.x;
    const int dim = QK_TBQ;
    const int n_vectors = k / dim;

    const int packed_bytes = (dim * bits + 7) / 8;
    const int block_bytes = 2 + packed_bytes;

    /* Load ENTIRE rotation matrix (FP16) to shared memory — 32 KB, one pass! */
    __shared__ half s_rot[QK_TBQ * QK_TBQ];  /* 128*128*2 = 32768 bytes */
    /* 128 threads load 16384 halfs = 128 halfs per thread */
    for (int i = tid; i < dim * dim; i += dim)
        s_rot[i] = rotation_t_fp16[i];
    __syncthreads();

    /* Centroid table */
    const float* codebook;
    if (bits == 2)      codebook = turbo_quant_centroids_2bit;
    else if (bits == 3)  codebook = turbo_quant_centroids_3bit;
    else                 codebook = turbo_quant_centroids_4bit;

    /* Unpack + dequantize 4 vectors */
    for (int vi = 0; vi < TBQ_VECTORS_PER_BLOCK; vi++) {
        int vec_id = block_vec_start + vi;
        if (vec_id >= n_vectors) continue;

        const uint8_t* block = (const uint8_t*)vx + (int64_t)vec_id * block_bytes;
        float norm = __half2float(*(const __half*)block);
        const uint8_t* packed = block + 2;

        /* Unpack this thread's index */
        int idx = 0;
        if (bits == 2) {
            idx = (packed[tid / 4] >> ((tid % 4) * 2)) & 0x3;
        } else if (bits == 3) {
            int bp = tid * 3, bi = bp / 8, bo = bp % 8;
            int val = packed[bi] >> bo;
            if (bo + 3 > 8 && bi + 1 < packed_bytes)
                val |= ((int)packed[bi + 1]) << (8 - bo);
            idx = val & 0x7;
        } else {
            idx = (packed[tid / 2] >> ((tid % 2) * 4)) & 0xF;
        }

        float centroid_val = codebook[idx];

        /* Share centroids across threads for rotation */
        __shared__ float s_centroids[QK_TBQ];
        s_centroids[tid] = centroid_val;
        __syncthreads();

        /* Inverse rotation: out[tid] = sum_c rot_t[tid*dim+c] * s_centroids[c] */
        float sum = 0.0f;
        const half* rot_row = s_rot + tid * dim;
        for (int c = 0; c < dim; c++)
            sum += __half2float(rot_row[c]) * s_centroids[c];

        y[vec_id * dim + tid] = sum * norm;
        __syncthreads();
    }
}

/* v1-compatible wrappers — auto-select v3 (FP16) or v2 (FP32) */
static void dequantize_row_tbq2_0_cuda(const void* vx, float* y, int64_t k, cudaStream_t stream) {
    const int n = k / QK_TBQ;
    if (n <= 0) return;
    const int n_blocks = (n + TBQ_VECTORS_PER_BLOCK - 1) / TBQ_VECTORS_PER_BLOCK;
    /* v2 (FP32 tiled) is faster than v3 (FP16 full) due to better occupancy */
    float* rot_f32 = nullptr;
    cudaMemcpyFromSymbol(&rot_f32, g_tbq_rotation_t, sizeof(float*));
    if (rot_f32) {
        k_dequantize_tbq_v2<2><<<n_blocks, QK_TBQ, 0, stream>>>(vx, y, rot_f32, k);
    }
}

static void dequantize_row_tbq3_0_cuda(const void* vx, float* y, int64_t k, cudaStream_t stream) {
    const int n = k / QK_TBQ;
    if (n <= 0) return;
    const int n_blocks = (n + TBQ_VECTORS_PER_BLOCK - 1) / TBQ_VECTORS_PER_BLOCK;
    /* v2 (FP32 tiled) is faster than v3 (FP16 full) due to better occupancy */
    float* rot_f32 = nullptr;
    cudaMemcpyFromSymbol(&rot_f32, g_tbq_rotation_t, sizeof(float*));
    if (rot_f32) {
        k_dequantize_tbq_v2<3><<<n_blocks, QK_TBQ, 0, stream>>>(vx, y, rot_f32, k);
    }
}

static void dequantize_row_tbq4_0_cuda(const void* vx, float* y, int64_t k, cudaStream_t stream) {
    const int n = k / QK_TBQ;
    if (n <= 0) return;
    const int n_blocks = (n + TBQ_VECTORS_PER_BLOCK - 1) / TBQ_VECTORS_PER_BLOCK;
    /* v2 (FP32 tiled) is faster than v3 (FP16 full) due to better occupancy */
    float* rot_f32 = nullptr;
    cudaMemcpyFromSymbol(&rot_f32, g_tbq_rotation_t, sizeof(float*));
    if (rot_f32) {
        k_dequantize_tbq_v2<4><<<n_blocks, QK_TBQ, 0, stream>>>(vx, y, rot_f32, k);
    }
}
