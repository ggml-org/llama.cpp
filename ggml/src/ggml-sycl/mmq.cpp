//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "mmq.hpp"
#include "vecdotq.hpp"
#include "mmq-esimd.hpp"

// Kernel names for VTune profiling
template<bool check> class mmq_q4_0_kernel;
template<int mmq_x, int mmq_y, int nwarps, bool check> class mmq_q4_0_soa_kernel;  // SoA layout variant
template<bool check> class mmq_q4_1_kernel;
template<bool check> class mmq_q5_0_kernel;
template<bool check> class mmq_q5_1_kernel;
template<bool check> class mmq_q8_0_kernel;
template<int mmq_x, int mmq_y, int nwarps, bool check> class mmq_q8_0_soa_kernel;  // SoA layout variant
template<bool check> class mmq_q2_K_kernel;
template<bool check> class mmq_q3_K_kernel;
template<bool check> class mmq_q4_K_kernel;
template<bool check> class mmq_q5_K_kernel;
template<bool check> class mmq_q6_K_kernel;
template<int mmq_x, int mmq_y, int nwarps, bool check> class mmq_q6_K_soa_kernel;  // SoA layout variant

// MMQ tile size in K dimension, decoupled from WARP_SIZE for portability.
// The K dimension of the tiles has either 1*MMQ_TILE_NE_K==32 or 2*MMQ_TILE_NE_K==64.
// This must be 32 because the quantization block sizes (QI4_K=32, QI5_K=32, QI6_K=32)
// were designed around CUDA's warp size of 32. MMQ_TILE_NE_K is kept as a separate
// constant for clarity and to match the CUDA implementation's tile sizing.
#define MMQ_TILE_NE_K 32

// MMQ iteration size in K dimension - matches CUDA's MMQ_ITER_K.
// Each iteration processes MMQ_ITER_K elements (256 for K-quants).
// This allows processing quantization blocks larger than WARP_SIZE.
#define MMQ_ITER_K 256

typedef void (*allocate_tiles_sycl_t)(
    int** x_ql,
    sycl::half2** x_dm,
    int** x_qh,
    int** x_sc);
typedef void (*load_tiles_sycl_t)(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row);
typedef float (*vec_dot_q_mul_mat_sycl_t)(
    const int* __restrict__ x_ql,
    const sycl::half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const sycl::half2* __restrict__ y_ms,
    const int& i,
    const int& j,
    const int& k);


template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q4_0, float *tile_x_d_q4_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;
    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (const block_q4_0 *) vx;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

// SoA (Structure of Arrays) loader for Q4_0
// In SoA layout:
//   - qs_base points to contiguous quantized values (4-bit nibbles)
//   - d_offset is the byte offset from qs_base to the scales (fp16)
// This is more cache-friendly than AoS for batched GEMM
// NOTE: d_offset is passed instead of d_base to avoid pointer capture issues
// during SYCL graph recording. The offset is computed inside the kernel.
// Q4_0 SoA tile loading with row_low support (matching Q8_0 pattern)
// row_low converts local row indices to absolute indices for SoA addressing
template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_0_soa(const uint8_t *__restrict__ qs_base,
                    const size_t d_offset,
                    int *__restrict__ x_ql,
                    sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                    int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                    const int &k, const int &blocks_per_row,
                    const int &row_offset, const int &block_offset,
                    const int &row_low) {
    (void)x_qh; (void)x_sc;
    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  MMQ_TILE_NE_K);

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    float * x_dmf = (float *) x_dm;

    // Compute d_base inside kernel to avoid pointer capture during graph recording
    const sycl::half * d_base = (const sycl::half *)(qs_base + d_offset);

    // Load quantized values from SoA layout
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        // In SoA: qs values are at qs_base + (row * blocks_per_row + block) * (QK4_0/2) + byte_offset
        // row_low converts local row indices to absolute indices for SoA addressing
        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbx;
        const uint8_t * qs_ptr = qs_base + global_block * (QK4_0/2);

        // get_int_from_uint8 reads 4 bytes at offset kqsx*4
        // Use MMQ_TILE_NE_K (32) for tile stride to match logical tile width
        x_ql[i * (MMQ_TILE_NE_K + 1) + k] = get_int_from_uint8(qs_ptr, kqsx);
    }

    // Load scales from SoA layout - must match AoS tile indexing for vec_dot compatibility
    const int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_0;  // 4 (matches AoS load_tiles_q4_0)
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        // In SoA: d values are at d_base + (row * blocks_per_row + block)
        // row_low converts local row indices to absolute indices for SoA addressing
        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbxd;

        // Use same tile indexing as AoS load_tiles_q4_0 for compatibility with vec_dot
        x_dmf[i * (MMQ_TILE_NE_K/QI4_0) + i / QI4_0 + kbxd] = d_base[global_block];
    }
}

static __dpct_inline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    // kyqs computes an index into y_qs that can reach up to ~28 for k=15
    // Use MMQ_TILE_NE_K (32) which matches CUDA tile sizing for quantized blocks
    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        // Use MMQ_TILE_NE_K (32) for y_qs stride and modulo to avoid wraparound on Intel
        u[2*l+0] = y_qs[j * MMQ_TILE_NE_K + (kyqs + l)         % MMQ_TILE_NE_K];
        u[2*l+1] = y_qs[j * MMQ_TILE_NE_K + (kyqs + l + QI4_0) % MMQ_TILE_NE_K];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (MMQ_TILE_NE_K/QI8_1) + (2*k/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
}

// SoA version of vec_dot - uses MMQ_TILE_NE_K (32) for tile indexing
// This matches the logical tile width regardless of hardware warp size
static __dpct_inline__ float vec_dot_q4_0_q8_1_mul_mat_soa(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * MMQ_TILE_NE_K + (kyqs + l)         % MMQ_TILE_NE_K];
        u[2*l+1] = y_qs[j * MMQ_TILE_NE_K + (kyqs + l + QI4_0) % MMQ_TILE_NE_K];
    }

    // Use MMQ_TILE_NE_K (32) for tile stride to match logical tile width
    // dm index: k/QI4_0 gives the block index within the tile (0-7 for 8 blocks)
    // Q4_0 needs both d and s from Y (half2), so read y_ds directly as half2
    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (MMQ_TILE_NE_K + 1) + k], u, x_dmf[i * (MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (MMQ_TILE_NE_K/QI8_1) + (2*k/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_1(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_1(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (const block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

static __dpct_inline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_0, float *tile_x_d_q5_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_binary<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_binary<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

static __dpct_inline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_1(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_1(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __dpct_inline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q8_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q8_0, float *tile_x_d_q8_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

// Q8_0 uses float* directly for x_df (not half2*) to avoid type-punning UB
// Use MMQ_TILE_NE_K (32) for tile indexing to match CUDA tile sizing
template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q8_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                float *__restrict__ x_df, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  MMQ_TILE_NE_K);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (MMQ_TILE_NE_K + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        // Store d as float directly
        x_df[i * (MMQ_TILE_NE_K/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

// Q8_0 vec_dot for MMQ - uses float* directly for both X and Y tile scales
// This avoids undefined behavior from casting half2* to float*
// vec_dot_q8_0_q8_1_impl takes (float d8_0, float d8_1) - both floats
static __dpct_inline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const float *__restrict__ x_df,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const float *__restrict__ y_df,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    // Both X and Y scales are stored and read as float - no casting needed
    // Use MMQ_TILE_NE_K (32) for tile indexing to match CUDA tile sizing
    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (MMQ_TILE_NE_K + 1) + k], &y_qs[j * MMQ_TILE_NE_K + k],
         x_df[i * (MMQ_TILE_NE_K/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (MMQ_TILE_NE_K/QI8_1) + k/QI8_1]);
}

// Q8_0 SoA (Structure of Arrays) tile loading
// Loads quantized data from SoA layout where all qs bytes come first, then all d values.
// d_offset is the byte offset from qs_base to the start of d values.
// Uses MMQ_TILE_NE_K stride to match AoS load_tiles_q8_0 and vec_dot patterns.
template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q8_0_soa(const int8_t *__restrict__ qs_base,
                    const size_t d_offset,
                    int *__restrict__ x_ql,
                    float *__restrict__ x_df,
                    const int &i_offset, const int &i_max,
                    const int &k, const int &blocks_per_row,
                    const int &row_offset, const int &block_offset,
                    const int &row_low,
                    const sycl::stream *debug_stream = nullptr, int *debug_counter = nullptr) {
    // Use MMQ_TILE_NE_K (32) for tile indexing to match CUDA tile sizing
    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  MMQ_TILE_NE_K);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;

    // Compute d_base inside kernel to avoid pointer capture during graph recording
    const sycl::half * d_base = (const sycl::half *)((const uint8_t *)qs_base + d_offset);

    // Load quantized values from SoA layout
    // Use MMQ_TILE_NE_K+1 stride (bank conflict avoidance) - matches AoS pattern
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        // In SoA: qs values are at qs_base + (row * blocks_per_row + block) * QK8_0 + byte_offset
        // Q8_0 uses 32 bytes per block for qs (32 int8 values)
        // row_low converts local row indices to absolute indices for SoA addressing
        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbx;
        const int8_t * qs_ptr = qs_base + global_block * QK8_0;

        // get_int_from_int8 reads 4 bytes at offset kqsx*4
        const int qs_val = get_int_from_int8(qs_ptr, kqsx);
        x_ql[i * (MMQ_TILE_NE_K + 1) + k] = qs_val;
    }

    // Load scales from SoA layout - matches AoS pattern
    const int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI8_0;  // 4 blocks per tile row
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        // In SoA: d values are at d_base + (row * blocks_per_row + block)
        // row_low converts local row indices to absolute indices for SoA addressing
        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbxd;
        const sycl::half d_val = d_base[global_block];

        // Store d as float directly - half->float conversion
        x_df[i * (MMQ_TILE_NE_K/QI8_0) + i / QI8_0 + kbxd] = d_val;
    }
}


template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q2_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    int *tile_x_sc_q2_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q2_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#define VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
static __dpct_inline__ float
vec_dot_q2_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const uint8_t *__restrict__ scales,
                           const sycl::half2 &dm2, const float &d8) {

    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const sycl::float2 dm2f =
        dm2.convert<float, sycl::rounding_mode::automatic>();

    return d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

static __dpct_inline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const int kbx = k / QI2_K;
    const int ky  = (k % QI2_K) * QR2_K;
    const float * y_df = (const float *) y_ds;

    int v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    const int kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    const int shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    for (int l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    const uint8_t * scales = ((const uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    const int index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    return vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q3_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    int *tile_x_qh_q3_K, int *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q3_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = dpct::vectorized_binary<sycl::char4>(
            sc_low | sc_high, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

#define VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
static __dpct_inline__ float
vec_dot_q3_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const int8_t *__restrict__ scales, const float &d3,
                           const float &d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

static __dpct_inline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {

    const int kbx  = k / QI3_K;
    const int ky  = (k % QI3_K) * QR3_K;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    int v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        const int kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        const int shift = 2 * ((ky % 32) / 8);
        const int vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        const int vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        const int vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_binary<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    const int index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    int *tile_x_sc_q4_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr int blocks_per_tile_x_row = QI4_K > WARP_SIZE ? 1 : WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}


#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
static __dpct_inline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const sycl::half2 *__restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        const sycl::float2 ds8f =
            ds8[i].convert<float, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


static __dpct_inline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    // Y-tile uses QR4_K * WARP_SIZE = 64 stride per column for 2-phase quants
    const int index_y = j * (QR4_K * WARP_SIZE) + QR4_K*k;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    int *tile_x_sc_q5_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    constexpr int blocks_per_tile_x_row = QI5_K > WARP_SIZE ? 1 : WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
static __dpct_inline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const sycl::half2 *__restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        const sycl::float2 ds8f =
            ds8[i].convert<float, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

static __dpct_inline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    const int index_y = j * (QR5_K * WARP_SIZE)   + QR5_K*k;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q6_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql, sycl::half2 *tile_x_dm, int *tile_x_sc) {
    (void)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q6_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        // Use MMQ_TILE_NE_K (32) for tile stride calculations
        // This matches CUDA tile sizing and quantization block dimensions (QI6_K=32)
        x_ql[i * (2 * MMQ_TILE_NE_K + 1) + kq0] =
            dpct::vectorized_binary<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * MMQ_TILE_NE_K + 1) + kq1] =
            dpct::vectorized_binary<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    // blocks_per_tile_x_row: QI6_K=32 == MMQ_TILE_NE_K=32, so this is 1
    constexpr int blocks_per_tile_x_row = QI6_K > MMQ_TILE_NE_K ? 1 : MMQ_TILE_NE_K / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        // MMQ_TILE_NE_K/QI6_K = 32/32 = 1 (was WARP_SIZE/QI6_K = 16/32 = 0!)
        x_dmf[i * (MMQ_TILE_NE_K/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        // MMQ_TILE_NE_K/8 = 32/8 = 4 (was WARP_SIZE/8 = 16/8 = 2)
        int i = (i0 + i_offset * 8 + k / (MMQ_TILE_NE_K/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (MMQ_TILE_NE_K/8)) / 4;

        x_sc[i * (MMQ_TILE_NE_K/8) + i / 8 + k % (MMQ_TILE_NE_K/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
static __dpct_inline__ float
vec_dot_q6_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const int8_t *__restrict__ sc, const float &d6,
                           const float *__restrict__ d8) {

    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x() = dpct::dp4a(v[2 * i + 0], u[2 * i + 0],
                                    sumi_d.x()); // SIMD dot product
            sumi_d.x() = dpct::dp4a(v[2 * i + 1], u[2 * i + 1],
                                    sumi_d.x()); // SIMD dot product

            sumi_d.y() = dpct::dp4a(v[2 * i + 4], u[2 * i + 4],
                                    sumi_d.y()); // SIMD dot product
            sumi_d.y() = dpct::dp4a(v[2 * i + 5], u[2 * i + 5],
                                    sumi_d.y()); // SIMD dot product
        }

        sumf_d += d8[i0 / 4] *
                  (sc[i0 / 2 + 0] * sumi_d.x() + sc[i0 / 2 + 1] * sumi_d.y());
    }

    return d6 * sumf_d;
}

static __dpct_inline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    // Use MMQ_TILE_NE_K (32) for tile indexing
    const int8_t * sc = ((const int8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k/8]);

    // x-tile stride uses MMQ_TILE_NE_K for Q6_K to match tile allocation
    const int index_x = i * (QR6_K*MMQ_TILE_NE_K + 1) + QR6_K*k;

    // Y-tile uses WARP_SIZE stride with modulo wrapping (same as Q4_0).
    // Each phase (ir=0,1) loads into the same tile positions [0, WARP_SIZE).
    // For k=0,8 (phase 0): ky % WARP_SIZE = 0,16
    // For k=16,24 (phase 1): ky % WARP_SIZE = 0,16 (phase 1 data now in tile)
    const int ky = QR6_K * k;  // = 2*k
    const int index_y = j * WARP_SIZE + ky % WARP_SIZE;
    const int index_y_ds = j * (WARP_SIZE / QI8_1) + (ky % WARP_SIZE) / QI8_1;

    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc,
        x_dmf[i * (MMQ_TILE_NE_K/QI6_K) + i/QI6_K], &y_df[index_y_ds]);
}

// Q6_K SoA tile loader
// SoA Layout for Q6_K (per tensor):
//   | all ql (nblocks * 128 bytes) | all qh (nblocks * 64 bytes) | all scales (nblocks * 16 bytes) | all d (nblocks * 2 bytes) |
// Parameters:
//   qs_base: base pointer to SoA data
//   qh_offset: offset in bytes from qs_base to qh section (= nblocks * 128)
//   scales_offset: offset in bytes from qs_base to scales section (= nblocks * 192)
//   d_offset: offset in bytes from qs_base to d section (= nblocks * 208)
template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q6_K_soa(const uint8_t *__restrict__ qs_base,
                    const size_t qh_offset,
                    const size_t scales_offset,
                    const size_t d_offset,
                    int *__restrict__ x_ql,
                    sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                    int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                    const int &k, const int &blocks_per_row,
                    const int &row_offset, const int &block_offset,
                    const int &row_low) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    // Compute bases for each section
    const uint8_t * qh_base = qs_base + qh_offset;
    const int8_t * scales_base = (const int8_t*)(qs_base + scales_offset);
    const sycl::half * d_base = (const sycl::half*)(qs_base + d_offset);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        // Global block index in SoA layout
        // row_low is added to convert local row index to absolute row index for SoA addressing
        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbx;

        // ql: each block has QK_K/2 = 128 bytes of low bits
        const uint8_t * ql_ptr = qs_base + global_block * (QK_K/2);
        // qh: each block has QK_K/4 = 64 bytes of high bits
        const uint8_t * qh_ptr = qh_base + global_block * (QK_K/4);

        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(ql_ptr, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(qh_ptr, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        // Use MMQ_TILE_NE_K (32) for tile stride calculations
        x_ql[i * (2 * MMQ_TILE_NE_K + 1) + kq0] =
            dpct::vectorized_binary<sycl::char4>(ql0 | qh0, 0x20202020, dpct::sub_sat());
        x_ql[i * (2 * MMQ_TILE_NE_K + 1) + kq1] =
            dpct::vectorized_binary<sycl::char4>(ql1 | qh1, 0x20202020, dpct::sub_sat());
    }

    // Load d values
    constexpr int blocks_per_tile_x_row = QI6_K > MMQ_TILE_NE_K ? 1 : MMQ_TILE_NE_K / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbxd;

        const float d_val = d_base[global_block];
        x_dmf[i * (MMQ_TILE_NE_K/QI6_K) + i / QI6_K + kbxd] = d_val;
    }

    // Load scales
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (MMQ_TILE_NE_K/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + (k % (MMQ_TILE_NE_K/8)) / 4;

        // scales: each block has QK_K/16 = 16 bytes
        const int8_t * sc_ptr = scales_base + global_block * (QK_K/16);

        const int sc_val = get_int_from_int8(sc_ptr, k % (QI6_K/8));
        x_sc[i * (MMQ_TILE_NE_K/8) + i / 8 + k % (MMQ_TILE_NE_K/8)] = sc_val;
    }
}

// Q6_K SoA kernel template
// Note: need_sum=false for Q6_K (matches AoS kernel at line 2747)
template <int mmq_x, int mmq_y, int nwarps, bool need_check, bool need_sum = false> static void
    mul_mat_q6_K_soa(
    const uint8_t * __restrict__ qs_base,
    const size_t qh_offset, const size_t scales_offset, const size_t d_offset,
    const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_y_unpadded, const int nrows_dst, const int row_low,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q6_K, sycl::half2 *tile_x_dm_q6_K,
    int *tile_x_sc_q6_K, int *tile_y_qs, sycl::half2 *tile_y_ds) {

    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q6_K, tile_x_dm_q6_K, tile_x_sc_q6_K);

    // Y SoA layout per column: [nrows_y_unpadded quants (int8)][ds values (half2)]
    // Explicit SoA stride: qs bytes + ds bytes (don't rely on sizeof(block_q8_1))
    const int y_col_stride = nrows_y + (nrows_y / QK8_1) * sizeof(sycl::half2);  // stride between Y columns

#if GGML_SYCL_MMQ_Q6K_DEBUG
    if (item_ct1.get_group(0) == 0 && item_ct1.get_group(1) == 0 && item_ct1.get_group(2) == 0 &&
        item_ct1.get_local_id(0) == 0 && item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(2) == 0) {
        sycl::ext::oneapi::experimental::printf("[MMQ_Q6K_SOA] nrows_x=%d ncols_x=%d ncols_y=%d nrows_dst=%d row_low=%d\n",
            nrows_x, ncols_x, ncols_y, nrows_dst, row_low);
        sycl::ext::oneapi::experimental::printf("[MMQ_Q6K_SOA] qh_offset=%zu scales_offset=%zu d_offset=%zu\n",
            qh_offset, scales_offset, d_offset);
        sycl::ext::oneapi::experimental::printf("[MMQ_Q6K_SOA] Y SoA: nrows_y_unpadded=%d y_col_stride=%d\n",
            nrows_y_unpadded, y_col_stride);
    }
#endif

    constexpr int qk = QK_K;
    constexpr int qr = QR6_K;
    constexpr int qi = QI6_K;
    constexpr int vdr = VDR_Q6_K_Q8_1_MMQ;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;

    constexpr int blocks_per_iter = (qi > WARP_SIZE) ? (MMQ_ITER_K / qk) : (WARP_SIZE / qi);
    static_assert(blocks_per_iter > 0, "blocks_per_iter must be positive");

    constexpr int phases_per_iter = (qi > WARP_SIZE) ? (qk / WARP_SIZE) : qr;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = item_ct1.get_group(2) * mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = item_ct1.get_group(1) * mmq_x;
    const int & col_y_0 = col_dst_0;

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_iter) {
        // Load X tiles from SoA layout
        // row_low converts local row indices to absolute indices for SoA addressing
        load_tiles_q6_K_soa<mmq_y, nwarps, need_check>(
            qs_base, qh_offset, scales_offset, d_offset,
            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
            item_ct1.get_local_id(1), nrows_x - row_x_0 - 1,
            item_ct1.get_local_id(2), blocks_per_row_x, row_x_0, ib0, row_low);

        // Y-tile loading using WARP_SIZE stride with modulo wrapping.
        // Y is in SoA layout per column: [nrows_y_unpadded quants][ds values]
        // Phase 0 (ir=0): loads kqs=0..31 into positions 0-31
        // Phase 1 (ir=1): loads kqs=32..63, kqs%32 wraps to 0-31, OVERWRITES phase 0
        // This is correct because vec_dot uses same modulo wrapping: ky % WARP_SIZE
#if GGML_SYCL_MMQ_Q6K_DEBUG
        // Debug: print Y source info BEFORE loading
        if (ib0 == 0 &&
            item_ct1.get_group(0) == 0 && item_ct1.get_group(1) == 0 && item_ct1.get_group(2) == 0 &&
            item_ct1.get_local_id(0) == 0 && item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(2) == 0) {
            sycl::ext::oneapi::experimental::printf("[Y_SRC_SOA] blocks_per_col_y=%d col_y_0=%d ncols_y=%d nrows_y=%d\n",
                blocks_per_col_y, col_y_0, ncols_y, nrows_y);
            // Print first few Y SoA values
            for (int col = 0; col < sycl::min(2, ncols_y); col++) {
                const int8_t * y_col_qs = (const int8_t*)vy + col * y_col_stride;
                const sycl::half2 * y_col_ds = (const sycl::half2*)((const char*)vy + col * y_col_stride + nrows_y_unpadded);
                sycl::ext::oneapi::experimental::printf("[Y_SRC_SOA] col=%d qs[0..3]=%d,%d,%d,%d ds[0]=[%.6f,%.6f] ds[1]=[%.6f,%.6f]\n",
                    col, y_col_qs[0], y_col_qs[1], y_col_qs[2], y_col_qs[3],
                    (float)y_col_ds[0][0], (float)y_col_ds[0][1],
                    (float)y_col_ds[1][0], (float)y_col_ds[1][1]);
            }
        }
#endif

#pragma unroll
        for (int ir = 0; ir < phases_per_iter; ++ir) {
            const int kqs = ir * WARP_SIZE + item_ct1.get_local_id(2);
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = dpct::min(
                    (unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1);

                // Y SoA access: quants at col_base + block*QK8_1 + elem
                const int block_idx = ib0 * (qk/QK8_1) + kbxd;
                const int8_t * y_col_qs = (const int8_t*)vy + col_y_eff * y_col_stride;
                const int8_t * y_block_qs = y_col_qs + block_idx * QK8_1;

                // Use WARP_SIZE stride and modulo to match vec_dot's tile indexing
                const int index_y = (item_ct1.get_local_id(1) + i) * WARP_SIZE +
                                    kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(
                    y_block_qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (WARP_SIZE / QI8_1)) % mmq_x;
                const int kby = item_ct1.get_local_id(2) % (WARP_SIZE / QI8_1);
                const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // Y SoA access: ds at col_base + nrows_y_unpadded + block*sizeof(half2)
                const int block_idx = ib0 * (qk/QK8_1) + ir * (WARP_SIZE / QI8_1) + kby;
                const char * y_col_base = (const char*)vy + col_y_eff * y_col_stride;
                const sycl::half2 * y_col_ds = (const sycl::half2*)(y_col_base + nrows_y_unpadded);
                const sycl::half2 dsi_val = y_col_ds[block_idx];

#if GGML_SYCL_MMQ_Q6K_DEBUG
                // Debug: trace the ds loading for first few
                if (ib0 == 0 && ir == 0 && ids0 == 0 &&
                    item_ct1.get_group(0) == 0 && item_ct1.get_group(1) == 0 && item_ct1.get_group(2) == 0 &&
                    item_ct1.get_local_id(0) == 0 && item_ct1.get_local_id(1) < 2 && item_ct1.get_local_id(2) < 4) {
                    const int tile_idx = ids * (WARP_SIZE / QI8_1) + kby;
                    sycl::ext::oneapi::experimental::printf("[DS_LOAD_SOA] warp=%d lane=%d ids=%d kby=%d col_y_eff=%d block_idx=%d tile_idx=%d src_ds=[%.6f,%.6f]\n",
                        (int)item_ct1.get_local_id(1), (int)item_ct1.get_local_id(2),
                        ids, kby, col_y_eff, block_idx, tile_idx,
                        (float)dsi_val[0], (float)dsi_val[1]);
                }
#endif

                // Use WARP_SIZE / QI8_1 = 4 stride per column for ds (matches AoS)
                sycl::half2 *dsi_dst = &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = dsi_val;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = dsi_val[0];
                }
            }

            item_ct1.barrier();

#if GGML_SYCL_MMQ_Q6K_DEBUG
            // Debug: print tile_y_ds values AFTER loading and barrier
            if (ib0 == 0 && ir == 0 &&
                item_ct1.get_group(0) == 0 && item_ct1.get_group(1) == 0 && item_ct1.get_group(2) == 0 &&
                item_ct1.get_local_id(0) == 0 && item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(2) == 0) {
                const float * y_df = (const float *) tile_y_ds;
                // Print tile_y_ds for columns 0-3 (first 16 values = 4 cols * 4 blocks)
                sycl::ext::oneapi::experimental::printf("[TILE_Y_DS] After load: col0=[%.6f,%.6f,%.6f,%.6f]\n",
                    y_df[0], y_df[1], y_df[2], y_df[3]);
                sycl::ext::oneapi::experimental::printf("[TILE_Y_DS] After load: col1=[%.6f,%.6f,%.6f,%.6f]\n",
                    y_df[4], y_df[5], y_df[6], y_df[7]);
                sycl::ext::oneapi::experimental::printf("[TILE_Y_DS] After load: col2=[%.6f,%.6f,%.6f,%.6f]\n",
                    y_df[8], y_df[9], y_df[10], y_df[11]);
                sycl::ext::oneapi::experimental::printf("[TILE_Y_DS] After load: col3=[%.6f,%.6f,%.6f,%.6f]\n",
                    y_df[12], y_df[13], y_df[14], y_df[15]);
            }
#endif

#if GGML_SYCL_MMQ_Q6K_DEBUG
            // Debug: print tile values after loading (first thread, first work-group, first iter)
            if (ib0 == 0 && ir == 0 &&
                item_ct1.get_group(0) == 0 && item_ct1.get_group(1) == 0 && item_ct1.get_group(2) == 0 &&
                item_ct1.get_local_id(0) == 0 && item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(2) == 0) {
                // X-tile values (first row)
                sycl::ext::oneapi::experimental::printf("[TILE_X] x_ql[0]=0x%08x x_ql[1]=0x%08x\n",
                    tile_x_ql[0], tile_x_ql[1]);
                float d_val = ((float*)tile_x_dm)[0];
                sycl::ext::oneapi::experimental::printf("[TILE_X] x_dm[0]=%.6f (isnan=%d isinf=%d)\n",
                    d_val, sycl::isnan(d_val) ? 1 : 0, sycl::isinf(d_val) ? 1 : 0);
                sycl::ext::oneapi::experimental::printf("[TILE_X] x_sc[0]=0x%08x\n", tile_x_sc[0]);
                // Y-tile values (first column)
                sycl::ext::oneapi::experimental::printf("[TILE_Y] y_qs[0]=0x%08x y_qs[1]=0x%08x\n",
                    tile_y_qs[0], tile_y_qs[1]);
                float y_d = ((float*)tile_y_ds)[0];
                sycl::ext::oneapi::experimental::printf("[TILE_Y] y_ds[0]=%.6f (isnan=%d isinf=%d)\n",
                    y_d, sycl::isnan(y_d) ? 1 : 0, sycl::isinf(y_d) ? 1 : 0);
            }
#endif

            // k-loop: compute dot products
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int iy = 0; iy < mmq_y / WARP_SIZE; ++iy) {
#pragma unroll
                    for (int ix = 0; ix < mmq_x / nwarps; ++ix) {
                        const float dot_result = vec_dot_q6_K_q8_1_mul_mat(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds,
                            item_ct1.get_local_id(2) + iy * WARP_SIZE,
                            item_ct1.get_local_id(1) + ix * nwarps, k);
#if GGML_SYCL_MMQ_Q6K_DEBUG
                        // Debug: print first vec_dot result and trace inner values
                        if (ib0 == 0 && ir == 0 && k == 0 && iy == 0 && ix == 0 &&
                            item_ct1.get_group(0) == 0 && item_ct1.get_group(1) == 0 && item_ct1.get_group(2) == 0 &&
                            item_ct1.get_local_id(0) == 0 && item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(2) == 0) {
                            const int i_idx = item_ct1.get_local_id(2) + iy * WARP_SIZE;
                            const int j_idx = item_ct1.get_local_id(1) + ix * nwarps;
                            const float * x_dmf = (const float *) tile_x_dm;
                            const float * y_df  = (const float *) tile_y_ds;

                            // D6 calculation
                            const int d6_idx = i_idx * (MMQ_TILE_NE_K/QI6_K) + i_idx/QI6_K;
                            const float d6 = x_dmf[d6_idx];

                            // Scales indexing
                            const int sc_base_idx = i_idx * (MMQ_TILE_NE_K/8) + i_idx/8 + k/8;
                            const int8_t * sc = ((const int8_t *) &tile_x_sc[sc_base_idx]);

                            // X and Y tile indices
                            const int index_x = i_idx * (QR6_K*MMQ_TILE_NE_K + 1) + QR6_K*k;
                            const int ky = QR6_K * k;
                            const int index_y = j_idx * WARP_SIZE + ky % WARP_SIZE;
                            const int index_y_ds = j_idx * (WARP_SIZE / QI8_1) + (ky % WARP_SIZE) / QI8_1;

                            sycl::ext::oneapi::experimental::printf("[VEC_DOT] k=%d result=%.6f (isnan=%d isinf=%d)\n",
                                k, dot_result, sycl::isnan(dot_result) ? 1 : 0, sycl::isinf(dot_result) ? 1 : 0);
                            sycl::ext::oneapi::experimental::printf("[VEC_DOT] i=%d j=%d d6_idx=%d d6=%.6f\n",
                                i_idx, j_idx, d6_idx, d6);
                            sycl::ext::oneapi::experimental::printf("[VEC_DOT] sc_base=%d sc[0..3]=%d,%d,%d,%d\n",
                                sc_base_idx, sc[0], sc[1], sc[2], sc[3]);
                            sycl::ext::oneapi::experimental::printf("[VEC_DOT] index_x=%d x_ql[0..3]=0x%08x,0x%08x,0x%08x,0x%08x\n",
                                index_x, tile_x_ql[index_x], tile_x_ql[index_x+1], tile_x_ql[index_x+2], tile_x_ql[index_x+3]);
                            sycl::ext::oneapi::experimental::printf("[VEC_DOT] index_y=%d y_qs[0..3]=0x%08x,0x%08x,0x%08x,0x%08x\n",
                                index_y, tile_y_qs[index_y], tile_y_qs[index_y+1], tile_y_qs[index_y+2], tile_y_qs[index_y+3]);
                            sycl::ext::oneapi::experimental::printf("[VEC_DOT] index_y_ds=%d y_df[0..1]=%.6f,%.6f\n",
                                index_y_ds, y_df[index_y_ds], y_df[index_y_ds+1]);
                        }
#endif
                        sum[iy][ix] += dot_result;
                    }
                }
            }

            item_ct1.barrier();
        }
    }

    // Output results
#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

#if GGML_SYCL_MMQ_Q6K_DEBUG
            // Debug: print first few output values from SoA kernel
            if (row_dst < 4 && col_dst < 4) {
                const float out_val = sum[i/WARP_SIZE][j/nwarps];
                sycl::ext::oneapi::experimental::printf("[SOA_OUT] row=%d col=%d val=%.6f (isnan=%d isinf=%d)\n",
                    row_dst, col_dst, out_val, sycl::isnan(out_val) ? 1 : 0, sycl::isinf(out_val) ? 1 : 0);
            }
#endif
            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x,
          int mmq_y, int nwarps, load_tiles_sycl_t load_tiles, int vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
static __dpct_inline__ void
mul_mat_q(const void *__restrict__ vx, const void *__restrict__ vy,
          float *__restrict__ dst, const int ncols_x, const int nrows_x,
          const int ncols_y, const int nrows_y, const int nrows_dst,
          int *tile_x_ql, sycl::half2 *tile_x_dm, int *tile_x_qh,
          int *tile_x_sc, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;

    // For quant types where qi > WARP_SIZE, use CUDA-style blocks_per_iter = MMQ_ITER_K/qk.
    // For quant types where WARP_SIZE >= qi, use WARP_SIZE / qi as per original design.
    // With WARP_SIZE=32, Q6_K (qi=32) uses the WARP_SIZE/qi path (32/32=1 block per iter).
    constexpr int blocks_per_iter = (qi > WARP_SIZE) ? (MMQ_ITER_K / qk) : (WARP_SIZE / qi);
    static_assert(blocks_per_iter > 0, "blocks_per_iter must be positive");

    // Number of phases needed per iteration
    // For K-quants where qi > WARP_SIZE, we need more phases to cover the full qk elements
    constexpr int phases_per_iter = (qi > WARP_SIZE) ? (qk / WARP_SIZE) : qr;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = item_ct1.get_group(2) * mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = item_ct1.get_group(1) * mmq_x;
    const int & col_y_0 = col_dst_0;

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_iter) {

        load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
                   nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
                   blocks_per_row_x);

        // Y-tile stride: Use WARP_SIZE for all quant types.
        // The vec_dot functions use modulo wrapping (e.g., (kyqs + l) % MMQ_TILE_NE_K)
        // to handle phase overlap, so we don't need separate phase storage.
        // Y-tile allocation is mmq_x * MMQ_TILE_NE_K = mmq_x * 32 elements.
        //
        // The vec_dot functions use MMQ_TILE_NE_K for indexing to match CUDA tile sizing
        // and ensure consistent behavior across different WARP_SIZE configurations.

#pragma unroll
        for (int ir = 0; ir < phases_per_iter; ++ir) {
            const int kqs = ir * WARP_SIZE + item_ct1.get_local_id(2);
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = dpct::min(
                    (unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                // Use WARP_SIZE stride and modulo to match vec_dot's tile indexing
                const int index_y = (item_ct1.get_local_id(1) + i) * WARP_SIZE +
                                    kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(
                    by0->qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (WARP_SIZE / QI8_1)) %
                    mmq_x;
                const int kby = item_ct1.get_local_id(2) % (WARP_SIZE / QI8_1);
                const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const sycl::half2 *dsi_src =
                    &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_SIZE / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst = &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = (*dsi_src)[0];
                }
            }

            /*
            DPCT1118:9: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE) {
                        sum[i / WARP_SIZE][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds, item_ct1.get_local_id(2) + i,
                            item_ct1.get_local_id(1) + j, k);
                    }
                }
            }

            /*
            DPCT1118:10: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 8
#else
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q4_0, float *tile_x_d_q4_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware

    const int mmq_x  =  MMQ_X_Q4_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
              load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

// SoA version of mul_mat_q4_0 kernel
// Handles Structure of Arrays weight layout for better memory coalescing
// NOTE: d_offset is passed instead of d_base to avoid pointer capture issues
// during SYCL graph recording. The derived pointer is computed inside the kernel.
// This is a self-contained implementation because load_tiles_q4_0_soa has
// a different signature than the generic load_tiles_sycl_t function pointer.
// row_low converts local row indices to absolute indices for SoA addressing (matching Q8_0 pattern)
template <int mmq_x, int mmq_y, int nwarps, bool need_check> static void
    mul_mat_q4_0_soa(
    const uint8_t * __restrict__ qs_base, const size_t d_offset,
    const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_y_unpadded, const int nrows_dst, const int row_low,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q4_0, float *tile_x_d_q4_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {

    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);

    // SoA-specific mul_mat_q implementation for both X and Y in SoA layout
    // Y SoA layout per column: [nrows_y qs bytes][ds values (half2 per block)]
    // Explicit SoA stride: qs bytes + ds bytes (don't rely on sizeof(block_q8_1))
    const int y_col_stride = nrows_y + (nrows_y / QK8_1) * sizeof(sycl::half2);  // SoA stride

    constexpr int qk = QK4_0;
    constexpr int qr = QR4_0;
    constexpr int qi = QI4_0;
    constexpr int vdr = VDR_Q4_0_Q8_1_MMQ;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;

    // blocks_per_iter: number of X blocks processed per outer loop iteration
    // For Q4_0: qi=4, so blocks_per_iter = WARP_SIZE/qi = 32/4 = 8
    constexpr int blocks_per_iter = (qi > WARP_SIZE) ? (MMQ_ITER_K / qk) : (WARP_SIZE / qi);
    static_assert(blocks_per_iter > 0, "blocks_per_iter must be positive");

    // phases_per_iter: number of Y loading phases per iteration
    // For Q4_0: qr=2, so phases_per_iter = 2 (loads 4 blocks per phase, 8 total)
    constexpr int phases_per_iter = (qi > WARP_SIZE) ? (qk / WARP_SIZE) : qr;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = item_ct1.get_group(2) * mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = item_ct1.get_group(1) * mmq_x;
    const int & col_y_0 = col_dst_0;

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_iter) {
        // SoA loader - pass d_offset instead of d_base to avoid graph capture issues
        // row_low converts local row indices to absolute indices for SoA addressing
        load_tiles_q4_0_soa<mmq_y, nwarps, need_check>(
            qs_base, d_offset, tile_x_ql, tile_x_dm,
            tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
            nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
            blocks_per_row_x, row_x_0, ib0, row_low);

        // Y-tile stride: Use MMQ_TILE_NE_K (32) to match vec_dot tile indexing.
        // vec_dot_q4_0_q8_1_mul_mat_soa reads: y_qs[j * MMQ_TILE_NE_K + ...]
        // So we must STORE with the same stride.

#pragma unroll
        for (int ir = 0; ir < phases_per_iter; ++ir) {
            const int kqs = ir * WARP_SIZE + item_ct1.get_local_id(2);
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = dpct::min(
                    (unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1);

                // Y SoA access: quants at col_base + block*QK8_1 + elem
                const int block_idx = ib0 * (qk/QK8_1) + kbxd;
                const int8_t * y_col_qs = (const int8_t*)vy + col_y_eff * y_col_stride;
                const int8_t * y_block_qs = y_col_qs + block_idx * QK8_1;

                // Use MMQ_TILE_NE_K stride to match vec_dot's tile indexing
                const int index_y = (item_ct1.get_local_id(1) + i) * MMQ_TILE_NE_K +
                                    kqs % MMQ_TILE_NE_K;
                tile_y_qs[index_y] = get_int_from_int8_aligned(
                    y_block_qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (MMQ_TILE_NE_K / QI8_1)) %
                    mmq_x;
                const int kby = item_ct1.get_local_id(2) % (MMQ_TILE_NE_K / QI8_1);
                const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // Y SoA access: ds at col_base + nrows_y_unpadded + block*sizeof(half2)
                const int block_idx = ib0 * (qk/QK8_1) + ir*(MMQ_TILE_NE_K/QI8_1) + kby;
                const char * y_col_base = (const char*)vy + col_y_eff * y_col_stride;
                const sycl::half2 * y_col_ds = (const sycl::half2*)(y_col_base + nrows_y_unpadded);

                // Q4_0 needs both d and s (half2) for vec_dot_q4_0_q8_1_impl
                // Store full half2, not just d component
                tile_y_ds[ids * (MMQ_TILE_NE_K / QI8_1) + kby] = y_col_ds[block_idx];
            }

            item_ct1.barrier();  // Full barrier to match AoS template

            // Critical: k-loop matches AoS mul_mat_q template
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int iy = 0; iy < mmq_y / WARP_SIZE; ++iy) {
#pragma unroll
                    for (int ix = 0; ix < mmq_x / nwarps; ++ix) {
                        sum[iy][ix] += vec_dot_q4_0_q8_1_mul_mat_soa(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds,
                            item_ct1.get_local_id(2) + iy * WARP_SIZE,
                            item_ct1.get_local_id(1) + ix * nwarps, k);
                    }
                }
            }

            item_ct1.barrier();  // Full barrier to match AoS template
        }
    }

// Output loop: must match AoS mul_mat_q column-major output format
#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 8
#else
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q4_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    const int nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
              load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_0_AMPERE 64
#define  MMQ_Y_Q5_0_AMPERE 128
#define NWARPS_Q5_0_AMPERE 8
#else
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

template <bool need_check> static void
    mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_0, float *tile_x_d_q5_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    const int nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
              load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_1_AMPERE 64
#define  MMQ_Y_Q5_1_AMPERE 128
#define NWARPS_Q5_1_AMPERE 8
#else
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

template <bool need_check> static void
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    const int nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
              load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q8_0_AMPERE 64
#define  MMQ_Y_Q8_0_AMPERE 128
#define NWARPS_Q8_0_AMPERE 8
#else
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

// Q8_0 uses float* directly for tile scales (not half2*) to avoid type-punning UB
template <bool need_check> static void
    mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q8_0, float *tile_x_d_q8_0,
    int *tile_y_qs, float *tile_y_df) {

    // Q8_0 uses float* directly - no half2* casting needed
    int * tile_x_ql = tile_x_qs_q8_0;
    float * tile_x_df = tile_x_d_q8_0;

    //sycl_todo: change according to hardware
    constexpr int mmq_x  =  MMQ_X_Q8_0_AMPERE;
    constexpr int mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    constexpr int nwarps = NWARPS_Q8_0_AMPERE;

    // Inline mul_mat_q logic to use float* directly
    const block_q8_0 * x = (const block_q8_0 *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    constexpr int qk = QK8_0;
    constexpr int qr = QR8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMQ;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;

    constexpr int blocks_per_iter = (qi > WARP_SIZE) ? (MMQ_ITER_K / qk) : (WARP_SIZE / qi);
    static_assert(blocks_per_iter > 0, "blocks_per_iter must be positive");

    constexpr int phases_per_iter = (qi > WARP_SIZE) ? (qk / WARP_SIZE) : qr;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = item_ct1.get_group(2) * mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = item_ct1.get_group(1) * mmq_x;
    const int & col_y_0 = col_dst_0;

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_iter) {
        // X tile loading - uses float* directly
        load_tiles_q8_0<mmq_y, nwarps, need_check>(
            x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_df,
            nullptr, nullptr, item_ct1.get_local_id(1),
            nrows_x - row_x_0 - 1, item_ct1.get_local_id(2), blocks_per_row_x);

        // Y tile loading - Q8_0 has qr=1, so single phase
        // Use MMQ_TILE_NE_K (32) for tile indexing to match CUDA tile sizing
#pragma unroll
        for (int ir = 0; ir < phases_per_iter; ++ir) {
            const int kqs = ir * MMQ_TILE_NE_K + item_ct1.get_local_id(2);
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = dpct::min(
                    (unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1);

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                tile_y_qs[(item_ct1.get_local_id(1) + i) * MMQ_TILE_NE_K + item_ct1.get_local_id(2)] =
                    get_int_from_int8_aligned(by0->qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (MMQ_TILE_NE_K / QI8_1)) %
                    mmq_x;
                const int kby = item_ct1.get_local_id(2) % (MMQ_TILE_NE_K / QI8_1);
                const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(MMQ_TILE_NE_K/QI8_1) + kby];

                // Store d as float directly - extract [0] from half2 ds
                tile_y_df[ids * (MMQ_TILE_NE_K / QI8_1) + kby] = by0->ds[0];
            }

            item_ct1.barrier();

            // K-loop for vec_dot with float* parameters
            for (int k = ir*MMQ_TILE_NE_K/qr; k < (ir+1)*MMQ_TILE_NE_K/qr; k += vdr) {
#pragma unroll
                for (int iy = 0; iy < mmq_y / WARP_SIZE; ++iy) {
#pragma unroll
                    for (int ix = 0; ix < mmq_x / nwarps; ++ix) {
                        sum[iy][ix] += vec_dot_q8_0_q8_1_mul_mat(
                            tile_x_ql, tile_x_df, nullptr, nullptr,
                            tile_y_qs, tile_y_df,
                            item_ct1.get_local_id(2) + iy * WARP_SIZE,
                            item_ct1.get_local_id(1) + ix * nwarps, k);
                    }
                }
            }

            item_ct1.barrier();
        }
    }

    // Output loop
#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE][j / nwarps];
        }
    }
}

// Q8_0 AoS version with debug output - mirrors SoA kernel structure for comparison
// Uses float* directly for both X and Y tile scales (no half2* casting)
template <int mmq_x, int mmq_y, int nwarps, bool need_check> static void
    mul_mat_q8_0_aos(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q8_0, float *tile_x_d_q8_0,
    int *tile_y_qs, float *tile_y_df,
    const sycl::stream *debug_stream = nullptr, int *debug_counter = nullptr) {

    // Q8_0 uses float* directly - no need for allocate_tiles_q8_0 casting hack
    int * tile_x_ql = tile_x_qs_q8_0;
    float * tile_x_df = tile_x_d_q8_0;

    // Inline the mul_mat_q logic so we can pass debug parameters to load_tiles_q8_0
    const block_q8_0 * x = (const block_q8_0 *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    constexpr int qk = QK8_0;
    constexpr int qr = QR8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMQ;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;

    constexpr int blocks_per_iter = (qi > WARP_SIZE) ? (MMQ_ITER_K / qk) : (WARP_SIZE / qi);
    static_assert(blocks_per_iter > 0, "blocks_per_iter must be positive");

    constexpr int phases_per_iter = (qi > WARP_SIZE) ? (qk / WARP_SIZE) : qr;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = item_ct1.get_group(2) * mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = item_ct1.get_group(1) * mmq_x;
    const int & col_y_0 = col_dst_0;

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_iter) {
        // Call load_tiles_q8_0_debug with float* for x_df
        // Note: AoS uses pre-offset pointer, so we pass (x + row_x_0 * blocks_per_row_x + ib0)
        load_tiles_q8_0_debug<mmq_y, nwarps, need_check>(
            x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_df,
            nullptr, nullptr, item_ct1.get_local_id(1),
            nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
            blocks_per_row_x, debug_stream, debug_counter, row_x_0, ib0);

        // Y tile loading - Q8_0 has qr=1, so single phase
        // Use MMQ_TILE_NE_K (32) for tile indexing to match CUDA tile sizing
#pragma unroll
        for (int ir = 0; ir < phases_per_iter; ++ir) {
            const int kqs = ir * MMQ_TILE_NE_K + item_ct1.get_local_id(2);
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = dpct::min(
                    (unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1);

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                tile_y_qs[(item_ct1.get_local_id(1) + i) * MMQ_TILE_NE_K + item_ct1.get_local_id(2)] =
                    get_int_from_int8_aligned(by0->qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (MMQ_TILE_NE_K / QI8_1)) %
                    mmq_x;
                const int kby = item_ct1.get_local_id(2) % (MMQ_TILE_NE_K / QI8_1);
                const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(MMQ_TILE_NE_K/QI8_1) + kby];

                // Store d as float - extract [0] from half2 ds
                tile_y_df[ids * (MMQ_TILE_NE_K / QI8_1) + kby] = by0->ds[0];
            }

            item_ct1.barrier();

            // K-loop for vec_dot
            for (int k = ir*MMQ_TILE_NE_K/qr; k < (ir+1)*MMQ_TILE_NE_K/qr; k += vdr) {
#pragma unroll
                for (int iy = 0; iy < mmq_y / WARP_SIZE; ++iy) {
#pragma unroll
                    for (int ix = 0; ix < mmq_x / nwarps; ++ix) {
                        sum[iy][ix] += vec_dot_q8_0_q8_1_mul_mat(
                            tile_x_ql, tile_x_df, nullptr, nullptr,
                            tile_y_qs, tile_y_df,
                            item_ct1.get_local_id(2) + iy * WARP_SIZE,
                            item_ct1.get_local_id(1) + ix * nwarps, k);
                    }
                }
            }

            item_ct1.barrier();
        }
    }

    // Output loop
#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE][j / nwarps];
        }
    }
}


// Q8_0 SoA version of mul_mat kernel
// Handles Structure of Arrays weight layout for better memory coalescing
// Uses float* directly for both X and Y tile scales (no half2* casting)
template <int mmq_x, int mmq_y, int nwarps, bool need_check> static void
    mul_mat_q8_0_soa(
    const int8_t * __restrict__ qs_base, const size_t d_offset,
    const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_y_unpadded, const int nrows_dst, const int row_low,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q8_0, float *tile_x_d_q8_0,
    int *tile_y_qs, float *tile_y_df,
    const sycl::stream *debug_stream = nullptr, int *debug_counter = nullptr) {

    // Q8_0 uses float* directly - matches AoS kernel pattern
    int * tile_x_ql = tile_x_qs_q8_0;
    float * tile_x_df = tile_x_d_q8_0;

    // SoA-specific mul_mat_q implementation for both X and Y in SoA layout
    // Y SoA layout per column: [nrows_y qs bytes][ds values (half2 per block)]
    // Explicit SoA stride: qs bytes + ds bytes (don't rely on sizeof(block_q8_1))
    const int y_col_stride = nrows_y + (nrows_y / QK8_1) * sizeof(sycl::half2);  // SoA stride

    constexpr int qk = QK8_0;
    constexpr int qr = QR8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMQ;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;

    constexpr int blocks_per_iter = (qi > WARP_SIZE) ? (MMQ_ITER_K / qk) : (WARP_SIZE / qi);
    static_assert(blocks_per_iter > 0, "blocks_per_iter must be positive");

    constexpr int phases_per_iter = (qi > WARP_SIZE) ? (qk / WARP_SIZE) : qr;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = item_ct1.get_group(2) * mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = item_ct1.get_group(1) * mmq_x;
    const int & col_y_0 = col_dst_0;

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_iter) {
        // SoA loader - pass d_offset instead of d_base to avoid graph capture issues
        // row_low converts local row indices to absolute indices for SoA addressing
        load_tiles_q8_0_soa<mmq_y, nwarps, need_check>(
            qs_base, d_offset, tile_x_ql, tile_x_df,
            item_ct1.get_local_id(1),
            nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
            blocks_per_row_x, row_x_0, ib0, row_low, debug_stream, debug_counter);

        // Y tile loading - Use MMQ_TILE_NE_K stride to match vec_dot_q8_0_q8_1_mul_mat
        // vec_dot reads: y_qs[j * MMQ_TILE_NE_K + k], y_df[j * (MMQ_TILE_NE_K/QI8_1) + k/QI8_1]
#pragma unroll
        for (int ir = 0; ir < phases_per_iter; ++ir) {
            const int kqs = ir * MMQ_TILE_NE_K + item_ct1.get_local_id(2);
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = dpct::min(
                    (unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1);

                // Y SoA access: quants at col_base + block*QK8_1 + elem
                const int block_idx = ib0 * (qk/QK8_1) + kbxd;
                const int8_t * y_col_qs = (const int8_t*)vy + col_y_eff * y_col_stride;
                const int8_t * y_block_qs = y_col_qs + block_idx * QK8_1;

                // Use MMQ_TILE_NE_K stride to match vec_dot's tile indexing
                const int index_y = (item_ct1.get_local_id(1) + i) * MMQ_TILE_NE_K +
                                    kqs % MMQ_TILE_NE_K;
                tile_y_qs[index_y] = get_int_from_int8_aligned(
                    y_block_qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (MMQ_TILE_NE_K / QI8_1)) %
                    mmq_x;
                const int kby = item_ct1.get_local_id(2) % (MMQ_TILE_NE_K / QI8_1);
                const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // Y SoA access: ds at col_base + nrows_y_unpadded + block*sizeof(half2)
                const int block_idx = ib0 * (qk/QK8_1) + ir*(MMQ_TILE_NE_K/QI8_1) + kby;
                const char * y_col_base = (const char*)vy + col_y_eff * y_col_stride;
                const sycl::half2 * y_col_ds = (const sycl::half2*)(y_col_base + nrows_y_unpadded);

                // Q8_0: Store d as float - extract [0] from half2 ds (matches AoS pattern)
                tile_y_df[ids * (MMQ_TILE_NE_K / QI8_1) + kby] = y_col_ds[block_idx][0];
            }

            item_ct1.barrier();

            // K-loop for vec_dot
            for (int k = ir*MMQ_TILE_NE_K/qr; k < (ir+1)*MMQ_TILE_NE_K/qr; k += vdr) {
#pragma unroll
                for (int iy = 0; iy < mmq_y / WARP_SIZE; ++iy) {
#pragma unroll
                    for (int ix = 0; ix < mmq_x / nwarps; ++ix) {
                        sum[iy][ix] += vec_dot_q8_0_q8_1_mul_mat(
                            tile_x_ql, tile_x_df, nullptr, nullptr,
                            tile_y_qs, tile_y_df,
                            item_ct1.get_local_id(2) + iy * WARP_SIZE,
                            item_ct1.get_local_id(1) + ix * nwarps, k);
                    }
                }
            }

            item_ct1.barrier();
        }
    }

    // Output loop
#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 8
#else
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

template <bool need_check> static void
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, int *tile_x_sc_q2_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q2_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    const int nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
              load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q3_K_AMPERE 64
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 8
#else
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

template <bool need_check> static void
mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, int *tile_x_qh_q3_K, int *tile_x_sc_q3_K,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q3_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    const int nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
              load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 8
#else
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, int *tile_x_sc_q4_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q4_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    const int nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
              load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 8
#else
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

template <bool need_check> static void
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, int *tile_x_sc_q5_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    const int nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
              load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 128
#define NWARPS_Q6_K_AMPERE 8
#else
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8

template <bool need_check> static void
    mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql, sycl::half2 *tile_x_dm,
    int *tile_x_sc, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q6_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    const int nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
              load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

static void ggml_mul_mat_q4_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0),
                    cgh);
                // Y-tile uses MMQ_TILE_NE_K=32 elements per column to match CUDA tile sizing
                // and quantization block dimensions (QK8_1=32)
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * MMQ_TILE_NE_K), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * MMQ_TILE_NE_K / QI8_1), cgh);

                cgh.parallel_for<mmq_q4_0_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q4_0_acc_ct1),
                            get_pointer(tile_x_d_q4_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0),
                    cgh);
                // Y-tile uses MMQ_TILE_NE_K=32 elements per column to match CUDA tile sizing
                // and quantization block dimensions (QK8_1=32)
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * MMQ_TILE_NE_K), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * MMQ_TILE_NE_K / QI8_1), cgh);

                cgh.parallel_for<mmq_q4_0_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q4_0_acc_ct1),
                            get_pointer(tile_x_d_q4_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// SoA version of Q4_0 MMQ dispatch function
// vx points to SoA layout: all qs first, then all d values
// d_offset = nrows_x * ncols_x / 2 (size of qs section in bytes)
// nrows_y = padded Y dimension (for stride), nrows_y_unpadded = actual Y dimension (for ds offset)
static void ggml_mul_mat_q4_0_q8_1_sycl_soa(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_y_unpadded,
                                        const int nrows_dst,
                                        const size_t d_offset,
                                        const int row_low,
                                        dpct::queue_ptr stream) try {
    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    // Get base pointer - don't pre-compute d_base to avoid graph capture issues
    // The kernel will compute d_base from qs_base + d_offset internally
    const uint8_t * qs_base = (const uint8_t *) vx;

    // Dispatch to architecture-specific kernel instantiation
    // Template parameters must be compile-time constants, so we use separate branches
#define LAUNCH_Q4_0_SOA_KERNEL(MMQ_X, MMQ_Y, NWARPS)                                          \
    do {                                                                                      \
        constexpr int mmq_x = MMQ_X;                                                          \
        constexpr int mmq_y = MMQ_Y;                                                          \
        constexpr int nwarps = NWARPS;                                                        \
        const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;                                \
        const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;                                \
        const sycl::range<3> block_nums(1, block_num_y, block_num_x);                         \
        const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);                                \
        const bool need_check = (nrows_x % mmq_y != 0);                                       \
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});             \
        if (!need_check) {                                                                    \
            stream->submit([&](sycl::handler &cgh) {                                          \
                sycl::local_accessor<int, 1> tile_x_qs(sycl::range<1>(mmq_y * MMQ_TILE_NE_K + mmq_y), cgh); \
                sycl::local_accessor<float, 1> tile_x_d(sycl::range<1>(mmq_y * (MMQ_TILE_NE_K/QI4_0) + mmq_y / QI4_0), cgh); \
                sycl::local_accessor<int, 1> tile_y_qs(sycl::range<1>(mmq_x * MMQ_TILE_NE_K), cgh); \
                sycl::local_accessor<sycl::half2, 1> tile_y_ds(sycl::range<1>(mmq_x * MMQ_TILE_NE_K / QI8_1), cgh); \
                cgh.parallel_for<mmq_q4_0_soa_kernel<mmq_x, mmq_y, nwarps, false>>(           \
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),                   \
                    [=](sycl::nd_item<3> item_ct1) {                                          \
                        mul_mat_q4_0_soa<mmq_x, mmq_y, nwarps, false>(                        \
                            qs_base, d_offset, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,   \
                            nrows_y_unpadded, nrows_dst, row_low, item_ct1,                   \
                            get_pointer(tile_x_qs), get_pointer(tile_x_d),                    \
                            get_pointer(tile_y_qs), get_pointer(tile_y_ds));                  \
                    });                                                                       \
            });                                                                               \
        } else {                                                                              \
            stream->submit([&](sycl::handler &cgh) {                                          \
                sycl::local_accessor<int, 1> tile_x_qs(sycl::range<1>(mmq_y * MMQ_TILE_NE_K + mmq_y), cgh); \
                sycl::local_accessor<float, 1> tile_x_d(sycl::range<1>(mmq_y * (MMQ_TILE_NE_K/QI4_0) + mmq_y / QI4_0), cgh); \
                sycl::local_accessor<int, 1> tile_y_qs(sycl::range<1>(mmq_x * MMQ_TILE_NE_K), cgh); \
                sycl::local_accessor<sycl::half2, 1> tile_y_ds(sycl::range<1>(mmq_x * MMQ_TILE_NE_K / QI8_1), cgh); \
                cgh.parallel_for<mmq_q4_0_soa_kernel<mmq_x, mmq_y, nwarps, true>>(            \
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),                   \
                    [=](sycl::nd_item<3> item_ct1) {                                          \
                        mul_mat_q4_0_soa<mmq_x, mmq_y, nwarps, true>(                         \
                            qs_base, d_offset, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,   \
                            nrows_y_unpadded, nrows_dst, row_low, item_ct1,                   \
                            get_pointer(tile_x_qs), get_pointer(tile_x_d),                    \
                            get_pointer(tile_y_qs), get_pointer(tile_y_ds));                  \
                    });                                                                       \
            });                                                                               \
        }                                                                                     \
    } while (0)

    // Intel SYCL backend: Use single optimized configuration for Intel Arc GPUs
    // The AMPERE/RDNA naming is legacy from CUDA port - these are just tile sizes
    // Must match AoS nwarps=8 for correct tile loading iteration counts
    (void)compute_capability;  // Unused for now - single config for all Intel GPUs
    LAUNCH_Q4_0_SOA_KERNEL(64, 128, 8);

#undef LAUNCH_Q4_0_SOA_KERNEL
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + +mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_1) + mmq_y / QI4_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q4_1_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q4_1_acc_ct1),
                            get_pointer(tile_x_dm_q4_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + +mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_1) + mmq_y / QI4_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q4_1_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q4_1_acc_ct1),
                            get_pointer(tile_x_dm_q4_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q5_0_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_0_acc_ct1),
                            get_pointer(tile_x_d_q5_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:25: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q5_0_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_0_acc_ct1),
                            get_pointer(tile_x_d_q5_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q5_1_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_1_acc_ct1),
                            get_pointer(tile_x_dm_q5_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:27: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q5_1_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_1_acc_ct1),
                            get_pointer(tile_x_dm_q5_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                // Q8_0 uses float* directly for tile_y_df (not half2*)
                sycl::local_accessor<float, 1> tile_y_df_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q8_0_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q8_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q8_0_acc_ct1),
                            get_pointer(tile_x_d_q8_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_df_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                // Q8_0 uses float* directly for tile_y_df (not half2*)
                sycl::local_accessor<float, 1> tile_y_df_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q8_0_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q8_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q8_0_acc_ct1),
                            get_pointer(tile_x_d_q8_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_df_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// SoA version of Q8_0 MMQ dispatch function
// vx points to SoA layout: all qs first, then all d values
// d_offset = nrows_x * ncols_x (size of qs section in bytes for Q8_0)
// nrows_y = padded Y dimension (for stride), nrows_y_unpadded = actual Y dimension (for ds offset)
static void ggml_mul_mat_q8_0_q8_1_sycl_soa(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_y_unpadded,
                                        const int nrows_dst,
                                        const size_t d_offset,
                                        const int row_low,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    // Get base pointer - don't pre-compute d_base to avoid graph capture issues
    // The kernel will compute d_base from qs_base + d_offset internally
    const int8_t * qs_base = (const int8_t *) vx;

    // Dispatch to architecture-specific kernel instantiation
    // Template parameters must be compile-time constants, so we use separate branches

#define LAUNCH_Q8_0_SOA_KERNEL(MMQ_X, MMQ_Y, NWARPS)                                          \
    do {                                                                                      \
        constexpr int mmq_x = MMQ_X;                                                          \
        constexpr int mmq_y = MMQ_Y;                                                          \
        constexpr int nwarps = NWARPS;                                                        \
        const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;                                \
        const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;                                \
        const sycl::range<3> block_nums(1, block_num_y, block_num_x);                         \
        const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);                                \
        const bool need_check = (nrows_x % mmq_y != 0);                                       \
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});             \
        if (!need_check) {                                                                    \
            stream->submit([&](sycl::handler &cgh) {                                          \
                sycl::local_accessor<int, 1> tile_x_qs(sycl::range<1>(mmq_y * MMQ_TILE_NE_K + mmq_y), cgh); \
                sycl::local_accessor<float, 1> tile_x_d(sycl::range<1>(mmq_y * (MMQ_TILE_NE_K/QI8_0) + mmq_y / QI8_0), cgh); \
                sycl::local_accessor<int, 1> tile_y_qs(sycl::range<1>(mmq_x * MMQ_TILE_NE_K), cgh); \
                sycl::local_accessor<float, 1> tile_y_df(sycl::range<1>(mmq_x * MMQ_TILE_NE_K / QI8_1), cgh); \
                cgh.parallel_for<mmq_q8_0_soa_kernel<mmq_x, mmq_y, nwarps, false>>(           \
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),                   \
                    [=](sycl::nd_item<3> item_ct1) {                                          \
                        mul_mat_q8_0_soa<mmq_x, mmq_y, nwarps, false>(                        \
                            qs_base, d_offset, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,   \
                            nrows_y_unpadded, nrows_dst, row_low, item_ct1,                   \
                            get_pointer(tile_x_qs), get_pointer(tile_x_d),                    \
                            get_pointer(tile_y_qs), get_pointer(tile_y_df),                   \
                            nullptr, nullptr);                                                \
                    });                                                                       \
            });                                                                               \
        } else {                                                                              \
            stream->submit([&](sycl::handler &cgh) {                                          \
                sycl::local_accessor<int, 1> tile_x_qs(sycl::range<1>(mmq_y * MMQ_TILE_NE_K + mmq_y), cgh); \
                sycl::local_accessor<float, 1> tile_x_d(sycl::range<1>(mmq_y * (MMQ_TILE_NE_K/QI8_0) + mmq_y / QI8_0), cgh); \
                sycl::local_accessor<int, 1> tile_y_qs(sycl::range<1>(mmq_x * MMQ_TILE_NE_K), cgh); \
                sycl::local_accessor<float, 1> tile_y_df(sycl::range<1>(mmq_x * MMQ_TILE_NE_K / QI8_1), cgh); \
                cgh.parallel_for<mmq_q8_0_soa_kernel<mmq_x, mmq_y, nwarps, true>>(            \
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),                   \
                    [=](sycl::nd_item<3> item_ct1) {                                          \
                        mul_mat_q8_0_soa<mmq_x, mmq_y, nwarps, true>(                         \
                            qs_base, d_offset, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,   \
                            nrows_y_unpadded, nrows_dst, row_low, item_ct1,                   \
                            get_pointer(tile_x_qs), get_pointer(tile_x_d),                    \
                            get_pointer(tile_y_qs), get_pointer(tile_y_df),                   \
                            nullptr, nullptr);                                                \
                    });                                                                       \
            });                                                                               \
        }                                                                                     \
    } while (0)

    // Use same tile config as Q4_0 SoA - works well on Intel Arc
    // Must match AoS nwarps=8 for correct tile loading iteration counts
    LAUNCH_Q8_0_SOA_KERNEL(64, 128, 8);

#undef LAUNCH_Q8_0_SOA_KERNEL
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI2_K) + mmq_y / QI2_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q2_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q2_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q2_K_acc_ct1),
                            get_pointer(tile_x_dm_q2_K_acc_ct1),
                            get_pointer(tile_x_sc_q2_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:31: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI2_K) + mmq_y / QI2_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q2_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q2_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q2_K_acc_ct1),
                            get_pointer(tile_x_dm_q2_K_acc_ct1),
                            get_pointer(tile_x_sc_q2_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

#if QK_K == 256

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI3_K) + mmq_y / QI3_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_qh_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 2) + mmq_y / 2), cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q3_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q3_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q3_K_acc_ct1),
                            get_pointer(tile_x_dm_q3_K_acc_ct1),
                            get_pointer(tile_x_qh_q3_K_acc_ct1),
                            get_pointer(tile_x_sc_q3_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:33: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI3_K) + mmq_y / QI3_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_qh_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 2) + mmq_y / 2), cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q3_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q3_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q3_K_acc_ct1),
                            get_pointer(tile_x_dm_q3_K_acc_ct1),
                            get_pointer(tile_x_qh_q3_K_acc_ct1),
                            get_pointer(tile_x_sc_q3_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * QR4_K * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * QR4_K * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q4_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q4_K_acc_ct1),
                            get_pointer(tile_x_dm_q4_K_acc_ct1),
                            get_pointer(tile_x_sc_q4_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:35: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * QR4_K * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * QR4_K * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q4_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q4_K_acc_ct1),
                            get_pointer(tile_x_dm_q4_K_acc_ct1),
                            get_pointer(tile_x_sc_q4_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * QR5_K * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * QR5_K * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q5_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_K_acc_ct1),
                            get_pointer(tile_x_dm_q5_K_acc_ct1),
                            get_pointer(tile_x_sc_q5_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:37: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * QR5_K * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * QR5_K * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q5_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_K_acc_ct1),
                            get_pointer(tile_x_dm_q5_K_acc_ct1),
                            get_pointer(tile_x_sc_q5_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                // Use MMQ_TILE_NE_K (32) for x-tile allocations to match CUDA tile sizing
                // and quantization block dimensions (QI6_K=32)
                sycl::local_accessor<int, 1> tile_x_ql_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * MMQ_TILE_NE_K) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                    sycl::range<1>(mmq_y * (MMQ_TILE_NE_K / QI6_K) + mmq_y / QI6_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_acc_ct1(
                    sycl::range<1>(mmq_y * (MMQ_TILE_NE_K / 8) + mmq_y / 8), cgh);
                // Y-tile uses QR6_K * WARP_SIZE = 64 stride per column for 2-phase quants.
                // Each phase writes WARP_SIZE elements, total 64 per column.
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * QR6_K * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * QR6_K * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q6_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q6_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_acc_ct1),
                            get_pointer(tile_x_dm_acc_ct1),
                            get_pointer(tile_x_sc_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:39: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                // Use MMQ_TILE_NE_K (32) for x-tile allocations to match CUDA tile sizing
                // and quantization block dimensions (QI6_K=32)
                sycl::local_accessor<int, 1> tile_x_ql_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * MMQ_TILE_NE_K) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                    sycl::range<1>(mmq_y * (MMQ_TILE_NE_K / QI6_K) + mmq_y / QI6_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_acc_ct1(
                    sycl::range<1>(mmq_y * (MMQ_TILE_NE_K / 8) + mmq_y / 8), cgh);
                // Y-tile uses QR6_K * WARP_SIZE = 64 stride per column for 2-phase quants.
                // Each phase writes WARP_SIZE elements, total 64 per column.
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * QR6_K * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * QR6_K * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for<mmq_q6_K_kernel<need_check>>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q6_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_acc_ct1),
                            get_pointer(tile_x_dm_acc_ct1),
                            get_pointer(tile_x_sc_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Q6_K SoA dispatch function - handles weight tensors reordered to Structure-of-Arrays layout
// row_low: absolute row offset for split tensor support (added to local row indices)
static void ggml_mul_mat_q6_K_q8_1_sycl_soa(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_y_unpadded,
                                        const int nrows_dst,
                                        const size_t qh_offset,
                                        const size_t scales_offset,
                                        const size_t d_offset,
                                        const int row_low,
                                        dpct::queue_ptr stream) try {
    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    // Get base pointer - all offsets computed from ql base
    const uint8_t * qs_base = (const uint8_t *) vx;

    // Dispatch macro for Q6_K SoA kernel with compile-time template parameters
#define LAUNCH_Q6_K_SOA_KERNEL(MMQ_X, MMQ_Y, NWARPS)                                           \
    do {                                                                                       \
        constexpr int mmq_x = MMQ_X;                                                           \
        constexpr int mmq_y = MMQ_Y;                                                           \
        constexpr int nwarps = NWARPS;                                                         \
        const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;                                 \
        const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;                                 \
        const sycl::range<3> block_nums(1, block_num_y, block_num_x);                          \
        const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);                                 \
        const bool need_check = (nrows_x % mmq_y != 0);                                        \
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});              \
        if (!need_check) {                                                                     \
            stream->submit([&](sycl::handler &cgh) {                                           \
                /* X-tile: ql+qh combined, dm (half2), scales - using MMQ_TILE_NE_K */         \
                sycl::local_accessor<int, 1> tile_x_qs(                                        \
                    sycl::range<1>(mmq_y * (2 * MMQ_TILE_NE_K) + mmq_y), cgh);                  \
                sycl::local_accessor<sycl::half2, 1> tile_x_dm(                                \
                    sycl::range<1>(mmq_y * (MMQ_TILE_NE_K / QI6_K) + mmq_y / QI6_K), cgh);      \
                sycl::local_accessor<int, 1> tile_x_sc(                                        \
                    sycl::range<1>(mmq_y * (MMQ_TILE_NE_K / 8) + mmq_y / 8), cgh);              \
                /* Y-tile: AoS block_q8_1 layout, same allocation as AoS kernel.               */ \
                /* Uses WARP_SIZE stride with modulo wrapping - each phase overwrites same  */ \
                /* 32 positions. vec_dot uses ky % WARP_SIZE to correctly access phase data.*/ \
                sycl::local_accessor<int, 1> tile_y_qs(                                        \
                    sycl::range<1>(mmq_x * QR6_K * WARP_SIZE), cgh);                           \
                sycl::local_accessor<sycl::half2, 1> tile_y_ds(                                \
                    sycl::range<1>(mmq_x * QR6_K * WARP_SIZE / QI8_1), cgh);                        \
                cgh.parallel_for<mmq_q6_K_soa_kernel<mmq_x, mmq_y, nwarps, false>>(            \
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),                    \
                    [=](sycl::nd_item<3> item_ct1) {                                           \
                        mul_mat_q6_K_soa<mmq_x, mmq_y, nwarps, false>(                         \
                            qs_base, qh_offset, scales_offset, d_offset,                       \
                            vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,                       \
                            nrows_y_unpadded, nrows_dst, row_low, item_ct1,                    \
                            get_pointer(tile_x_qs), get_pointer(tile_x_dm),                    \
                            get_pointer(tile_x_sc),                                            \
                            get_pointer(tile_y_qs), get_pointer(tile_y_ds));                   \
                    });                                                                        \
            });                                                                                \
        } else {                                                                               \
            stream->submit([&](sycl::handler &cgh) {                                           \
                sycl::local_accessor<int, 1> tile_x_qs(                                        \
                    sycl::range<1>(mmq_y * (2 * MMQ_TILE_NE_K) + mmq_y), cgh);                  \
                sycl::local_accessor<sycl::half2, 1> tile_x_dm(                                \
                    sycl::range<1>(mmq_y * (MMQ_TILE_NE_K / QI6_K) + mmq_y / QI6_K), cgh);      \
                sycl::local_accessor<int, 1> tile_x_sc(                                        \
                    sycl::range<1>(mmq_y * (MMQ_TILE_NE_K / 8) + mmq_y / 8), cgh);              \
                /* Y-tile: AoS block_q8_1 layout, same allocation as AoS kernel.               */ \
                /* Uses WARP_SIZE stride with modulo wrapping - each phase overwrites same  */ \
                /* 32 positions. vec_dot uses ky % WARP_SIZE to correctly access phase data.*/ \
                sycl::local_accessor<int, 1> tile_y_qs(                                        \
                    sycl::range<1>(mmq_x * QR6_K * WARP_SIZE), cgh);                           \
                sycl::local_accessor<sycl::half2, 1> tile_y_ds(                                \
                    sycl::range<1>(mmq_x * QR6_K * WARP_SIZE / QI8_1), cgh);                        \
                cgh.parallel_for<mmq_q6_K_soa_kernel<mmq_x, mmq_y, nwarps, true>>(             \
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),                    \
                    [=](sycl::nd_item<3> item_ct1) {                                           \
                        mul_mat_q6_K_soa<mmq_x, mmq_y, nwarps, true>(                          \
                            qs_base, qh_offset, scales_offset, d_offset,                       \
                            vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,                       \
                            nrows_y_unpadded, nrows_dst, row_low, item_ct1,                    \
                            get_pointer(tile_x_qs), get_pointer(tile_x_dm),                    \
                            get_pointer(tile_x_sc),                                            \
                            get_pointer(tile_y_qs), get_pointer(tile_y_ds));                   \
                    });                                                                        \
            });                                                                                \
        }                                                                                      \
    } while (0)

    // Intel SYCL backend: Use optimized configuration for Intel Arc GPUs
    (void)compute_capability;  // Single config for all Intel GPUs
    // Must match AoS nwarps=8 for correct tile loading iteration counts
    LAUNCH_Q6_K_SOA_KERNEL(64, 128, 8);

#undef LAUNCH_Q6_K_SOA_KERNEL
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Helper to traverse view_src chain and get the underlying storage tensor
static const ggml_tensor * get_storage_tensor(const ggml_tensor * t) {
    const ggml_tensor * current = t;
    while (current->view_src != nullptr) {
        current = current->view_src;
    }
    return current;
}

void ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) try {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    const int64_t nrows_dst = device_id == ctx.device ? ne0 : row_diff;

    // ESIMD path for Q4_0 - reduces L3 cache misses via unified block loading
    // V1 uses single work-item per output, V2 is disabled (needs WARP_SIZE=32 redesign)
    if (src0->type == GGML_TYPE_Q4_0 && mmq_esimd_enabled() && mmq_esimd_available()) {

        // Check for kernel version via GGML_SYCL_MMQ_ESIMD=1,2,3
        const int esimd_ver = mmq_esimd_version();

        if (esimd_ver == 3) {
            // V3 kernel - ESIMD-native with better memory access patterns
            bool esimd_launched = launch_mmq_q4_0_esimd_v3(
                reinterpret_cast<const block_q4_0*>(src0_dd_i),
                reinterpret_cast<const block_q8_1*>(src1_ddq_i),
                dst_dd_i,
                row_diff,      // nrows
                src1_ncols,    // ncols
                ne00,          // k (inner dimension)
                nrows_dst,     // output stride
                *stream);
            if (esimd_launched) {
                return;
            }
            // V3 rejected - fall through to standard MMQ
        } else if (esimd_ver == 2) {
            // V2 kernel - tiled with SLM caching (redesigned for WARP_SIZE=32)
            bool esimd_launched = launch_mmq_q4_0_esimd_v2(
                reinterpret_cast<const block_q4_0*>(src0_dd_i),
                reinterpret_cast<const block_q8_1*>(src1_ddq_i),
                dst_dd_i,
                row_diff,      // nrows
                src1_ncols,    // ncols
                ne00,          // k (inner dimension)
                nrows_dst,     // output stride
                *stream);
            if (esimd_launched) {
                return;
            }
            // V2 rejected - fall through to standard MMQ
        } else {
            // V1 kernel (single work-item per output) - only when ESIMD=1
            bool esimd_launched = launch_mmq_q4_0_esimd(
                reinterpret_cast<const block_q4_0*>(src0_dd_i),
                reinterpret_cast<const block_q8_1*>(src1_ddq_i),
                dst_dd_i,
                row_diff,      // nrows
                src1_ncols,    // ncols
                ne00,          // k (inner dimension)
                nrows_dst,     // output stride
                *stream);
            if (esimd_launched) {
                return;
            }
        }
        // Fall through to standard MMQ
    }

    // Check for SoA reordering (applies to supported quantization types)
    // NOTE: SoA layout only works when processing the full tensor because the src0_dd_i
    // pointer calculation assumes AoS layout. For row slices (row_low > 0), the pointer
    // arithmetic is incompatible with SoA layout.
    auto * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    const int64_t nrows_full = src0->ne[1];
    
    // We can support views if we correctly calculate the base pointer and offsets
    const bool use_soa = src0_extra && src0_extra->optimized_feature.is_soa();

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            if (use_soa) {
                // SoA layout: all qs values first, then all d (scale) values
                // Pattern matches Q6_K - pass storage base, d_offset, and global_row_low
                const ggml_tensor * storage = get_storage_tensor(src0);
                const int64_t storage_ne01 = storage->ne[1];
                const int64_t storage_ne00 = storage->ne[0];

                // Q4_0: explicit nblocks pattern (matches Q6_K)
                const size_t nblocks = static_cast<size_t>(storage_ne01) * static_cast<size_t>(storage_ne00 / QK4_0);
                const size_t total_qs_bytes = nblocks * (QK4_0 / 2);  // 16 bytes per block
                const size_t d_offset = total_qs_bytes;

                // Get storage base pointer
                const void * storage_base = ggml_sycl_get_data_ptr(storage, device_id);

                // Calculate global row_low (matches Q6_K pattern)
                int64_t view_row_offset = 0;
                if (src0->view_src != nullptr) {
                    view_row_offset = src0->view_offs / src0->nb[1];
                }
                const int global_row_low = static_cast<int>(row_low + view_row_offset);

                GGML_SYCL_KTRACE("mmq_q4_0_soa", " ne00=%lld row_low=%lld row_diff=%lld ncols=%lld d_offset=%zu global_row=%d",
                    (long long)ne00, (long long)row_low, (long long)row_diff, (long long)src1_ncols, d_offset, global_row_low);

                ggml_mul_mat_q4_0_q8_1_sycl_soa(storage_base, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, ne10, nrows_dst, d_offset, global_row_low, stream);
            } else {
                GGML_SYCL_KTRACE("mmq_q4_0_aos", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
                ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            }
            break;
        case GGML_TYPE_Q4_1:
            GGML_SYCL_KTRACE("mmq_q4_1", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            GGML_SYCL_KTRACE("mmq_q5_0", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            GGML_SYCL_KTRACE("mmq_q5_1", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            if (use_soa) {
                // SoA layout: all qs values first, then all d (scale) values
                const ggml_tensor * storage = get_storage_tensor(src0);
                const int64_t storage_ne01 = storage->ne[1];
                const int64_t ncols = storage->ne[0];
                // Q8_0: explicit nblocks pattern (matches Q6_K)
                const size_t nblocks = static_cast<size_t>(storage_ne01) * static_cast<size_t>(ncols / QK8_0);
                const size_t total_qs_bytes = nblocks * QK8_0;  // 32 bytes per block
                const size_t d_offset = total_qs_bytes;
                
                // Get storage base pointer
                const void * storage_base = ggml_sycl_get_data_ptr(storage, device_id);
                
                // Calculate global row_low
                int64_t view_row_offset = 0;
                if (src0->view_src != nullptr) {
                    view_row_offset = src0->view_offs / src0->nb[1];
                }
                const int global_row_low = static_cast<int>(row_low + view_row_offset);

                GGML_SYCL_KTRACE("mmq_q8_0_soa", " ne00=%lld row_low=%lld row_diff=%lld ncols=%lld d_offset=%zu global_row=%d",
                    (long long)ne00, (long long)row_low, (long long)row_diff, (long long)src1_ncols, d_offset, global_row_low);

                ggml_mul_mat_q8_0_q8_1_sycl_soa(storage_base, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, ne10, nrows_dst, d_offset, global_row_low, stream);
            } else {
                GGML_SYCL_KTRACE("mmq_q8_0_aos", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
                ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            }
            break;
        case GGML_TYPE_Q2_K:
            GGML_SYCL_KTRACE("mmq_q2_k", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            GGML_SYCL_KTRACE("mmq_q3_k", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            GGML_SYCL_KTRACE("mmq_q4_k", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            GGML_SYCL_KTRACE("mmq_q5_k", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            // Q6_K SoA requires full_tensor like Q4_0/Q8_0
            if (use_soa) {
                const ggml_tensor * storage = get_storage_tensor(src0);
                const int64_t storage_ne01 = storage->ne[1];
                const int64_t storage_ne00 = storage->ne[0];
                
                // Q6_K SoA layout: | all ql (nblocks * 128) | all qh (nblocks * 64) | all scales (nblocks * 16) | all d (nblocks * 2) |
                // Block size: QK_K = 256 elements, nblocks = nrows * ncols / QK_K
                const size_t nblocks = static_cast<size_t>(storage_ne01) * static_cast<size_t>(storage_ne00) / QK_K;
                const size_t qh_offset = nblocks * 128;      // after ql section
                const size_t scales_offset = nblocks * 192;  // after ql + qh sections
                const size_t d_offset = nblocks * 208;       // after ql + qh + scales sections
                
                // Get storage base pointer
                const void * soa_base = ggml_sycl_get_data_ptr(storage, device_id);
                
                // Calculate global row_low
                int64_t view_row_offset = 0;
                if (src0->view_src != nullptr) {
                    view_row_offset = src0->view_offs / src0->nb[1];
                }
                const int global_row_low = static_cast<int>(row_low + view_row_offset);

                GGML_SYCL_KTRACE("mmq_q6_k_soa", " ne00=%lld row_low=%lld row_diff=%lld ncols=%lld qh=%zu scales=%zu d=%zu global_row=%d",
                    (long long)ne00, (long long)row_low, (long long)row_diff, (long long)src1_ncols, qh_offset, scales_offset, d_offset, global_row_low);
                ggml_mul_mat_q6_K_q8_1_sycl_soa(soa_base, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, ne10, nrows_dst, qh_offset, scales_offset, d_offset, global_row_low, stream);
            } else {
                GGML_SYCL_KTRACE("mmq_q6_k_aos", " ne00=%lld row_diff=%lld ncols=%lld", (long long)ne00, (long long)row_diff, (long long)src1_ncols);
                ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
