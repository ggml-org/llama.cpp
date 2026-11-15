#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

void quantize_row_q8_1_cuda(
        const float * x, void * vy,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// src1 of the few-row MMQ in the layout of the MMA fragments. Per channel, 256 values and group of MMQ_FEW_NCOLS columns:
//   qs: int4[4][32], [j][lane] holds the B fragments of sub-blocks 2j and 2j+1 of lane (g, t): the values [4t, 4t + 4) and [16 + 4t, 16 + 4t + 4) of column g.
//   ds: int4[4][4], [t][j] holds d (D4, float) or (d, sum) (DS4, half2) of the columns 2t and 2t+1 for sub-block 2j, then 2j+1.
// The column groups of a unit are adjacent, columns beyond ne1 are zero.
static constexpr int MMQ_FEW_NCOLS     = 8; // src1 columns per column group
static constexpr int MMQ_FEW_Q8_1_INT4 = 4*WARP_SIZE + 4*4; // int4 per 256 values and column group

void quantize_mmq_few_q8_1_cuda(
        const float * x, void * vy, mmq_q8_1_ds_layout ds_layout, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// src1 of the few-row MMQ for AMD GPUs, in the layout of the WMMA and MFMA fragments. Both multiply 16 src1 columns and use the same total size.
// RDNA WMMA, per channel, 256 values and group of MMQ_FEW_HIP_NCOLS columns:
//   qs: int4[8][16][2], [s][m][k] holds the values [16k, 16k + 16) of sub-block s of column m, the A fragments of lanes m and m + 16.
//   ds: int4[8][2][2], [s][h][k] holds d (D4, float) or (d, sum) (DS4, half2) of sub-block s of the columns 2v + h, v = 4k .. 4k + 3.
// CDNA, where lane group g = l / 16 decodes 2 sub-blocks of src0 as chains 0 and 1 of 8 MFMAs:
//   qs, 4-block MFMA: int4[4][64], [k][l] holds the ints 4k .. 4k + 3 of lane l: int 8c + i holds the values [4i, 4i + 4) of column l % 16 of the sub-block of chain c.
//       The sub-block of chain c is 2g + c, for Q6_K 4*(g/2) + 2c + g%2.
//   d:  float[8][16], [s][m] holds d of sub-block s of column m.
//   sum: float[8][16], [s][m] holds d times the sum of the quantized values of sub-block s of column m.
// CDNA narrow, for the 16-block MFMA with up to 8 columns in groups of 4: the same d and sum,
//   qs: int4[4][2][2][4][4], [cg][c][h][g][r] holds the ints 4h .. 4h + 3 of chain c of slot g of the column 4cg + r.
static constexpr int MMQ_FEW_HIP_NCOLS     = 16;
static constexpr int MMQ_FEW_HIP_Q8_1_INT4 = 8*16*2 + 8*16*2/4;

static constexpr int MMQ_FEW_CDNA_ORDER_Q6_K   = 1;
static constexpr int MMQ_FEW_CDNA_ORDER_NARROW = 2;

void quantize_mmq_few_hip_q8_1_cuda(
        const float * x, void * vy, ggml_type type_src0, bool narrow, mmq_q8_1_ds_layout ds_layout, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);
