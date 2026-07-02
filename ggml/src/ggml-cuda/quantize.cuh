#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             ggml_type       type_src0,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// Gather already-quantized block_fp4_mmq blocks from a per-token "unique" buffer (row-major:
// block (token, kb) at token*blocks_per_col + kb) into the compact expert-sorted buffer expected
// by MMQ (column-block-major: block (row i, kb) at kb*nrows_dst + i), using ids_src1[i] -> token.
// Used to avoid re-quantizing the same broadcast activation once per routed expert (MoE gate/up).
void gather_mmq_fp4_blocks_cuda(const void *    src_unique,
                                const int32_t * ids_src1,
                                void *          dst,
                                int64_t         nrows_dst,
                                int64_t         blocks_per_col,
                                cudaStream_t    stream);

// Build inverse map token -> its compact expert-sorted rows. cnt[n_tokens] is scratch (zeroed),
// tok2c[n_tokens*n_expert_used] is filled so tok2c[t*n_expert_used + j] = compact row i whose
// ids_src1[i] == t (unused slots stay -1). Feeds the fused quantize+scatter path below.
void build_tok2c_cuda(const int32_t * ids_src1,
                      int32_t *       cnt,
                      int32_t *       tok2c,
                      int64_t         nrows_dst,
                      int64_t         n_tokens,
                      int             n_expert_used,
                      cudaStream_t    stream);

// Fused NVFP4 quantize + scatter: quantize each unique token's activation once and write the
// resulting block_fp4_mmq directly to all of that token's compact rows (kb*nrows_dst + i), using
// tok2c. Replaces quantize-unique + gather with one kernel (no intermediate unique buffer).
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * tok2c,
                                   void *          vy,
                                   ggml_type       type_src0,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);
