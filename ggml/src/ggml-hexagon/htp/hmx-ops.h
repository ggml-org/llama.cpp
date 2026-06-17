// HMX operation entry-point declarations.
// Ported from htp-ops-lib/include/dsp/ops.h (renamed, benchmark kernels removed). (https://github.com/haozixu/htp-ops-lib)

#ifndef HMX_OPS_H
#define HMX_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "htp-ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float        *dst;
    const float  *activation;
    const __fp16 *weight;
    int           m;
    int           k;
    int           n;
    int           act_stride;
    int           weight_stride;
    int           dst_stride;
    int           ne02;
    int           ne03;
    int           ne12;
    int           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        src1_nb2;
    size_t        src1_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
} hmx_matmul_f16_f32_batched_params_t;

// Batched F16 wrapper over hmx_matmul_2d_f32.
// Batch semantics match ggml_mul_mat(): src0 broadcasts to src1 in dims 2/3.
int hmx_matmul_f16_f32_batched(struct htp_context *ctx, const hmx_matmul_f16_f32_batched_params_t *params,
                               int m_chunk, int n_chunk, int use_pipeline, int num_threads, int act_threads, int spad_size);

int hmx_matmul_2d_f32(struct htp_context *ctx,
                                  float *restrict dst,
                                  const float *activation,
                                  const uint8_t *weight,
                                  int m, int k, int n,
                                  int act_stride,
                                  int weight_stride,
                                  int weight_type,
                                  int k_valid,
                                  int dst_stride,
                                  int dst_cols,
                                  int m_chunk,
                                  int n_chunk,
                                  int use_pipeline,
                                  int num_threads,
                                  int act_threads,
                                  int tile_size,
                                  int aligned_tile_size,
                                  int spad_size);

struct mmid_row_mapping;

int hmx_matmul_id_2d_f32(struct htp_context *ctx,
                                         float *restrict dst,
                                         const float *activation,
                                         const uint8_t *weight,
                                         int m, int k, int n,
                                         int k_valid,
                                         int ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         int weight_stride,
                                         int weight_type,
                                         const struct mmid_row_mapping *matrix_rows,
                                         int cur_a,
                                         int mapping_stride);

// HMX flash attention
int hmx_flash_attn_ext(struct htp_ops_context * octx);

#ifdef __cplusplus
}
#endif

#endif // HMX_OPS_H
