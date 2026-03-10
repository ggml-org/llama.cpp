// HMX operation entry-point declarations.
// Ported from htp-ops-lib/include/dsp/ops.h (renamed, benchmark kernels removed).

#ifndef HMX_OPS_H
#define HMX_OPS_H

#include <stddef.h>
#include <stdint.h>
#include "hmx-quants.h"

#ifndef restrict
#  define restrict __restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct htp_context;  // forward declaration

typedef struct {
    float        *dst;
    const float  *activation;
    const __fp16 *permuted_weight;
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
} hmx_matmul_w16a32_batched_params_t;

// HMX matrix multiplication — tile-permuted FP16 weights, FP32 activation/output
// act_stride: activation row stride in elements (= k for contiguous, or
//             nb[1]/sizeof(float) for permuted tensors like attention Q).
// weight_stride: weight row stride in elements (= k for compact weights, or
//                nb[1]/sizeof(__fp16) for permuted KV-cache views used by QK).
int hmx_mat_mul_permuted_w16a32(struct htp_context *ctx,
                                float *restrict dst,
                                const float *activation,
                                const __fp16 *permuted_weight,
                                int m, int k, int n,
                                int act_stride,
                                int weight_stride);

// Batched F16 wrapper over hmx_mat_mul_permuted_w16a32.
// Batch semantics match ggml_mul_mat(): src0 broadcasts to src1 in dims 2/3.
int hmx_mat_mul_permuted_w16a32_batched(struct htp_context *ctx,
                                        const hmx_matmul_w16a32_batched_params_t *params);

// HMX matrix multiplication — tile-permuted quantised weights (Q4_0/Q8_0/IQ4_NL)
int hmx_mat_mul_permuted_qk_0_d16a32(struct htp_context *ctx,
                                      float *restrict dst,
                                      const float *activation,
                                      const uint8_t *permuted_weight,
                                      int m, int k, int n,
                                      int weight_type);

#ifdef __cplusplus
}
#endif

#endif // HMX_OPS_H
