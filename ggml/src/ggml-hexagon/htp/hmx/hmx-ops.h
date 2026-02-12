// HMX operation entry-point declarations.
// Ported from htp-ops-lib/include/dsp/ops.h (renamed, benchmark kernels removed).

#ifndef HMX_OPS_H
#define HMX_OPS_H

#include <stdint.h>
#include "hmx-quants.h"

#ifndef restrict
#  define restrict __restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

// RMS normalisation (HVX-accelerated, used by both HMX and fallback paths)
int hvx_rms_norm_f32(float *restrict dst, const float *restrict src, int ne0, int ne1);

// HMX matrix multiplication — tile-permuted FP16 weights, FP32 activation/output
int hmx_mat_mul_permuted_w16a32(float *restrict dst,
                                const float *activation,
                                const __fp16 *permuted_weight,
                                int m, int k, int n);

// HMX matrix multiplication — tile-permuted quantised weights (Q4_0/Q8_0/IQ4_NL)
int hmx_mat_mul_permuted_qk_0_d16a32(float *restrict dst,
                                      const float *activation,
                                      const uint8_t *permuted_weight,
                                      int m, int k, int n,
                                      int weight_type);

// HMX flash attention — FP16 in/out
int simple_flash_attn(__fp16 *restrict O,
                      const __fp16 *restrict Q,
                      const __fp16 *restrict K,
                      const __fp16 *restrict V,
                      const __fp16 *restrict mask,
                      int qo_len, int kv_len,
                      int n_heads, int n_kv_heads,
                      int head_dim);

// Precomputed table initialisation (exp2 LUT for safe softmax).
// vtcm_table points to the reserved VTCM tail area.
void init_precomputed_tables(uint8_t *vtcm_table);

#ifdef __cplusplus
}
#endif

#endif // HMX_OPS_H
