// HVX-accelerated RMS normalisation for HMX backend.
// Ported from htp-ops-lib/src/dsp/ops/rms_norm.c.
//
// Changes from the original:
//   - vmem / VLEN / is_aligned / l2fetch replaced with hmx_ prefixed versions
//   - Include paths adapted for the merged htp/ directory layout.

#include <math.h>

#include "hmx-hvx-internal.h"
#include "hmx-utils.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / HMX_VLEN)

static void hvx_rms_norm_f32_inner(float *restrict dst, const float *restrict src, int ne0) {
  // TODO(hzx): make eps an input
  const float eps = 1e-5;

  int n_vecs        = ne0 / 32;
  int leftover      = ne0 & 31;
  int leftover_size = leftover * sizeof(float);

  const HVX_Vector *pv_in  = (const HVX_Vector *) src;
  HVX_Vector       *pv_out = (HVX_Vector *) dst;
  HVX_Vector        v_x, v_scale, v_sum = Q6_V_vzero(), v_zero = Q6_V_vzero();

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        hmx_l2fetch(pv_in + PREFETCH_N_VECS, HMX_VLEN, HMX_VLEN, prefetch_n_vecs, 0);
      }
    }

    v_x   = *pv_in++;
    v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_Vqf32_vmpy_VsfVsf(v_x, v_x));  // s += x*x
  }

  if (leftover > 0) {
    v_x   = Q6_V_valign_VVR(*pv_in, v_zero, leftover_size);
    v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_Vqf32_vmpy_VsfVsf(v_x, v_x));
  }

  float tmp[32] __attribute__((aligned(HMX_VLEN)));
  float sum = 0;

  // 32-way reduce sum
  for (int s = 64; s >= 4; s >>= 1) {
    v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_V_vlalign_VVR(v_sum, v_zero, s));
  }
  v_sum = Q6_Vsf_equals_Vqf32(v_sum);

  hmx_vmem(tmp) = v_sum;
  sum            = tmp[31];

  float mean  = sum / ne0;
  float scale = 1.0f / sqrtf(mean + eps);
  v_scale     = Q6_V_vsplat_R(*(int32_t *) &scale);  // fp32_to_bits

  // assume original input is still in L2 cache
  pv_in          = (const HVX_Vector *) src;
  // assume all buffer sizes are multiples of HMX_VLEN
  int n_vecs_out = n_vecs + (leftover > 0 ? 1 : 0);
  for (int i = 0; i < n_vecs_out; ++i) {
    *pv_out++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*pv_in++, v_scale));
  }
}

int hvx_rms_norm_f32(float *restrict dst, const float *restrict src, int ne0, int ne1) {
  if (!dst || !src || !ne0 || !ne1) {
    return -1;
  }
  if (!hmx_is_aligned(dst, HMX_VLEN) || !hmx_is_aligned(src, HMX_VLEN)) {
    return -1;
  }

  // TODO(hzx): parallelize outer loop
  for (int j = 0; j < ne1; ++j) {
    float       *output = dst + j * ne0;
    const float *input  = src + j * ne0;
    hvx_rms_norm_f32_inner(output, input, ne0);
  }

  return 0;
}
