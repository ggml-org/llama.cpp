#ifndef HVX_UTILS_H
#define HVX_UTILS_H

#include <stdbool.h>
#include <stdint.h>

#include "hex-utils.h"

#include "hvx-types.h"
#include "hvx-copy.h"
#include "hvx-scale.h"
#include "hvx-exp.h"
#include "hvx-inverse.h"
#include "hvx-base.h"
#include "hvx-reduce.h"
#include "hvx-sigmoid.h"

#define RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#define RSQRT_ONE_HALF     0x3f000000  // 0.5
#define RSQRT_THREE_HALVES 0x3fc00000  // 1.5

static inline HVX_Vector hvx_vec_rsqrt_fp32(HVX_Vector in_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>2)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_Vector rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_Vector x2, y, ypower2, temp;

    x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(in_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    return Q6_Vsf_equals_Vqf32(temp);
}

float hvx_sum_of_squares_f32(const uint8_t * restrict src, const int num_elems);
void  hvx_mul_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_mul_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_mul_mul_f32_opt(const uint8_t * restrict src0,
                          const uint8_t * restrict src1,
                          const uint8_t * restrict src2,
                          uint8_t * restrict dst,
                          const int num_elems);
void  hvx_mul_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_add_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_add_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_add_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_sub_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_sub_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_sub_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
float hvx_self_max_f32(const uint8_t * restrict src, const int num_elems);
float hvx_self_sum_f32(const uint8_t * restrict src, const int num_elems);
void  hvx_min_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_clamp_scalar_f32(const uint8_t * restrict src,
                           const float limit_left,
                           const float limit_right,
                           uint8_t * restrict dst,
                           const int num_elems);

#endif /* HVX_UTILS_H */
