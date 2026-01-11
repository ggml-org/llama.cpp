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
#include "hvx-reduce.h"
#include "hvx-sigmoid.h"
#include "hvx-sqrt.h"
#include "hvx-base.h"

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
