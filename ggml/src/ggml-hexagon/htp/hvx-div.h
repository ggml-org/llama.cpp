#ifndef HVX_DIV_H
#define HVX_DIV_H

#include <HAP_farf.h>

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"
#include "hex-utils.h"
#include "hvx-inverse.h"
#include "hvx-arith.h"

#if __HVX_ARCH__ < 79
#define HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

#define hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src0_type * restrict vsrc0 = (src0_type *) src0;                             \
        src1_type * restrict vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        const uint32_t nvec = n / VLEN_FP32;                                         \
        const uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; i++) {                                                      \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL(vsrc0[i], inv_src1);                         \
            vdst[i] = res;                                                           \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL(vsrc0[i], inv_src1);                         \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } while(0)

// 3-letter suffix variants
static inline void hvx_div_f32_aaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((uintptr_t) dst % 128 == 0);
    assert((uintptr_t) src0 % 128 == 0);
    assert((uintptr_t) src1 % 128 == 0);
    hvx_div_f32_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_div_f32_aau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((uintptr_t) dst % 128 == 0);
    assert((uintptr_t) src0 % 128 == 0);
    hvx_div_f32_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_div_f32_aua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((uintptr_t) dst % 128 == 0);
    assert((uintptr_t) src1 % 128 == 0);
    hvx_div_f32_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_div_f32_auu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((uintptr_t) dst % 128 == 0);
    hvx_div_f32_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_div_f32_uaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((uintptr_t) src0 % 128 == 0);
    assert((uintptr_t) src1 % 128 == 0);
    hvx_div_f32_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_div_f32_uau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((uintptr_t) src0 % 128 == 0);
    hvx_div_f32_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_div_f32_uua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((uintptr_t) src1 % 128 == 0);
    hvx_div_f32_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_div_f32_uuu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    hvx_div_f32_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_div_f32(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint32_t num_elems) {
    if (hex_is_aligned((void *) dst, 128)) {
        if (hex_is_aligned((void *) src0, 128)) {
            if (hex_is_aligned((void *) src1, 128)) hvx_div_f32_aaa(dst, src0, src1, num_elems);
            else                                    hvx_div_f32_aau(dst, src0, src1, num_elems);
        } else {
            if (hex_is_aligned((void *) src1, 128)) hvx_div_f32_aua(dst, src0, src1, num_elems);
            else                                    hvx_div_f32_auu(dst, src0, src1, num_elems);
        }
    } else {
        if (hex_is_aligned((void *) src0, 128)) {
            if (hex_is_aligned((void *) src1, 128)) hvx_div_f32_uaa(dst, src0, src1, num_elems);
            else                                    hvx_div_f32_uau(dst, src0, src1, num_elems);
        } else {
            if (hex_is_aligned((void *) src1, 128)) hvx_div_f32_uua(dst, src0, src1, num_elems);
            else                                    hvx_div_f32_uuu(dst, src0, src1, num_elems);
        }
    }
}

#undef HVX_OP_MUL

#endif // HVX_DIV_H
