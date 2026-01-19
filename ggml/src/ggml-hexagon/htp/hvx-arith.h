#ifndef HVX_ARITH_H
#define HVX_ARITH_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "hvx-base.h"
#include "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#define hvx_arith_loop_body(dst_type, src0_type, src1_type, vec_store, vec_op) \
    do {                                                                       \
        dst_type * restrict vdst  = (dst_type *) dst;                          \
        src0_type * restrict vsrc0 = (src0_type *) src0;                       \
        src1_type * restrict vsrc1 = (src1_type *) src1;                       \
                                                                               \
        const uint32_t elem_size = sizeof(float);                              \
        const uint32_t epv  = 128 / elem_size;                                 \
        const uint32_t nvec = n / epv;                                         \
        const uint32_t nloe = n % epv;                                         \
                                                                               \
        uint32_t i = 0;                                                        \
                                                                               \
        _Pragma("unroll(4)")                                                   \
        for (; i < nvec; i++) {                                                \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                              \
        }                                                                      \
        if (nloe) {                                                            \
            HVX_Vector v = vec_op(vsrc0[i], vsrc1[i]);                         \
            vec_store((void *) &vdst[i], nloe * elem_size, v);                 \
        }                                                                      \
    } while(0)

#if __HVX_ARCH__ < 79
#define HVX_OP_ADD(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_ADD(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

// ADD variants

static inline void hvx_add_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, hvx_vec_store_a, HVX_OP_ADD);
}

static inline void hvx_add_f32_au(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, hvx_vec_store_a, HVX_OP_ADD);
}

static inline void hvx_add_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, hvx_vec_store_u, HVX_OP_ADD);
}

static inline void hvx_add_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u, HVX_OP_ADD);
}

// SUB variants

static inline void hvx_sub_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, hvx_vec_store_a, HVX_OP_SUB);
}

static inline void hvx_sub_f32_au(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, hvx_vec_store_a, HVX_OP_SUB);
}

static inline void hvx_sub_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, hvx_vec_store_u, HVX_OP_SUB);
}

static inline void hvx_sub_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u, HVX_OP_SUB);
}

// MUL variants

static inline void hvx_mul_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, hvx_vec_store_a, HVX_OP_MUL);
}

static inline void hvx_mul_f32_au(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, hvx_vec_store_a, HVX_OP_MUL);
}

static inline void hvx_mul_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, hvx_vec_store_u, HVX_OP_MUL);
}

static inline void hvx_mul_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u, HVX_OP_MUL);
}

// Dispatchers

static inline void hvx_add_f32(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint32_t num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src0, 128)) {
        if (hex_is_aligned((void *) src1, 128)) {
            hvx_add_f32_aa(dst, src0, src1, num_elems);
        } else {
            hvx_add_f32_au(dst, src0, src1, num_elems);
        }
    } else if (hex_is_aligned((void *) src0, 128) && hex_is_aligned((void *) src1, 128)) {
        hvx_add_f32_ua(dst, src0, src1, num_elems);
    } else {
        hvx_add_f32_uu(dst, src0, src1, num_elems);
    }
}

static inline void hvx_sub_f32(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint32_t num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src0, 128)) {
        if (hex_is_aligned((void *) src1, 128)) {
            hvx_sub_f32_aa(dst, src0, src1, num_elems);
        } else {
            hvx_sub_f32_au(dst, src0, src1, num_elems);
        }
    } else if (hex_is_aligned((void *) src0, 128) && hex_is_aligned((void *) src1, 128)) {
        hvx_sub_f32_ua(dst, src0, src1, num_elems);
    } else {
        hvx_sub_f32_uu(dst, src0, src1, num_elems);
    }
}

static inline void hvx_mul_f32(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint32_t num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src0, 128)) {
        if (hex_is_aligned((void *) src1, 128)) {
            hvx_mul_f32_aa(dst, src0, src1, num_elems);
        } else {
            hvx_mul_f32_au(dst, src0, src1, num_elems);
        }
    } else if (hex_is_aligned((void *) src0, 128) && hex_is_aligned((void *) src1, 128)) {
        hvx_mul_f32_ua(dst, src0, src1, num_elems);
    } else {
        hvx_mul_f32_uu(dst, src0, src1, num_elems);
    }
}

// Mul-Mul Optimized

static inline void hvx_mul_mul_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint8_t * restrict src2, const uint32_t num_elems) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);
    assert((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    const uint32_t elem_size = sizeof(float);
    const uint32_t epv  = 128 / elem_size;
    const uint32_t nvec = num_elems / epv;
    const uint32_t nloe = num_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    for (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    if (nloe) {
        HVX_Vector v1 = HVX_OP_MUL(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL(v1, vsrc2[i]);
        hvx_vec_store_a((void *) &vdst[i], nloe * elem_size, v2);
    }
}

// Scalar Operations

#define hvx_scalar_loop_body(dst_type, src_type, vec_store, scalar_op_macro)   \
    do {                                                                       \
        dst_type * restrict vdst = (dst_type *) dst;                           \
        src_type * restrict vsrc = (src_type *) src;                           \
                                                                               \
        const uint32_t elem_size = sizeof(float);                              \
        const uint32_t epv  = 128 / elem_size;                                 \
        const uint32_t nvec = n / epv;                                         \
        const uint32_t nloe = n % epv;                                         \
                                                                               \
        uint32_t i = 0;                                                        \
                                                                               \
        _Pragma("unroll(4)")                                                   \
        for (; i < nvec; i++) {                                                \
            HVX_Vector v = vsrc[i];                                            \
            vdst[i] = scalar_op_macro(v);                                      \
        }                                                                      \
        if (nloe) {                                                            \
            HVX_Vector v = vsrc[i];                                            \
            v = scalar_op_macro(v);                                            \
            vec_store((void *) &vdst[i], nloe * elem_size, v);                 \
        }                                                                      \
    } while(0)

#define HVX_OP_ADD_SCALAR(v) \
    ({ \
        const HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD(v, val_vec); \
        Q6_V_vmux_QVV(pred_inf, inf, out); \
    })

#define HVX_OP_MUL_SCALAR(v) HVX_OP_MUL(v, val_vec)
#define HVX_OP_SUB_SCALAR(v) HVX_OP_SUB(v, val_vec)

// Add Scalar Variants

static inline void hvx_add_scalar_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    const HVX_Vector inf = hvx_vec_splat_f32(INFINITY);
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a, HVX_OP_ADD_SCALAR);
}

static inline void hvx_add_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    const HVX_Vector inf = hvx_vec_splat_f32(INFINITY);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a, HVX_OP_ADD_SCALAR);
}

static inline void hvx_add_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    const HVX_Vector inf = hvx_vec_splat_f32(INFINITY);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u, HVX_OP_ADD_SCALAR);
}

static inline void hvx_add_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    static const float kInf = INFINITY;
    const HVX_Vector inf = hvx_vec_splat_f32(kInf);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u, HVX_OP_ADD_SCALAR);
}

// Sub Scalar Variants

static inline void hvx_sub_scalar_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a, HVX_OP_SUB_SCALAR);
}

static inline void hvx_sub_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a, HVX_OP_SUB_SCALAR);
}

static inline void hvx_sub_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u, HVX_OP_SUB_SCALAR);
}

static inline void hvx_sub_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u, HVX_OP_SUB_SCALAR);
}

// Mul Scalar Variants

static inline void hvx_mul_scalar_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a, HVX_OP_MUL_SCALAR);
}

static inline void hvx_mul_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a, HVX_OP_MUL_SCALAR);
}

static inline void hvx_mul_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u, HVX_OP_MUL_SCALAR);
}

static inline void hvx_mul_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u, HVX_OP_MUL_SCALAR);
}

static inline void hvx_add_scalar_f32(uint8_t * restrict dst, const uint8_t * restrict src, const float val, const int num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) {
        hvx_add_scalar_f32_aa(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) dst, 128)) {
        hvx_add_scalar_f32_au(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) src, 128)) {
        hvx_add_scalar_f32_ua(dst, src, val, num_elems);
    } else {
        hvx_add_scalar_f32_uu(dst, src, val, num_elems);
    }
}

static inline void hvx_mul_scalar_f32(uint8_t * restrict dst, const uint8_t * restrict src, const float val, const int num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) {
        hvx_mul_scalar_f32_aa(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) dst, 128)) {
        hvx_mul_scalar_f32_au(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) src, 128)) {
        hvx_mul_scalar_f32_ua(dst, src, val, num_elems);
    } else {
        hvx_mul_scalar_f32_uu(dst, src, val, num_elems);
    }
}

static inline void hvx_sub_scalar_f32(uint8_t * restrict dst, const uint8_t * restrict src, const float val, const int num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) {
        hvx_sub_scalar_f32_aa(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) dst, 128)) {
        hvx_sub_scalar_f32_au(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) src, 128)) {
        hvx_sub_scalar_f32_ua(dst, src, val, num_elems);
    } else {
        hvx_sub_scalar_f32_uu(dst, src, val, num_elems);
    }
}

// MIN Scalar variants

#define HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

static inline void hvx_min_scalar_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32(uint8_t * restrict dst, const uint8_t * restrict src, const float val, const int num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, num_elems);
    } else {
        hvx_min_scalar_f32_uu(dst, src, val, num_elems);
    }
}

// CLAMP Scalar variants

#define HVX_OP_CLAMP_SCALAR(v) \
    ({ \
        HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_right, max_vec, v); \
        Q6_V_vmux_QVV(pred_cap_left, min_vec, tmp); \
    })

static inline void hvx_clamp_scalar_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, const int num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, num_elems);
    } else if (hex_is_aligned((void *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, num_elems);
    } else if (hex_is_aligned((void *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, num_elems);
    } else {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, num_elems);
    }
}

#undef HVX_OP_ADD
#undef HVX_OP_SUB
#undef HVX_OP_MUL
#undef hvx_arith_loop_body
#undef HVX_OP_ADD_SCALAR
#undef HVX_OP_SUB_SCALAR
#undef HVX_OP_MUL_SCALAR
#undef hvx_scalar_loop_body
#undef HVX_OP_MIN_SCALAR
#undef HVX_OP_CLAMP_SCALAR

#endif // HVX_ARITH_H
