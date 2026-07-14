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

#define UNUSED(x) (void)(x)

#define hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_size, vec_store, vec_op) \
    do {                                                                       \
        dst_type * restrict vdst  = (dst_type *) dst;                          \
        src0_type * restrict vsrc0 = (src0_type *) src0;                       \
        src1_type * restrict vsrc1 = (src1_type *) src1;                       \
                                                                               \
        const uint32_t epv  = 128 / (elem_size);                               \
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
            vec_store((void *) &vdst[i], nloe * (elem_size), v);               \
        }                                                                      \
    } while(0)

#if __HVX_ARCH__ < 79

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#else

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#endif

#define HVX_OP_ADD_F16(a, b) hvx_vec_add_f16_f16(a, b)
#define HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#define HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_BINARY_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE) \
static inline void OP_NAME##_aaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src0 % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_aau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src0 % 128 == 0); \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_aua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_auu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_uaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src0 % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \
static inline void OP_NAME##_uau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src0 % 128 == 0); \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \
static inline void OP_NAME##_uua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src1 % 128 == 0); \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \
static inline void OP_NAME##_uuu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \

DEFINE_HVX_BINARY_OP_VARIANTS(hvx_add_f32, HVX_OP_ADD_F32, float)
DEFINE_HVX_BINARY_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, float)
DEFINE_HVX_BINARY_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, float)

DEFINE_HVX_BINARY_OP_VARIANTS(hvx_add_f16, HVX_OP_ADD_F16, _Float16)
DEFINE_HVX_BINARY_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_BINARY_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#define HVX_BINARY_DISPATCHER(OP_NAME) \
static inline void OP_NAME(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint32_t num_elems) { \
    if (hex_is_aligned((void *) dst, 128)) { \
        if (hex_is_aligned((void *) src0, 128)) { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_aau(dst, src0, src1, num_elems); \
        } else { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_aua(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_auu(dst, src0, src1, num_elems); \
        } \
    } else { \
        if (hex_is_aligned((void *) src0, 128)) { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_uau(dst, src0, src1, num_elems); \
        } else { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_uua(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_uuu(dst, src0, src1, num_elems); \
        } \
    } \
}

HVX_BINARY_DISPATCHER(hvx_add_f32)
HVX_BINARY_DISPATCHER(hvx_sub_f32)
HVX_BINARY_DISPATCHER(hvx_mul_f32)

HVX_BINARY_DISPATCHER(hvx_add_f16)
HVX_BINARY_DISPATCHER(hvx_sub_f16)
HVX_BINARY_DISPATCHER(hvx_mul_f16)

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
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    if (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((void *) &vdst[i], nloe * elem_size, v2);
    }
}

// Scalar Operations

#define hvx_scalar_loop_body(dst_type, src_type, elem_size, vec_store, scalar_op_macro)   \
    do {                                                                       \
        dst_type * restrict vdst = (dst_type *) dst;                           \
        src_type * restrict vsrc = (src_type *) src;                           \
                                                                               \
        const uint32_t epv  = 128 / (elem_size);                               \
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
            vec_store((void *) &vdst[i], nloe * (elem_size), v);               \
        }                                                                      \
    } while(0)

#define HVX_OP_ADD_SCALAR_F32(v) \
    ({ \
        const HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F32(v, val_vec); \
        Q6_V_vmux_QVV(pred_inf, inf, out); \
    })

#define HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#define HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#define HVX_OP_ADD_SCALAR_F16(v) \
    ({ \
        const HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F16(v, val_vec); \
        Q6_V_vmux_QVV(pred_inf, inf, out); \
    })

#define HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#define HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE) \
static inline void OP_NAME##_aa(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, uint32_t n) { \
    const HVX_Vector val_vec = SPLAT_MACRO(val); \
    const HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf); \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src % 128 == 0); \
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_au(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, uint32_t n) { \
    const HVX_Vector val_vec = SPLAT_MACRO(val); \
    const HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf); \
    assert((uintptr_t) dst % 128 == 0); \
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_ua(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, uint32_t n) { \
    const HVX_Vector val_vec = SPLAT_MACRO(val); \
    const HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf); \
    assert((uintptr_t) src % 128 == 0); \
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \
static inline void OP_NAME##_uu(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, uint32_t n) { \
    const HVX_Vector val_vec = SPLAT_MACRO(val); \
    const HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf); \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \

DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_add_scalar_f32, HVX_OP_ADD_SCALAR_F32, hvx_vec_splat_f32, float)
DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, float)
DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, float)

DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_add_scalar_f16, HVX_OP_ADD_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#define HVX_BINARY_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE) \
static inline void OP_NAME(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, const uint32_t num_elems) { \
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) { \
        OP_NAME##_aa(dst, src, val, num_elems); \
    } else if (hex_is_aligned((void *) dst, 128)) { \
        OP_NAME##_au(dst, src, val, num_elems); \
    } else if (hex_is_aligned((void *) src, 128)) { \
        OP_NAME##_ua(dst, src, val, num_elems); \
    } else { \
        OP_NAME##_uu(dst, src, val, num_elems); \
    } \
}

HVX_BINARY_SCALAR_DISPATCHER(hvx_add_scalar_f32, float)
HVX_BINARY_SCALAR_DISPATCHER(hvx_sub_scalar_f32, float)
HVX_BINARY_SCALAR_DISPATCHER(hvx_mul_scalar_f32, float)

HVX_BINARY_SCALAR_DISPATCHER(hvx_add_scalar_f16, _Float16)
HVX_BINARY_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_BINARY_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#define HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

static inline void hvx_min_scalar_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, sizeof(float), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, sizeof(float), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, sizeof(float), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, sizeof(float), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
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
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, sizeof(float), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, sizeof(float), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, sizeof(float), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, sizeof(float), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
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

//
// Square
//

#define hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)           \
    do {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        const uint32_t elem_size = sizeof(float);                          \
        const uint32_t epv  = 128 / elem_size;                             \
        const uint32_t nvec = n / epv;                                     \
        const uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        for (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                        \
        }                                                                  \
        if (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                   \
            vec_store((void *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } while(0)

static inline void hvx_sqr_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_sqr_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_sqr_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_sqr_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_sqr_f32(uint8_t * restrict dst, const uint8_t * restrict src, const uint32_t num_elems) {
    if (hex_is_aligned((void *) dst, 128)) {
        if (hex_is_aligned((void *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, num_elems);
        } else {
            hvx_sqr_f32_au(dst, src, num_elems);
        }
    } else {
        if (hex_is_aligned((void *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, num_elems);
        } else {
            hvx_sqr_f32_uu(dst, src, num_elems);
        }
    }
}

#undef HVX_OP_ADD_F32
#undef HVX_OP_SUB_F32
#undef HVX_OP_MUL_F32
#undef HVX_OP_ADD_F16
#undef HVX_OP_SUB_F16
#undef HVX_OP_MUL_F16
#undef hvx_arith_loop_body
#undef HVX_OP_ADD_SCALAR_F32
#undef HVX_OP_SUB_SCALAR_F32
#undef HVX_OP_MUL_SCALAR_F32
#undef HVX_OP_ADD_SCALAR_F16
#undef HVX_OP_SUB_SCALAR_F16
#undef HVX_OP_MUL_SCALAR_F16
#undef hvx_scalar_loop_body
#undef HVX_OP_MIN_SCALAR
#undef HVX_OP_CLAMP_SCALAR
#undef DEFINE_HVX_BINARY_OP_VARIANTS
#undef HVX_BINARY_DISPATCHER
#undef UNUSED

#endif // HVX_ARITH_H
