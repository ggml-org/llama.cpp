#ifndef HVX_SCALE_H
#define HVX_SCALE_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"

#define hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        const uint32_t elem_size = sizeof(float);                                    \
        const uint32_t epv = 128 / elem_size;                                        \
        const uint32_t nvec = n / epv;                                               \
        const uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((void *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } while(0)

static inline void hvx_scale_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } else {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } else {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#define hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                   \
                                                                                     \
        const uint32_t elem_size = sizeof(float);                                    \
        const uint32_t epv = 128 / elem_size;                                        \
        const uint32_t nvec = n / epv;                                               \
        const uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                        \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((void *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } while(0)

static inline void hvx_scale_offset_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}

#endif // HVX_SCALE_H
