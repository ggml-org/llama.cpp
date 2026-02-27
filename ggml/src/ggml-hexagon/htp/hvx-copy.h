#ifndef HVX_COPY_H
#define HVX_COPY_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"

#define hvx_splat_loop_body(dst_type, vec_store)                 \
    do {                                                         \
        dst_type * restrict vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_size);                   \
        uint32_t nloe = n % (128 / elem_size);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        _Pragma("unroll(4)")                                     \
        for (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        if (nloe) {                                              \
            vec_store((void *) &vdst[i], nloe * elem_size, src); \
        }                                                        \
    } while(0)

static inline void hvx_splat_a(uint8_t * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_splat_loop_body(HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_splat_u(uint8_t * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_splat_f32_a(uint8_t * restrict dst, float v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, sizeof(float));
}

static inline void hvx_splat_f32_u(uint8_t * restrict dst, float v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, sizeof(float));
}

static inline void hvx_splat_f16_a(uint8_t * restrict dst, float v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
}

static inline void hvx_splat_f16_u(uint8_t * restrict dst, float v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
}

#define hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    do {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        const uint32_t epv  = 128 / elem_size;                       \
        const uint32_t nvec = n / epv;                               \
        const uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        for (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        if (nloe) {                                                  \
            vec_store((void *) &vdst[i], nloe * elem_size, vsrc[i]); \
        }                                                            \
    } while(0)

// Generic copy routines
static inline void hvx_copy_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_copy_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_copy_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_copy_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(__fp16));
}

// copy n fp32 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is aligned, destination is unaligned
static inline void hvx_copy_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is unaligned, destination is aligned
static inline void hvx_copy_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is unaligned, destination unaligned
static inline void hvx_copy_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(float));
}

//// fp32 -> fp16

#define hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    do {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        const uint32_t elem_size = sizeof(__fp16);                                  \
        const uint32_t epv  = 128 / elem_size;                                      \
        const uint32_t nvec = n / epv;                                              \
        const uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        for (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        if (nloe) {                                                                 \
            HVX_Vector v = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((void *) &vdst[i], nloe * elem_size, v);                      \
        }                                                                           \
    } while(0)

// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is aligned
static inline void hvx_copy_f16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is aligned
static inline void hvx_copy_f16_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is unaligned
static inline void hvx_copy_f16_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is unaligned
static inline void hvx_copy_f16_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#define hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    do {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        const HVX_Vector one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        const uint32_t elem_size = sizeof(__fp16);                                  \
        const uint32_t epv  = 128 / elem_size;                                      \
        const uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        for (i = 0; i < nvec; ++i) {                                                \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        if (nloe) {                                                                 \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_Vector vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            if (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            if (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * sizeof(float), vd);                \
            }                                                                       \
        }                                                                           \
    } while(0)

// copy/convert n fp16 elements into n fp32 elements : source is aligned, destination is aligned
static inline void hvx_copy_f32_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : source is unaligned, destination is aligned
static inline void hvx_copy_f32_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : source is aligned, destination is unaligned
static inline void hvx_copy_f32_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : source is unaligned, destination is unaligned
static inline void hvx_copy_f32_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#endif // HVX_COPY_H
