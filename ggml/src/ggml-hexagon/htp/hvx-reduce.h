#ifndef HVX_REDUCE_H
#define HVX_REDUCE_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

#include "hex-utils.h"
#include "hvx-base.h"
#include "hvx-types.h"

static inline HVX_Vector hvx_vec_reduce_sum_n_i32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // int32

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);     // rotate right
        sum   = Q6_Vw_vadd_VwVw(sum_t, sum);  // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_reduce_sum_i32(HVX_Vector in) {
    return hvx_vec_reduce_sum_n_i32(in, 32);
}

static inline HVX_Vector hvx_vec_reduce_sum_n_qf32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum), width);  // rotate right
        sum   = Q6_Vqf32_vadd_Vqf32Vsf(sum, sum_t);             // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_reduce_sum_qf32(HVX_Vector in) {
    return hvx_vec_reduce_sum_n_qf32(in, 32);
}

#if __HVX_ARCH__ > 75

static inline HVX_Vector hvx_vec_reduce_sum_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  sum_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 2));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 4));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 8));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 16));
    return sum_sf;
}

static inline HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);       // rotate right
        sum   = Q6_Vsf_vadd_VsfVsf(sum, sum_t); // elementwise sum
        width = width << 1;
    }
    return sum;
}

#else

static inline HVX_Vector hvx_vec_reduce_sum_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  sum_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 2));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 4));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 8));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 16));
    return Q6_Vsf_equals_Vqf32(sum_qf);
}

static inline HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);                               // rotate right
        sum   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, sum_t));  // elementwise sum
        width = width << 1;
    }
    return sum;
}

#endif

static inline HVX_Vector hvx_vec_reduce_sum_f32(HVX_Vector in) {
    return hvx_vec_reduce_sum_n_f32(in, 32);
}

static inline HVX_Vector hvx_vec_reduce_max_f16(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp16 nbytes

    HVX_Vector _max = in, _max_t;
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max2_f16(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(in, _max);
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max_f32(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max = in, _max_t;
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max2_f32(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(in, _max);
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

#define hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    do {                                                                                    \
        src_type * restrict vsrc = (src_type *) src;                                        \
        HVX_Vector acc = init_vec;                                                          \
                                                                                            \
        const uint32_t elem_size = sizeof(float);                                           \
        const uint32_t epv  = 128 / elem_size;                                              \
        const uint32_t nvec = num_elems / epv;                                              \
        const uint32_t nloe = num_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        for (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        if (nloe) {                                                                         \
            const float * srcf = (const float *) src + i * epv;                             \
            HVX_Vector in = *(HVX_UVector *) srcf;                                          \
            HVX_Vector temp = Q6_V_valign_VVR(in, pad_vec, nloe * elem_size);               \
            acc = vec_op(acc, temp);                                                        \
        }                                                                                   \
        HVX_Vector v = reduce_op(acc);                                                      \
        return scalar_reduce(v);                                                            \
    } while(0)

#define HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#define HVX_REDUCE_SUM_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#define HVX_SUM_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#define HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#define HVX_REDUCE_SUM_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

static inline float hvx_reduce_max_f32_a(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((const float *) src)[0]);
    assert((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

static inline float hvx_reduce_max_f32_u(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((const float *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

static inline float hvx_reduce_max_f32(const uint8_t * restrict src, const int num_elems) {
    if (hex_is_aligned((void *) src, 128)) {
        return hvx_reduce_max_f32_a(src, num_elems);
    } else {
        return hvx_reduce_max_f32_u(src, num_elems);
    }
}

// Sum variants

static inline float hvx_reduce_sum_f32_a(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    assert((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_SUM_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

static inline float hvx_reduce_sum_f32_u(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_SUM_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

static inline float hvx_reduce_sum_f32(const uint8_t * restrict src, const int num_elems) {
    if (hex_is_aligned((void *) src, 128)) {
        return hvx_reduce_sum_f32_a(src, num_elems);
    } else {
        return hvx_reduce_sum_f32_u(src, num_elems);
    }
}

// Sum of squares variants

static inline float hvx_sum_of_squares_f32_a(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    assert((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_SUM_SQ_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

static inline float hvx_sum_of_squares_f32_u(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_SUM_SQ_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

static inline float hvx_sum_of_squares_f32(const uint8_t * restrict src, const int num_elems) {
    if (hex_is_aligned((void *) src, 128)) {
        return hvx_sum_of_squares_f32_a(src, num_elems);
    } else {
        return hvx_sum_of_squares_f32_u(src, num_elems);
    }
}

#undef hvx_reduce_loop_body
#undef HVX_REDUCE_MAX_OP
#undef HVX_REDUCE_SUM_OP
#undef HVX_REDUCE_MAX_SCALAR
#undef HVX_REDUCE_SUM_SCALAR
#undef HVX_SUM_SQ_OP

#endif /* HVX_REDUCE_H */
