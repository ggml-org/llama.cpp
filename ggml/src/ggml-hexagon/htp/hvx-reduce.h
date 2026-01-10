#ifndef HVX_REDUCE_H
#define HVX_REDUCE_H

#include <stdbool.h>
#include <stdint.h>

#include "hvx-types.h"

static inline HVX_Vector hvx_vec_int32_reduce_sum_n(HVX_Vector in, unsigned int n) {
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

static inline HVX_Vector hvx_vec_int32_reduce_sum(HVX_Vector in) {
    return hvx_vec_int32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_qf32_reduce_sum_n(HVX_Vector in, unsigned int n) {
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

static inline HVX_Vector hvx_vec_qf32_reduce_sum(HVX_Vector in) {
    return hvx_vec_qf32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_fp32_reduce_sum_n(HVX_Vector in, unsigned int n) {
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

static inline HVX_Vector hvx_vec_fp32_reduce_sum(HVX_Vector in) {
    return hvx_vec_fp32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_reduce_max_fp16(HVX_Vector in) {
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

static inline HVX_Vector hvx_vec_reduce_max2_fp16(HVX_Vector in, HVX_Vector _max) {
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

static inline HVX_Vector hvx_vec_reduce_max_fp32(HVX_Vector in) {
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

static inline HVX_Vector hvx_vec_reduce_max2_fp32(HVX_Vector in, HVX_Vector _max) {
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

#endif /* HVX_REDUCE_H */
