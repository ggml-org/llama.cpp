#ifndef HVX_BASE_H
#define HVX_BASE_H

#include <stdbool.h>
#include <stdint.h>

#include "hex-utils.h"
#include "hvx-types.h"

static inline void hvx_vec_store_u(void * restrict dst, uint32_t n, HVX_Vector v) {
    // Rotate as needed.
    v = Q6_V_vlalign_VVR(v, v, (size_t) dst);

    uint32_t left_off  = (size_t) dst & 127;
    uint32_t right_off = left_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) dst);
    HVX_VectorPred qr     = Q6_Q_vsetq2_R(right_off);

    if (right_off > 128) {
        Q6_vmem_QRIV(qr, (HVX_Vector *) dst + 1, v);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(v, v);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector *) dst, v);
}

static inline void hvx_vec_store_a(void * restrict dst, uint32_t n, HVX_Vector v) {
    assert((unsigned long) dst % 128 == 0);
    HVX_VectorPred m = Q6_Q_or_QQn(Q6_Q_vsetq_R((unsigned long) dst), Q6_Q_vsetq2_R(n));
    Q6_vmem_QnRIV(m, (HVX_Vector *) dst, v);
}

static inline HVX_Vector hvx_vec_splat_f32(float v) {
    union { float  f; uint32_t i; } u = { .f = v };
    return Q6_V_vsplat_R(u.i);
}

static inline HVX_Vector hvx_vec_splat_f16(float v) {
    union { __fp16 f; uint16_t i; } u = { .f = v };
    return Q6_Vh_vsplat_R(u.i);
}

static inline HVX_Vector hvx_vec_repl4(HVX_Vector v) {
    // vdelta control to replicate first 4 bytes across all elements
    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };

    HVX_Vector ctrl = *(HVX_Vector *) repl;
    return Q6_V_vdelta_VV(v, ctrl);
}

static inline float hvx_vec_get_f32(HVX_Vector v) {
    float __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    return x;
}

static inline int32_t hvx_vec_get_i32(HVX_Vector v) {
    int32_t __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    return x;
}

static inline HVX_Vector hvx_vec_abs_f16(HVX_Vector v) {
    // abs by clearing the fp16 sign bit
    HVX_Vector mask = Q6_Vh_vsplat_R(0x7fff);
    return Q6_V_vand_VV(v, mask);
}

static inline HVX_Vector hvx_vec_neg_f16(HVX_Vector v) {
    // neg by setting the fp16 sign bit
    HVX_Vector mask = Q6_Vh_vsplat_R(0x8000);
    return Q6_V_vxor_VV(v, mask);
}

static inline HVX_Vector hvx_vec_abs_f32(HVX_Vector v) {
    // abs by clearing the fp32 sign bit
    HVX_Vector mask = Q6_V_vsplat_R(0x7fffffff);
    return Q6_V_vand_VV(v, mask);
}

static inline HVX_Vector hvx_vec_neg_f32(HVX_Vector v) {
#if __HVX_ARCH__ > 75
    return Q6_Vsf_vfneg_Vsf(v);
#else
    // neg by setting the fp32 sign bit
    HVX_Vector mask = Q6_V_vsplat_R(0x80000000);
    return Q6_V_vxor_VV(v, mask);
#endif  // __HVX_ARCH__ > 75
}

static inline HVX_VectorPred hvx_vec_is_nan_f16(HVX_Vector v) {
    const HVX_Vector vnan_exp  = Q6_Vh_vsplat_R(0x7C00);
    const HVX_Vector vnan_frac = Q6_Vh_vsplat_R(0x7FFF);

    // get pred of which are NaN, i.e., exponent bits all 1s and fraction bits non 0s
    HVX_VectorPred p_exp  = Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, vnan_exp), vnan_exp);
    HVX_VectorPred p_frac = Q6_Q_not_Q(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, vnan_frac), vnan_exp));
    return Q6_Q_and_QQ(p_exp, p_frac);
}

static inline HVX_Vector hvx_vec_f32_to_f16(HVX_Vector v0, HVX_Vector v1) {
    const HVX_Vector zero = Q6_V_vsplat_R(0);
    HVX_Vector q0 = Q6_Vqf32_vadd_VsfVsf(v0, zero);
    HVX_Vector q1 = Q6_Vqf32_vadd_VsfVsf(v1, zero);
    HVX_Vector  v = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0)));

#if __HVX_ARCH__ < 79
    // replace NaNs with -INF, older arches produce NaNs for (-INF + 0.0)
    const HVX_Vector neg_inf = hvx_vec_splat_f16(-INFINITY);
    HVX_VectorPred nan = hvx_vec_is_nan_f16(v);
    v = Q6_V_vmux_QVV(nan, neg_inf, v);
#endif

    return v;
}

/* Q6_Vsf_equals_Vw is only available on v73+.*/
#if __HVX_ARCH__ < 73
static inline HVX_Vector hvx_vec_i32_to_qf32(HVX_Vector const in)
{
    HVX_Vector const vzero = Q6_V_vzero();
    HVX_VectorPred is_zero = Q6_Q_vcmp_eq_VwVw(in, vzero);
    HVX_Vector lshift = Q6_Vw_vnormamt_Vw(in);
    HVX_Vector normalized = Q6_Vw_vasl_VwVw(in, lshift);
    HVX_Vector vexp = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7f + 30), lshift);
    HVX_Vector mant = Q6_V_vand_VV(Q6_V_vsplat_R(0xFFFFFF00), normalized);
    HVX_Vector ret = Q6_V_vmux_QVV(is_zero, vzero, Q6_Vw_vadd_VwVw(mant, vexp));
    return ret;
}

static inline HVX_Vector Q6_Vsf_equals_Vw(HVX_Vector const in)
{
    return Q6_Vsf_equals_Vqf32(hvx_vec_i32_to_qf32(in));
}
#endif

static inline HVX_Vector hvx_vec_i16_from_hf_rnd_sat(HVX_Vector vin) {
    // This looks complicated.
    // Ideally should just be Q6_Vh_equals_Vhf(vin)
    // but that instruction does not do proper rounding.

    // convert to qf32, multiplying by 1.0 in the process.
    HVX_VectorPair v32 = Q6_Wqf32_vmpy_VhfVhf(vin, Q6_Vh_vsplat_R(0x3C00));

    // 'in-range' values are +/32752.
    // add 192K to it, convert to sf
    HVX_Vector v192K = Q6_V_vsplat_R(0x48400000);
    HVX_Vector vsf_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(v32), v192K));
    HVX_Vector vsf_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(v32), v192K));

    // for in-range cases, result is {163858... 229360} so the exponent is always 144.
    // if we extract bits 21..0 as a signed quantity, and round 6 bits off, that will be the answer.
    // Start by <<10 to get the final 'sign' bit in bit 15...
    vsf_0 = Q6_Vw_vasl_VwR(vsf_0, 10);
    vsf_1 = Q6_Vw_vasl_VwR(vsf_1, 10);

    // now round down to 16
    return Q6_Vh_vround_VwVw_sat(vsf_1, vsf_0);
}

#endif /* HVX_BASE_H */
