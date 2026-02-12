// HVX internal helpers used by HMX infrastructure (math, conversion, etc.).
// Ported from htp-ops-lib/include/dsp/hvx_internal.h.
//
// Provides HVX_DV union, vmem macro, VLEN constants, floor/ceil/truncate
// functions, and vqf16_from_int / vqf32_from_int converters that are
// prerequisites for hmx-hvx-math.h and hmx-hvx-convert.h.

#ifndef HMX_HVX_INTERNAL_H
#define HMX_HVX_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>

#define hmx_vmem(A)   *((HVX_Vector *)(A))
#define hmx_vmemu(A)  *((HVX_UVector *)(A))

#define HMX_HVX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

#ifndef HMX_LOG2VLEN
#define HMX_LOG2VLEN 7
#endif
#define HMX_VLEN       (1 << HMX_LOG2VLEN)       // 128 bytes
#define HMX_VLEN_SHORT ((1 << HMX_LOG2VLEN) >> 1) // 64 half-words
#define HMX_VLEN_WORD  ((1 << HMX_LOG2VLEN) >> 2) // 32 words

#define HMX_IEEE_VSF_EXPLEN    8
#define HMX_IEEE_VSF_EXPBIAS   127
#define HMX_IEEE_VSF_EXPMASK   0xFF
#define HMX_IEEE_VSF_MANTLEN   23
#define HMX_IEEE_VSF_MANTMASK  0x7FFFFF
#define HMX_IEEE_VSF_MIMPMASK  0x800000

#define HMX_IEEE_VHF_EXPLEN    5
#define HMX_IEEE_VHF_EXPBIAS   15
#define HMX_IEEE_VHF_EXPMASK   0x1F
#define HMX_IEEE_VHF_MANTLEN   10
#define HMX_IEEE_VHF_MANTMASK  0x3FF
#define HMX_IEEE_VHF_MIMPMASK  0x400

typedef union {
    HVX_VectorPair VV;
    struct {
        HVX_Vector lo;
        HVX_Vector hi;
    } V;
} hmx_HVX_DV;

static HMX_HVX_INLINE_ALWAYS void hmx_l2fetch(const void *p, uint32_t stride,
                                               uint32_t width, uint32_t height,
                                               uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__(" l2fetch(%0,%1) " : : "r"(p), "r"(control));
}

static HMX_HVX_INLINE_ALWAYS int32_t hmx_is_aligned(const void *addr, uint32_t align) {
    return ((size_t)addr & (align - 1)) == 0;
}

// Store first n bytes of vin to addr (arbitrary alignment).
static HMX_HVX_INLINE_ALWAYS void hmx_vstu_variable(void *addr, uint32_t n,
                                                     HVX_Vector vin) {
    vin = Q6_V_vlalign_VVR(vin, vin, (size_t)addr);

    uint32_t left_off  = (size_t)addr & 127;
    uint32_t right_off = left_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t)addr);
    HVX_VectorPred qr     = Q6_Q_vsetq2_R(right_off);

    if (right_off > 128) {
        Q6_vmem_QRIV(qr, (HVX_Vector *)addr + 1, vin);
        qr = Q6_Q_vcmp_eq_VbVb(vin, vin); // all 1's
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector *)addr, vin);
}

static HMX_HVX_INLINE_ALWAYS void hmx_vstdu_variable(void *addr, uint32_t n,
                                                      HVX_VectorPair vin) {
    hmx_vstu_variable(addr, n > 128 ? 128 : n, Q6_V_lo_W(vin));
    if (n > 128) {
        hmx_vstu_variable((HVX_Vector *)addr + 1, n - 128, Q6_V_hi_W(vin));
    }
}

// 32×32 fractional multiply (expands to two ops).
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_Vw_vmpy_VwVw_s1_rnd_sat(HVX_Vector vu,
                                                                      HVX_Vector vv) {
    return Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(vu, vv), vu, vv);
}

static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_Vw_vmpy_VwVw_s1_sat(HVX_Vector vu,
                                                                  HVX_Vector vv) {
    return Q6_Vw_vmpyoacc_VwVwVh_s1_sat_shift(Q6_Vw_vmpye_VwVuh(vu, vv), vu, vv);
}

static HMX_HVX_INLINE_ALWAYS HVX_VectorPair hmx_W_vmpy_VwVw(HVX_Vector vu,
                                                              HVX_Vector vv) {
    return Q6_W_vmpyoacc_WVwVh(Q6_W_vmpye_VwVuh(vu, vv), vu, vv);
}

static HMX_HVX_INLINE_ALWAYS uint16_t hmx_fp16_to_bits(__fp16 *x) {
    union { __fp16 f; uint16_t i; } fp16 = { .f = *x };
    return fp16.i;
}

// --- floor with return of both integer and float result ---

static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_Vw_vfloor_VsfVsf(HVX_Vector Vu, HVX_Vector *Vd) {
    HVX_Vector o_i_v, o_f_v, expval_v, mantissa_v, mantissa_shift_v, mask;
    HVX_Vector const_zero_v = Q6_V_vzero();
    HVX_Vector const_v;
    HVX_Vector round_f = Vu;

    const_v = Q6_V_vsplat_R(0x7FFFFFFF);
    HVX_VectorPred qpred_negative_vq = Q6_Q_vcmp_gt_VuwVuw(round_f, const_v);

    expval_v = round_f >> 23;
    expval_v &= 0xFF;
    expval_v -= 127;

    HVX_VectorPred qpred_negativexp_vq = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    expval_v = Q6_Vw_vmax_VwVw(expval_v, const_zero_v);

    mantissa_shift_v = 23 - expval_v;
    mantissa_shift_v = Q6_Vw_vmax_VwVw(mantissa_shift_v, const_zero_v);

    mantissa_v = round_f;
    mantissa_v &= ((1 << 23) - 1);
    mantissa_v >>= mantissa_shift_v;

    o_i_v = 1 << expval_v;
    o_i_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, o_i_v);
    o_i_v += mantissa_v;

    HVX_Vector negative_i_v = -o_i_v;
    o_i_v = Q6_V_vmux_QVV(qpred_negative_vq, negative_i_v, o_i_v);

    mask = (1 << mantissa_shift_v);
    mask = -mask;
    round_f &= mask;

    o_f_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, round_f);

    *Vd = o_f_v;
    return o_i_v;
}

static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_Vh_vfloor_VhfVhf(HVX_Vector Vu, HVX_Vector *Vd) {
    HVX_Vector o_i_v, o_f_v, expval_v, mantissa_v, mantissa_shift_v, mask;
    HVX_Vector const_zero_v = Q6_V_vzero();
    HVX_Vector const_v;
    HVX_Vector round_f = Vu;

    const_v = Q6_Vh_vsplat_R(0x7FFF);
    HVX_VectorPred qpred_negative_vq = Q6_Q_vcmp_gt_VuhVuh(round_f, const_v);

    expval_v = Q6_Vuh_vlsr_VuhR(round_f, 10);
    const_v  = Q6_Vh_vsplat_R(0x1F);
    expval_v = Q6_V_vand_VV(expval_v, const_v);
    const_v  = Q6_Vh_vsplat_R(0x000F); // 15
    expval_v = Q6_Vh_vsub_VhVh(expval_v, const_v);

    HVX_VectorPred qpred_negativexp_vq = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);
    expval_v = Q6_Vh_vmax_VhVh(expval_v, const_zero_v);

    const_v          = Q6_Vh_vsplat_R(0x000A); // 10
    mantissa_shift_v = Q6_Vh_vsub_VhVh(const_v, expval_v);
    mantissa_shift_v = Q6_Vh_vmax_VhVh(mantissa_shift_v, const_zero_v);

    mantissa_v = round_f;
    const_v    = Q6_Vh_vsplat_R(0x03FF);
    mantissa_v = Q6_V_vand_VV(mantissa_v, const_v);
    mantissa_v = Q6_Vh_vlsr_VhVh(mantissa_v, mantissa_shift_v);

    const_v = Q6_Vh_vsplat_R(0x0001);
    o_i_v   = Q6_Vh_vasl_VhVh(const_v, expval_v);
    o_i_v   = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, o_i_v);
    o_i_v   = Q6_Vh_vadd_VhVh(o_i_v, mantissa_v);

    HVX_Vector negative_i_v = Q6_Vh_vsub_VhVh(const_zero_v, o_i_v);
    o_i_v = Q6_V_vmux_QVV(qpred_negative_vq, negative_i_v, o_i_v);

    mask    = Q6_Vh_vasl_VhVh(Q6_Vh_vsplat_R(0x0001), mantissa_shift_v);
    mask    = Q6_Vh_vsub_VhVh(const_zero_v, mask);
    round_f = Q6_V_vand_VV(round_f, mask);

    o_f_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, round_f);

    *Vd = o_f_v;
    return o_i_v;
}

// --- integer ↔ qf conversions ---

static inline HVX_Vector hmx_vqf32_from_int(HVX_Vector src) {
    HVX_Vector const_126 = Q6_V_vsplat_R(0x0000007e);
    HVX_Vector const31   = Q6_V_vsplat_R(31);
    HVX_Vector mant = src;
    HVX_Vector exp  = Q6_Vw_vnormamt_Vw(mant);
    mant = Q6_Vw_vasl_VwVw(mant, exp);
    exp  = Q6_Vw_vsub_VwVw(const31, exp);
    exp  = Q6_Vw_vadd_VwVw(exp, const_126);
    return Q6_V_vor_VV(mant, exp);
}

static inline HVX_Vector hmx_vqf16_from_int(HVX_Vector src) {
    HVX_Vector const_14 = Q6_Vh_vsplat_R(0x000e);
    HVX_Vector const15  = Q6_Vh_vsplat_R(15);
    HVX_Vector mant = src;
    HVX_Vector exp  = Q6_Vh_vnormamt_Vh(mant);
    mant = Q6_Vh_vasl_VhVh(mant, exp);
    exp  = Q6_Vh_vsub_VhVh(const15, exp);
    exp  = Q6_Vh_vadd_VhVh(exp, const_14);
    return Q6_V_vor_VV(mant, exp);
}

#endif // HMX_HVX_INTERNAL_H
