#ifndef HVX_FA_KERNELS_H
#define HVX_FA_KERNELS_H

#include <assert.h>
#include <math.h>
#include "hvx-utils.h"

// Little inner kernels for HVX

#if __HVX_ARCH__ < 79
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
static __attribute__((unused)) __attribute__((noinline)) void hvx_vec_f32_to_f16_a(void *ptr, HVX_Vector v0, HVX_Vector v1)
{
    *(HVX_Vector *) ptr = hvx_vec_f32_to_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
static inline void hvx_dot_f16_f16_aa(float * restrict r, const void * restrict x, const void * restrict y, unsigned int n, float s) {
    const HVX_Vector * restrict vx = (const HVX_Vector * restrict) x; // fp16
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_VectorPair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_Vector rsum = HVX_OP_ADD_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx4(const void * restrict y,
                                                const uint8_t * restrict x,
                                                const size_t stride_x,
                                                const size_t nvec,
                                                const size_t nloe) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) x;                   // fp16
    const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) (x + stride_x);      // fp16
    const HVX_Vector * restrict vx2 = (const HVX_Vector * restrict) (x + stride_x * 2);  // fp16
    const HVX_Vector * restrict vx3 = (const HVX_Vector * restrict) (x + stride_x * 3);  // fp16
    const HVX_Vector * restrict vy  = (const HVX_Vector * restrict) y;                   // fp16

    HVX_VectorPair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf  = vy[i];
        HVX_Vector x0_hf = vx0[i];
        HVX_Vector x1_hf = vx1[i];
        HVX_Vector x2_hf = vx2[i];
        HVX_Vector x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    if (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_Vector     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_Vector     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_Vector     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_Vector rsum0 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_Vector rsum1 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_Vector rsum2 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_Vector rsum3 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_Vector_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    return hvx_vec_reduce_sum_f32x4(rsum0123);
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx32(const void * restrict y,
                                                 const uint8_t * restrict x,
                                                 const size_t stride_x,
                                                 const size_t n,
                                                 float        s) {

    const size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    const size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector   sums = Q6_V_vzero();
    const size_t stride_x_4 = stride_x * 4;
    for (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_Vector     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_VectorPred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    return HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
static inline void hvx_mad_f32_f16_aa(float * restrict y, const void * restrict x, const __fp16 * restrict s, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x;

    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
static inline void hvx_mad_f32_f16_aa_rx2(float * restrict y, const void * restrict x0, const void * restrict x1,
                                          const __fp16 * restrict s0, const __fp16 * restrict s1, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict vx1 = (const HVX_Vector *) x1;

    HVX_VectorPair * restrict vy_p  = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s0);
    HVX_Vector S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

static inline void hvx_scale_vec_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const uint32_t n, HVX_Vector vs) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);

    const HVX_Vector * restrict vsrc = (const HVX_Vector * restrict) src;
    HVX_Vector * restrict vdst       = (HVX_Vector * restrict) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #pragma unroll(4)
    for (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

// 5th-order Horner polynomial for exp2(x) in qf16/hf16 domain.  Input must be
// <= 0 (safe softmax invariant - overflow handling omitted).  ~18 ALU ops per
// 64 fp16 lanes, fully parallel across HVX threads (no scatter/gather engine).
// Replaces the F32 round-trip (qf16->f32->exp->f32->f16, ~44 ops for 2x32 lanes).
static inline HVX_Vector hvx_exp2_hf(HVX_Vector x_v) {
    const HVX_Vector zero_v    = Q6_V_vzero();
    const HVX_Vector half_hf_v = Q6_Vh_vsplat_R(0x3800);  // fp16 0.5

    // k = round_toward_neg_inf(x);  f = (float)k;  frac = x - f
    HVX_Vector x_minus_half = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x_v, half_hf_v));
    HVX_Vector k_v          = Q6_Vh_equals_Vhf(x_minus_half);  // truncate to int16
    HVX_Vector f_v          = Q6_Vhf_equals_Vh(k_v);           // back to fp16

    HVX_Vector x_qf16 = Q6_Vqf16_vsub_VhfVhf(x_v, f_v);        // fractional part in qf16

    // Horner: y = ((((E5*x + E4)*x + E3)*x + E2)*x + E1)*x + E0
    HVX_Vector y = Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vh_vsplat_R(0x5082), x_qf16); // E5*x
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x157d));        // + E4
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x20ed));        // + E3
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x2b1b));        // + E2
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x33b0));        // + E1
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x398c));        // + E0
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);                      // y = y * x
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x3c00));        // + 1.0

    // Combine polynomial (mantissa) with integer part (exponent): result = y * 2^k
    y                          = Q6_Vhf_equals_Vqf16(y);
    HVX_Vector y_exp           = Q6_Vuh_vlsr_VuhR(Q6_Vh_vasl_VhR(y, 1), 11);
    y_exp                      = Q6_Vh_vadd_VhVh(k_v, y_exp);
    HVX_VectorPred q_underflow = Q6_Q_vcmp_gt_VhVh(zero_v, y_exp);
    y                          = Q6_Vh_vaslacc_VhVhR(y, k_v, 10);
    return Q6_V_vmux_QVV(q_underflow, zero_v, y);
}

#endif /* HVX_FA_KERNELS_H */
