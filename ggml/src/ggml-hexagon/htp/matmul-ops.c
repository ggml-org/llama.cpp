#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hvx-dump.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "matmul-ops.h"
#include "vtcm-utils.h"
#include "hmx-ops.h"

static inline HVX_Vector hvx_vec_f16_to_f32_lower32(HVX_Vector v) {
    HVX_VectorPair pair = hvx_vec_f16_to_f32_shuff(v);
    return Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(pair), Q6_V_lo_W(pair), -4));
}

static inline HVX_Vector hvx_vec_mul_f16_f16_to_f32_lower32(HVX_Vector v1, HVX_Vector v2) {
#if __HVX_ARCH__ >= 79
    HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    return Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#else
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    return Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#endif
}

struct htp_mm_context {
    const char * type;
    struct htp_ops_context * octx;

    void (*vec_dot_1x1)(const int n, float * restrict s0,
         const void * restrict vx0,
         const void * restrict vy0);

    void (*vec_dot_2x1)(const int n, float * restrict s0,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0);

    void (*vec_dot_2x2)(const int n, float * restrict s0, float * restrict s1,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0, const void * restrict vy1);

    void (*vec_dot_32x1)(const int n, float * restrict s,
         const void * restrict vx,
         const void * restrict vy, int valid_rows);


    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src1_nrows_per_thread;

    struct fastdiv_values mm_div_ne12_ne1;
    struct fastdiv_values mm_div_ne1;
    struct fastdiv_values mm_div_r2;
    struct fastdiv_values mm_div_r3;
    struct fastdiv_values mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t quant_ib_first[MAX_NUM_WORKERS];
    uint32_t quant_ib_last[MAX_NUM_WORKERS];
    uint32_t quant_r[MAX_NUM_WORKERS];
    uint32_t quant_c[MAX_NUM_WORKERS];

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    const uint32_t * matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows;

    // Dynamic VTCM pointers allocated sequentially
    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_src2;
    uint8_t * vtcm_src3;
    uint8_t * vtcm_dst;

    // Cached strides
    uint32_t vtcm_src0_stride;
    uint32_t vtcm_src1_stride;
    uint32_t vtcm_src2_stride;
    uint32_t vtcm_src3_stride;
    uint32_t vtcm_dst_stride;

    // Cached thread offsets/sizes
    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_src2_size_per_thread;
    uint32_t vtcm_src3_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    bool hmx_eligible;
};

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

// IQ4_NL dequantization LUT: maps 4-bit index (0-15) to int8 kvalue
// kvalues: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
static const uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};



#if __HVX_ARCH__ < 79
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

static void vec_dot_f32_f32_aa_1x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const HVX_Vector * restrict x = (const HVX_Vector *) vx;
    const HVX_Vector * restrict y = (const HVX_Vector *) vy;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_Vector rsum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_Vector prod = HVX_OP_MUL_F32(x[i], y[i]);
        rsum = HVX_OP_ADD_F32(rsum, prod);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector x_sf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_sf = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector prod = HVX_OP_MUL_F32(x_sf, y_sf);
        rsum = HVX_OP_ADD_F32(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
}

static void vec_dot_f32_f32_aa_2x1(const int n, float * restrict s0,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y  = (const HVX_Vector *) vy0;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_Vector rsum0 = Q6_V_vzero();
    HVX_Vector rsum1 = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_sf = y[i];
        HVX_Vector prod0 = HVX_OP_MUL_F32(x0[i], y_sf);
        HVX_Vector prod1 = HVX_OP_MUL_F32(x1[i], y_sf);
        rsum0 = HVX_OP_ADD_F32(rsum0, prod0);
        rsum1 = HVX_OP_ADD_F32(rsum1, prod1);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector y_sf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector x0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector x1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector prod0 = HVX_OP_MUL_F32(x0_sf, y_sf);
        HVX_Vector prod1 = HVX_OP_MUL_F32(x1_sf, y_sf);
        rsum0 = HVX_OP_ADD_F32(rsum0, prod0);
        rsum1 = HVX_OP_ADD_F32(rsum1, prod1);
    }

    HVX_Vector rsum = hvx_vec_reduce_sum_f32x2(rsum0, rsum1);
    HVX_VectorAlias va;
    va.v = rsum;
    s0[0] = va.fp32[0];
    s0[1] = va.fp32[1];
}

static void vec_dot_f32_f32_aa_2x2(const int n, float * restrict s0, float * restrict s1,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0, const void * restrict vy1) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y0 = (const HVX_Vector *) vy0;
    const HVX_Vector * restrict y1 = (const HVX_Vector *) vy1;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_Vector r0_c0_sum = Q6_V_vzero();
    HVX_Vector r0_c1_sum = Q6_V_vzero();
    HVX_Vector r1_c0_sum = Q6_V_vzero();
    HVX_Vector r1_c1_sum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector r0_sf = x0[i];
        HVX_Vector r1_sf = x1[i];
        HVX_Vector c0_sf = y0[i];
        HVX_Vector c1_sf = y1[i];

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);

        HVX_Vector r0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector r1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector c0_sf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_Vector c1_sf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    // Reduce and store results
    HVX_Vector r0_r1_c0_sum = hvx_vec_reduce_sum_f32x2(r0_c0_sum, r1_c0_sum);
    HVX_Vector r0_r1_c1_sum = hvx_vec_reduce_sum_f32x2(r0_c1_sum, r1_c1_sum);

    HVX_VectorAlias va0, va1;
    va0.v = r0_r1_c0_sum;
    va1.v = r0_r1_c1_sum;
    s0[0] = va0.fp32[0];
    s0[1] = va0.fp32[1];
    s1[0] = va1.fp32[0];
    s1[1] = va1.fp32[1];
}

static void vec_dot_f32_f32_uu_1x1(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x;
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_Vector       rsum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector x_sf = vx[i];
        HVX_Vector y_sf = vy[i];

        rsum = HVX_OP_ADD_F32(rsum, HVX_OP_MUL_F32(x_sf, y_sf));
    }

    if (nloe) {
        HVX_Vector x_sf = vx[i];
        HVX_Vector y_sf = vy[i];

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        x_sf = Q6_V_vand_QV(bmask, x_sf);
        y_sf = Q6_V_vand_QV(bmask, y_sf);

        rsum = HVX_OP_ADD_F32(rsum, HVX_OP_MUL_F32(x_sf, y_sf));
    }

    rsum = hvx_vec_reduce_sum_f32(rsum);
    hvx_vec_store_u(&s[0], 4, rsum);
}

static void vec_dot_f16_f16_aa_1x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const HVX_Vector * restrict x = (const HVX_Vector *) vx;
    const HVX_Vector * restrict y = (const HVX_Vector *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_VectorPair rsum_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x[i], y[i]);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, y[i]);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_Vector rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    hvx_vec_store_u(s, 4, hvx_vec_reduce_sum_f32(rsum));
}

static void vec_dot_f16_f16_aa_2x1(const int n, float * restrict s0,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y  = (const HVX_Vector *) vy0;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_VectorPair rsum0_p = Q6_W_vzero();
    HVX_VectorPair rsum1_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf = y[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0[i], y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1[i], y_hf);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
    }

    HVX_Vector rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_Vector rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    HVX_Vector rsum  = hvx_vec_reduce_sum_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

static void vec_dot_f16_f16_aa_2x2(const int n, float * restrict s0, float * restrict s1,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0, const void * restrict vy1) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y0 = (const HVX_Vector *) vy0;
    const HVX_Vector * restrict y1 = (const HVX_Vector *) vy1;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    // Row sums (sf) - 4 accumulators for 2×2 tile
    HVX_VectorPair r0_c0_sum_p = Q6_W_vzero();
    HVX_VectorPair r0_c1_sum_p = Q6_W_vzero();
    HVX_VectorPair r1_c0_sum_p = Q6_W_vzero();
    HVX_VectorPair r1_c1_sum_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector r0_hf = x0[i];
        HVX_Vector r1_hf = x1[i];
        HVX_Vector c0_hf = y0[i];
        HVX_Vector c1_hf = y1[i];

        // Compute 4 dot products: r0×c0, r0×c1, r1×c0, r1×c1
        r0_c0_sum_p = hvx_vec_mpyacc_f32_f16(r0_c0_sum_p, r0_hf, c0_hf);
        r0_c1_sum_p = hvx_vec_mpyacc_f32_f16(r0_c1_sum_p, r0_hf, c1_hf);
        r1_c0_sum_p = hvx_vec_mpyacc_f32_f16(r1_c0_sum_p, r1_hf, c0_hf);
        r1_c1_sum_p = hvx_vec_mpyacc_f32_f16(r1_c1_sum_p, r1_hf, c1_hf);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_Vector r0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector r1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector c0_hf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_Vector c1_hf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_sum_p = hvx_vec_mpyacc_f32_f16(r0_c0_sum_p, r0_hf, c0_hf);
        r0_c1_sum_p = hvx_vec_mpyacc_f32_f16(r0_c1_sum_p, r0_hf, c1_hf);
        r1_c0_sum_p = hvx_vec_mpyacc_f32_f16(r1_c0_sum_p, r1_hf, c0_hf);
        r1_c1_sum_p = hvx_vec_mpyacc_f32_f16(r1_c1_sum_p, r1_hf, c1_hf);
    }

    HVX_Vector r0_c0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c0_sum_p), Q6_V_hi_W(r0_c0_sum_p)));
    HVX_Vector r0_c1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c1_sum_p), Q6_V_hi_W(r0_c1_sum_p)));
    HVX_Vector r1_c0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c0_sum_p), Q6_V_hi_W(r1_c0_sum_p)));
    HVX_Vector r1_c1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c1_sum_p), Q6_V_hi_W(r1_c1_sum_p)));

    // Reduce and store results
    HVX_Vector r0_r1_c0_sum = hvx_vec_reduce_sum_f32x2(r0_c0_sum, r1_c0_sum);
    HVX_Vector r0_r1_c1_sum = hvx_vec_reduce_sum_f32x2(r0_c1_sum, r1_c1_sum);

    hvx_vec_store_u(&s0[0], 8, r0_r1_c0_sum);  // row0,col0 row1,col0
    hvx_vec_store_u(&s1[0], 8, r0_r1_c1_sum);  // row0,col1 row1,col1
}

static void vec_dot_f16_f16_uu_1x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const HVX_UVector * restrict x = (const HVX_UVector *) vx;
    const HVX_UVector * restrict y = (const HVX_UVector *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector rsum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x[i], y[i]);
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, y[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(rsum));
    hvx_vec_store_u(&s[0], 4, rsum);
}

static void vec_dot_f16_f32_uu_1x1(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x;
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vzero();

    HVX_Vector       rsum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf)));

        // Load x (fp16)
        HVX_Vector x_hf  = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf)));

        // Load x (fp16)
        HVX_Vector x_hf  = vx[i];

        // Zero-out unused elements
        // Note that we need to clear both x and y because they may contain NANs
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        x_hf = Q6_V_vand_QV(bmask, x_hf);
        y_hf = Q6_V_vand_QV(bmask, y_hf);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    // Convert into fp32 and reduce
    rsum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(rsum));
    hvx_vec_store_u(&s[0], 4, rsum);
}

#define htp_matmul_tensors_preamble                             \
    const struct htp_tensor * restrict src0 = octx->src[0];     \
    const struct htp_tensor * restrict src1 = octx->src[1];     \
    const struct htp_tensor * restrict src2 = octx->src[2];     \
    const struct htp_tensor * restrict  dst = octx->dst;        \
                                                                \
    const uint32_t ne00 = src0->ne[0];                          \
    const uint32_t ne01 = src0->ne[1];                          \
    const uint32_t ne02 = src0->ne[2];                          \
    const uint32_t ne03 = src0->ne[3];                          \
                                                                \
    const uint32_t ne10 = src1->ne[0];                          \
    const uint32_t ne11 = src1->ne[1];                          \
    const uint32_t ne12 = src1->ne[2];                          \
    const uint32_t ne13 = src1->ne[3];                          \
                                                                \
    const uint32_t ne20 = src2->ne[0];                          \
    const uint32_t ne21 = src2->ne[1];                          \
    const uint32_t ne22 = src2->ne[2];                          \
    const uint32_t ne23 = src2->ne[3];                          \
                                                                \
    const uint32_t ne0 = dst->ne[0];                            \
    const uint32_t ne1 = dst->ne[1];                            \
    const uint32_t ne2 = dst->ne[2];                            \
    const uint32_t ne3 = dst->ne[3];                            \
                                                                \
    const uint32_t nb00 = src0->nb[0];                          \
    const uint32_t nb01 = src0->nb[1];                          \
    const uint32_t nb02 = src0->nb[2];                          \
    const uint32_t nb03 = src0->nb[3];                          \
                                                                \
    const uint32_t nb10 = src1->nb[0];                          \
    const uint32_t nb11 = src1->nb[1];                          \
    const uint32_t nb12 = src1->nb[2];                          \
    const uint32_t nb13 = src1->nb[3];                          \
                                                                \
    const uint32_t nb0 = dst->nb[0];                            \
    const uint32_t nb1 = dst->nb[1];                            \
    const uint32_t nb2 = dst->nb[2];                            \
    const uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                             \
    struct htp_mm_context * mmctx = data;                           \
    struct htp_ops_context * octx  = mmctx->octx;                       \
    htp_matmul_tensors_preamble;                                        \
    dma_queue *dma_queue           = octx->ctx->dma[ith];               \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;

// *** matmul with support for 4d tensors and full broadcasting

static void matmul_4d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const uint32_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const uint32_t nr1 = ne1 * ne2 * ne3;

    // distribute the thread work across the inner or outer loop based on which one is larger
    uint32_t nchunk0 = nr0 > nr1 ? nth : 1;  // parallelize by src0 rows
    uint32_t nchunk1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows

    // The number of elements in each chunk
    const uint32_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const uint32_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    uint32_t current_chunk = ith;

    const uint32_t ith0 = current_chunk % nchunk0;
    const uint32_t ith1 = current_chunk / nchunk0;

    const uint32_t ir0_start = dr0 * ith0;
    const uint32_t ir0_end   = MIN(ir0_start + dr0, nr0);

    const uint32_t ir1_start = dr1 * ith1;
    const uint32_t ir1_end   = MIN(ir1_start + dr1, nr1);

    // no work for this thread
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);

    // block-tiling attempt
    const uint32_t blck_0 = 64;
    const uint32_t blck_1 = 64;

    for (uint32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (uint32_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (uint32_t ir1 = iir1; ir1 < MIN(iir1 + blck_1, ir1_end); ir1++) {
                const uint32_t i13 = fastdiv(ir1, &mmctx->mm_div_ne12_ne1);
                const uint32_t i12 = fastdiv(ir1 - i13 * ne12 * ne1, &mmctx->mm_div_ne1);
                const uint32_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const uint32_t i03 = fastdiv(i13, &mmctx->mm_div_r3);
                const uint32_t i02 = fastdiv(i12, &mmctx->mm_div_r2);

                const uint32_t i1 = i11;
                const uint32_t i2 = i12;
                const uint32_t i3 = i13;

                const uint8_t * restrict src0_base = (const uint8_t *) src0->data + (0 + i02 * nb02 + i03 * nb03);
                const uint8_t * restrict src1_col  = (const uint8_t *) src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13);
                float * dst_col = (float *) ((uint8_t * restrict) dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                const uint32_t ir0_block_end = MIN(iir0 + blck_0, ir0_end);
                for (uint32_t ir0 = iir0; ir0 < ir0_block_end; ir0++) {
                    const uint8_t * restrict src0_row = src0_base + ir0 * nb01;
                    mmctx->vec_dot_1x1(ne00, &dst_col[ir0], src0_row, src1_col);
                }
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-4d %d/%d: %ux%ux%ux%u (%u:%u %u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0_start, ir0_end, ir1_start, ir1_end, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);
}

static inline HVX_Vector unpack_and_interleave_4bit(HVX_Vector v_a, HVX_Vector v_b, HVX_Vector mask_h4) {
    HVX_Vector v_W0 = Q6_V_vand_VV(v_a, mask_h4);
    HVX_Vector v_W1 = Q6_Vub_vlsr_VubR(v_a, 4);
    HVX_Vector v_W2 = Q6_V_vand_VV(v_b, mask_h4);
    HVX_Vector v_W3 = Q6_Vub_vlsr_VubR(v_b, 4);

    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_W1, v_W0, -1);
    HVX_VectorPair v23_pair = Q6_W_vshuff_VVR(v_W3, v_W2, -1);
    HVX_VectorPair v0123_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(v23_pair), Q6_V_lo_W(v01_pair), -2);
    return Q6_V_lo_W(v0123_pair);
}

static inline HVX_VectorPair unpack_and_interleave_4bit_x2(HVX_Vector v_src, HVX_Vector mask_h4) {
    HVX_Vector v_lo = Q6_V_vand_VV(v_src, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(v_src, 4);
    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);

    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    return Q6_W_vcombine_VV(v_W1, v_W0);
}

static inline HVX_Vector accum_4bit_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_Vector accum_4bit_32x1_lut(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_VectorPair accum_4bit_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static inline HVX_VectorPair accum_4bit_32x2_lut(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static inline HVX_Vector accum_q8_0_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act
) {
    HVX_Vector v_sum = Q6_V_vzero();
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum = Q6_Vw_vrmpyacc_VwVbVb(v_sum, v_W, v_act[g]);
    }
    return v_sum;
}

static inline HVX_VectorPair accum_q8_0_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W, v_act1[g]);
    }
    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

#include "matmul-kernels-tiled.h"
#include "matmul-kernels-flat.h"

// Specialized repacked matmul macros
#define MATMUL_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                                \
static void matmul_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                          \
    htp_matmul_preamble;                                                                                                            \
                                                                                                                                    \
    const uint32_t src0_nrows = ne01 * ne02 * ne03;                                                                                 \
    const uint32_t src1_nrows = ne11 * ne12 * ne13;                                                                                 \
                                                                                                                                    \
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;                                                                   \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                                       \
                                                                                                                                    \
    if (src0_start_row >= src0_end_row) {                                                                                           \
        return;                                                                                                                     \
    }                                                                                                                               \
                                                                                                                                    \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                                       \
                                                                                                                                    \
    const size_t dst_row_size  = nb1;                                                                                               \
    const size_t src1_row_size = nb11;                                                                                              \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                             \
                                                                                                                                    \
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;                                            \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                            \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                                \
                                                                                                                                    \
    uint64_t t1, t2;                                                                                                                \
    t1 = HAP_perf_get_qtimer_count();                                                                                               \
                                                                                                                                    \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                               \
                                                                                                                                    \
    const uint32_t tile_size = TILE_SIZE;                                                                                           \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                                \
                                                                                                                                    \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                               \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                               \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                             \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                      \
                                                                                                                                    \
    uint32_t ct_start = src0_start_row / 32;                                                                                        \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                   \
                                                                                                                                    \
    uint32_t push_ct = ct_start;                                                                                                    \
    for (uint32_t d = 0; d < HTP_MM_DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {                                                       \
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                                      \
                       src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                \
    }                                                                                                                               \
                                                                                                                                    \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                               \
        const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                      \
                                                                                                                                    \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                                 \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                   \
                                                                                                                                    \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                     \
        uint32_t ir1 = 0;                                                                                                           \
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                    \
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                             \
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                             \
            float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                           \
            float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                           \
                                                                                                                                    \
            float * dst_ptr0 = &dst_row0[ct * 32];                                                                                  \
            float * dst_ptr1 = &dst_row1[ct * 32];                                                                                  \
                                                                                                                                    \
            DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows);                                            \
        }                                                                                                                           \
                                                                                                                                    \
        for (; ir1 < src1_nrows; ++ir1) {                                                                                           \
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                  \
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));                                       \
            float * dst_ptr = &dst_row[ct * 32];                                                                                    \
                                                                                                                                    \
            DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows);                                                                   \
        }                                                                                                                           \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                      \
                                                                                                                                    \
        if (push_ct < ct_end) {                                                                                                     \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),                        \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                   \
            push_ct++;                                                                                                              \
        }                                                                                                                           \
    }                                                                                                                               \
                                                                                                                                    \
    t2 = HAP_perf_get_qtimer_count();                                                                                               \
                                                                                                                                    \
    FARF(HIGH, "matmul-repacked-%s %u/%u: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mmctx->type, ith, nth,       \
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],                \
         src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],                                                  \
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));                                                                          \
}


#define MATVEC_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X1)                                                                         \
static void matvec_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                          \
    htp_matmul_preamble;                                                                                                            \
                                                                                                                                    \
    const uint32_t src0_nrows = ne01;                                                                                               \
                                                                                                                                    \
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;                                                                   \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                                       \
                                                                                                                                    \
    if (src0_start_row >= src0_end_row) {                                                                                           \
        return;                                                                                                                     \
    }                                                                                                                               \
                                                                                                                                    \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                                       \
                                                                                                                                    \
    const size_t dst_row_size  = nb1;                                                                                               \
    const size_t src1_row_size = nb11;                                                                                              \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                             \
                                                                                                                                    \
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;                                                       \
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                                     \
    uint8_t * src1_data = mmctx->vtcm_src1;                                                                                         \
                                                                                                                                    \
    volatile uint64_t t1, t2;                                                                                                       \
    t1 = HAP_perf_get_qtimer_count();                                                                                               \
                                                                                                                                    \
    float * tmp = (float *) vtcm_dst_ptr;                                                                                               \
                                                                                                                                    \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                               \
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;                                                                \
    float * restrict dst_col          = (float *) dst->data;                                                                        \
                                                                                                                                    \
    const uint32_t tile_size = TILE_SIZE;                                                                                           \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                                \
                                                                                                                                    \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                               \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                               \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                             \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                      \
                                                                                                                                    \
    uint32_t ct_start = src0_start_row / 32;                                                                                        \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                   \
                                                                                                                                    \
    uint32_t push_ct = ct_start;                                                                                                    \
    for (uint32_t d = 0; d < HTP_MM_DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {                                                       \
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                                      \
                       src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                \
    }                                                                                                                               \
                                                                                                                                    \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                               \
        const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                      \
                                                                                                                                    \
        float * dst_ptr = &tmp[ct * 32 - src0_start_row];                                                                           \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                                 \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                   \
                                                                                                                                    \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                     \
        DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows);                                                                       \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                      \
                                                                                                                                    \
        if (push_ct < ct_end) {                                                                                                     \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),                        \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                   \
            push_ct++;                                                                                                              \
        }                                                                                                                           \
    }                                                                                                                               \
                                                                                                                                    \
    int copy_cnt = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                                               \
    if (copy_cnt > 0) {                                                                                                             \
        hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);                                           \
    }                                                                                                                               \
                                                                                                                                    \
    t2 = HAP_perf_get_qtimer_count();                                                                                               \
                                                                                                                                    \
    FARF(HIGH, "matvec-repacked-%s %u/%u: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mmctx->type, ith, nth,       \
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],                \
         src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],                                                  \
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));                                                                          \
}


#define MATMUL_QKV_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                                \
static void matmul_qkv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                          \
    struct htp_mm_context * mmctx = data;                                                                                           \
    struct htp_ops_context * octx = mmctx->octx;                                                                                        \
                                                                                                                                        \
    const struct htp_tensor * restrict src0 = octx->src[0]; /* Wk */                                                                    \
    const struct htp_tensor * restrict src1 = octx->src[1]; /* x */                                                                     \
    const struct htp_tensor * restrict src2 = octx->src[2]; /* Wv */                                                                    \
    const struct htp_tensor * restrict src3 = octx->src[3]; /* Wq */                                                                    \
    const struct htp_tensor * restrict dst_k = octx->dsts[0];                                                                           \
    const struct htp_tensor * restrict dst_v = octx->dsts[1];                                                                           \
    const struct htp_tensor * restrict dst_q = octx->dsts[2];                                                                           \
                                                                                                                                        \
    const uint32_t ne00 = src0->ne[0];                                                                                                  \
    const uint32_t ne10 = src1->ne[0];                                                                                                  \
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];                                                                \
                                                                                                                                        \
    const size_t dst_row_size  = dst_k->nb[1];                                                                                          \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                                 \
                                                                                                                                        \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                                \
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;                                                \
    uint8_t * restrict vtcm_src3_ptr = mmctx->vtcm_src3 + mmctx->vtcm_src3_size_per_thread * ith;                                                \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                                    \
                                                                                                                                        \
    volatile uint64_t t1, t2;                                                                                                           \
    t1 = HAP_perf_get_qtimer_count();                                                                                                   \
                                                                                                                                        \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                                           \
                                                                                                                                        \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                                   \
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;                                                                   \
    const uint8_t * restrict src3_row = (const uint8_t *) src3->data;                                                                   \
                                                                                                                                        \
    const uint32_t tile_size = TILE_SIZE;                                                                                               \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                                    \
                                                                                                                                        \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                                   \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                                   \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                                 \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                          \
                                                                                                                                        \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                                        \
                                                                                                                                        \
    /* 1. Process K and V together */                                                                                                   \
    const uint32_t src0_nrows_kv = src0->ne[1] * src0->ne[2] * src0->ne[3]; /* src0 is Wk */                                            \
    uint32_t src0_nrows_per_thread_kv = (src0_nrows_kv + nth - 1) / nth;                                                                \
    src0_nrows_per_thread_kv = hex_round_up(src0_nrows_per_thread_kv, 32);                                                              \
                                                                                                                                        \
    const uint32_t start_row_kv = src0_nrows_per_thread_kv * ith;                                                                       \
    const uint32_t end_row_kv   = MIN(start_row_kv + src0_nrows_per_thread_kv, src0_nrows_kv);                                          \
                                                                                                                                        \
    if (start_row_kv < end_row_kv) {                                                                                                    \
        uint32_t ct_start_kv = start_row_kv / 32;                                                                                       \
        uint32_t ct_end_kv   = (end_row_kv + 31) / 32;                                                                                  \
                                                                                                                                        \
        uint32_t push_ct = ct_start_kv;                                                                                                 \
        for (uint32_t d = 0; d < HTP_MM_DMA_DEPTH && push_ct < ct_end_kv; d++, push_ct++) {                                                    \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                                      \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + d * tile_row_transfer_size_aligned,                                      \
                           src2_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                \
        }                                                                                                                               \
                                                                                                                                        \
        for (uint32_t ct = ct_start_kv; ct < ct_end_kv; ct++) {                                                                         \
            const uint8_t * w_tile_k = dma_queue_pop(dma_queue).dst;                                                                    \
            const uint8_t * w_tile_v = dma_queue_pop(dma_queue).dst;                                                                    \
                                                                                                                                        \
            int valid_rows = (int)src0->ne[1] - (int)(ct * 32);                                                                         \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                                   \
                                                                                                                                        \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                     \
            uint32_t ir1 = 0;                                                                                                           \
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                    \
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                             \
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                             \
                                                                                                                                        \
                float * restrict dst_row0_k = (float *) (dst_k->data + ((ir1+0) * dst_row_size));                                       \
                float * restrict dst_row1_k = (float *) (dst_k->data + ((ir1+1) * dst_row_size));                                       \
                float * dst_ptr0_k = &dst_row0_k[ct * 32];                                                                              \
                float * dst_ptr1_k = &dst_row1_k[ct * 32];                                                                              \
                                                                                                                                        \
                float * restrict dst_row0_v = (float *) (dst_v->data + ((ir1+0) * dst_row_size));                                       \
                float * restrict dst_row1_v = (float *) (dst_v->data + ((ir1+1) * dst_row_size));                                       \
                float * dst_ptr0_v = &dst_row0_v[ct * 32];                                                                              \
                float * dst_ptr1_v = &dst_row1_v[ct * 32];                                                                              \
                                                                                                                                        \
                DOT_2X2(ne10, dst_ptr0_k, dst_ptr1_k, w_tile_k, src1_col0, src1_col1, valid_rows);                                      \
                DOT_2X2(ne10, dst_ptr0_v, dst_ptr1_v, w_tile_v, src1_col0, src1_col1, valid_rows);                                      \
            }                                                                                                                           \
                                                                                                                                        \
            for (; ir1 < src1_nrows; ++ir1) {                                                                                           \
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                  \
                                                                                                                                        \
                float * restrict dst_row_k = (float *) (dst_k->data + (ir1 * dst_row_size));                                            \
                float * dst_ptr_k = &dst_row_k[ct * 32];                                                                                \
                                                                                                                                        \
                float * restrict dst_row_v = (float *) (dst_v->data + (ir1 * dst_row_size));                                            \
                float * dst_ptr_v = &dst_row_v[ct * 32];                                                                                \
                                                                                                                                        \
                DOT_2X1(ne10, dst_ptr_k, w_tile_k, src1_col, valid_rows);                                                               \
                DOT_2X1(ne10, dst_ptr_v, w_tile_v, src1_col, valid_rows);                                                               \
            }                                                                                                                           \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                      \
                                                                                                                                        \
            if (push_ct < ct_end_kv) {                                                                                                  \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_k, src0_row + push_ct * tile_row_stride),                      \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                   \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_v, src2_row + push_ct * tile_row_stride),                      \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                   \
                push_ct++;                                                                                                              \
            }                                                                                                                           \
        }                                                                                                                               \
    }                                                                                                                                   \
                                                                                                                                        \
    /* 2. Process Q separately */                                                                                                       \
    const uint32_t src0_nrows_q = src3->ne[1] * src3->ne[2] * src3->ne[3]; /* src3 is Wq */                                             \
    uint32_t src0_nrows_per_thread_q = (src0_nrows_q + nth - 1) / nth;                                                                  \
    src0_nrows_per_thread_q = hex_round_up(src0_nrows_per_thread_q, 32);                                                                \
                                                                                                                                        \
    const uint32_t start_row_q = src0_nrows_per_thread_q * ith;                                                                         \
    const uint32_t end_row_q   = MIN(start_row_q + src0_nrows_per_thread_q, src0_nrows_q);                                              \
                                                                                                                                        \
    if (start_row_q < end_row_q) {                                                                                                      \
        uint32_t ct_start_q = start_row_q / 32;                                                                                         \
        uint32_t ct_end_q   = (end_row_q + 31) / 32;                                                                                    \
                                                                                                                                        \
        uint32_t push_ct = ct_start_q;                                                                                                  \
        for (uint32_t d = 0; d < HTP_MM_DMA_DEPTH && push_ct < ct_end_q; d++, push_ct++) {                                                     \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + d * tile_row_transfer_size_aligned,                                      \
                           src3_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                \
        }                                                                                                                               \
                                                                                                                                        \
        for (uint32_t ct = ct_start_q; ct < ct_end_q; ct++) {                                                                           \
            const uint8_t * w_tile_q = dma_queue_pop(dma_queue).dst;                                                                    \
                                                                                                                                        \
            int valid_rows = (int)src3->ne[1] - (int)(ct * 32);                                                                         \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                                   \
                                                                                                                                        \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                     \
            uint32_t ir1 = 0;                                                                                                           \
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                    \
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                             \
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                             \
                                                                                                                                        \
                float * restrict dst_row0_q = (float *) (dst_q->data + ((ir1+0) * dst_row_size));                                       \
                float * restrict dst_row1_q = (float *) (dst_q->data + ((ir1+1) * dst_row_size));                                       \
                float * dst_ptr0_q = &dst_row0_q[ct * 32];                                                                              \
                float * dst_ptr1_q = &dst_row1_q[ct * 32];                                                                              \
                                                                                                                                        \
                DOT_2X2(ne10, dst_ptr0_q, dst_ptr1_q, w_tile_q, src1_col0, src1_col1, valid_rows);                                      \
            }                                                                                                                           \
                                                                                                                                        \
            for (; ir1 < src1_nrows; ++ir1) {                                                                                           \
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                  \
                                                                                                                                        \
                float * restrict dst_row_q = (float *) (dst_q->data + (ir1 * dst_row_size));                                            \
                float * dst_ptr_q = &dst_row_q[ct * 32];                                                                                \
                                                                                                                                        \
                DOT_2X1(ne10, dst_ptr_q, w_tile_q, src1_col, valid_rows);                                                               \
            }                                                                                                                           \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                      \
                                                                                                                                        \
            if (push_ct < ct_end_q) {                                                                                                   \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_q, src3_row + push_ct * tile_row_stride),                      \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                   \
                push_ct++;                                                                                                              \
            }                                                                                                                           \
        }                                                                                                                               \
    }                                                                                                                                   \
                                                                                                                                        \
    t2 = HAP_perf_get_qtimer_count();                                                                                                   \
                                                                                                                                        \
    FARF(HIGH, "matmul-qkv-repacked-%s %u/%u: Wk:%ux%ux%ux%u Wv:%ux%ux%ux%u Wq:%ux%ux%ux%u * %ux%ux%ux%u -> usec %u\n",                 \
         mmctx->type, ith, nth,                                                                                                         \
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],                                                                            \
         src2->ne[0], src2->ne[1], src2->ne[2], src2->ne[3],                                                                            \
         src3->ne[0], src3->ne[1], src3->ne[2], src3->ne[3],                                                                            \
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],                                                                            \
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));                                                                              \
}


#define MATMUL_FFN_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                                \
static void matmul_ffn_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                          \
    struct htp_mm_context * mmctx = data;                                                                                           \
    struct htp_ops_context * octx = mmctx->octx;                                                                                        \
                                                                                                                                        \
    const struct htp_tensor * restrict src0 = octx->src[0]; /* Wgate */                                                                 \
    const struct htp_tensor * restrict src1 = octx->src[1]; /* y */                                                                     \
    const struct htp_tensor * restrict src2 = octx->src[2]; /* Wup */                                                                   \
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];                                                                        \
    const struct htp_tensor * restrict dst_up = octx->dsts[1];                                                                          \
                                                                                                                                        \
    const uint32_t ne00 = src0->ne[0];                                                                                                  \
    const uint32_t ne01 = src0->ne[1];                                                                                                  \
    const uint32_t ne10 = src1->ne[0];                                                                                                  \
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];                                                                \
                                                                                                                                        \
    const size_t dst_row_size  = dst_gate->nb[1];                                                                                       \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                                 \
                                                                                                                                        \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                                \
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;                                                \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                                    \
                                                                                                                                        \
    volatile uint64_t t1, t2;                                                                                                           \
    t1 = HAP_perf_get_qtimer_count();                                                                                                   \
                                                                                                                                        \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                                           \
                                                                                                                                        \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                                   \
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;                                                                   \
                                                                                                                                        \
    const uint32_t tile_size = TILE_SIZE;                                                                                               \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                                    \
                                                                                                                                        \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                                   \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                                   \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                                 \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                          \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                                        \
                                                                                                                                        \
    const uint32_t src0_nrows = ne01 * src0->ne[2] * src0->ne[3];                                                                       \
    const uint32_t src0_start_row = mmctx->src0_nrows_per_thread * ith;                                                                 \
    const uint32_t src0_end_row   = MIN(src0_start_row + mmctx->src0_nrows_per_thread, src0_nrows);                                     \
                                                                                                                                        \
    uint32_t ct_start = src0_start_row / 32;                                                                                            \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                       \
                                                                                                                                        \
    uint32_t push_ct = ct_start;                                                                                                        \
    for (uint32_t d = 0; d < HTP_MM_DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {                                                           \
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),   \
                       aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                           \
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + d * tile_row_transfer_size_aligned, src2_row + push_ct * tile_row_stride),   \
                       aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                           \
    }                                                                                                                                   \
                                                                                                                                        \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                                   \
        const uint8_t * w_tile_gate = dma_queue_pop(dma_queue).dst;                                                                     \
        const uint8_t * w_tile_up   = dma_queue_pop(dma_queue).dst;                                                                     \
                                                                                                                                        \
        int valid_rows = (int)ne01 - (int)(ct * 32);                                                                                    \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                       \
                                                                                                                                        \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                         \
        uint32_t ir1 = 0;                                                                                                               \
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                        \
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                                 \
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                                 \
                                                                                                                                        \
            float * restrict dst_row0_gate = (float *) (dst_gate->data + ((ir1+0) * dst_row_size));                                     \
            float * restrict dst_row1_gate = (float *) (dst_gate->data + ((ir1+1) * dst_row_size));                                     \
            float * dst_ptr0_gate = &dst_row0_gate[ct * 32];                                                                            \
            float * dst_ptr1_gate = &dst_row1_gate[ct * 32];                                                                            \
                                                                                                                                        \
            float * restrict dst_row0_up = (float *) (dst_up->data + ((ir1+0) * dst_row_size));                                         \
            float * restrict dst_row1_up = (float *) (dst_up->data + ((ir1+1) * dst_row_size));                                         \
            float * dst_ptr0_up = &dst_row0_up[ct * 32];                                                                                \
            float * dst_ptr1_up = &dst_row1_up[ct * 32];                                                                                \
                                                                                                                                        \
            DOT_2X2(ne10, dst_ptr0_gate, dst_ptr1_gate, w_tile_gate, src1_col0, src1_col1, valid_rows);                                 \
            DOT_2X2(ne10, dst_ptr0_up, dst_ptr1_up, w_tile_up, src1_col0, src1_col1, valid_rows);                                       \
        }                                                                                                                               \
                                                                                                                                        \
        for (; ir1 < src1_nrows; ++ir1) {                                                                                               \
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                      \
                                                                                                                                        \
            float * restrict dst_row_gate = (float *) (dst_gate->data + (ir1 * dst_row_size));                                          \
            float * dst_ptr_gate = &dst_row_gate[ct * 32];                                                                              \
                                                                                                                                        \
            float * restrict dst_row_up = (float *) (dst_up->data + (ir1 * dst_row_size));                                              \
            float * dst_ptr_up = &dst_row_up[ct * 32];                                                                                  \
                                                                                                                                        \
            DOT_2X1(ne10, dst_ptr_gate, w_tile_gate, src1_col, valid_rows);                                                             \
            DOT_2X1(ne10, dst_ptr_up, w_tile_up, src1_col, valid_rows);                                                                 \
        }                                                                                                                               \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                          \
                                                                                                                                        \
        if (push_ct < ct_end) {                                                                                                         \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_gate, src0_row + push_ct * tile_row_stride),                       \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                       \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_up, src2_row + push_ct * tile_row_stride),                         \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                       \
            push_ct++;                                                                                                                  \
        }                                                                                                                               \
    }                                                                                                                                   \
                                                                                                                                        \
    t2 = HAP_perf_get_qtimer_count();                                                                                                   \
                                                                                                                                        \
    FARF(HIGH, "matmul-ffn-repacked-%s %u/%u: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mmctx->type, ith, nth,       \
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],                    \
         src1->ne[2], src1->ne[3], dst_gate->ne[0], dst_gate->ne[1], dst_gate->ne[2], dst_gate->ne[3],                                  \
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));                                                                              \
}


MATMUL_2D_REPACKED_IMPL(q4_0,  576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1,  640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0,  1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl, 576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4, 544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,  flat_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,  flat_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,  flat_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2, flat_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2, flat_vec_dot_mxfp4_32x1)


MATVEC_2D_REPACKED_IMPL(q4_0,  576,  tiled_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1,  640,  tiled_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0,  1088, tiled_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl, 576,  tiled_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4, 544,  tiled_vec_dot_mxfp4_32x1)

MATVEC_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x1)


MATMUL_QKV_2D_REPACKED_IMPL(q4_0,  576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q4_1,  640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q8_0,  1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(iq4nl, 576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(mxfp4, 544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_QKV_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,  flat_vec_dot_q4_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,  flat_vec_dot_q4_1_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,  flat_vec_dot_q8_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2, flat_vec_dot_iq4nl_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2, flat_vec_dot_mxfp4_32x1)


MATMUL_FFN_2D_REPACKED_IMPL(q4_0,  576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q4_1,  640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q8_0,  1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(iq4nl, 576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(mxfp4, 544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_FFN_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,  flat_vec_dot_q4_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,  flat_vec_dot_q4_1_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,  flat_vec_dot_q8_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2, flat_vec_dot_iq4nl_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2, flat_vec_dot_mxfp4_32x1)

// src1 tensor is already in VTCM
static void matmul_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows
    const uint32_t src1_nrows = ne11 * ne12 * ne13;  // src1 rows

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    volatile uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;

    // Prefill spad with src0 rows
    #pragma unroll(4)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= HTP_MM_VTCM_SRC0_NROWS) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);
        // Process src1 columns in pairs (2×2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);
            float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);

        // Prefetch next (n + vtcm_nrows) row
        const int pr0 = (ir0 + HTP_MM_VTCM_SRC0_NROWS);
        const int is0 = (pr0 - src0_start_row) % HTP_MM_VTCM_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) % HTP_MM_VTCM_SRC0_NROWS;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);
        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mmctx->type, ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],
         src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// q8_0_tiled/q8_1_tiled src1 tensor is already in VTCM
static void matvec_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = ne01;

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * src1_data = mmctx->vtcm_src1;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    float * tmp = (float *) vtcm_dst_ptr;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
    float * restrict dst_col          = (float *) dst->data;

    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // Prefill spad with 2x src0 rows
    #pragma unroll(2)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint32_t is0 = (ir0 - src0_start_row);
        if (is0 >= HTP_MM_VTCM_SRC0_NROWS) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);

        // Prefetch next (n + vtcm_nrows) row
        const uint32_t pr0 = (ir0 + HTP_MM_VTCM_SRC0_NROWS);
        const uint32_t is0 = (pr0 - src0_start_row) % HTP_MM_VTCM_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row) % HTP_MM_VTCM_SRC0_NROWS;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);
    }

    hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, src0_end_row - src0_start_row);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matvec-%s %u/%u: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mmctx->type, ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],
         src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ids->ne[0] * ids->ne[1] + (i1)]

struct mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// src1 tensor is already in VTCM
static void matmul_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows = ne01;  // src0 rows per expert
    const uint32_t src1_nrows = ne11;

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;

    const uint32_t n_ids = ids->ne[0];  // n_expert_used
    const uint32_t n_as  = ne02;        // n_expert

    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    const size_t dst_row_size  = nb1;
    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        if (mmctx->hmx_eligible) {
            continue;
        }

        const uint8_t * src0_row = (const uint8_t *) src0->data + cur_a * nb02;

        const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < HTP_MM_DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);
            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // src1 row idx
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);

            if (push_ct < ct_end) {
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-id-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u usec %u\n", mmctx->type,
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// src1 tensor is already in VTCM
static void matvec_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows = ne01;  // src0 rows per expert

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;

    assert(ne13 % ne03 == 0);

    const size_t dst_row_size  = nb1;
    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const uint32_t n_aids = src2->ne[0];  // num activated experts
    const uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const int32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
        if (eid < 0) {
            continue;
        }
        assert(eid < (int32_t) n_ids);

        const uint8_t * restrict src0_row = (const uint8_t *) src0->data + eid * nb02;
        const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
        float * restrict dst_row          = (float *) (dst->data + ie1 * nb1);

        const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < HTP_MM_DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);

            if (push_ct < ct_end) {
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matvec-id-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u usec %u\n", mmctx->type,
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], src2->ne[0], src2->ne[1], src2->ne[2], src2->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void quantize_f32_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = mmctx->vtcm_src1;
    uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;
    uint32_t dst_stride = mmctx->vtcm_src1_stride;

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = ne0 * sizeof(float);
    const size_t src_stride   = src->nb[1];

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_stride * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst       + (dst_stride * ir_first);

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-f32-f32: %u/%u : n-rows %u (%u:%u) row-size %u (%u) -> %u usec %u\n", ith, nth, nrows, ir_first,
        ir_last, src_row_size, src_stride, dst_stride, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);
}

static void quantize_f32_f16(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = mmctx->vtcm_src1;
    uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;
    uint32_t dst_stride = mmctx->vtcm_src1_stride;

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = ne0 * sizeof(float);
    const size_t src_stride   = src->nb[1];

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_stride * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst       + (dst_stride * ir_first);

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f16_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-f32-f16: %u/%u : n-rows %u (%u:%u) row-size %u (%u) -> %u usec %u\n", ith, nth, nrows, ir_first,
        ir_last, src_row_size, src_stride, dst_stride, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);
}

// TODO just a plain copy that should be done via the DMA during the Op setup
static void quantize_f16_f16(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = mmctx->vtcm_src1;
    uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;
    uint32_t dst_stride = mmctx->vtcm_src1_stride;

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = ne0 * sizeof(float);
    const size_t src_stride   = src->nb[1];

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_stride * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst       + (dst_stride * ir_first);

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f16_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-f16-f16: %u/%u : n-rows %u (%u:%u) row-size %u (%u) -> %u usec %u\n", ith, nth, nrows, ir_first,
        ir_last, src_row_size, src_stride, dst_stride, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);
}


static inline bool htp_is_permuted(const struct htp_tensor * t) {
    return t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

static int htp_mminit_vec_dot(struct htp_mm_context * mmctx, enum htp_data_type type) {
    switch (type) {
        case HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            return 0;
        case HTP_TYPE_Q4_1:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            return 0;
        case HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            return 0;
        case HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            return 0;
        case HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            return 0;
        default:
            return -1;
    }
}

static void htp_mminit_spad(struct htp_ops_context * octx,
                                 size_t dst_row_size,
                                 size_t src0_row_size_padded,
                                 size_t src1_row_size,
                                 uint32_t src1_nrows,
                                 size_t src2_vtcm_size_per_thread) {
    octx->dst_spad.size_per_thread  = hex_round_up(HTP_MM_VTCM_DST_NROWS * dst_row_size, 256);
    octx->src0_spad.size_per_thread = hex_round_up(HTP_MM_VTCM_SRC0_NROWS * src0_row_size_padded, 256);
    octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);

    if (src2_vtcm_size_per_thread > 0) {
        octx->src2_spad.size_per_thread = src2_vtcm_size_per_thread;
        octx->src2_spad.size            = octx->src2_spad.size_per_thread;
    }

    // src0 spad is also used in dynamic quantizer to store padded src1 rows
    size_t src1_row_size_padded = hex_round_up(src1_row_size, QK_Q8_0_TILED * sizeof(float));
    if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
        octx->src0_spad.size_per_thread = src1_row_size_padded;
    }

    octx->src1_spad.size = octx->src1_spad.size_per_thread;
    octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
    octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
}

static int op_matmul_hvx(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;
    size_t       src1_row_size = nb11;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
    size_t       src1_row_size_padded;

    worker_callback_t quant_job_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_jobs = 1;
    if (src1_nrows > 1) {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = matmul_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = matmul_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = matmul_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = matmul_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = matmul_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = matmul_2d;
        }
    } else {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = matvec_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = matvec_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = matvec_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = matvec_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = matvec_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = matvec_2d;
        }
    }

    bool need_quant = true;

    switch (kparams->kernel_type) {
        case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_job_func     = (src1->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            mmctx->type        = "f16-f16";
            mmctx->vec_dot_1x1 = vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1 = vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2 = vec_dot_f16_f16_aa_2x2;
            src1_row_size      = hex_round_up(ne10 * 2, 128);
            break;

        case HTP_MM_KERNEL_HVX_F16_F32_DDR:
            quant_job_func = NULL;
            mmctx->type        = "f16-f32";
            mmctx->vec_dot_1x1 = vec_dot_f16_f32_uu_1x1;
            matmul_job_func    = matmul_4d;
            src1_row_size = nb11;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            need_quant = false;
            break;

        case HTP_MM_KERNEL_HVX_F16_F16_DDR:
            quant_job_func = NULL;
            mmctx->type        = "f16-f16";
            mmctx->vec_dot_1x1 = vec_dot_f16_f16_uu_1x1;
            matmul_job_func    = matmul_4d;
            src1_row_size = nb11;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            need_quant = false;
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_job_func     = quantize_f32_f32;
            mmctx->type        = "f32-f32";
            mmctx->vec_dot_1x1 = vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1 = vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2 = vec_dot_f32_f32_aa_2x2;
            src1_row_size      = hex_round_up(ne10 * 4, 128);
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_DDR:
            quant_job_func = NULL;
            mmctx->type        = "f32-f32";
            mmctx->vec_dot_1x1 = vec_dot_f32_f32_uu_1x1;
            matmul_job_func    = matmul_4d;
            src1_row_size = nb11;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            need_quant = false;
            break;

        case HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT: {
            n_quant_jobs = MIN(src1_nrows, octx->n_threads);
            quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
            src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);
            
            if (src1_nrows > 1) {
                switch (src0->type) {
                    case HTP_TYPE_Q4_0:   matmul_job_func = matmul_2d_repacked_q4_0_flat;   break;
                    case HTP_TYPE_Q4_1:   matmul_job_func = matmul_2d_repacked_q4_1_flat;   break;
                    case HTP_TYPE_Q8_0:   matmul_job_func = matmul_2d_repacked_q8_0_flat;   break;
                    case HTP_TYPE_IQ4_NL: matmul_job_func = matmul_2d_repacked_iq4nl_flat;  break;
                    case HTP_TYPE_MXFP4:  matmul_job_func = matmul_2d_repacked_mxfp4_flat;  break;
                    default:              return HTP_STATUS_NO_SUPPORT;
                }
            } else {
                switch (src0->type) {
                    case HTP_TYPE_Q4_0:   matmul_job_func = matvec_2d_repacked_q4_0_flat;   break;
                    case HTP_TYPE_Q4_1:   matmul_job_func = matvec_2d_repacked_q4_1_flat;   break;
                    case HTP_TYPE_Q8_0:   matmul_job_func = matvec_2d_repacked_q8_0_flat;   break;
                    case HTP_TYPE_IQ4_NL: matmul_job_func = matvec_2d_repacked_iq4nl_flat;  break;
                    case HTP_TYPE_MXFP4:  matmul_job_func = matvec_2d_repacked_mxfp4_flat;  break;
                    default:              return HTP_STATUS_NO_SUPPORT;
                }
            }
            break;
        }

        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        case HTP_MM_KERNEL_HVX_QUANT_ROW:
        default:
            if (htp_mminit_vec_dot(mmctx, src0->type) != 0) {
                return HTP_STATUS_NO_SUPPORT;
            }

            const uint32_t qk = QK_Q8_0_TILED;
            const uint32_t nb = (ne10 + qk - 1) / qk;
            const uint32_t total_nb = src1_nrows * nb;

            if (src1_nrows < octx->n_threads) {
                n_quant_jobs = MIN(total_nb, octx->n_threads);
                quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
                for (uint32_t ith = 0; ith < n_quant_jobs; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_jobs;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_jobs;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } else {
                n_quant_jobs = MIN(src1_nrows, octx->n_threads);
                quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
            }
            src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
            break;
    }

    size_t src0_sz = 0, src1_sz = 0, dst_sz = 0;
    if (kparams->vtcm_src0_size > 0 || kparams->vtcm_src1_size > 0 || kparams->vtcm_dst_size > 0) {
        src0_sz = kparams->vtcm_src0_size;
        src1_sz = kparams->vtcm_src1_size;
        dst_sz  = kparams->vtcm_dst_size;
    } else {
        hvx_get_vtcm_sizes(
            kparams->kernel_type, src0->type, ne10, src1_nrows, octx->n_threads,
            dst_row_size, src0_row_size, src1_row_size,
            &src0_sz, &src1_sz, &dst_sz
        );
    }

    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_src1_size_per_thread = src1_sz;
    } else {
        mmctx->vtcm_src1_size_per_thread = src1_sz / octx->n_threads;
    }
    mmctx->vtcm_src0_size_per_thread = src0_sz / octx->n_threads;
    mmctx->vtcm_dst_size_per_thread  = dst_sz / octx->n_threads;

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : (src1_sz + src0_sz + dst_sz);

    FARF(HIGH, "matmul-%s : src0-spad-size %zu src1-spad-size %zu dst-spad-size %zu (%zu)\n", mmctx->type,
         src0_sz, src1_sz, dst_sz, vtcm_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, src1->data, dst->data);

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = vtcm_seq_alloc(&vtcm_ptr, src1_sz);
    mmctx->vtcm_src0 = vtcm_seq_alloc(&vtcm_ptr, src0_sz);
    mmctx->vtcm_dst  = vtcm_seq_alloc(&vtcm_ptr, dst_sz);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)
        return HTP_STATUS_OK;

    if (need_quant) {
        mmctx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, mmctx, n_quant_jobs);
    }

    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}

int op_matmul(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_mm_kernel_params * kparams = (struct htp_mm_kernel_params *)octx->kernel_params;

    if (kparams->n_hmx) {
        int k = (int) src0->ne[0];
        int n = (int) src0->ne[1];
        const int m_total = (int) src1->ne[1];
        const int act_stride = (int)(src1->nb[1] / sizeof(float));
        const int wgt_stride = (int)(src0->nb[1] / sizeof(__fp16));

        if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
            return HTP_STATUS_OK;
        }

        // Force dynamic quantization cache to clear since HMX will overwrite VTCM
        octx->src1_spad.src = NULL;

        int ret = -1;
        const int num_threads = MIN(kparams->num_threads, (int) octx->n_threads);
        if (kparams->kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
            htp_mm_hmx_f16_f32_batched_params_t batch_params = {
                .dst             = (float *) dst->data,
                .activation      = (float *) src1->data,
                .weight          = (const __fp16 *) src0->data,
                .m               = m_total,
                .k               = k,
                .n               = n,
                .act_stride      = act_stride,
                .weight_stride   = wgt_stride,
                .dst_stride      = (int) (dst->nb[1] / sizeof(float)),
                .ne02            = ne02,
                .ne03            = ne03,
                .ne12            = ne12,
                .ne13            = ne13,
                .src0_nb2        = src0->nb[2],
                .src0_nb3        = src0->nb[3],
                .src1_nb2        = src1->nb[2],
                .src1_nb3        = src1->nb[3],
                .dst_nb2         = dst->nb[2],
                .dst_nb3         = dst->nb[3],
            };
            ret = htp_mm_hmx_f16_f32_batched(octx->ctx, &batch_params,
                                             kparams->m_chunk, kparams->n_chunk,
                                             kparams->use_pipeline, num_threads,
                                             kparams->act_threads,
                                             kparams->vtcm_size);
        } else {
            ret = htp_mm_hmx_2d_f32(
                octx->ctx, (float*) dst->data, (float*) src1->data, (const uint8_t *) src0->data,
                m_total, k, n, act_stride, (int) src0->nb[1], (int) src0->type, (int) src1->ne[0],
                (int)(dst->nb[1] / sizeof(float)), (int)dst->ne[0],
                kparams->m_chunk, kparams->n_chunk, kparams->use_pipeline, num_threads,
                kparams->act_threads,
                kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size
            );
        }

        if (ret != 0) {
            FARF(HIGH, "HMX matmul failed (ret=%d), falling back to HVX", ret);
            return op_matmul_hvx(octx);
        }
        return 0;
    }

    return op_matmul_hvx(octx);
}

int op_matmul_id(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const struct htp_tensor * restrict ids = octx->src[2];

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = ne01;  // per expert
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    worker_callback_t quant_job_func;
    worker_callback_t matmul_id_job_func = src1_nrows > 1 ? matmul_id : matvec_id;

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    mmctx->src0_nrows_per_thread  = hex_round_up(mmctx->src0_nrows_per_thread, 32);

    size_t src1_row_size;
    size_t src1_row_size_padded;

    // row groups
    const int n_ids = ids->ne[0];  // n_expert_used
    const int n_as  = ne02;        // n_expert

    size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
    size_t matrix_row_map_size    = n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping);
    const size_t total_map_size   = matrix_row_counts_size + matrix_row_map_size;

    void * mapping_buf = NULL;
    bool must_free_mapping = false;

    if (octx->ctx->ddr_spad_base && total_map_size <= octx->ctx->ddr_spad_size) {
        mapping_buf = octx->ctx->ddr_spad_base;
    } else {
        mapping_buf = memalign(128, total_map_size);
        if (mapping_buf) {
            must_free_mapping = true;
        } else {
            return HTP_STATUS_INTERNAL_ERR;
        }
    }

    uint32_t *                matrix_row_counts = (uint32_t *) mapping_buf;
    struct mmid_row_mapping * matrix_rows       = (struct mmid_row_mapping *) ((uint8_t *) mapping_buf + matrix_row_counts_size);

    mmctx->matrix_row_counts = matrix_row_counts;
    mmctx->matrix_rows       = matrix_rows;
    mmctx->mm_div_ne11       = kparams->div_ne11;

    if (htp_mminit_vec_dot(mmctx, src0->type) != 0) {
        if (must_free_mapping) free(mapping_buf);
        return HTP_STATUS_NO_SUPPORT;
    }

    if (src1_nrows > 1) {
        // initialize matrix_row_counts and map
        memset(matrix_row_counts, 0, n_as * sizeof(uint32_t));

        // group rows by src0 matrix
        for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {  // token idx
            for (uint32_t id = 0; id < n_ids; ++id) {         // expert idx
                const int32_t i02 = *(const int32_t *) ((const uint8_t *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                if (i02 < 0) {
                    continue;
                }
                assert(i02 < n_as);

                matrix_rows[i02 * n_ids * ids->ne[1] + matrix_row_counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
                matrix_row_counts[i02] += 1;
            }
        }
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        if (must_free_mapping) free(mapping_buf);
        return HTP_STATUS_OK;
    }

    bool hmx_eligible = kparams->n_hmx;

    mmctx->hmx_eligible = hmx_eligible;

    if (hmx_eligible) {
        for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
            const int32_t cne1 = matrix_row_counts[cur_a];
            if (cne1 == 0) continue;

            int ret = htp_mm_hmx_id_2d_f32(octx->ctx, (float*) dst->data, (float*) src1->data,
                                           (const uint8_t *) src0->data + cur_a * nb02,
                                           cne1, ne00, ne01,
                                           ne10,
                                           ne11,
                                           nb11, nb12,
                                           nb1, nb2,
                                           (int) src0->nb[1], (int) src0->type,
                                           matrix_rows, cur_a, n_ids * ids->ne[1]);
            if (ret != 0) {
                FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
                if (must_free_mapping) free(mapping_buf);
                return HTP_STATUS_NO_SUPPORT;
            }
        }

        // HMX has overwritten VTCM, so force dynamic quantization cache to clear
        octx->src1_spad.src = NULL;

        if (must_free_mapping) free(mapping_buf);
        return HTP_STATUS_OK;
    }

    // --- HVX Fallback Path ---
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne10 + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    uint32_t n_quant_jobs = 1;
    if (src1_nrows < octx->n_threads) {
        n_quant_jobs = MIN(total_nb, octx->n_threads);
        quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_jobs; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_jobs;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_jobs;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }
    src1_row_size  = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

    size_t dst_sz_per_thread  = hex_round_up(HTP_MM_VTCM_DST_NROWS * 0, 256);
    size_t src0_sz_per_thread = hex_round_up(HTP_MM_VTCM_SRC0_NROWS * src0_row_size_padded, 256);
    size_t src1_sz_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);
    size_t src2_sz_per_thread = 0; // We moved the mapping to DDR!

    // Ensure src0 spad has enough size to host temporary transposed src1 columns
    src1_row_size_padded = hex_round_up(src1_row_size, QK_Q8_0_TILED * sizeof(float));
    if (src0_sz_per_thread < src1_row_size_padded) {
        src0_sz_per_thread = src1_row_size_padded;
    }

    const bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                              src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                              src0->type == HTP_TYPE_MXFP4);
    if (is_repacked) {
        const uint32_t n_k_tiles = ne10 / 32;
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
        size_t repacked_vtcm_size = hex_round_up(HTP_MM_DMA_DEPTH * tile_row_size, 256);
        if (repacked_vtcm_size < src1_row_size_padded) {
            repacked_vtcm_size = src1_row_size_padded;
        }
        src0_sz_per_thread = repacked_vtcm_size;
    }

    size_t src1_sz = src1_sz_per_thread;
    size_t src0_sz = src0_sz_per_thread * octx->n_threads;
    size_t src2_sz = src2_sz_per_thread * octx->n_threads;
    size_t dst_sz  = dst_sz_per_thread * octx->n_threads;

    size_t vtcm_size = src2_sz + src1_sz + src0_sz + dst_sz;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu src1-spad-size %zu src2-spad-size %zu dst-spad-size %zu (%zu)\n", mmctx->type,
         src0_sz, src1_sz, src2_sz, dst_sz, vtcm_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        if (must_free_mapping) free(mapping_buf);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = vtcm_seq_alloc(&vtcm_ptr, src1_sz);
    mmctx->vtcm_src0 = vtcm_seq_alloc(&vtcm_ptr, src0_sz);
    mmctx->vtcm_src2 = vtcm_seq_alloc(&vtcm_ptr, src2_sz);
    mmctx->vtcm_dst  = vtcm_seq_alloc(&vtcm_ptr, dst_sz);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = src0_sz_per_thread;
    mmctx->vtcm_src1_size_per_thread = src1_sz_per_thread;
    mmctx->vtcm_src2_size_per_thread = src2_sz_per_thread;
    mmctx->vtcm_dst_size_per_thread  = dst_sz_per_thread;

    mmctx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
    worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, mmctx, n_quant_jobs);

    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_pool_run_func(octx->ctx->worker_pool, matmul_id_job_func, mmctx, n_matmul_jobs);

    if (must_free_mapping) free(mapping_buf);
    return HTP_STATUS_OK;
}

static void matmul_qkv_2d(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;

    const struct htp_tensor * restrict src0 = octx->src[0]; // Wk
    const struct htp_tensor * restrict src1 = octx->src[1]; // x
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wv
    const struct htp_tensor * restrict src3 = octx->src[3]; // Wq
    const struct htp_tensor * restrict dst_k = octx->dsts[0];
    const struct htp_tensor * restrict dst_v = octx->dsts[1];
    const struct htp_tensor * restrict dst_q = octx->dsts[2];

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t ne13 = src1->ne[3];

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    const uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const size_t dst_row_size  = dst_k->nb[1];
    const size_t src0_row_size = src0->nb[1];
    const size_t src2_row_size = src2->nb[1];
    const size_t src3_row_size = src3->nb[1];

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src2_stride = mmctx->vtcm_src2_stride;
    const size_t src3_stride = mmctx->vtcm_src3_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;
    uint8_t * restrict vtcm_src3_ptr = mmctx->vtcm_src3 + mmctx->vtcm_src3_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    dma_queue * dma_queue = octx->ctx->dma[ith];

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;
    const uint8_t * restrict src3_row = (const uint8_t *) src3->data;

    // Prefill spad with src0, src2, src3 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= HTP_MM_VTCM_SRC0_NROWS) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                       src2_stride, src2_row_size, src2_row_size, 2);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + is0 * src3_stride, src3_row + ir0 * src3_row_size),
                       src3_stride, src3_row_size, src3_row_size, 2);
    }

    // Process rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss2 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss3 = dma_queue_pop(dma_queue).dst;

        // Process src1 columns in pairs (2×2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);

            float * restrict dst_row0_q = (float *) (dst_q->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1_q = (float *) (dst_q->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_q[ir0], &dst_row1_q[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);

            float * restrict dst_row0_k = (float *) (dst_k->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1_k = (float *) (dst_k->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_k[ir0], &dst_row1_k[ir0], ss2, ss2 + src2_stride, src1_col0, src1_col1);

            float * restrict dst_row0_v = (float *) (dst_v->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1_v = (float *) (dst_v->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_v[ir0], &dst_row1_v[ir0], ss3, ss3 + src3_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_q          = (float *) (dst_q->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_q[ir0], ss0, ss0 + src0_stride, src1_col);

            float * restrict dst_row_k          = (float *) (dst_k->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_k[ir0], ss2, ss2 + src2_stride, src1_col);

            float * restrict dst_row_v          = (float *) (dst_v->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_v[ir0], ss3, ss3 + src3_stride, src1_col);
        }

        // Prefetch next (n + vtcm_nrows) rows
        const int pr0 = (ir0 + HTP_MM_VTCM_SRC0_NROWS);
        const int is0 = (pr0 - src0_start_row) % HTP_MM_VTCM_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + pr0 * src2_row_size),
                           src2_stride, src2_row_size, src2_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + is0 * src3_stride, src3_row + pr0 * src3_row_size),
                           src3_stride, src3_row_size, src3_row_size, 2);
        }
    }

    // Process last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) % HTP_MM_VTCM_SRC0_NROWS;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                       src2_stride, src2_row_size, src2_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + is0 * src3_stride, src3_row + ir0 * src3_row_size),
                       src3_stride, src3_row_size, src3_row_size, 1);

        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss2 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss3 = dma_queue_pop(dma_queue).dst;

        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_q          = (float *) (dst_q->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_q[ir0], ss0, src1_col);

            float * restrict dst_row_k          = (float *) (dst_k->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_k[ir0], ss2, src1_col);

            float * restrict dst_row_v          = (float *) (dst_v->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_v[ir0], ss3, src1_col);
        }
    }
}

static void matmul_ffn_2d(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;

    const struct htp_tensor * restrict src0 = octx->src[0]; // Wgate
    const struct htp_tensor * restrict src1 = octx->src[1]; // y
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wup
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];
    const struct htp_tensor * restrict dst_up = octx->dsts[1];

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t ne13 = src1->ne[3];

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    const uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const size_t dst_row_size  = dst_gate->nb[1];
    const size_t src0_row_size = src0->nb[1];
    const size_t src2_row_size = src2->nb[1];

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src2_stride = mmctx->vtcm_src2_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    dma_queue * dma_queue = octx->ctx->dma[ith];

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;

    // Prefill spad with src0, src2 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= HTP_MM_VTCM_SRC0_NROWS) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                       src2_stride, src2_row_size, src2_row_size, 2);
    }

    // Process rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss2 = dma_queue_pop(dma_queue).dst;

        // Process src1 columns in pairs (2×2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);

            float * restrict dst_row0_gate = (float *) (dst_gate->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1_gate = (float *) (dst_gate->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_gate[ir0], &dst_row1_gate[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);

            float * restrict dst_row0_up = (float *) (dst_up->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1_up = (float *) (dst_up->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_up[ir0], &dst_row1_up[ir0], ss2, ss2 + src2_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_gate      = (float *) (dst_gate->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_gate[ir0], ss0, ss0 + src0_stride, src1_col);

            float * restrict dst_row_up        = (float *) (dst_up->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_up[ir0], ss2, ss2 + src2_stride, src1_col);
        }

        // Prefetch next rows
        const int pr0 = (ir0 + HTP_MM_VTCM_SRC0_NROWS);
        const int is0 = (pr0 - src0_start_row) % HTP_MM_VTCM_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + pr0 * src2_row_size),
                           src2_stride, src2_row_size, src2_row_size, 2);
        }
    }

    // Process last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) % HTP_MM_VTCM_SRC0_NROWS;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                       src2_stride, src2_row_size, src2_row_size, 1);

        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss2 = dma_queue_pop(dma_queue).dst;

        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_gate      = (float *) (dst_gate->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_gate[ir0], ss0, src1_col);

            float * restrict dst_row_up        = (float *) (dst_up->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_up[ir0], ss2, src1_col);
        }
    }
}

int op_matmul_qkv(struct htp_ops_context * octx) {
    const struct htp_tensor * restrict src0 = octx->src[0]; // Wk
    const struct htp_tensor * restrict src1 = octx->src[1]; // x
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wv
    const struct htp_tensor * restrict src3 = octx->src[3]; // Wq
    const struct htp_tensor * restrict dst_k = octx->dsts[0];
    const struct htp_tensor * restrict dst_v = octx->dsts[1];
    const struct htp_tensor * restrict dst_q = octx->dsts[2];

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    if (htp_mminit_vec_dot(mmctx, src0->type) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (src1->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    worker_callback_t quant_job_func;
    uint32_t n_quant_jobs = 1;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
    } else if (src1_nrows < octx->n_threads) {
        n_quant_jobs = MIN(total_nb, octx->n_threads);
        quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_jobs; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_jobs;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_jobs;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }

    size_t src1_row_size;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(src1->ne[0]) : htp_mm_q8_0_flat_row_size(src1->ne[0]);
    } else {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(src1->ne[0]) : htp_mm_q8_0_tiled_row_size(src1->ne[0]);
    }

    // Set up scratchpads using precomputed sizes from the host
    size_t src0_sz = kparams->vtcm_src0_size;
    size_t src1_sz = kparams->vtcm_src1_size;
    size_t src2_sz = kparams->vtcm_src2_size;
    size_t src3_sz = kparams->vtcm_src3_size;
    size_t vtcm_size = kparams->vtcm_size;

    size_t src0_sz_per_thread = src0_sz / octx->n_threads;
    size_t src1_sz_per_thread = src1_sz;
    size_t src2_sz_per_thread = src2_sz / octx->n_threads;
    size_t src3_sz_per_thread = src3_sz / octx->n_threads;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-qkv: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = vtcm_seq_alloc(&vtcm_ptr, src1_sz);
    mmctx->vtcm_src0 = vtcm_seq_alloc(&vtcm_ptr, src0_sz);
    mmctx->vtcm_src2 = vtcm_seq_alloc(&vtcm_ptr, src2_sz);
    mmctx->vtcm_src3 = vtcm_seq_alloc(&vtcm_ptr, src3_sz);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->src3_spad.src  = NULL;

    mmctx->vtcm_src0_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src2_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src3_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = src0_sz_per_thread;
    mmctx->vtcm_src1_size_per_thread = src1_sz_per_thread;
    mmctx->vtcm_src2_size_per_thread = src2_sz_per_thread;
    mmctx->vtcm_src3_size_per_thread = src3_sz_per_thread;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)
        return HTP_STATUS_OK;

    // Run quantization once
    mmctx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
    worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, mmctx, n_quant_jobs);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_callback_t matmul_job_func;
    if (is_repacked) {
        if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = matmul_qkv_2d_repacked_q4_0_flat;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = matmul_qkv_2d_repacked_q4_1_flat;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = matmul_qkv_2d_repacked_q8_0_flat;   break;
                case HTP_TYPE_IQ4_NL:  matmul_job_func = matmul_qkv_2d_repacked_iq4nl_flat;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = matmul_qkv_2d_repacked_mxfp4_flat;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = matmul_qkv_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = matmul_qkv_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = matmul_qkv_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL:  matmul_job_func = matmul_qkv_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = matmul_qkv_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        }
    } else {
        matmul_job_func = matmul_qkv_2d;
    }
    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}

int op_matmul_ffn(struct htp_ops_context * octx) {
    const struct htp_tensor * restrict src0 = octx->src[0]; // Wgate
    const struct htp_tensor * restrict src1 = octx->src[1]; // y
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wup
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];
    const struct htp_tensor * restrict dst_up = octx->dsts[1];

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    if (htp_mminit_vec_dot(mmctx, src0->type) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (src1->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    worker_callback_t quant_job_func;
    uint32_t n_quant_jobs = 1;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
    } else if (src1_nrows < octx->n_threads) {
        n_quant_jobs = MIN(total_nb, octx->n_threads);
        quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_jobs; ++ith) {
            uint32_t ib_first = (total_nb * (ith + 0)) / n_quant_jobs;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_jobs;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }

    size_t src1_row_size;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(src1->ne[0]) : htp_mm_q8_0_flat_row_size(src1->ne[0]);
    } else {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(src1->ne[0]) : htp_mm_q8_0_tiled_row_size(src1->ne[0]);
    }

    // Set up scratchpads using precomputed sizes from the host
    size_t src0_sz = kparams->vtcm_src0_size;
    size_t src1_sz = kparams->vtcm_src1_size;
    size_t src2_sz = kparams->vtcm_src2_size;
    size_t vtcm_size = kparams->vtcm_size;

    size_t src0_sz_per_thread = src0_sz / octx->n_threads;
    size_t src1_sz_per_thread = src1_sz;
    size_t src2_sz_per_thread = src2_sz / octx->n_threads;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-ffn: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = vtcm_seq_alloc(&vtcm_ptr, src1_sz);
    mmctx->vtcm_src0 = vtcm_seq_alloc(&vtcm_ptr, src0_sz);
    mmctx->vtcm_src2 = vtcm_seq_alloc(&vtcm_ptr, src2_sz);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;

    mmctx->vtcm_src0_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src2_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = src0_sz_per_thread;
    mmctx->vtcm_src1_size_per_thread = src1_sz_per_thread;
    mmctx->vtcm_src2_size_per_thread = src2_sz_per_thread;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)
        return HTP_STATUS_OK;

    // Run quantization once
    mmctx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
    worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, mmctx, n_quant_jobs);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_callback_t matmul_job_func;
    if (is_repacked) {
        if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = matmul_ffn_2d_repacked_q4_0_flat;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = matmul_ffn_2d_repacked_q4_1_flat;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = matmul_ffn_2d_repacked_q8_0_flat;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = matmul_ffn_2d_repacked_iq4nl_flat;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = matmul_ffn_2d_repacked_mxfp4_flat;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = matmul_ffn_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = matmul_ffn_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = matmul_ffn_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = matmul_ffn_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = matmul_ffn_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        }
    } else {
        matmul_job_func = matmul_ffn_2d;
    }
    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}
