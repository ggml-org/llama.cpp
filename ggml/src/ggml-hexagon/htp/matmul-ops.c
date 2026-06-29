#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_compute_res.h>

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

typedef struct {
    float        *dst;
    const float  *activation;
    const __fp16 *weight;
    int           m;
    int           k;
    int           n;
    int           act_stride;
    int           weight_stride;
    int           dst_stride;
    int           ne02;
    int           ne03;
    int           ne12;
    int           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        src1_nb2;
    size_t        src1_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
} hmx_mm_f16_f32_batched_params_t;

struct htp_mm_context {
    const char * type;
    struct htp_ops_context * octx;

    void (*vec_dot_1x1)(const uint32_t n, float * restrict s0,
         const void * restrict vx0,
         const void * restrict vy0);

    void (*vec_dot_2x1)(const uint32_t n, float * restrict s0,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0);

    void (*vec_dot_2x2)(const uint32_t n, float * restrict s0, float * restrict s1,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0, const void * restrict vy1);

    void (*vec_dot_32x1)(const uint32_t n, float * restrict s,
         const void * restrict vx,
         const void * restrict vy, uint32_t valid_rows);

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

    // Cached thread offsets/sizes
    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_src2_size_per_thread;
    uint32_t vtcm_src3_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;
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

static void vec_dot_f32_f32_aa_1x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy) {
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

static void vec_dot_f32_f32_aa_2x1(const uint32_t n, float * restrict s0,
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

static void vec_dot_f32_f32_aa_2x2(const uint32_t n, float * restrict s0, float * restrict s1,
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

static void vec_dot_f32_f32_uu_1x1(const uint32_t n, float * restrict s, const void * restrict x, const void * restrict y) {
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

static void vec_dot_f16_f16_aa_1x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy) {
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

static void vec_dot_f16_f16_aa_2x1(const uint32_t n, float * restrict s0,
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

static void vec_dot_f16_f16_aa_2x2(const uint32_t n, float * restrict s0, float * restrict s1,
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

static void vec_dot_f16_f16_uu_1x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy) {
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

static void vec_dot_f16_f32_uu_1x1(const uint32_t n, float * restrict s, const void * restrict x, const void * restrict y) {
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

#define htp_matmul_tensors_preamble                                 \
    const struct htp_tensor * restrict src0 = octx->src[0];         \
    const struct htp_tensor * restrict src1 = octx->src[1];         \
    const struct htp_tensor * restrict src2 = octx->src[2];         \
    const struct htp_tensor * restrict  dst = octx->dst;            \
                                                                    \
    const uint32_t ne00 = src0->ne[0];                              \
    const uint32_t ne01 = src0->ne[1];                              \
    const uint32_t ne02 = src0->ne[2];                              \
    const uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    const uint32_t ne10 = src1->ne[0];                              \
    const uint32_t ne11 = src1->ne[1];                              \
    const uint32_t ne12 = src1->ne[2];                              \
    const uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    const uint32_t ne20 = src2->ne[0];                              \
    const uint32_t ne21 = src2->ne[1];                              \
    const uint32_t ne22 = src2->ne[2];                              \
    const uint32_t ne23 = src2->ne[3];                              \
                                                                    \
    const uint32_t ne0 = dst->ne[0];                                \
    const uint32_t ne1 = dst->ne[1];                                \
    const uint32_t ne2 = dst->ne[2];                                \
    const uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    const uint32_t nb00 = src0->nb[0];                              \
    const uint32_t nb01 = src0->nb[1];                              \
    const uint32_t nb02 = src0->nb[2];                              \
    const uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    const uint32_t nb10 = src1->nb[0];                              \
    const uint32_t nb11 = src1->nb[1];                              \
    const uint32_t nb12 = src1->nb[2];                              \
    const uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    const uint32_t nb0 = dst->nb[0];                                \
    const uint32_t nb1 = dst->nb[1];                                \
    const uint32_t nb2 = dst->nb[2];                                \
    const uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                         \
    struct htp_mm_context * mmctx  = data;                          \
    struct htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue *dma_queue           = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;

// *** matmul with support for 4d tensors and full broadcasting

static void hvx_mm_4d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

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
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0_start);

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

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0_start);
}

#include "hmx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-flat.h"

// Specialized repacked matmul macros
#define MATMUL_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                              \
static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                        \
    htp_matmul_preamble;                                                                                                          \
                                                                                                                                  \
    const uint32_t src0_nrows = ne01 * ne02 * ne03;                                                                               \
    const uint32_t src1_nrows = ne11 * ne12 * ne13;                                                                               \
                                                                                                                                  \
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;                                                                 \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                                     \
                                                                                                                                  \
    if (src0_start_row >= src0_end_row) {                                                                                         \
        return;                                                                                                                   \
    }                                                                                                                             \
                                                                                                                                  \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                                     \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const size_t dst_row_size  = nb1;                                                                                             \
    const size_t src1_row_size = nb11;                                                                                            \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;                                 \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                              \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    uint32_t ct_start = src0_start_row / 32;                                                                                      \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                 \
                                                                                                                                  \
    uint32_t push_ct = ct_start;                                                                                                  \
    for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                    \
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                                \
                       src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);              \
    }                                                                                                                             \
                                                                                                                                  \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                             \
        const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                    \
                                                                                                                                  \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                               \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                 \
                                                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
        uint32_t ir1 = 0;                                                                                                         \
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                  \
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                           \
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                           \
            float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                         \
            float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                         \
                                                                                                                                  \
            float * dst_ptr0 = &dst_row0[ct * 32];                                                                                \
            float * dst_ptr1 = &dst_row1[ct * 32];                                                                                \
                                                                                                                                  \
            DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows);                                          \
        }                                                                                                                         \
                                                                                                                                  \
        for (; ir1 < src1_nrows; ++ir1) {                                                                                         \
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                \
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));                                     \
            float * dst_ptr = &dst_row[ct * 32];                                                                                  \
                                                                                                                                  \
            DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows);                                                                 \
        }                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                     \
                                                                                                                                  \
        if (push_ct < ct_end) {                                                                                                   \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),                      \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            push_ct++;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
}

#define MATVEC_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X1)                                                                       \
static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                        \
    htp_matmul_preamble;                                                                                                          \
                                                                                                                                  \
    const uint32_t src0_nrows = ne01;                                                                                             \
                                                                                                                                  \
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;                                                                 \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                                     \
                                                                                                                                  \
    if (src0_start_row >= src0_end_row) {                                                                                         \
        return;                                                                                                                   \
    }                                                                                                                             \
                                                                                                                                  \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                                     \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const size_t dst_row_size  = nb1;                                                                                             \
    const size_t src1_row_size = nb11;                                                                                            \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;                                            \
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                          \
    uint8_t * src1_data = mmctx->vtcm_src1;                                                                                       \
                                                                                                                                  \
    float * tmp = (float *) vtcm_dst_ptr;                                                                                         \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;                                                              \
    float * restrict dst_col          = (float *) dst->data;                                                                      \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    uint32_t ct_start = src0_start_row / 32;                                                                                      \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                 \
                                                                                                                                  \
    uint32_t push_ct = ct_start;                                                                                                  \
    for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                    \
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                                \
                       src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);              \
    }                                                                                                                             \
                                                                                                                                  \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                             \
        const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                    \
                                                                                                                                  \
        float * dst_ptr = &tmp[ct * 32 - src0_start_row];                                                                         \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                               \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                 \
                                                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
        DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows);                                                                     \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                     \
                                                                                                                                  \
        if (push_ct < ct_end) {                                                                                                   \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),                      \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            push_ct++;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    int copy_cnt = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                                             \
    if (copy_cnt > 0) {                                                                                                           \
        hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);                                         \
    }                                                                                                                             \
}

#define MATMUL_QKV_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                          \
static void hvx_mm_qkv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                    \
    struct htp_mm_context * mmctx = data;                                                                                         \
    struct htp_ops_context * octx = mmctx->octx;                                                                                  \
                                                                                                                                  \
    const struct htp_tensor * restrict src0 = octx->src[0]; /* Wk */                                                              \
    const struct htp_tensor * restrict src1 = octx->src[1]; /* x */                                                               \
    const struct htp_tensor * restrict src2 = octx->src[2]; /* Wv */                                                              \
    const struct htp_tensor * restrict src3 = octx->src[3]; /* Wq */                                                              \
    const struct htp_tensor * restrict dst_k = octx->dsts[0];                                                                     \
    const struct htp_tensor * restrict dst_v = octx->dsts[1];                                                                     \
    const struct htp_tensor * restrict dst_q = octx->dsts[2];                                                                     \
                                                                                                                                  \
    const uint32_t ne00 = src0->ne[0];                                                                                            \
    const uint32_t ne10 = src1->ne[0];                                                                                            \
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];                                                          \
                                                                                                                                  \
    const size_t dst_k_row_size = dst_k->nb[1]; /* K and V share output width */                                                  \
    const size_t dst_q_row_size = dst_q->nb[1]; /* Q may be wider (GQA) */                                                        \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;                                 \
    uint8_t * restrict vtcm_src3_ptr = mmctx->vtcm_src3 + mmctx->vtcm_src3_size_per_thread * ith;                                 \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                              \
                                                                                                                                  \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                                     \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;                                                             \
    const uint8_t * restrict src3_row = (const uint8_t *) src3->data;                                                             \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                                  \
                                                                                                                                  \
    /* 1. Process K and V together */                                                                                             \
    const uint32_t src0_nrows_kv = src0->ne[1] * src0->ne[2] * src0->ne[3]; /* src0 is Wk */                                      \
    uint32_t src0_nrows_per_thread_kv = (src0_nrows_kv + nth - 1) / nth;                                                          \
    src0_nrows_per_thread_kv = hex_round_up(src0_nrows_per_thread_kv, 32);                                                        \
                                                                                                                                  \
    const uint32_t start_row_kv = src0_nrows_per_thread_kv * ith;                                                                 \
    const uint32_t end_row_kv   = MIN(start_row_kv + src0_nrows_per_thread_kv, src0_nrows_kv);                                    \
                                                                                                                                  \
    if (start_row_kv < end_row_kv) {                                                                                              \
        uint32_t ct_start_kv = start_row_kv / 32;                                                                                 \
        uint32_t ct_end_kv   = (end_row_kv + 31) / 32;                                                                            \
                                                                                                                                  \
        uint32_t push_ct = ct_start_kv;                                                                                           \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end_kv; d++, push_ct++) {                                             \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                            \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + d * tile_row_transfer_size_aligned,                            \
                           src2_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
                                                                                                                                  \
        for (uint32_t ct = ct_start_kv; ct < ct_end_kv; ct++) {                                                                   \
            const uint8_t * w_tile_k = dma_queue_pop(dma_queue).dst;                                                              \
            const uint8_t * w_tile_v = dma_queue_pop(dma_queue).dst;                                                              \
                                                                                                                                  \
            int valid_rows = (int)src0->ne[1] - (int)(ct * 32);                                                                   \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                             \
                                                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                               \
            uint32_t ir1 = 0;                                                                                                     \
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                              \
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                       \
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                       \
                                                                                                                                  \
                float * restrict dst_row0_k = (float *) (dst_k->data + ((ir1+0) * dst_k_row_size));                               \
                float * restrict dst_row1_k = (float *) (dst_k->data + ((ir1+1) * dst_k_row_size));                               \
                float * dst_ptr0_k = &dst_row0_k[ct * 32];                                                                        \
                float * dst_ptr1_k = &dst_row1_k[ct * 32];                                                                        \
                                                                                                                                  \
                float * restrict dst_row0_v = (float *) (dst_v->data + ((ir1+0) * dst_k_row_size));                               \
                float * restrict dst_row1_v = (float *) (dst_v->data + ((ir1+1) * dst_k_row_size));                               \
                float * dst_ptr0_v = &dst_row0_v[ct * 32];                                                                        \
                float * dst_ptr1_v = &dst_row1_v[ct * 32];                                                                        \
                                                                                                                                  \
                DOT_2X2(ne10, dst_ptr0_k, dst_ptr1_k, w_tile_k, src1_col0, src1_col1, valid_rows);                                \
                DOT_2X2(ne10, dst_ptr0_v, dst_ptr1_v, w_tile_v, src1_col0, src1_col1, valid_rows);                                \
            }                                                                                                                     \
                                                                                                                                  \
            for (; ir1 < src1_nrows; ++ir1) {                                                                                     \
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                            \
                                                                                                                                  \
                float * restrict dst_row_k = (float *) (dst_k->data + (ir1 * dst_k_row_size));                                    \
                float * dst_ptr_k = &dst_row_k[ct * 32];                                                                          \
                                                                                                                                  \
                float * restrict dst_row_v = (float *) (dst_v->data + (ir1 * dst_k_row_size));                                    \
                float * dst_ptr_v = &dst_row_v[ct * 32];                                                                          \
                                                                                                                                  \
                DOT_2X1(ne10, dst_ptr_k, w_tile_k, src1_col, valid_rows);                                                         \
                DOT_2X1(ne10, dst_ptr_v, w_tile_v, src1_col, valid_rows);                                                         \
            }                                                                                                                     \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                \
                                                                                                                                  \
            if (push_ct < ct_end_kv) {                                                                                            \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_k, src0_row + push_ct * tile_row_stride),                \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                             \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_v, src2_row + push_ct * tile_row_stride),                \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                             \
                push_ct++;                                                                                                        \
            }                                                                                                                     \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    /* 2. Process Q separately */                                                                                                 \
    const uint32_t src0_nrows_q = src3->ne[1] * src3->ne[2] * src3->ne[3]; /* src3 is Wq */                                       \
    uint32_t src0_nrows_per_thread_q = (src0_nrows_q + nth - 1) / nth;                                                            \
    src0_nrows_per_thread_q = hex_round_up(src0_nrows_per_thread_q, 32);                                                          \
                                                                                                                                  \
    const uint32_t start_row_q = src0_nrows_per_thread_q * ith;                                                                   \
    const uint32_t end_row_q   = MIN(start_row_q + src0_nrows_per_thread_q, src0_nrows_q);                                        \
                                                                                                                                  \
    if (start_row_q < end_row_q) {                                                                                                \
        uint32_t ct_start_q = start_row_q / 32;                                                                                   \
        uint32_t ct_end_q   = (end_row_q + 31) / 32;                                                                              \
                                                                                                                                  \
        uint32_t push_ct = ct_start_q;                                                                                            \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end_q; d++, push_ct++) {                                              \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + d * tile_row_transfer_size_aligned,                            \
                           src3_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
                                                                                                                                  \
        for (uint32_t ct = ct_start_q; ct < ct_end_q; ct++) {                                                                     \
            const uint8_t * w_tile_q = dma_queue_pop(dma_queue).dst;                                                              \
                                                                                                                                  \
            int valid_rows = (int)src3->ne[1] - (int)(ct * 32);                                                                   \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                             \
                                                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                \
            uint32_t ir1 = 0;                                                                                                     \
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                              \
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                       \
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                       \
                                                                                                                                  \
                float * restrict dst_row0_q = (float *) (dst_q->data + ((ir1+0) * dst_q_row_size));                               \
                float * restrict dst_row1_q = (float *) (dst_q->data + ((ir1+1) * dst_q_row_size));                               \
                float * dst_ptr0_q = &dst_row0_q[ct * 32];                                                                        \
                float * dst_ptr1_q = &dst_row1_q[ct * 32];                                                                        \
                                                                                                                                  \
                DOT_2X2(ne10, dst_ptr0_q, dst_ptr1_q, w_tile_q, src1_col0, src1_col1, valid_rows);                                \
            }                                                                                                                     \
                                                                                                                                  \
            for (; ir1 < src1_nrows; ++ir1) {                                                                                     \
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                            \
                                                                                                                                  \
                float * restrict dst_row_q = (float *) (dst_q->data + (ir1 * dst_q_row_size));                                    \
                float * dst_ptr_q = &dst_row_q[ct * 32];                                                                          \
                                                                                                                                  \
                DOT_2X1(ne10, dst_ptr_q, w_tile_q, src1_col, valid_rows);                                                         \
            }                                                                                                                     \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                 \
                                                                                                                                  \
            if (push_ct < ct_end_q) {                                                                                             \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_q, src3_row + push_ct * tile_row_stride),                \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                             \
                push_ct++;                                                                                                        \
            }                                                                                                                     \
        }                                                                                                                         \
    }                                                                                                                             \
}

#define MATMUL_FFN_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                          \
static void hvx_mm_ffn_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                    \
    struct htp_mm_context * mmctx = data;                                                                                         \
    struct htp_ops_context * octx = mmctx->octx;                                                                                  \
                                                                                                                                  \
    const struct htp_tensor * restrict src0 = octx->src[0]; /* Wgate */                                                           \
    const struct htp_tensor * restrict src1 = octx->src[1]; /* y */                                                               \
    const struct htp_tensor * restrict src2 = octx->src[2]; /* Wup */                                                             \
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];                                                                  \
    const struct htp_tensor * restrict dst_up = octx->dsts[1];                                                                    \
                                                                                                                                  \
    const uint32_t ne00 = src0->ne[0];                                                                                            \
    const uint32_t ne01 = src0->ne[1];                                                                                            \
    const uint32_t ne10 = src1->ne[0];                                                                                            \
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];                                                          \
                                                                                                                                  \
    const size_t dst_row_size  = dst_gate->nb[1];                                                                                 \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;                                 \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                              \
                                                                                                                                  \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                                     \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;                                                             \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                                  \
                                                                                                                                  \
    const uint32_t src0_nrows = ne01 * src0->ne[2] * src0->ne[3];                                                                 \
    const uint32_t src0_start_row = mmctx->src0_nrows_per_thread * ith;                                                           \
    const uint32_t src0_end_row   = MIN(src0_start_row + mmctx->src0_nrows_per_thread, src0_nrows);                               \
                                                                                                                                  \
    uint32_t ct_start = src0_start_row / 32;                                                                                      \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                 \
                                                                                                                                  \
    uint32_t push_ct = ct_start;                                                                                                  \
    for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                    \
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                                \
                       src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);              \
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + d * tile_row_transfer_size_aligned,                                \
                       src2_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);              \
    }                                                                                                                             \
                                                                                                                                  \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                             \
        const uint8_t * w_tile_gate = dma_queue_pop(dma_queue).dst;                                                               \
        const uint8_t * w_tile_up   = dma_queue_pop(dma_queue).dst;                                                               \
                                                                                                                                  \
        int valid_rows = (int)ne01 - (int)(ct * 32);                                                                              \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                 \
                                                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
        uint32_t ir1 = 0;                                                                                                         \
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                  \
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                           \
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                           \
                                                                                                                                  \
            float * restrict dst_row0_gate = (float *) (dst_gate->data + ((ir1+0) * dst_row_size));                               \
            float * restrict dst_row1_gate = (float *) (dst_gate->data + ((ir1+1) * dst_row_size));                               \
            float * dst_ptr0_gate = &dst_row0_gate[ct * 32];                                                                      \
            float * dst_ptr1_gate = &dst_row1_gate[ct * 32];                                                                      \
                                                                                                                                  \
            float * restrict dst_row0_up = (float *) (dst_up->data + ((ir1+0) * dst_row_size));                                   \
            float * restrict dst_row1_up = (float *) (dst_up->data + ((ir1+1) * dst_row_size));                                   \
            float * dst_ptr0_up = &dst_row0_up[ct * 32];                                                                          \
            float * dst_ptr1_up = &dst_row1_up[ct * 32];                                                                          \
                                                                                                                                  \
            DOT_2X2(ne10, dst_ptr0_gate, dst_ptr1_gate, w_tile_gate, src1_col0, src1_col1, valid_rows);                           \
            DOT_2X2(ne10, dst_ptr0_up, dst_ptr1_up, w_tile_up, src1_col0, src1_col1, valid_rows);                                 \
        }                                                                                                                         \
                                                                                                                                  \
        for (; ir1 < src1_nrows; ++ir1) {                                                                                         \
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                \
                                                                                                                                  \
            float * restrict dst_row_gate = (float *) (dst_gate->data + (ir1 * dst_row_size));                                    \
            float * dst_ptr_gate = &dst_row_gate[ct * 32];                                                                        \
                                                                                                                                  \
            float * restrict dst_row_up = (float *) (dst_up->data + (ir1 * dst_row_size));                                        \
            float * dst_ptr_up = &dst_row_up[ct * 32];                                                                            \
                                                                                                                                  \
            DOT_2X1(ne10, dst_ptr_gate, w_tile_gate, src1_col, valid_rows);                                                       \
            DOT_2X1(ne10, dst_ptr_up, w_tile_up, src1_col, valid_rows);                                                           \
        }                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                     \
                                                                                                                                  \
        if (push_ct < ct_end) {                                                                                                   \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_gate, src0_row + push_ct * tile_row_stride),                 \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_up, src2_row + push_ct * tile_row_stride),                   \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            push_ct++;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
}

MATMUL_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,   flat_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,   flat_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,   flat_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2,  flat_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2,  flat_vec_dot_mxfp4_32x1)

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_expr)                                        \
static void name(unsigned int nth, unsigned int ith, void * data) {                                        \
    struct htp_mm_context * mmctx = data;                                                                  \
    struct htp_ops_context * octx = mmctx->octx;                                                           \
    const struct htp_tensor * src = octx->src[1];                                                          \
    const uint32_t ne0 = src->ne[0];                                                                       \
    const uint32_t ne1 = src->ne[1];                                                                       \
    const uint32_t ne2 = src->ne[2];                                                                       \
    const uint32_t ne3 = src->ne[3];                                                                       \
    const uint32_t nrows = ne1 * ne2 * ne3;                                                                \
    const uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;                                        \
                                                                                                           \
    const uint32_t ir_first = nrows_per_thread * ith;                                                      \
    if (ir_first >= nrows) {                                                                               \
        return;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                              \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_src1;                                                             \
    const uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    const size_t src_row_size = src->nb[1];                                                                \
    const size_t dst_row_size = (dst_row_size_expr);                                                       \
    const uint8_t * restrict src_data = (const uint8_t *) src->data + (src_row_size * ir_first);           \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    uint8_t * restrict tmp_data = (uint8_t *) mmctx->vtcm_src0 + (mmctx->vtcm_src0_size_per_thread * ith); \
    kernel_fn(src_data, dst_data, tmp_data, ne0, ir_last - ir_first, src_row_size, dst_row_size);          \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled, "quantize-f32-q8_0_tiled", quantize_f32_q8_0_tiled_kernel, htp_mm_q8_0_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled, "quantize-f32-q8_1_tiled", quantize_f32_q8_1_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_0_flat,  "quantize-f32-q8_0_flat",  quantize_f32_q8_0_flat_kernel,  htp_mm_q8_0_flat_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_flat,  "quantize-f32-q8_1_flat",  quantize_f32_q8_1_flat_kernel,  htp_mm_q8_1_flat_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_f32_flat,   "quantize-f32-f32",        quantize_f32_f32_flat_kernel,   mmctx->vtcm_src1_stride)
QUANTIZE_IMPL(quantize_f32_f16_flat,   "quantize-f32-f16",        quantize_f32_f16_flat_kernel,   mmctx->vtcm_src1_stride)
QUANTIZE_IMPL(quantize_f16_f16_flat,   "quantize-f16-f16",        quantize_f16_f16_flat_kernel,   mmctx->vtcm_src1_stride)

static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = octx->src[1];

    quantize_f32_q8_0_tiled_block_kernel(
        (const float *) src->data,
        mmctx->vtcm_src1,
        (uint8_t *) mmctx->vtcm_src0 + (mmctx->vtcm_src0_size_per_thread * ith),
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        src->nb[1],
        htp_mm_q8_0_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = octx->src[1];

    quantize_f32_q8_1_tiled_block_kernel(
        (const float *) src->data,
        mmctx->vtcm_src1,
        (uint8_t *) mmctx->vtcm_src0 + (mmctx->vtcm_src0_size_per_thread * ith),
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        src->nb[1],
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

MATVEC_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x1)

MATVEC_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x1)


MATMUL_QKV_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_QKV_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,   flat_vec_dot_q4_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,   flat_vec_dot_q4_1_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,   flat_vec_dot_q8_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2,  flat_vec_dot_iq4nl_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2,  flat_vec_dot_mxfp4_32x1)


MATMUL_FFN_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_FFN_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,   flat_vec_dot_q4_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,   flat_vec_dot_q4_1_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,   flat_vec_dot_q8_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2,  flat_vec_dot_iq4nl_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2,  flat_vec_dot_mxfp4_32x1)

static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

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
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data     = mmctx->vtcm_src1;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;

    // Prefill vtcm with src0 rows
    #pragma unroll(4)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= (int)n_prefetch) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
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
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const int pr0 = (ir0 + n_prefetch);
        const int is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
}

static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
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
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * src1_data     = mmctx->vtcm_src1;

    float * tmp = (float *) vtcm_dst_ptr;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
    float * restrict dst_col          = (float *) dst->data;

    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    #pragma unroll(2)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint32_t is0 = (ir0 - src0_start_row);
        if (is0 >= n_prefetch) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const uint32_t pr0 = (ir0 + n_prefetch);
        const uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, src0_end_row - src0_start_row);
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ids->ne[0] * ids->ne[1] + (i1)]

static void hvx_mm_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows      = ne01;  // src0 rows per expert
    const uint32_t src1_nrows      = ne11;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

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
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // src1 row idx
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }
}

static void hvx_mv_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    const uint32_t src0_nrows      = ne01;  // src0 rows per expert
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

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
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }
}

static int hvx_mm_init_vec_dot(struct htp_mm_context * mmctx, enum htp_data_type type) {
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

static int hvx_mm_matmul(struct htp_ops_context * octx) {
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
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mm_2d;
        }
    } else {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mv_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mv_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mv_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mv_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mv_2d;
        }
    }

    bool need_quant = true;

    switch (kparams->kernel_type) {
        case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_job_func         = (src1->type == HTP_TYPE_F32) ? quantize_f32_f16_flat : quantize_f16_f16_flat;
            mmctx->type            = "f16-f16";
            mmctx->vec_dot_1x1     = vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f16_f16_aa_2x2;
            src1_row_size          = hex_round_up(ne10 * 2, 128);
            break;

        case HTP_MM_KERNEL_HVX_F16_F32_DDR:
            mmctx->type            = "f16-f32";
            mmctx->vec_dot_1x1     = vec_dot_f16_f32_uu_1x1;
            matmul_job_func        = hvx_mm_4d;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            need_quant             = false;
            quant_job_func         = NULL;
            src1_row_size          = nb11;
            break;

        case HTP_MM_KERNEL_HVX_F16_F16_DDR:
            mmctx->type            = "f16-f16";
            mmctx->vec_dot_1x1     = vec_dot_f16_f16_uu_1x1;
            matmul_job_func        = hvx_mm_4d;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            src1_row_size          = nb11;
            need_quant             = false;
            quant_job_func         = NULL;
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_job_func         = quantize_f32_f32_flat;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f32_f32_aa_2x2;
            src1_row_size          = hex_round_up(ne10 * 4, 128);
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_DDR:
            quant_job_func         = NULL;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = vec_dot_f32_f32_uu_1x1;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            src1_row_size          = nb11;
            need_quant             = false;
            matmul_job_func        = hvx_mm_4d;
            break;

        case HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT: {
            n_quant_jobs = MIN(src1_nrows, octx->n_threads);
            quant_job_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
            src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

            if (src1_nrows > 1) {
                switch (src0->type) {
                    case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0_flat;   break;
                    case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_2d_repacked_q4_1_flat;   break;
                    case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0_flat;   break;
                    case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_2d_repacked_iq4nl_flat;  break;
                    case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_2d_repacked_mxfp4_flat;  break;
                    default:              return HTP_STATUS_NO_SUPPORT;
                }
            } else {
                switch (src0->type) {
                    case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mv_2d_repacked_q4_0_flat;   break;
                    case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mv_2d_repacked_q4_1_flat;   break;
                    case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0_flat;   break;
                    case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mv_2d_repacked_iq4nl_flat;  break;
                    case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mv_2d_repacked_mxfp4_flat;  break;
                    default:              return HTP_STATUS_NO_SUPPORT;
                }
            }
            break;
        }

        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        case HTP_MM_KERNEL_HVX_QUANT_ROW:
        default:
            if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
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
        const uint32_t n_prefetch = kparams->n_prefetch;
        assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
        htp_mm_hvx_get_vtcm_sizes(
            kparams->kernel_type, src0->type, ne10, src1_nrows, octx->n_threads,
            dst_row_size, src0_row_size, src1_row_size, n_prefetch,
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
    mmctx->vtcm_dst_size_per_thread  = dst_sz  / octx->n_threads;

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : (src1_sz + src0_sz + dst_sz);

    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
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

static void hvx_mm_qkv_2d(unsigned int nth, unsigned int ith, void * data) {
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

    const size_t dst_k_row_size  = dst_k->nb[1]; // K and V share output width
    const size_t dst_q_row_size  = dst_q->nb[1]; // Q may be wider (GQA)
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

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;
    const uint8_t * restrict src3_row = (const uint8_t *) src3->data;

    // Prefill spad with src0, src2, src3 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= (int)n_prefetch) {
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

            float * restrict dst_row0_k = (float *) (dst_k->data + ((ir1+0) * dst_k_row_size));
            float * restrict dst_row1_k = (float *) (dst_k->data + ((ir1+1) * dst_k_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_k[ir0], &dst_row1_k[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);

            float * restrict dst_row0_v = (float *) (dst_v->data + ((ir1+0) * dst_k_row_size));
            float * restrict dst_row1_v = (float *) (dst_v->data + ((ir1+1) * dst_k_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_v[ir0], &dst_row1_v[ir0], ss2, ss2 + src2_stride, src1_col0, src1_col1);

            float * restrict dst_row0_q = (float *) (dst_q->data + ((ir1+0) * dst_q_row_size));
            float * restrict dst_row1_q = (float *) (dst_q->data + ((ir1+1) * dst_q_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_q[ir0], &dst_row1_q[ir0], ss3, ss3 + src3_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_k          = (float *) (dst_k->data + (ir1 * dst_k_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_k[ir0], ss0, ss0 + src0_stride, src1_col);

            float * restrict dst_row_v          = (float *) (dst_v->data + (ir1 * dst_k_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_v[ir0], ss2, ss2 + src2_stride, src1_col);

            float * restrict dst_row_q          = (float *) (dst_q->data + (ir1 * dst_q_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_q[ir0], ss3, ss3 + src3_stride, src1_col);
        }

        // Prefetch next (n + vtcm_nrows) rows
        const int pr0 = (ir0 + n_prefetch);
        const int is0 = (pr0 - src0_start_row) & prefetch_mask;
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
        const int is0 = (ir0 - src0_start_row) & prefetch_mask;
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

            float * restrict dst_row_k          = (float *) (dst_k->data + (ir1 * dst_k_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_k[ir0], ss0, src1_col);

            float * restrict dst_row_v          = (float *) (dst_v->data + (ir1 * dst_k_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_v[ir0], ss2, src1_col);

            float * restrict dst_row_q          = (float *) (dst_q->data + (ir1 * dst_q_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_q[ir0], ss3, src1_col);
        }
    }
}

static void hvx_mm_ffn_2d(unsigned int nth, unsigned int ith, void * data) {
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

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;

    // Prefill spad with src0, src2 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= (int)n_prefetch) {
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

            float * restrict dst_row0_up   = (float *) (dst_up->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1_up   = (float *) (dst_up->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_up[ir0], &dst_row1_up[ir0], ss2, ss2 + src2_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_gate     = (float *) (dst_gate->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_gate[ir0], ss0, ss0 + src0_stride, src1_col);

            float * restrict dst_row_up       = (float *) (dst_up->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_up[ir0], ss2, ss2 + src2_stride, src1_col);
        }

        // Prefetch next rows
        const int pr0 = (ir0 + n_prefetch);
        const int is0 = (pr0 - src0_start_row) & prefetch_mask;
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
        const int is0 = (ir0 - src0_start_row) & prefetch_mask;
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

#define DEQUANTIZE_WORKER_LOOP_IMPL(SUFFIX)                                                     \
static void dequantize_tiled_worker_loop_##SUFFIX(unsigned int n, unsigned int i, void *data) { \
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;                         \
    struct htp_thread_trace * tr = state->traces ? &state->traces[i] : NULL;                    \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                  \
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {      \
        int start = task_id * state->n_tiles_per_task;                                          \
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);              \
        dequantize_tiled_weight_to_fp16_task_##SUFFIX(state, start, end);                       \
    }                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                   \
}

DEQUANTIZE_WORKER_LOOP_IMPL(q4_0)
DEQUANTIZE_WORKER_LOOP_IMPL(q4_1)
DEQUANTIZE_WORKER_LOOP_IMPL(iq4_nl)
DEQUANTIZE_WORKER_LOOP_IMPL(mxfp4)
DEQUANTIZE_WORKER_LOOP_IMPL(q8_0)

static void convert_f16_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;
    struct htp_thread_trace * tr = state->traces ? &state->traces[i] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        convert_f16_weight_to_fp16_tiles_task(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

static void quantize_f32_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;

    struct htp_thread_trace * tr = state->traces ? &state->traces[i] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);

    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        quantize_f32_weight_to_fp16_tiles_task(state, start, end);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);
}

static void transfer_output_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;

    int start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        float        *dst      = st->dst      + chunk_idx * st->dst_stride;
        transfer_output_chunk_fp16_to_fp32(dst, st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->dst_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;

    int start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, start_chunk_idx);

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        const float *src = st->src + chunk_idx * st->k_stride;

        if (st->vtcm_f32_act) {
            float *thread_f32_act = st->vtcm_f32_act + i * HTP_MM_DMA_ACT_MULTIPLIER * st->k_block;
            transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
                st->ctx->dma[i], dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_f32_act
            );
        } else {
            transfer_activation_chunk_fp32_to_fp16(dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, start_chunk_idx);
}

static void transfer_activation_chunk_gathered_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int start_row = st->start_row + chunk_idx * chunk_size;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_activation_chunk_gathered_worker_flat_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int start_row = st->start_row + chunk_idx * chunk_size;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_output_chunk_scattered_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_scattered_task_state_t *st = data;
    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int start_row = st->start_row + chunk_idx * chunk_size;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

static void dequantize_tiled_weight_chunk_to_fp16_tiles(
        struct htp_context *ctx, __fp16 *vtcm_dst,
        const void *weight_src_ddr,
        int n_cols, int k_block,
        size_t row_stride, int weight_type,
        int n_k_tiles, struct fastdiv_values n_k_tiles_div,
        worker_callback_t dequant_worker_fn, int n_threads) {

    assert(n_cols  % HTP_MM_HMX_TILE_N_COLS == 0);
    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_col_tiles = n_cols / HTP_MM_HMX_TILE_N_COLS;
    size_t n_tot_tiles = n_col_tiles * n_k_tiles;

    size_t n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (const uint8_t *)weight_src_ddr;
    state.n_cols           = n_cols;
    state.k_block          = k_block;
    state.row_stride       = row_stride;
    state.weight_type      = weight_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_size = htp_mm_get_weight_tile_size(weight_type);
    state.aligned_tile_size = htp_mm_get_weight_aligned_tile_size(weight_type);

    if (state.n_tasks == 1 || n_threads == 1) {
        dequant_worker_fn(1, 0, &state);
    } else {
        int n_tasks = hex_smin((int) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, dequant_worker_fn, &state, n_tasks);
    }
}

static void transfer_output_chunk_threaded(struct htp_context *ctx, float *dst, const __fp16 *vtcm_src,
                                              int n_rows, int n_cols, int dst_stride, int dst_cols, int n_threads) {
    assert(n_cols % HTP_MM_HMX_TILE_N_COLS == 0);

    if (n_rows <= 0) return;

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : hmx_ceil_div(n_rows, n_threads);
    n_chunks_per_task        = hex_align_up(n_chunks_per_task, 2);

    int actual_threads = hmx_ceil_div(n_rows, n_chunks_per_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_threads;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_per_task = n_chunks_per_task;
    state.dst               = dst;
    state.vtcm_src          = vtcm_src;
    state.n_cols            = n_cols;
    state.dst_stride        = dst_stride;
    state.dst_cols          = dst_cols;
    state.traces            = ctx->trace;

    if (actual_threads <= 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, actual_threads);
    }
}

static void transfer_activation_chunk_threaded(
        struct htp_context *ctx,
        __fp16 *dst,
        const float *src,
        int n_rows,
        int k_block,
        int k_stride,
        int n_threads,
        int k_valid,
        float *vtcm_f32_act) {
    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks           = (n_tot_chunks + n_chunks_per_task - 1) / n_chunks_per_task;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_per_task = n_chunks_per_task;
    state.dst               = dst;
    state.src               = src;
    state.k_block           = k_block;
    state.k_stride          = k_stride;
    state.k_valid           = k_valid;
    state.traces            = ctx->trace;
    state.ctx               = ctx;
    state.vtcm_f32_act      = vtcm_f32_act;

    if (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } else {
        int n_tasks = hex_smin((int) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, n_tasks);
    }
}

static int hmx_mm_2d_f32(struct htp_context *ctx,
                                  float *restrict dst,
                                  const float *activation,
                                  const uint8_t *weight,
                                  int m, int k, int n,
                                  int act_stride,
                                  int weight_stride,
                                  int weight_type,
                                  int k_valid,
                                  int dst_stride,
                                  int dst_cols,
                                  int m_chunk,
                                  int n_chunk,
                                  int pipeline,
                                  int n_threads,
                                  int act_threads,
                                  int tile_size,
                                  int aligned_tile_size,
                                  int vtcm_size) {
    if (k % 32 != 0 || n % 32 != 0) { return -1; }
    if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { return -1; }

    size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_quant     = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = vtcm_size;

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;

    const size_t act_f32_size     = hex_align_up((size_t)act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * sizeof(float), HTP_MM_HMX_TILE_SIZE);

    const size_t weight_area_size = is_quant
        ? hex_align_up((n_chunk_n_cols / 32) * n_k_tiles * aligned_tile_size, HTP_MM_HMX_TILE_SIZE)
        : hex_align_up(n_chunk_n_cols * row_stride, HTP_MM_HMX_TILE_SIZE);
    const size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    const size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);

    size_t scratch0_size, scratch1_size, scratch2_size;
    scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);  // dequant buf 0
    scratch1_size = pipeline ? scratch0_size : 0;                                 // dequant buf 1
    scratch2_size = pipeline ? output_area_size : 0;                              // output  buf 1

    uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = { NULL, NULL };
    if (weight_area_size) {
        if (pipeline) {
            vtcm_weight_raw[0] = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
            vtcm_weight_raw[1] = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
        } else {
            vtcm_weight_raw[0] = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
        }
    }
    __fp16  *vtcm_f16_act = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    float   *vtcm_f32_act    = (float *) vtcm_seq_alloc(&vtcm_ptr, act_f32_size);
    __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    void    *vtcm_scratch1   = scratch1_size ? vtcm_seq_alloc(&vtcm_ptr, scratch1_size) : NULL;
    void    *vtcm_scratch2   = scratch2_size ? vtcm_seq_alloc(&vtcm_ptr, scratch2_size) : NULL;
    __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        return -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "hmx-mm-2d-precomputed: standard : m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    int n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    if (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors

        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            void *vtcm_weight_bufs[2] = { vtcm_scratch0, vtcm_scratch1 };
            void *vtcm_output_bufs[2] = { vtcm_output,   vtcm_scratch2 };

            transfer_activation_chunk_threaded(ctx, vtcm_f16_act, activation + mr * act_stride, n_rows, k, act_stride, act_threads, k_valid, vtcm_f32_act);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            const size_t n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            if (is_quant) {
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight), aligned_tile_size, tile_size, tile_size, (n_cols_A0 / 32) * n_k_tiles);
            } else {
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight), row_stride, weight_stride, row_stride, n_cols_A0);
            }

            if (1 < n_chunk_cnt) {
                const size_t n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                if (is_quant) {
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride), aligned_tile_size, tile_size, tile_size, (n_cols_A1 / 32) * n_k_tiles);
                } else {
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride), row_stride, weight_stride, row_stride, n_cols_A1);
                }
            }

            // pop A0 -> dequantize A0 -> submit C0
            dma_queue_pop(ctx->dma[0]);
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_weight_bufs[0], vtcm_weight_raw[0],
                n_cols_A0, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

            hmx_matmul_job_init(&job_slots[0], (__fp16 *) vtcm_output_bufs[0], (__fp16 *) vtcm_f16_act,
                                (__fp16 *) vtcm_weight_bufs[0], vtcm_scales,
                                hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                hmx_ceil_div(n_cols_A0, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[0]));

            // Main loop: pop/dequantize A_{i+1} -> push A_{i+2} -> submit C_{i+1} -> wait C_i and store D_i
            for (int i = 0; i < n_chunk_cnt; ++i) {
                const size_t nc    = i * n_chunk_n_cols;
                const size_t nc_p1 = nc + 1 * n_chunk_n_cols;
                const size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                const size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_cols_p1 = hex_smin(n - nc_p1, n_chunk_n_cols);
                const size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_{i+1} and dequantize it (if i+1 < n_chunk_cnt)
                if (i + 1 < n_chunk_cnt) {
                    dma_queue_pop(ctx->dma[0]);
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[(i + 1) % 2], vtcm_weight_raw[(i + 1) % 2],
                        n_cols_p1, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);
                }

                // 2. push A_{i+2} (if i+2 < n_chunk_cnt)
                if (i + 2 < n_chunk_cnt) {
                    if (is_quant) {
                        dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[(i + 2) % 2], weight + nc_p2 * weight_stride), aligned_tile_size, tile_size, tile_size, (n_cols_p2 / 32) * n_k_tiles);
                    } else {
                        dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[(i + 2) % 2], weight + nc_p2 * weight_stride), row_stride, weight_stride, row_stride, n_cols_p2);
                    }
                }

                // 3. submit C_{i+1} (if i+1 < n_chunk_cnt)
                if (i + 1 < n_chunk_cnt) {
                    hmx_matmul_job_init(&job_slots[(i + 1) % 2], (__fp16 *) vtcm_output_bufs[(i + 1) % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[(i + 1) % 2],
                                        vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                        hmx_ceil_div(n_cols_p1, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[(i + 1) % 2]));
                }

                // 4. wait C_i and store D_i (multi-thread HVX, parallel with C_{i+1})
                hmx_queue_pop(ctx->hmx_queue);
                float *output_chunk = dst + (mr * dst_stride + nc);
                int chunk_dst_cols = dst_cols - (int)nc;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, vtcm_output_bufs[i % 2], n_rows, n_cols, dst_stride, chunk_dst_cols, n_threads);
                }
            }
        }
        hmx_queue_suspend(ctx->hmx_queue);
    } else {
        // --- Synchronous Un-pipelined loop (m <= 32 or fallback) ---
        HAP_compute_res_hmx_lock(ctx->vtcm_rctx);
        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            transfer_activation_chunk_threaded(ctx, vtcm_f16_act, activation + mr * act_stride, n_rows, k, act_stride, act_threads, k_valid, vtcm_f32_act);

            for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
                const size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Weight DMA (Synchronous)
                if (is_quant) {
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight + nc * weight_stride), aligned_tile_size, tile_size, tile_size, (n_cols / 32) * n_k_tiles);
                } else {
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight + nc * weight_stride), row_stride, weight_stride, row_stride, n_cols);
                }
                dma_queue_pop(ctx->dma[0]);

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_scratch0, vtcm_weight_raw[0],
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // C: HMX Compute (Synchronous)
                core_dot_chunk_fp16(vtcm_output, vtcm_f16_act, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);

                // D: Output Store
                float *output_chunk = dst + (mr * dst_stride + nc);
                int chunk_dst_cols = dst_cols - (int)nc;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, vtcm_output, n_rows, n_cols, dst_stride, chunk_dst_cols, n_threads);
                }
            }
        }
        HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
    }

    return 0;
}

static inline int hmx_mm_batch_r2(const hmx_mm_f16_f32_batched_params_t *params) {
    return params->ne02 > 0 ? params->ne12 / params->ne02 : 1;
}

static inline int hmx_mm_batch_r3(const hmx_mm_f16_f32_batched_params_t *params) {
    return params->ne03 > 0 ? params->ne13 / params->ne03 : 1;
}

static inline const __fp16 *hmx_mm_weight_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
                                                        int dst_b2, int dst_b3) {
    const int r2 = hmx_mm_batch_r2(params);
    const int r3 = hmx_mm_batch_r3(params);
    return (const __fp16 *) ((const uint8_t *) params->weight +
                             (size_t) (dst_b2 / r2) * params->src0_nb2 +
                             (size_t) (dst_b3 / r3) * params->src0_nb3);
}

static inline const float *hmx_mm_activation_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
                                                           int dst_b2, int dst_b3) {
    return (const float *) ((const uint8_t *) params->activation +
                            (size_t) dst_b2 * params->src1_nb2 +
                            (size_t) dst_b3 * params->src1_nb3);
}

static inline float *hmx_mm_dst_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
                                              int dst_b2, int dst_b3) {
    return (float *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

static int hmx_mm_f16_f32_batched_simple(struct htp_context *ctx,
                                                        const hmx_mm_f16_f32_batched_params_t *params,
                                                        int m_chunk, int n_chunk, int pipeline, int n_threads, int act_threads, int vtcm_size) {
    int ret = 0;
    for (int b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        for (int b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            ret = hmx_mm_2d_f32(ctx, hmx_mm_dst_batch_ptr(params, b2, b3),
                                           hmx_mm_activation_batch_ptr(params, b2, b3),
                                           (const uint8_t *)hmx_mm_weight_batch_ptr(params, b2, b3),
                                           params->m, params->k, params->n,
                                           params->act_stride, params->weight_stride * (int)sizeof(__fp16),
                                           HTP_TYPE_F16, params->k, params->n, params->n,
                                           m_chunk, n_chunk, pipeline, n_threads, act_threads,
                                           0, 0, vtcm_size);
        }
    }
    return ret;
}

static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_batched_params_t *params,
                               int m_chunk, int n_chunk, int pipeline, int n_threads, int act_threads, int vtcm_size) {
    if (params->act_stride < params->k || params->weight_stride < params->k || params->dst_stride < params->n) { return -1; }
    if (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { return -1; }
    if (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { return -1; }
    if (params->k % 32 != 0 || params->n % 32 != 0) { return -1; }
    if (!hex_is_aligned(params->dst, VLEN) || !hex_is_aligned(params->activation, VLEN)) { return -1; }

    const int group_size = hmx_mm_batch_r2(params);
    const size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (group_size > 1 && (size_t)vtcm_size <= vtcm_budget);
    if (!run_grouped) {
        return hmx_mm_f16_f32_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size);
    }

    const size_t vec_dot_size = params->k * sizeof(__fp16);

    const bool use_dma_activation = (params->act_stride > params->k);
    const size_t f32_scratch_size = use_dma_activation
        ? hex_align_up((size_t)act_threads * HTP_MM_DMA_ACT_MULTIPLIER * (size_t) params->k * sizeof(float), HTP_MM_HMX_TILE_SIZE) : 0;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used = vtcm_size;

    const size_t act_head_stride      = m_chunk_n_rows * (size_t) params->k;  // fp16 elements between heads
    const size_t weight_area_size     = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    const size_t activation_area_size = hex_align_up(group_size * m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    const size_t output_area_size     = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);
    const size_t scratch_area_size    = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);

    uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
    __fp16  *vtcm_f16_act    = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
    __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
    void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
    __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
    float   *vtcm_f32_act    = use_dma_activation ? (float *) vtcm_seq_alloc(&vtcm_ptr, f32_scratch_size) : NULL;

    if ((size_t) (vtcm_ptr - (uint8_t *) ctx->vtcm_base) > vtcm_budget) {
        FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        return hmx_mm_f16_f32_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            (size_t) (vtcm_ptr - (uint8_t *) ctx->vtcm_base), vtcm_budget);

    const size_t fp16_row_bytes   = (size_t) params->k * sizeof(__fp16);
    const size_t weight_row_bytes = (size_t) params->weight_stride * sizeof(__fp16);

    HAP_compute_res_hmx_lock(ctx->vtcm_rctx);

    for (int b3 = 0; b3 < params->ne13; ++b3) {
        for (int b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            const __fp16 *weight_group = hmx_mm_weight_batch_ptr(params, b2_base, b3);

            for (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
                const size_t n_rows = hex_smin((size_t) params->m - mr, m_chunk_n_rows);
                const size_t n_row_tiles = hmx_ceil_div((int) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                for (int g = 0; g < group_size; ++g) {
                    const float *activation_chunk = hmx_mm_activation_batch_ptr(params, b2_base + g, b3) + mr * params->act_stride;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * act_head_stride;
                    if (use_dma_activation) {
                        transfer_activation_chunk_threaded(ctx, vtcm_act_g,
                                                               activation_chunk, (int) n_rows,
                                                               params->k, params->act_stride, act_threads, params->k, vtcm_f32_act);
                    } else {
                        transfer_activation_chunk_threaded(ctx, vtcm_act_g,
                                                               activation_chunk, (int) n_rows,
                                                               params->k, params->act_stride, act_threads, params->k, NULL);
                    }
                }

                void *buf_curr = vtcm_scratch0;
                void *buf_next = vtcm_scratch1;

                {
                    const size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);
                    dma_queue_push(ctx->dma[0], dma_make_ptr(buf_curr, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }

                for (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    const size_t n_cols = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    const size_t n_col_tiles = hmx_ceil_div((int) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        dma_queue_pop(ctx->dma[0]);

                        const size_t nc_next = nc + n_chunk_n_cols;
                        if (nc_next < (size_t) params->n) {
                            const size_t n_cols_next = hex_smin((size_t) params->n - nc_next, n_chunk_n_cols);
                            const __fp16 *next_weight_chunk = weight_group + nc_next * params->weight_stride;

                            dma_queue_push(ctx->dma[0], dma_make_ptr(buf_next, next_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }

                        hmx_interleave_rows_to_tiles(vtcm_weight, (const __fp16 *) buf_curr, n_cols, params->k, params->k, 0, n_cols);
                        hex_swap_ptr(&buf_curr, &buf_next);
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    for (int g = 0; g < group_size; ++g) {
                        struct htp_thread_trace * tr = &ctx->trace[HTP_MAX_NTHREADS];
                        htp_trace_event_start(tr, HTP_TRACE_EVT_HMX_COMP, g);
                        {
                            const __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * act_head_stride;
                            core_dot_chunk_fp16(vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles,
                                                params->k / 32);
                        }
                        htp_trace_event_stop(tr, HTP_TRACE_EVT_HMX_COMP, g);

                        {
                            float *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            int chunk_dst_cols = params->n - (int)nc;
                            if (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, vtcm_output, (int) n_rows, (int) n_cols, params->dst_stride, chunk_dst_cols, ctx->n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);

    return 0;
}

static void transfer_activation_chunk_gathered_threaded(
            struct htp_context *ctx,
            __fp16 *dst,
            const float *src,
            int start_row,
            int n_rows,
            int k_block,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            int ne11,
            size_t nb11,
            size_t nb12,
            int cne1,
            int n_threads,
            int k_valid) {
    if (n_rows <= 0) return;
    int chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    int actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_threads,
        .n_tot_chunks      = n_rows,
        .n_chunks_per_task = chunks_per_thread,
        .k_block           = k_block,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_values(ne11) : (struct fastdiv_values){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    worker_callback_t worker_fn = ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn :
                                              transfer_activation_chunk_gathered_worker_fn;

    if (actual_threads <= 1) {
        worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, worker_fn, &state, actual_threads);
    }
}

static void transfer_output_chunk_scattered_threaded(
            struct htp_context *ctx,
            float *dst,
            const __fp16 *vtcm_src,
            int start_row,
            int n_rows,
            int n_cols,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            int cne1,
            int n_threads) {
    if (n_rows <= 0) return;
    int chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    int actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_threads,
        .n_tot_chunks      = n_rows,
        .n_chunks_per_task = chunks_per_thread,
        .n_cols            = n_cols,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .start_row         = start_row,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    if (actual_threads <= 1) {
        transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_scattered_worker_fn, &state, actual_threads);
    }
}

static int hmx_mm_id_2d_f32(struct htp_context *ctx,
                                         float *restrict dst,
                                         const float *activation,
                                         const uint8_t *weight,
                                         int m, int k, int n,
                                         int k_valid,
                                         int ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         int weight_stride,
                                         int weight_type,
                                         const struct mmid_row_mapping *matrix_rows,
                                         int cur_a,
                                         int mapping_stride) {
    const int cne1 = m;
    const int m_padded = hex_align_up(m, 32);

    if (k % 32 != 0 || n % 32 != 0) { return -1; }
    if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { return -1; }

    size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const int n_threads = ctx->n_threads;
    const bool is_quant   = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);

    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;
    size_t vtcm_used = 0;

    int tile_size = htp_mm_get_weight_tile_size(weight_type);
    int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(weight_type);

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;
    const size_t weight_row_stride = is_quant ? qweight_row_stride : row_stride;

    size_t size_per_n = 0, size_per_m = 0, size_per_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_size,
                                  &size_per_n, &size_per_m, &size_per_mn);

    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    if (htp_mm_hmx_compute_chunks(vtcm_budget, /*overhead=*/256, size_per_n, size_per_m, size_per_mn,
                           m_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_padded, k, n, vtcm_budget);
        return -1;
    }

    const size_t weight_area_size = hex_align_up(n_chunk_n_cols * weight_row_stride, HTP_MM_HMX_TILE_SIZE);
    const size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    const size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);

    size_t scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);

    uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = weight_area_size ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : NULL;
    __fp16  *vtcm_f16_act = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        return -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    HAP_compute_res_hmx_lock(ctx->vtcm_rctx);

    for (size_t mr = 0; mr < (size_t) m_padded; mr += m_chunk_n_rows) {
        const size_t n_rows = hex_smin(m_padded - mr, m_chunk_n_rows);
        const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (int) mr, (int) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_threads, k_valid);

        for (size_t nc = 0; nc < (size_t) n; nc += n_chunk_n_cols) {
            const size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
            const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            if (is_quant) {
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight, weight + nc * weight_stride), aligned_tile_size, tile_size, tile_size, (n_cols / 32) * n_k_tiles);
            } else {
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight, weight + nc * weight_stride), row_stride, weight_stride, row_stride, n_cols);
            }
            dma_queue_pop(ctx->dma[0]);

            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_scratch0, vtcm_weight,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads
            );

            struct htp_thread_trace * tr = &ctx->trace[HTP_MAX_NTHREADS];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HMX_COMP, nc);
            core_dot_chunk_fp16(vtcm_output, vtcm_f16_act, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HMX_COMP, nc);

            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (int) mr, (int) n_rows, (int) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_threads);
        }
    }

    HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
    return 0;
}


// --- Dispatchers and Public Entry Points ---

static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    int k = (int) src0->ne[0];
    int n = (int) src0->ne[1];
    const int m_total    = (int) src1->ne[1];
    const int act_stride = (int)(src1->nb[1] / sizeof(float));
    const int wgt_stride = (int)(src0->nb[1] / sizeof(__fp16));

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    int ret = -1;
    const int n_threads = MIN(kparams->n_threads, (int) octx->n_threads);
    if (kparams->kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
        hmx_mm_f16_f32_batched_params_t batch_params = {
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
        ret = hmx_mm_f16_f32_batched(octx->ctx, &batch_params,
                                     kparams->m_chunk, kparams->n_chunk,
                                     kparams->pipeline, n_threads,
                                     kparams->n_act_threads,
                                     kparams->vtcm_size);
    } else {
        ret = hmx_mm_2d_f32(
            octx->ctx, (float*) dst->data, (float*) src1->data, (const uint8_t *) src0->data,
            m_total, k, n, act_stride, (int) src0->nb[1], (int) src0->type, (int) src1->ne[0],
            (int)(dst->nb[1] / sizeof(float)), (int)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size
        );
    }

    if (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        return HTP_STATUS_INTERNAL_ERR;
    }
    return HTP_STATUS_OK;
}

int op_matmul(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    if (kparams->n_hmx) {
        return hmx_mm_op_matmul(octx, kparams);
    }

    return hvx_mm_matmul(octx);
}

static int hmx_mm_op_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    const uint32_t * matrix_row_counts,
    const struct mmid_row_mapping * matrix_rows,
    void * mapping_buf,
    bool must_free_mapping
) {
    htp_matmul_tensors_preamble;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const int n_ids = octx->src[2]->ne[0];
    const int n_as  = ne02;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        int ret = hmx_mm_id_2d_f32(octx->ctx, (float*) dst->data, (float*) src1->data,
                                   (const uint8_t *) src0->data + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (int) src0->nb[1], (int) src0->type,
                                   matrix_rows, cur_a, n_ids * octx->src[2]->ne[1]);
        if (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            if (must_free_mapping) free(mapping_buf);
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    if (must_free_mapping) free(mapping_buf);
    return HTP_STATUS_OK;
}

static int hvx_mm_op_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    size_t src0_row_size_padded,
    uint32_t src1_nrows,
    worker_callback_t matmul_id_job_func,
    void * mapping_buf,
    bool must_free_mapping
) {
    htp_matmul_tensors_preamble;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const struct htp_tensor * restrict ids = octx->src[2];
    const size_t src0_row_size = nb01;

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne10 + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    worker_callback_t quant_job_func;
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
    size_t src1_row_size  = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

    // Scratchpad sizes are computed on the host (htp_mm_hvx_id_get_vtcm_sizes) and passed in.
    // The ID layout is routing-independent, so the host has exact visibility -- consume it here
    // rather than recomputing, to keep host budgeting and device allocation in lockstep.
    size_t src0_sz = kparams->vtcm_src0_size;
    size_t src1_sz = kparams->vtcm_src1_size;
    size_t src2_sz = 0; // mapping lives in DDR
    size_t dst_sz  = 0; // ID kernels scatter straight to DDR
    size_t vtcm_size = kparams->vtcm_size;

    size_t src0_sz_per_thread = src0_sz / octx->n_threads;
    size_t src1_sz_per_thread = src1_sz;
    size_t src2_sz_per_thread = 0;
    size_t dst_sz_per_thread  = 0;

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
    worker_callback_t matmul_id_job_func = src1_nrows > 1 ? hvx_mm_id : hvx_mv_id;

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    mmctx->src0_nrows_per_thread  = hex_round_up(mmctx->src0_nrows_per_thread, 32);

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

    if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
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

    if (kparams->n_hmx) {
        return hmx_mm_op_matmul_id(octx, mmctx, matrix_row_counts, matrix_rows, mapping_buf, must_free_mapping);
    }

    return hvx_mm_op_matmul_id(octx, mmctx, src0_row_size_padded, src1_nrows, matmul_id_job_func, mapping_buf, must_free_mapping);
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

    if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
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
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_qkv_2d_repacked_q4_0_flat;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_qkv_2d_repacked_q4_1_flat;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_qkv_2d_repacked_q8_0_flat;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_qkv_2d_repacked_iq4nl_flat;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_qkv_2d_repacked_mxfp4_flat;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_qkv_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_qkv_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_qkv_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_qkv_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_qkv_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        }
    } else {
        matmul_job_func = hvx_mm_qkv_2d;
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

    if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
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
        FARF(ERROR, "matmul-ffn: current VTCM reservation %zu is too small, needed %zu\n", octx->ctx->vtcm_size, vtcm_size);
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
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_ffn_2d_repacked_q4_0_flat;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_ffn_2d_repacked_q4_1_flat;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_ffn_2d_repacked_q8_0_flat;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_ffn_2d_repacked_iq4nl_flat;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_ffn_2d_repacked_mxfp4_flat;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_ffn_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_ffn_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_ffn_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_ffn_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_ffn_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        }
    } else {
        matmul_job_func = hvx_mm_ffn_2d;
    }
    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}
