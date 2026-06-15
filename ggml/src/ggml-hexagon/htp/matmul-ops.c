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
#include "htp-ops.h"
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

#define MM_SPAD_SRC0_NROWS 16
#define MM_SPAD_SRC1_NROWS 16
#define MM_SPAD_DST_NROWS  2
#define DMA_DEPTH          4

struct htp_matmul_context {
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

    uint32_t tile_size;

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

static inline size_t q8_0_tiled_row_size(uint32_t ne) {
    const uint32_t nb_32 = (ne + 31) / 32;
    return nb_32 * 1152;
}

static inline size_t q8_1_tiled_row_size(uint32_t ne) {
    const uint32_t nb_32 = (ne + 31) / 32;
    return nb_32 * 1280;
}

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
    struct htp_spad * restrict src0_spad = &octx->src0_spad;    \
    struct htp_spad * restrict src1_spad = &octx->src1_spad;    \
    struct htp_spad * restrict dst_spad  = &octx->dst_spad;     \
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
    struct htp_matmul_context * mmctx = data;                           \
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
    const HVX_UVector * restrict vptr,
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
    const HVX_UVector * restrict vptr,
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
    const HVX_UVector * restrict vptr,
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
    const HVX_UVector * restrict vptr,
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
    const HVX_UVector * restrict vptr,
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
    const HVX_UVector * restrict vptr,
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


static void tiled_vec_dot_q4_0_32x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 576);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, i8);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_q4_0_32x2(const int n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_qf16     = Q6_V_vzero();
    HVX_Vector zero           = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);
    HVX_VectorPred scale_mask = Q6_Q_vsetq_R(valid_rows * 2);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 576);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, i8);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

#if __HVX_ARCH__ >= 81
        HVX_Vector v_sum_qf_c0 = Q6_Vqf32_equals_Vsf(v_sum_sf_c0);
        HVX_Vector v_sum_qf_c1 = Q6_Vqf32_equals_Vsf(v_sum_sf_c1);
#else
        HVX_Vector v_sum_qf_c0 = Q6_Vqf32_vadd_VsfVsf(v_sum_sf_c0, zero);
        HVX_Vector v_sum_qf_c1 = Q6_Vqf32_vadd_VsfVsf(v_sum_sf_c1, zero);
#endif

        HVX_Vector v_sum_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_sum_qf_c1, v_sum_qf_c0)));

        HVX_Vector v_scale_w = vptr[4];
        v_scale_w = Q6_V_vmux_QVV(scale_mask, v_scale_w, zero);
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_a_comb = Q6_V_valign_VVR(v_scale_a_c1, v_scale_a_c0, 64);
        HVX_Vector v_scale_w_upper = Q6_V_valign_VVR(v_scale_w, zero, 64);
        HVX_Vector v_scale_w_dup = Q6_V_vor_VV(v_scale_w_upper, Q6_V_vror_VR(v_scale_w_upper, 64));

        HVX_Vector v_scale_comb_qf16 = Q6_Vqf16_vmpy_VhfVhf(v_scale_w_dup, v_scale_a_comb);
        HVX_Vector v_sum_scaled_qf16 = Q6_Vqf16_vmpy_Vqf16Vhf(v_scale_comb_qf16, v_sum_hf);

        v_sum_qf16 = Q6_Vqf16_vadd_Vqf16Vqf16(v_sum_qf16, v_sum_scaled_qf16);
    }

    HVX_Vector v_sum_float_hf_val = Q6_Vhf_equals_Vqf16(v_sum_qf16);
    HVX_Vector one = hvx_vec_splat_f16(1.0f);
    HVX_VectorPair v_sum_float_pair = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sum_float_hf_val), one);

    hvx_vec_store_u(s0, valid_rows * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_sum_float_pair)));
    hvx_vec_store_u(s1, valid_rows * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_sum_float_pair)));
}

static void tiled_vec_dot_q4_1_32x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1280);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_sum_a   = v_act[9];

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_q4_1_32x2(const int n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_qf16     = Q6_V_vzero();
    HVX_Vector zero           = Q6_V_vzero();
    HVX_VectorPred scale_mask = Q6_Q_vsetq_R(valid_rows * 2);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1280);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1280);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

#if __HVX_ARCH__ >= 81
        HVX_Vector v_sum_qf_c0 = Q6_Vqf32_equals_Vsf(v_sum_sf_c0);
        HVX_Vector v_sum_qf_c1 = Q6_Vqf32_equals_Vsf(v_sum_sf_c1);
#else
        HVX_Vector v_sum_qf_c0 = Q6_Vqf32_vadd_VsfVsf(v_sum_sf_c0, zero);
        HVX_Vector v_sum_qf_c1 = Q6_Vqf32_vadd_VsfVsf(v_sum_sf_c1, zero);
#endif

        HVX_Vector v_sum_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_sum_qf_c1, v_sum_qf_c0)));

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);
        v_scale = Q6_V_vmux_QVV(scale_mask, v_scale, zero);
        v_offset = Q6_V_vmux_QVV(scale_mask, v_offset, zero);

        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_sum_a_c0   = v_act0[9];
        HVX_Vector v_scale_a_c1 = v_act1[8];
        HVX_Vector v_sum_a_c1   = v_act1[9];

        HVX_Vector v_scale_a_comb = Q6_V_valign_VVR(v_scale_a_c1, v_scale_a_c0, 64);
        HVX_Vector v_sum_a_comb   = Q6_V_valign_VVR(v_sum_a_c1, v_sum_a_c0, 64);

        HVX_Vector v_scale_upper  = Q6_V_valign_VVR(v_scale, zero, 64);
        HVX_Vector v_scale_dup    = Q6_V_vor_VV(v_scale_upper, Q6_V_vror_VR(v_scale_upper, 64));
        HVX_Vector v_offset_upper = Q6_V_valign_VVR(v_offset, zero, 64);
        HVX_Vector v_offset_dup   = Q6_V_vor_VV(v_offset_upper, Q6_V_vror_VR(v_offset_upper, 64));

        HVX_Vector v_scale_comb_qf16  = Q6_Vqf16_vmpy_VhfVhf(v_scale_dup, v_scale_a_comb);
        HVX_Vector v_offset_comb_qf16 = Q6_Vqf16_vmpy_VhfVhf(v_offset_dup, v_sum_a_comb);

        HVX_Vector v_scaled_dot_qf16  = Q6_Vqf16_vmpy_Vqf16Vhf(v_scale_comb_qf16, v_sum_hf);
        HVX_Vector v_sum_scaled_qf16  = Q6_Vqf16_vadd_Vqf16Vqf16(v_scaled_dot_qf16, v_offset_comb_qf16);

        v_sum_qf16 = Q6_Vqf16_vadd_Vqf16Vqf16(v_sum_qf16, v_sum_scaled_qf16);
    }

    HVX_Vector v_sum_float_hf_val = Q6_Vhf_equals_Vqf16(v_sum_qf16);
    HVX_Vector one = hvx_vec_splat_f16(1.0f);
    HVX_VectorPair v_sum_float_pair = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sum_float_hf_val), one);

    hvx_vec_store_u(s0, valid_rows * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_sum_float_pair)));
    hvx_vec_store_u(s1, valid_rows * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_sum_float_pair)));
}

static void tiled_vec_dot_q8_0_32x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 1088);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_q8_0_32x1(vptr, v_act);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_q8_0_32x2(const int n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_qf16     = Q6_V_vzero();
    HVX_Vector zero           = Q6_V_vzero();
    HVX_VectorPred scale_mask = Q6_Q_vsetq_R(valid_rows * 2);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 1088);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q8_0_32x2(vptr, v_act0, v_act1);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(v_sum_c0, 7));
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(v_sum_c1, 7));

#if __HVX_ARCH__ >= 81
        HVX_Vector v_sum_qf_c0 = Q6_Vqf32_equals_Vsf(v_sum_sf_c0);
        HVX_Vector v_sum_qf_c1 = Q6_Vqf32_equals_Vsf(v_sum_sf_c1);
#else
        HVX_Vector v_sum_qf_c0 = Q6_Vqf32_vadd_VsfVsf(v_sum_sf_c0, zero);
        HVX_Vector v_sum_qf_c1 = Q6_Vqf32_vadd_VsfVsf(v_sum_sf_c1, zero);
#endif

        HVX_Vector v_sum_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_sum_qf_c1, v_sum_qf_c0)));

        HVX_Vector v_scale_w = vptr[8];
        v_scale_w = Q6_V_vmux_QVV(scale_mask, v_scale_w, zero);
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_a_comb = Q6_V_valign_VVR(v_scale_a_c1, v_scale_a_c0, 64);
        HVX_Vector v_scale_w_upper = Q6_V_valign_VVR(v_scale_w, zero, 64);
        HVX_Vector v_scale_w_dup = Q6_V_vor_VV(v_scale_w_upper, Q6_V_vror_VR(v_scale_w_upper, 64));

        HVX_Vector factor = hvx_vec_splat_f16(128.0f);
        HVX_Vector v_scale_comb_qf16 = Q6_Vqf16_vmpy_VhfVhf(v_scale_w_dup, v_scale_a_comb);
        v_scale_comb_qf16 = Q6_Vqf16_vmpy_Vqf16Vhf(v_scale_comb_qf16, factor);
        HVX_Vector v_sum_scaled_qf16 = Q6_Vqf16_vmpy_Vqf16Vhf(v_scale_comb_qf16, v_sum_hf);

        v_sum_qf16 = Q6_Vqf16_vadd_Vqf16Vqf16(v_sum_qf16, v_sum_scaled_qf16);
    }

    HVX_Vector v_sum_float_hf_val = Q6_Vhf_equals_Vqf16(v_sum_qf16);
    HVX_Vector one = hvx_vec_splat_f16(1.0f);
    HVX_VectorPair v_sum_float_pair = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sum_float_hf_val), one);

    hvx_vec_store_u(s0, valid_rows * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_sum_float_pair)));
    hvx_vec_store_u(s1, valid_rows * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_sum_float_pair)));
}

static void tiled_vec_dot_iq4nl_32x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 576);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_iq4nl_32x2(const int n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_qf16     = Q6_V_vzero();
    HVX_Vector zero           = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;
    HVX_VectorPred scale_mask = Q6_Q_vsetq_R(valid_rows * 2);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 576);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(v_sum_c0, 7));
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(v_sum_c1, 7));

#if __HVX_ARCH__ >= 81
        HVX_Vector v_sum_qf_c0 = Q6_Vqf32_equals_Vsf(v_sum_sf_c0);
        HVX_Vector v_sum_qf_c1 = Q6_Vqf32_equals_Vsf(v_sum_sf_c1);
#else
        HVX_Vector v_sum_qf_c0 = Q6_Vqf32_vadd_VsfVsf(v_sum_sf_c0, zero);
        HVX_Vector v_sum_qf_c1 = Q6_Vqf32_vadd_VsfVsf(v_sum_sf_c1, zero);
#endif

        HVX_Vector v_sum_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_sum_qf_c1, v_sum_qf_c0)));

        HVX_Vector v_scale_w = vptr[4];
        v_scale_w = Q6_V_vmux_QVV(scale_mask, v_scale_w, zero);
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_a_comb = Q6_V_valign_VVR(v_scale_a_c1, v_scale_a_c0, 64);
        HVX_Vector v_scale_w_upper = Q6_V_valign_VVR(v_scale_w, zero, 64);
        HVX_Vector v_scale_w_dup = Q6_V_vor_VV(v_scale_w_upper, Q6_V_vror_VR(v_scale_w_upper, 64));

        HVX_Vector factor = hvx_vec_splat_f16(128.0f);
        HVX_Vector v_scale_comb_qf16 = Q6_Vqf16_vmpy_VhfVhf(v_scale_w_dup, v_scale_a_comb);
        v_scale_comb_qf16 = Q6_Vqf16_vmpy_Vqf16Vhf(v_scale_comb_qf16, factor);
        HVX_Vector v_sum_scaled_qf16 = Q6_Vqf16_vmpy_Vqf16Vhf(v_scale_comb_qf16, v_sum_hf);

        v_sum_qf16 = Q6_Vqf16_vadd_Vqf16Vqf16(v_sum_qf16, v_sum_scaled_qf16);
    }

    HVX_Vector v_sum_float_hf_val = Q6_Vhf_equals_Vqf16(v_sum_qf16);
    HVX_Vector one = hvx_vec_splat_f16(1.0f);
    HVX_VectorPair v_sum_float_pair = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sum_float_hf_val), one);

    hvx_vec_store_u(s0, valid_rows * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_sum_float_pair)));
    hvx_vec_store_u(s1, valid_rows * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_sum_float_pair)));
}

static void tiled_vec_dot_mxfp4_32x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 544);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = * (const HVX_UVector *) (tile_ptr + kt * 544 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_f16 = v_act[8];
        HVX_VectorPair p_scale_a_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_f16);
        HVX_Vector v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_Vector v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    v_sum_float = hvx_vec_mul_f32_f32(v_sum_float, hvx_vec_splat_f32(0.5f));

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_mxfp4_32x2(const int n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, int valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_UVector * restrict vptr = (const HVX_UVector *) (tile_ptr + kt * 544);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = * (const HVX_UVector *) (tile_ptr + kt * 544 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_c0_f16 = v_act0[8];
        HVX_Vector v_scale_a_c1_f16 = v_act1[8];

        HVX_VectorPair p_scale_a_c0_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c0_f16);
        HVX_VectorPair p_scale_a_c1_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c1_f16);

        HVX_Vector v_scale_a_c0 = Q6_V_lo_W(p_scale_a_c0_f32);
        HVX_Vector v_scale_a_c1 = Q6_V_lo_W(p_scale_a_c1_f32);

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    v_sum_float_c0 = hvx_vec_mul_f32_f32(v_sum_float_c0, hvx_vec_splat_f32(0.5f));
    v_sum_float_c1 = hvx_vec_mul_f32_f32(v_sum_float_c1, hvx_vec_splat_f32(0.5f));

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

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
                                                                                                                                    \
    const size_t src1_stride = src1_spad->stride;                                                                                   \
                                                                                                                                    \
    uint8_t * restrict spad_dst  = dst_spad->data  + dst_spad->size_per_thread  * ith;                                              \
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;                                              \
    uint8_t * restrict src1_data = src1_spad->data;                                                                                 \
                                                                                                                                    \
    volatile uint64_t t1, t2;                                                                                                       \
    t1 = HAP_perf_get_qtimer_count();                                                                                               \
                                                                                                                                    \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                               \
                                                                                                                                    \
    const uint32_t tile_size = TILE_SIZE;                                                                                           \
                                                                                                                                    \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                               \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                               \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                             \
    uint32_t tile_row_transfer_size = n_k_tiles_a * tile_size;                                                                      \
                                                                                                                                    \
    uint32_t ct_start = src0_start_row / 32;                                                                                        \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                   \
                                                                                                                                    \
    uint32_t push_ct = ct_start;                                                                                                    \
    for (uint32_t d = 0; d < DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {                                                       \
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + d * tile_row_transfer_size, src0_row + push_ct * tile_row_stride),       \
                       tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                  \
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
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                              \
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
                                                                                                                                    \
    const size_t src1_stride = src1_spad->stride;                                                                                   \
                                                                                                                                    \
    uint8_t * spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;                                                         \
    uint8_t * spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;                                                       \
    uint8_t * src1_data = src1_spad->data;                                                                                          \
                                                                                                                                    \
    volatile uint64_t t1, t2;                                                                                                       \
    t1 = HAP_perf_get_qtimer_count();                                                                                               \
                                                                                                                                    \
    float * tmp = (float *) spad_dst;                                                                                               \
                                                                                                                                    \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                               \
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;                                                                \
    float * restrict dst_col          = (float *) dst->data;                                                                        \
                                                                                                                                    \
    const uint32_t tile_size = TILE_SIZE;                                                                                           \
                                                                                                                                    \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                               \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                               \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                             \
    uint32_t tile_row_transfer_size = n_k_tiles_a * tile_size;                                                                      \
                                                                                                                                    \
    uint32_t ct_start = src0_start_row / 32;                                                                                        \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                   \
                                                                                                                                    \
    uint32_t push_ct = ct_start;                                                                                                    \
    for (uint32_t d = 0; d < DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {                                                       \
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + d * tile_row_transfer_size, src0_row + push_ct * tile_row_stride),       \
                       tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                  \
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
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                              \
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
    struct htp_matmul_context * mmctx = data;                                                                                           \
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
    struct htp_spad * restrict src0_spad = &octx->src0_spad; /* Wk */                                                                   \
    struct htp_spad * restrict src2_spad = &octx->src2_spad; /* Wv */                                                                   \
    struct htp_spad * restrict src3_spad = &octx->src3_spad; /* Wq */                                                                   \
    struct htp_spad * restrict src1_spad = &octx->src1_spad; /* x */                                                                    \
                                                                                                                                        \
    const uint32_t ne00 = src0->ne[0];                                                                                                  \
    const uint32_t ne10 = src1->ne[0];                                                                                                  \
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];                                                                \
                                                                                                                                        \
    const size_t dst_row_size  = dst_k->nb[1];                                                                                          \
    const size_t src1_stride = src1_spad->stride;                                                                                       \
                                                                                                                                        \
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;                                                  \
    uint8_t * restrict spad_src2 = src2_spad->data + src2_spad->size_per_thread * ith;                                                  \
    uint8_t * restrict spad_src3 = src3_spad->data + src3_spad->size_per_thread * ith;                                                  \
    uint8_t * restrict src1_data = src1_spad->data;                                                                                     \
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
                                                                                                                                        \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                                   \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                                   \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                                 \
    uint32_t tile_row_transfer_size = n_k_tiles_a * tile_size;                                                                          \
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
        for (uint32_t d = 0; d < DMA_DEPTH && push_ct < ct_end_kv; d++, push_ct++) {                                                    \
            dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + d * tile_row_transfer_size, src0_row + push_ct * tile_row_stride),       \
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                  \
            dma_queue_push(dma_queue, dma_make_ptr(spad_src2 + d * tile_row_transfer_size, src2_row + push_ct * tile_row_stride),       \
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                  \
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
                               tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                              \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_v, src2_row + push_ct * tile_row_stride),                      \
                               tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                              \
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
        for (uint32_t d = 0; d < DMA_DEPTH && push_ct < ct_end_q; d++, push_ct++) {                                                     \
            dma_queue_push(dma_queue, dma_make_ptr(spad_src3 + d * tile_row_transfer_size, src3_row + push_ct * tile_row_stride),       \
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                  \
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
                               tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                              \
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
    struct htp_matmul_context * mmctx = data;                                                                                           \
    struct htp_ops_context * octx = mmctx->octx;                                                                                        \
                                                                                                                                        \
    const struct htp_tensor * restrict src0 = octx->src[0]; /* Wgate */                                                                 \
    const struct htp_tensor * restrict src1 = octx->src[1]; /* y */                                                                     \
    const struct htp_tensor * restrict src2 = octx->src[2]; /* Wup */                                                                   \
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];                                                                        \
    const struct htp_tensor * restrict dst_up = octx->dsts[1];                                                                          \
                                                                                                                                        \
    struct htp_spad * restrict src0_spad = &octx->src0_spad; /* Wgate */                                                                \
    struct htp_spad * restrict src2_spad = &octx->src2_spad; /* Wup */                                                                  \
    struct htp_spad * restrict src1_spad = &octx->src1_spad; /* y */                                                                    \
                                                                                                                                        \
    const uint32_t ne00 = src0->ne[0];                                                                                                  \
    const uint32_t ne01 = src0->ne[1];                                                                                                  \
    const uint32_t ne10 = src1->ne[0];                                                                                                  \
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];                                                                \
                                                                                                                                        \
    const size_t dst_row_size  = dst_gate->nb[1];                                                                                       \
    const size_t src1_stride = src1_spad->stride;                                                                                       \
                                                                                                                                        \
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;                                                  \
    uint8_t * restrict spad_src2 = src2_spad->data + src2_spad->size_per_thread * ith;                                                  \
    uint8_t * restrict src1_data = src1_spad->data;                                                                                     \
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
                                                                                                                                        \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                                   \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                                   \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                                 \
    uint32_t tile_row_transfer_size = n_k_tiles_a * tile_size;                                                                          \
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
    for (uint32_t d = 0; d < DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {                                                           \
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + d * tile_row_transfer_size, src0_row + push_ct * tile_row_stride),           \
                       tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                      \
        dma_queue_push(dma_queue, dma_make_ptr(spad_src2 + d * tile_row_transfer_size, src2_row + push_ct * tile_row_stride),           \
                       tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                      \
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
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                  \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_up, src2_row + push_ct * tile_row_stride),                         \
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);                                  \
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


MATVEC_2D_REPACKED_IMPL(q4_0,  576,  tiled_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1,  640,  tiled_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0,  1088, tiled_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl, 576,  tiled_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4, 544,  tiled_vec_dot_mxfp4_32x1)


MATMUL_QKV_2D_REPACKED_IMPL(q4_0,  576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q4_1,  640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q8_0,  1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(iq4nl, 576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(mxfp4, 544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)


MATMUL_FFN_2D_REPACKED_IMPL(q4_0,  576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q4_1,  640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q8_0,  1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(iq4nl, 576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(mxfp4, 544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)



// src1 tensor is already in VTCM spad
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

    const size_t src0_stride = src0_spad->stride;
    const size_t src1_stride = src1_spad->stride;

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * restrict spad_dst  = dst_spad->data  + dst_spad->size_per_thread  * ith;
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    volatile uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;

    // Prefill spad with src0 rows
    #pragma unroll(4)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= MM_SPAD_SRC0_NROWS) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
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

        // Prefetch next (n + spad_nrows) row
        const int pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
        const int is0 = (pr0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
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

// q8_0_tiled/q8_1_tiled src1 tensor is already in VTCM spad
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

    const size_t src0_stride = src0_spad->stride;
    const size_t src1_stride = src1_spad->stride;

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * src1_data = src1_spad->data;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    float * tmp = (float *) spad_dst;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
    float * restrict dst_col          = (float *) dst->data;

    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // Prefill spad with 2x src0 rows
    #pragma unroll(2)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint32_t is0 = (ir0 - src0_start_row);
        if (is0 >= MM_SPAD_SRC0_NROWS) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);

        // Prefetch next (n + spad_nrows) row
        const uint32_t pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
        const uint32_t is0 = (pr0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
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

// src1 tensor is already in VTCM spad
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
    const size_t src1_row_size = q8_0_tiled_row_size(ne10);

    const size_t src1_stride = src1_spad->stride;

    // Per-thread VTCM scratchpads for all tensors
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        if (mmctx->hmx_eligible) {
            continue;
        }

        const uint8_t * src0_row = (const uint8_t *) src0->data + cur_a * nb02;

        const uint32_t tile_size = mmctx->tile_size;
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size = n_k_tiles_a * tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + d * tile_row_transfer_size, src0_row + push_ct * tile_row_stride),
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);
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
                               tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);
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

// src1 tensor is already in VTCM spad
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
    const size_t src1_row_size = q8_0_tiled_row_size(ne10);

    const uint32_t n_aids = src2->ne[0];  // num activated experts
    const uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCM scratchpads for all tensors
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const int32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
        if (eid < 0) {
            continue;
        }
        assert(eid < (int32_t) n_ids);

        const uint8_t * restrict src0_row = (const uint8_t *) src0->data + eid * nb02;
        const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
        float * restrict dst_row          = (float *) (dst->data + ie1 * nb1);

        const uint32_t tile_size = mmctx->tile_size;
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size = n_k_tiles_a * tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < DMA_DEPTH && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + d * tile_row_transfer_size, src0_row + push_ct * tile_row_stride),
                           tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);
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
                               tile_row_transfer_size, tile_row_transfer_size, tile_row_transfer_size, 1);
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

// *** dynamic quant

static inline void quantize_block_f32_q8_1_tiled(float * restrict x, uint8_t * restrict y_block) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero = Q6_V_vzero();

    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    const HVX_Vector ones = Q6_Vb_vsplat_R(1);
    HVX_Vector v_sums = Q6_Vw_vrmpy_VbVb(vx_i8, ones);
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 4));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 8));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 16));

    float vmax0[32] __attribute__((aligned(128)));
    float vmax1[32] __attribute__((aligned(128)));
    float vmax2[32] __attribute__((aligned(128)));
    float vmax3[32] __attribute__((aligned(128)));
    int32_t sums[32] __attribute__((aligned(128)));

    hvx_vec_store_u(vmax0, 128, vmax0_sf);
    hvx_vec_store_u(vmax1, 128, vmax1_sf);
    hvx_vec_store_u(vmax2, 128, vmax2_sf);
    hvx_vec_store_u(vmax3, 128, vmax3_sf);
    hvx_vec_store_u(sums, 128, v_sums);

    float d0 = vmax0[0] / 127.0f;
    float d1 = vmax1[0] / 127.0f;
    float d2 = vmax2[0] / 127.0f;
    float d3 = vmax3[0] / 127.0f;

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
    HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;

    for (int b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);
        
        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4), v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8), v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        __fp16 scale_h, offset_h;
        if (b == 0) {
            scale_h = (__fp16) d0;
            offset_h = (__fp16) (sums[0] * d0);
        } else if (b == 1) {
            scale_h = (__fp16) d1;
            offset_h = (__fp16) (sums[8] * d1);
        } else if (b == 2) {
            scale_h = (__fp16) d2;
            offset_h = (__fp16) (sums[16] * d2);
        } else {
            scale_h = (__fp16) d3;
            offset_h = (__fp16) (sums[24] * d3);
        }

        HVX_Vector r_scale = Q6_Vh_vsplat_R(*(int16_t *)&scale_h);
        HVX_Vector r_offset = Q6_Vh_vsplat_R(*(int16_t *)&offset_h);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1280);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
        dst[9] = r_offset;
    }
}

static inline void quantize_block_f32_q8_0_tiled(float * restrict x, uint8_t * restrict y_block) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero   = Q6_V_vzero();

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vmax_hf = hvx_vec_reduce_max_f16(hvx_vec_abs_f16(vx01_hf));
    vmax_hf            = hvx_vec_reduce_max2_f16(hvx_vec_abs_f16(vx23_hf), vmax_hf);

    HVX_Vector vd_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_Vector vd_hf   = Q6_Vhf_equals_Vqf16(vd_qf16);

    HVX_Vector vd_inv_hf = hvx_vec_inverse_f16(vd_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    HVX_Vector r_scale = hvx_vec_repl_f16(vd_hf);

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
    HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;

    for (int b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);
        
        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4), v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8), v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1152);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
    }
}

// Overrides input x
// Overrides input x
static void quantize_row_f32_q8_0_tiled(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (k + qk - 1) / qk;

    for (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block0 = y + i * 8 * 1152;
        uint8_t * restrict y_block1 = y + (i * 8 + 4) * 1152;
        quantize_block_f32_q8_0_tiled(x + (i*2 + 0) * qk/2, y_block0);
        quantize_block_f32_q8_0_tiled(x + (i*2 + 1) * qk/2, y_block1);
    }
}

static void quantize_row_f32_q8_1_tiled(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (k + qk - 1) / qk;

    for (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block0 = y + i * 8 * 1280;
        uint8_t * restrict y_block1 = y + (i * 8 + 4) * 1280;
        quantize_block_f32_q8_1_tiled(x + (i*2 + 0) * qk/2, y_block0);
        quantize_block_f32_q8_1_tiled(x + (i*2 + 1) * qk/2, y_block1);
    }
}

static void quantize_f32_q8_0_tiled(unsigned int nth, unsigned int ith, void * data) {
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = octx->src1_spad.data;
    struct htp_spad * spad = &octx->src0_spad;
    uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = src->nb[1];
    const size_t dst_row_size = q8_0_tiled_row_size(ne0);

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_row_size * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);
    uint8_t * restrict tmp_data = (uint8_t *) spad->data + (spad->size_per_thread * ith);

    const size_t src_row_size_padded = hex_round_up(src_row_size, QK_Q8_0_TILED * sizeof(float));
    hvx_splat_f32_a(tmp_data, 0.0f, src_row_size_padded / sizeof(float));  // zero-out temp row data for padding

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        hex_l2fetch(src_data, src_row_size, src_row_size, 2);
        hvx_copy_f32_aa(tmp_data, src_data, ne0);

        quantize_row_f32_q8_0_tiled((float *) tmp_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-f32-q8_0_tiled: %u/%u : n-rows %u (%u:%u) row-size %u -> %u usec %u\n", ith, nth, nrows, ir_first,
         ir_last, src_row_size, dst_row_size, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);
}


static void quantize_f32_q8_1_tiled(unsigned int nth, unsigned int ith, void * data) {
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = octx->src1_spad.data;
    struct htp_spad * spad = &octx->src0_spad;
    uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = src->nb[1];
    const size_t dst_row_size = q8_1_tiled_row_size(ne0);

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_row_size * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);
    uint8_t * restrict tmp_data = (uint8_t *) spad->data + (spad->size_per_thread * ith);

    const size_t src_row_size_padded = hex_round_up(src_row_size, QK_Q8_0_TILED * sizeof(float));
    hvx_splat_f32_a(tmp_data, 0.0f, src_row_size_padded / sizeof(float));  // zero-out temp row data for padding

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        hex_l2fetch(src_data, src_row_size, src_row_size, 2);
        hvx_copy_f32_aa(tmp_data, src_data, ne0);

        quantize_row_f32_q8_1_tiled((float *) tmp_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-f32-q8_1_tiled: %u/%u : n-rows %u (%u:%u) row-size %u -> %u usec %u\n", ith, nth, nrows, ir_first,
         ir_last, src_row_size, dst_row_size, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);
}

static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = octx->src1_spad.data;
    struct htp_spad * spad = &octx->src0_spad;

    const uint32_t ne0 = src->ne[0];
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne0 + qk - 1) / qk;

    const uint32_t ib_first = mmctx->quant_ib_first[ith];
    const uint32_t ib_last  = mmctx->quant_ib_last[ith];

    const size_t src_row_size = src->nb[1];
    const size_t dst_row_size = q8_0_tiled_row_size(ne0);
    uint8_t * restrict tmp_data = (uint8_t *) spad->data + (spad->size_per_thread * ith);

    uint32_t r = mmctx->quant_r[ith];
    uint32_t c = mmctx->quant_c[ith];

    for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        const uint8_t * restrict src_ptr = (const uint8_t *) src->data + r * src_row_size + c * qk * sizeof(float);
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 8 * 1152;

        hex_l2fetch(src_ptr, qk * sizeof(float), qk * sizeof(float), 1);

        if (c == nb - 1) {
            uint32_t active_elements = ne0 - c * qk;
            hvx_splat_f32_a(tmp_data, 0.0f, qk);
            hvx_copy_f32_aa(tmp_data, src_ptr, active_elements);
        } else {
            hvx_copy_f32_aa(tmp_data, src_ptr, qk);
        }

        quantize_block_f32_q8_0_tiled((float *) tmp_data + 0, dst_ptr + 0);
        quantize_block_f32_q8_0_tiled((float *) tmp_data + qk/2, dst_ptr + 4 * 1152);

        c++;
        if (c == nb) {
            c = 0;
            r++;
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);
}

static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = octx->src1_spad.data;
    struct htp_spad * spad = &octx->src0_spad;

    const uint32_t ne0 = src->ne[0];
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne0 + qk - 1) / qk;

    const uint32_t ib_first = mmctx->quant_ib_first[ith];
    const uint32_t ib_last  = mmctx->quant_ib_last[ith];

    const size_t src_row_size = src->nb[1];
    const size_t dst_row_size = q8_1_tiled_row_size(ne0);
    uint8_t * restrict tmp_data = (uint8_t *) spad->data + (spad->size_per_thread * ith);

    uint32_t r = mmctx->quant_r[ith];
    uint32_t c = mmctx->quant_c[ith];

    for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        const uint8_t * restrict src_ptr = (const uint8_t *) src->data + r * src_row_size + c * qk * sizeof(float);
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 8 * 1280;

        hex_l2fetch(src_ptr, qk * sizeof(float), qk * sizeof(float), 1);

        if (c == nb - 1) {
            uint32_t active_elements = ne0 - c * qk;
            hvx_splat_f32_a(tmp_data, 0.0f, qk);
            hvx_copy_f32_aa(tmp_data, src_ptr, active_elements);
        } else {
            hvx_copy_f32_aa(tmp_data, src_ptr, qk);
        }

        quantize_block_f32_q8_1_tiled((float *) tmp_data + 0, dst_ptr + 0);
        quantize_block_f32_q8_1_tiled((float *) tmp_data + qk/2, dst_ptr + 4 * 1280);

        c++;
        if (c == nb) {
            c = 0;
            r++;
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);
}

static void quantize_f32_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = octx->src1_spad.data;
    uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;
    uint32_t dst_stride = octx->src1_spad.stride;

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
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = octx->src1_spad.data;
    uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;
    uint32_t dst_stride = octx->src1_spad.stride;

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
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ith);

    const struct htp_tensor * src = octx->src[1];
    uint8_t * restrict dst = octx->src1_spad.data;
    uint32_t nrows_per_thread = mmctx->src1_nrows_per_thread;
    uint32_t dst_stride = octx->src1_spad.stride;

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

static int htp_mminit_vec_dot(struct htp_matmul_context * mmctx, enum htp_data_type type) {
    switch (type) {
        case HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->tile_size    = 576;
            return 0;
        case HTP_TYPE_Q4_1:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->tile_size    = 640;
            return 0;
        case HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->tile_size    = 1088;
            return 0;
        case HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->tile_size    = 576;
            return 0;
        case HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->tile_size    = 544;
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
                                 size_t src2_spad_size_per_thread) {
    octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
    octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
    octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);

    if (src2_spad_size_per_thread > 0) {
        octx->src2_spad.size_per_thread = src2_spad_size_per_thread;
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

    struct htp_matmul_context mmctx_struct = {0};
    struct htp_matmul_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

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

    if (src0->type == HTP_TYPE_F16) {
        // Try optimized f16-f16 path first (src1 in VTCM)
        const size_t f16_src1_row_size  = hex_round_up(ne10 * 2, 128);
        const size_t f16_src1_spad_size = hex_round_up(f16_src1_row_size * src1_nrows, 256);
        const size_t f16_src0_spad_size = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256) * octx->n_threads;
        const size_t f16_dst_spad_size  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256) * octx->n_threads;

        const size_t f16_total_size = f16_src1_spad_size + f16_src0_spad_size + f16_dst_spad_size;

        // Default matmul implementation does not support multi-batch src0 (N-vs-N broadcasting).
        // It only supports 1-vs-N broadcasting (src0 is 2D) or standard 2D matmul.
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = htp_is_permuted(octx->src[0]) || htp_is_permuted(octx->src[1]);

        if (!is_batched && !is_permuted && f16_total_size <= octx->ctx->vtcm_size) {
            // Optimized path
            quant_job_func     = (src1->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            mmctx->type        = "f16-f16";
            mmctx->vec_dot_1x1 = vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1 = vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2 = vec_dot_f16_f16_aa_2x2;

            src1_row_size = f16_src1_row_size;  // row size post quantization

            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);

            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
        } else {
            // Fallback to f16/f32 (DDR) if src1 doesn't fit in VTCM or broadcasting is required
            quant_job_func = NULL;
            if (src1->type == HTP_TYPE_F32) {
                mmctx->type        = "f16-f32";
                mmctx->vec_dot_1x1 = vec_dot_f16_f32_uu_1x1;
                matmul_job_func    = matmul_4d;
            } else {
                mmctx->type        = "f16-f16";
                mmctx->vec_dot_1x1 = vec_dot_f16_f16_uu_1x1;
                matmul_job_func    = matmul_4d;
            }

            src1_row_size = nb11;  // original row size in DDR

            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size, 256);
            octx->src1_spad.size_per_thread = hex_round_up(MM_SPAD_SRC1_NROWS * src1_row_size, 256);

            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->src1_spad.size = octx->src1_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;

            // Init fastdiv for matmul_4d (supports broadcasting)
            mmctx->mm_div_ne12_ne1 = init_fastdiv_values(src1->ne[2] * dst->ne[1]);
            mmctx->mm_div_ne1      = init_fastdiv_values(dst->ne[1]);
            mmctx->mm_div_r2       = init_fastdiv_values(src1->ne[2] / src0->ne[2]);
            mmctx->mm_div_r3       = init_fastdiv_values(src1->ne[3] / src0->ne[3]);

            need_quant = false;
        }
    } else if (src0->type == HTP_TYPE_F32) {
        // Try optimized f32-f32 path first (src1 in VTCM)
        const size_t f32_src1_row_size  = hex_round_up(ne10 * 4, 128);
        const size_t f32_src1_spad_size = hex_round_up(f32_src1_row_size * src1_nrows, 256);
        const size_t f32_src0_spad_size = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256) * octx->n_threads;
        const size_t f32_dst_spad_size  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256) * octx->n_threads;

        const size_t f32_total_size = f32_src1_spad_size + f32_src0_spad_size + f32_dst_spad_size;

        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = htp_is_permuted(octx->src[0]) || htp_is_permuted(octx->src[1]);

        if (!is_batched && !is_permuted && f32_total_size <= octx->ctx->vtcm_size) {
            // Optimized path
            quant_job_func     = quantize_f32_f32;
            mmctx->type        = "f32-f32";
            mmctx->vec_dot_1x1 = vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1 = vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2 = vec_dot_f32_f32_aa_2x2;

            src1_row_size = f32_src1_row_size;

            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);

            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
        } else {
            // Fallback to DDR / broadcasting
            quant_job_func = NULL;
            mmctx->type        = "f32-f32";
            mmctx->vec_dot_1x1 = vec_dot_f32_f32_uu_1x1;
            matmul_job_func    = matmul_4d;

            src1_row_size = nb11;

            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size, 256);
            octx->src1_spad.size_per_thread = hex_round_up(MM_SPAD_SRC1_NROWS * src1_row_size, 256);

            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->src1_spad.size = octx->src1_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;

            // Init fastdiv for matmul_4d (supports broadcasting)
            mmctx->mm_div_ne12_ne1 = init_fastdiv_values(src1->ne[2] * dst->ne[1]);
            mmctx->mm_div_ne1      = init_fastdiv_values(dst->ne[1]);
            mmctx->mm_div_r2       = init_fastdiv_values(src1->ne[2] / src0->ne[2]);
            mmctx->mm_div_r3       = init_fastdiv_values(src1->ne[3] / src0->ne[3]);

            need_quant = false;
        }
    } else {
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
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? q8_1_tiled_row_size(ne10) : q8_0_tiled_row_size(ne10);
        htp_mminit_spad(octx, dst_row_size, src0_row_size_padded, src1_row_size, src1_nrows, 0);

        if (is_repacked) {
            uint32_t tile_size;
            if (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_IQ4_NL) { tile_size = 576; }
            else if (src0->type == HTP_TYPE_Q4_1) { tile_size = 640; }
            else if (src0->type == HTP_TYPE_Q8_0) { tile_size = 1088; }
            else if (src0->type == HTP_TYPE_MXFP4) { tile_size = 544; }
            else { tile_size = 0; }
            uint32_t n_k_tiles = ne10 / 32;
            uint32_t tile_row_size = n_k_tiles * tile_size;
            octx->src0_spad.size_per_thread = hex_round_up(DMA_DEPTH * tile_row_size, 256);
            octx->src0_spad.size            = octx->src0_spad.size_per_thread * octx->n_threads;
        }
    }

    // VTCM scratchpads for all tensors
    size_t spad_size = octx->src1_spad.size + octx->src0_spad.size + octx->dst_spad.size;

    FARF(HIGH, "matmul-%s : src0-spad-size %u src1-spad-size %u dst-spad-size %u (%zu)\n", mmctx->type,
         octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size, spad_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    // Place src1 spad first. We use it for dyn.quant and may reuse between ops
    octx->src1_spad.data = octx->ctx->vtcm_base;
    octx->src0_spad.data = octx->src1_spad.data + octx->src1_spad.size;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size;

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    octx->src0_spad.stride = src0_row_size_padded;
    octx->src1_spad.stride = src1_row_size;

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

#ifndef HTP_HAS_HMX
    return op_matmul_hvx(octx);
#else
    if (!octx->ctx->hmx_enabled) {
        return op_matmul_hvx(octx);
    }

    // HMX weight tile requires N to be 32-aligned.
    if (src0->ne[1] % 32 != 0) {
        return op_matmul_hvx(octx);
    }

    // HMX supports F16, F32, Q4_0, Q8_0, IQ4_NL, MXFP4 weights.
    // Other types fall back to HVX.
    uint32_t wtype = src0->type;
    if (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_Q4_0 && wtype != HTP_TYPE_Q4_1 && wtype != HTP_TYPE_Q8_0 && wtype != HTP_TYPE_IQ4_NL && wtype != HTP_TYPE_MXFP4) {
        return op_matmul_hvx(octx);
    }

    // Quantised HMX path requires K aligned to 256 (tiled super-block).
    // F16 and F32 HMX paths require K aligned to 32 (tile width).
    if (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && src0->ne[0] % 256 != 0) {
        return op_matmul_hvx(octx);
    }

    if ((wtype == HTP_TYPE_F16 || wtype == HTP_TYPE_F32) && src0->ne[0] % 32 != 0) {
        return op_matmul_hvx(octx);
    }

    const bool is_batched = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    // Quantised HMX kernels only handle flat 2D matmul (host already rejects
    // batched quantised, but guard here too).  F16 batched matmul is handled
    // by the dedicated wrapper in hmx-matmul-ops.c.
    if (is_batched && src0->type != HTP_TYPE_F16) {
        return op_matmul_hvx(octx);
    }

    // HMX assumes contiguous row-major layout.  Fall back for permuted
    // tensors where strides are non-monotonic (e.g. transposed KV cache).
    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        return op_matmul_hvx(octx);
    }

    // M alignment: Use HMX when M >= 32, the last partial tile (m_total % 32 rows)
    //  is handled by HMX itself; when M < 32  fall back to HVX (except for repacked weight types).
    const int m_total = (int) src1->ne[1];
    const int m_hmx   = m_total & ~31;   // 0 when M < 32
    if (m_total <= 4) {
        return op_matmul_hvx(octx);
    }

    // Always re-quantize src1 since HMX kernel overwrites vtcm/spad,
    // so any previously cached quantized data is invalid.
    octx->src1_spad.src = NULL;

    int k = (int) src0->ne[0];  // inner dimension
    int n = (int) src0->ne[1];  // weight columns

    int ret = -1;

    // Row strides in elements. For compact tensors these equal k; for
    // permuted attention views they can be larger, so pass the real stride.
    const int act_stride = (int)(src1->nb[1] / sizeof(float));
    const int wgt_stride = (int)(src0->nb[1] / sizeof(__fp16));

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    if (is_batched) {
        if (src0->type == HTP_TYPE_F16) {
            hmx_matmul_f16_f32_batched_params_t batch_params = {
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
            ret = hmx_matmul_f16_f32_batched(octx->ctx, &batch_params);
        } else {
            return op_matmul_hvx(octx);
        }
    } else {
        ret = hmx_matmul_2d_f32(octx->ctx, (float*) dst->data, (float*) src1->data, (const uint8_t *) src0->data,
                    m_total, k, n, act_stride, (int) src0->nb[1], (int) src0->type, (int) src1->ne[0],
                    (int)(dst->nb[1] / sizeof(float)), (int)dst->ne[0]);
    }

    if (ret != 0) {
        FARF(HIGH, "HMX matmul failed (ret=%d), falling back to HVX", ret);
        return op_matmul_hvx(octx);
    }

    return 0;
#endif // HTP_HAS_HMX
}

int op_matmul_id(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_matmul_context mmctx_struct = {0};
    struct htp_matmul_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

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
    mmctx->mm_div_ne11       = init_fastdiv_values(ne11);

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

    bool hmx_eligible = false;
#ifdef HTP_HAS_HMX
    if (octx->ctx->hmx_enabled && src1_nrows > 4) {
        uint32_t wtype = src0->type;
        if (ne01 % 32 == 0 && ne10 % 32 == 0 &&
            (wtype == HTP_TYPE_F16 || wtype == HTP_TYPE_F32 || wtype == HTP_TYPE_Q4_0 || wtype == HTP_TYPE_Q4_1 || wtype == HTP_TYPE_Q8_0 || wtype == HTP_TYPE_IQ4_NL || wtype == HTP_TYPE_MXFP4)) {
            if ((wtype == HTP_TYPE_F16 || wtype == HTP_TYPE_F32) && ne00 % 32 == 0) {
                hmx_eligible = true;
            } else if (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && ne00 % 256 == 0) {
                hmx_eligible = true;
            }
        }
    }
#endif

    mmctx->hmx_eligible = hmx_eligible;

    if (hmx_eligible) {
        for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
            const int32_t cne1 = matrix_row_counts[cur_a];
            if (cne1 == 0) continue;

            int ret = hmx_matmul_id_2d_f32(octx->ctx, (float*) dst->data, (float*) src1->data,
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
    src1_row_size  = (src0->type == HTP_TYPE_Q4_1) ? q8_1_tiled_row_size(ne10) : q8_0_tiled_row_size(ne10);

    const size_t src2_spad_size_per_thread = 0; // We moved the mapping to DDR!
    htp_mminit_spad(octx, 0, src0_row_size_padded, src1_row_size, src1_nrows, src2_spad_size_per_thread);

    const bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                              src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                              src0->type == HTP_TYPE_MXFP4);
    if (is_repacked) {
        const uint32_t n_k_tiles = ne10 / 32;
        const uint32_t tile_row_size = n_k_tiles * mmctx->tile_size;
        octx->src0_spad.size_per_thread = hex_round_up(DMA_DEPTH * tile_row_size, 256);
        octx->src0_spad.size            = octx->src0_spad.size_per_thread * octx->n_threads;
    }

    size_t spad_size = octx->src2_spad.size + octx->src1_spad.size + octx->src0_spad.size + octx->dst_spad.size;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %u src1-spad-size %u src2-spad-size %u dst-spad-size %u (%zu)\n", mmctx->type,
         octx->src0_spad.size, octx->src1_spad.size, octx->src2_spad.size, octx->dst_spad.size, spad_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, spad_size);
        if (must_free_mapping) free(mapping_buf);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    // Place src1 spad first. We use it for dyn.quant and may reuse in subseq ops.
    octx->src1_spad.data = octx->ctx->vtcm_base;
    octx->src0_spad.data = octx->src1_spad.data + octx->src1_spad.size;
    octx->src2_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src2_spad.data + octx->src2_spad.size;

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    octx->src0_spad.stride = src0_row_size_padded;
    octx->src1_spad.stride = src1_row_size;

    mmctx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
    worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, mmctx, n_quant_jobs);

    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_pool_run_func(octx->ctx->worker_pool, matmul_id_job_func, mmctx, n_matmul_jobs);

    if (must_free_mapping) free(mapping_buf);
    return HTP_STATUS_OK;
}



static void matmul_qkv_2d(unsigned int nth, unsigned int ith, void * data) {
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;

    const struct htp_tensor * restrict src0 = octx->src[0]; // Wk
    const struct htp_tensor * restrict src1 = octx->src[1]; // x
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wv
    const struct htp_tensor * restrict src3 = octx->src[3]; // Wq
    const struct htp_tensor * restrict dst_k = octx->dsts[0];
    const struct htp_tensor * restrict dst_v = octx->dsts[1];
    const struct htp_tensor * restrict dst_q = octx->dsts[2];

    struct htp_spad * restrict src0_spad = &octx->src0_spad; // Wk
    struct htp_spad * restrict src2_spad = &octx->src2_spad; // Wv
    struct htp_spad * restrict src3_spad = &octx->src3_spad; // Wq
    struct htp_spad * restrict src1_spad = &octx->src1_spad; // x

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

    const size_t src0_stride = src0_spad->stride;
    const size_t src2_stride = src2_spad->stride;
    const size_t src3_stride = src3_spad->stride;
    const size_t src1_stride = src1_spad->stride;

    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict spad_src2 = src2_spad->data + src2_spad->size_per_thread * ith;
    uint8_t * restrict spad_src3 = src3_spad->data + src3_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    dma_queue * dma_queue = octx->ctx->dma[ith];

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;
    const uint8_t * restrict src3_row = (const uint8_t *) src3->data;

    // Prefill spad with src0, src2, src3 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= MM_SPAD_SRC0_NROWS) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
        dma_queue_push(dma_queue, dma_make_ptr(spad_src2 + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                       src2_stride, src2_row_size, src2_row_size, 2);
        dma_queue_push(dma_queue, dma_make_ptr(spad_src3 + is0 * src3_stride, src3_row + ir0 * src3_row_size),
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

        // Prefetch next (n + spad_nrows) rows
        const int pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
        const int is0 = (pr0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(spad_src2 + is0 * src2_stride, src2_row + pr0 * src2_row_size),
                           src2_stride, src2_row_size, src2_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(spad_src3 + is0 * src3_stride, src3_row + pr0 * src3_row_size),
                           src3_stride, src3_row_size, src3_row_size, 2);
        }
    }

    // Process last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(spad_src2 + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                       src2_stride, src2_row_size, src2_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(spad_src3 + is0 * src3_stride, src3_row + ir0 * src3_row_size),
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
    struct htp_matmul_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;

    const struct htp_tensor * restrict src0 = octx->src[0]; // Wgate
    const struct htp_tensor * restrict src1 = octx->src[1]; // y
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wup
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];
    const struct htp_tensor * restrict dst_up = octx->dsts[1];

    struct htp_spad * restrict src0_spad = &octx->src0_spad; // Wgate
    struct htp_spad * restrict src2_spad = &octx->src2_spad; // Wup
    struct htp_spad * restrict src1_spad = &octx->src1_spad; // y

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

    const size_t src0_stride = src0_spad->stride;
    const size_t src2_stride = src2_spad->stride;
    const size_t src1_stride = src1_spad->stride;

    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict spad_src2 = src2_spad->data + src2_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    dma_queue * dma_queue = octx->ctx->dma[ith];

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;

    // Prefill spad with src0, src2 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= MM_SPAD_SRC0_NROWS) {
            break;
        }
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
        dma_queue_push(dma_queue, dma_make_ptr(spad_src2 + is0 * src2_stride, src2_row + ir0 * src2_row_size),
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
        const int pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
        const int is0 = (pr0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(spad_src2 + is0 * src2_stride, src2_row + pr0 * src2_row_size),
                           src2_stride, src2_row_size, src2_row_size, 2);
        }
    }

    // Process last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        dma_queue_push(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(spad_src2 + is0 * src2_stride, src2_row + ir0 * src2_row_size),
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

    struct htp_matmul_context mmctx_struct = {0};
    struct htp_matmul_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

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

    size_t src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? q8_1_tiled_row_size(src1->ne[0]) : q8_0_tiled_row_size(src1->ne[0]);

    // Set up scratchpads
    if (is_repacked) {
        uint32_t tile_size;
        if (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_IQ4_NL) { tile_size = 576; }
        else if (src0->type == HTP_TYPE_Q4_1) { tile_size = 640; }
        else if (src0->type == HTP_TYPE_Q8_0) { tile_size = 1088; }
        else if (src0->type == HTP_TYPE_MXFP4) { tile_size = 544; }
        else { tile_size = 0; }
        uint32_t n_k_tiles = src1->ne[0] / 32;
        uint32_t tile_row_size = n_k_tiles * tile_size;

        octx->src0_spad.size_per_thread = hex_round_up(DMA_DEPTH * tile_row_size, 256);
        octx->src0_spad.size            = octx->src0_spad.size_per_thread * octx->n_threads;

        octx->src2_spad.size_per_thread = hex_round_up(DMA_DEPTH * tile_row_size, 256);
        octx->src2_spad.size            = octx->src2_spad.size_per_thread * octx->n_threads;

        octx->src3_spad.size_per_thread = hex_round_up(DMA_DEPTH * tile_row_size, 256);
        octx->src3_spad.size            = octx->src3_spad.size_per_thread * octx->n_threads;
    } else {
        octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
        octx->src0_spad.size            = octx->src0_spad.size_per_thread * octx->n_threads;

        octx->src2_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
        octx->src2_spad.size            = octx->src2_spad.size_per_thread * octx->n_threads;

        octx->src3_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
        octx->src3_spad.size            = octx->src3_spad.size_per_thread * octx->n_threads;
    }

    octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);
    octx->src1_spad.size            = octx->src1_spad.size_per_thread;

    octx->dst_spad.size_per_thread  = 0;
    octx->dst_spad.size            = 0;

    size_t spad_size = octx->src0_spad.size + octx->src1_spad.size + octx->src2_spad.size + octx->src3_spad.size;

    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "matmul-qkv: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    // Place src1 first
    octx->src1_spad.data = octx->ctx->vtcm_base;
    octx->src0_spad.data = octx->src1_spad.data + octx->src1_spad.size;
    octx->src2_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->src3_spad.data = octx->src2_spad.data + octx->src2_spad.size;

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->src3_spad.src  = NULL;

    octx->src0_spad.stride = is_repacked ? 0 : src0_row_size_padded;
    octx->src2_spad.stride = is_repacked ? 0 : src0_row_size_padded;
    octx->src3_spad.stride = is_repacked ? 0 : src0_row_size_padded;
    octx->src1_spad.stride = src1_row_size;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)
        return HTP_STATUS_OK;

    // Run quantization once
    mmctx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
    worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, mmctx, n_quant_jobs);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
        worker_callback_t matmul_job_func;
    if (is_repacked) {
        switch (src0->type) {
            case HTP_TYPE_Q4_0:   matmul_job_func = matmul_qkv_2d_repacked_q4_0;   break;
            case HTP_TYPE_Q4_1:   matmul_job_func = matmul_qkv_2d_repacked_q4_1;   break;
            case HTP_TYPE_Q8_0:   matmul_job_func = matmul_qkv_2d_repacked_q8_0;   break;
            case HTP_TYPE_IQ4_NL:  matmul_job_func = matmul_qkv_2d_repacked_iq4nl;  break;
            case HTP_TYPE_MXFP4:  matmul_job_func = matmul_qkv_2d_repacked_mxfp4;  break;
            default:              return HTP_STATUS_NO_SUPPORT;
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

    struct htp_matmul_context mmctx_struct = {0};
    struct htp_matmul_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

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

    size_t src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? q8_1_tiled_row_size(src1->ne[0]) : q8_0_tiled_row_size(src1->ne[0]);

    // Set up scratchpads
    if (is_repacked) {
        uint32_t tile_size;
        if (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_IQ4_NL) { tile_size = 576; }
        else if (src0->type == HTP_TYPE_Q4_1) { tile_size = 640; }
        else if (src0->type == HTP_TYPE_Q8_0) { tile_size = 1088; }
        else if (src0->type == HTP_TYPE_MXFP4) { tile_size = 544; }
        else { tile_size = 0; }
        uint32_t n_k_tiles = src1->ne[0] / 32;
        uint32_t tile_row_size = n_k_tiles * tile_size;

        octx->src0_spad.size_per_thread = hex_round_up(DMA_DEPTH * tile_row_size, 256);
        octx->src0_spad.size            = octx->src0_spad.size_per_thread * octx->n_threads;

        octx->src2_spad.size_per_thread = hex_round_up(DMA_DEPTH * tile_row_size, 256);
        octx->src2_spad.size            = octx->src2_spad.size_per_thread * octx->n_threads;
    } else {
        octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
        octx->src0_spad.size            = octx->src0_spad.size_per_thread * octx->n_threads;

        octx->src2_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
        octx->src2_spad.size            = octx->src2_spad.size_per_thread * octx->n_threads;
    }

    octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);
    octx->src1_spad.size            = octx->src1_spad.size_per_thread;

    octx->dst_spad.size_per_thread  = 0;
    octx->dst_spad.size            = 0;

    size_t spad_size = octx->src0_spad.size + octx->src1_spad.size + octx->src2_spad.size;

    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "matmul-ffn: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    // Place src1 first
    octx->src1_spad.data = octx->ctx->vtcm_base;
    octx->src0_spad.data = octx->src1_spad.data + octx->src1_spad.size;
    octx->src2_spad.data = octx->src0_spad.data + octx->src0_spad.size;

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;

    octx->src0_spad.stride = is_repacked ? 0 : src0_row_size_padded;
    octx->src2_spad.stride = is_repacked ? 0 : src0_row_size_padded;
    octx->src1_spad.stride = src1_row_size;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)
        return HTP_STATUS_OK;

    // Run quantization once
    mmctx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
    worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, mmctx, n_quant_jobs);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
        worker_callback_t matmul_job_func;
    if (is_repacked) {
        switch (src0->type) {
            case HTP_TYPE_Q4_0:   matmul_job_func = matmul_ffn_2d_repacked_q4_0;   break;
            case HTP_TYPE_Q4_1:   matmul_job_func = matmul_ffn_2d_repacked_q4_1;   break;
            case HTP_TYPE_Q8_0:   matmul_job_func = matmul_ffn_2d_repacked_q8_0;   break;
            case HTP_TYPE_IQ4_NL:  matmul_job_func = matmul_ffn_2d_repacked_iq4nl;  break;
            case HTP_TYPE_MXFP4:  matmul_job_func = matmul_ffn_2d_repacked_mxfp4;  break;
            default:              return HTP_STATUS_NO_SUPPORT;
        }
    } else {
        matmul_job_func = matmul_ffn_2d;
    }
    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}
