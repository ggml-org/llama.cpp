#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "binary-ops.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Context for binary operations
struct htp_binary_context {
    struct htp_ops_context * octx;
    struct htp_binary_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;

    struct fastdiv_values src0_dim1_div; // ne01
    struct fastdiv_values src0_dim2_div; // ne02
    struct fastdiv_values src0_dim12_div;// ne03

    struct fastdiv_values src1_dim1_div; // ne11
    struct fastdiv_values src1_dim2_div; // ne12
    struct fastdiv_values src1_dim3_div; // ne13

    uint32_t block_max;
    uint32_t nrows_per_hvx;
    uint32_t total_rows;
    uint32_t row_start;
    uint32_t n_srcs;
    size_t   src0_row_size_aligned;
    size_t   src1_row_size_aligned;
    size_t   dst_row_size_aligned;
    size_t   src0_row_size;
    size_t   src1_row_size;
    size_t   dst_row_size;
    uint32_t src0_elem_size;
    uint32_t src1_elem_size;
    uint32_t dst_elem_size;
    uint32_t src1_table_elem_size;

    bool split_at_ne01;
    bool split_at_ne02;

    void * compute;
};

#define htp_binary_preamble                        \
    const struct htp_tensor * src0 = octx->src[0]; \
    const struct htp_tensor * src1 = octx->src[1]; \
    const struct htp_tensor * dst  = octx->dst;    \
                                                   \
    const uint32_t ne00 = src0->ne[0];             \
    const uint32_t ne01 = src0->ne[1];             \
    const uint32_t ne02 = src0->ne[2];             \
    const uint32_t ne03 = src0->ne[3];             \
                                                   \
    const uint32_t ne10 = src1->ne[0];             \
    const uint32_t ne11 = src1->ne[1];             \
    const uint32_t ne12 = src1->ne[2];             \
    const uint32_t ne13 = src1->ne[3];             \
                                                   \
    const uint32_t nb01 = src0->nb[1];             \
    const uint32_t nb02 = src0->nb[2];             \
    const uint32_t nb03 = src0->nb[3];             \
                                                   \
    const uint32_t nb11 = src1->nb[1];             \
    const uint32_t nb12 = src1->nb[2];             \
    const uint32_t nb13 = src1->nb[3];             \
                                                   \
    const uint32_t nb1 = dst->nb[1];               \
    const uint32_t nb2 = dst->nb[2];               \
    const uint32_t nb3 = dst->nb[3];

static inline uint32_t calc_block_size(struct htp_binary_context * bctx, uint32_t ir, uint32_t end_row, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t rows_left = end_row - ir;
    uint32_t block_limit = rows_left;

    if (bctx->split_at_ne01) {
        block_limit = MIN(block_limit, ne01 - i01);
    }
    if (bctx->split_at_ne02) {
         uint32_t rows_in_plane = (ne02 * ne01) - rem;
         block_limit = MIN(block_limit, rows_in_plane);
    }

    return MIN(bctx->block_max, block_limit);
}

// Out-of-line compute micro-kernels

typedef void (*compute_scalar_t)(
    uint8_t * dst, const uint8_t * src0, const void * s1_table,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
static void compute_scalar_##NAME(                              \
    uint8_t * dst, const uint8_t * src0, const void * s1_table, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    const TYPE * table = (const TYPE *) s1_table;               \
    for (uint32_t r = 0; r < n_rows; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        const uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = table[cur_i11];                              \
        HVX_STMT;                                               \
        if (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(add_f32, float,    hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(add_f16, _Float16, hvx_add_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, float,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, float,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, float,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

typedef void (*compute_same_shape_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
static void compute_same_shape_##NAME(                                                           \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1,                                   \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    for (uint32_t r = 0; r < n_rows; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(add_f32, hvx_add_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(add_f16, hvx_add_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

typedef void (*compute_row_bcast_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_ROW_BCAST(NAME, HVX_FN)                                               \
static void compute_row_bcast_##NAME(                                                        \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1,                               \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    for (uint32_t r = 0; r < n_rows; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_ROW_BCAST(add_f32, hvx_add_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(add_f16, hvx_add_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(div_f16, hvx_div_f16_aaa)

typedef void (*compute_complex_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
static void compute_complex_##NAME(                                                          \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t src1_stride,  \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    for (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        const uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(add_f32, hvx_add_f32_aau)
DEFINE_COMPUTE_COMPLEX(add_f16, hvx_add_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

typedef void (*compute_repeat_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#define DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
static void compute_repeat_##NAME(                                                           \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t src1_stride,  \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    for (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        const uint8_t * r_src1_row = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        const uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        for (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * sizeof(TYPE), r_src0 + c * sizeof(TYPE), r_src1_row, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(add_f32, float,    hvx_add_f32_uuu)
DEFINE_COMPUTE_REPEAT(add_f16, _Float16, hvx_add_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, float,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, float,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, float,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

static inline HVX_Vector binary_clamp_f16_finite_vec(HVX_Vector v, HVX_Vector max_f16, HVX_Vector min_f16, HVX_Vector inf) {
    const HVX_VectorPred finite = Q6_Q_vcmp_gt_VsfVsf(inf, hvx_vec_abs_f32(v));
    const HVX_VectorPred high   = Q6_Q_and_QQ(finite, Q6_Q_vcmp_gt_VsfVsf(v, max_f16));
    const HVX_VectorPred low    = Q6_Q_and_QQ(finite, Q6_Q_vcmp_gt_VsfVsf(min_f16, v));
    v = Q6_V_vmux_QVV(high, max_f16, v);
    return Q6_V_vmux_QVV(low, min_f16, v);
}

#define BINARY_OP_ADD(a, b, nan_inf_mask) ((void) (nan_inf_mask), hvx_vec_add_f32_f32(a, b))
#define BINARY_OP_SUB(a, b, nan_inf_mask) ((void) (nan_inf_mask), hvx_vec_sub_f32_f32(a, b))
#define BINARY_OP_MUL(a, b, nan_inf_mask) ((void) (nan_inf_mask), hvx_vec_mul_f32_f32(a, b))
#define BINARY_OP_DIV(a, b, nan_inf_mask) hvx_vec_mul_f32_f32(a, hvx_vec_inverse_f32_guard(b, nan_inf_mask))

#define BINARY_LOAD64_A_F16(vsrc, k, s_lo, s_hi)              \
    HVX_VectorPair p_##s_lo = hvx_vec_f16_to_f32((vsrc)[k]);  \
    HVX_Vector s_lo = Q6_V_lo_W(p_##s_lo);                    \
    HVX_Vector s_hi = Q6_V_hi_W(p_##s_lo)

#define BINARY_LOAD64_A_F32(vsrc, k, s_lo, s_hi)              \
    HVX_Vector s_lo = (vsrc)[2 * (k) + 0];                    \
    HVX_Vector s_hi = (vsrc)[2 * (k) + 1]

#define BINARY_LOAD64_U_F16(src, k, s_lo, s_hi)                                               \
    HVX_VectorPair p_##s_lo = hvx_vec_f16_to_f32(hvx_vmemu((src) + (size_t) (k) * 128));      \
    HVX_Vector s_lo = Q6_V_lo_W(p_##s_lo);                                                    \
    HVX_Vector s_hi = Q6_V_hi_W(p_##s_lo)

#define BINARY_LOAD64_U_F32(src, k, s_lo, s_hi)                                               \
    HVX_Vector s_lo = hvx_vmemu((src) + (size_t) (2 * (k) + 0) * 128);                        \
    HVX_Vector s_hi = hvx_vmemu((src) + (size_t) (2 * (k) + 1) * 128)

#define BINARY_TAIL_LO_A_F16(vsrc, nvec, s_lo, s_hi)          \
    BINARY_LOAD64_A_F16(vsrc, nvec, s_lo, s_hi)
#define BINARY_TAIL_HI_A_F16(vsrc, nvec, s_hi)                \
    ((void) (vsrc), (void) (nvec), (void) (s_hi))

#define BINARY_TAIL_LO_A_F32(vsrc, nvec, s_lo, s_hi)          \
    HVX_Vector s_lo = (vsrc)[2 * (nvec)];                     \
    HVX_Vector s_hi
#define BINARY_TAIL_HI_A_F32(vsrc, nvec, s_hi)                \
    s_hi = (vsrc)[2 * (nvec) + 1]

#define BINARY_TAIL_LO_U_F16(src, nvec, s_lo, s_hi)           \
    BINARY_LOAD64_U_F16(src, nvec, s_lo, s_hi)
#define BINARY_TAIL_HI_U_F16(src, nvec, s_hi)                 \
    ((void) (src), (void) (nvec), (void) (s_hi))

#define BINARY_TAIL_LO_U_F32(src, nvec, s_lo, s_hi)                   \
    HVX_Vector s_lo = hvx_vmemu((src) + (size_t) (2 * (nvec)) * 128); \
    HVX_Vector s_hi
#define BINARY_TAIL_HI_U_F32(src, nvec, s_hi)                 \
    s_hi = hvx_vmemu((src) + (size_t) (2 * (nvec) + 1) * 128)

#define DEFINE_BINARY_MIXED_F32(NAME, T0, T1, OP, UNROLL)                                              \
static void NAME(uint8_t * dst, const uint8_t * src0, const uint8_t * src1, uint32_t n) {              \
    const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                                         \
    const uint32_t nvec = n / 64;                                                                      \
    const uint32_t nloe = n % 64;                                                                      \
    if (hex_is_aligned(dst, 128) && hex_is_aligned(src0, 128) && hex_is_aligned(src1, 128)) {          \
        const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;                                 \
        const HVX_Vector * restrict vsrc1 = (const HVX_Vector *) src1;                                 \
        HVX_Vector * restrict vdst        = (HVX_Vector *) dst;                                        \
        UNROLL                                                                                         \
        for (uint32_t k = 0; k < nvec; k++) {                                                          \
            BINARY_LOAD64_A_##T0(vsrc0, k, s0_0, s0_1);                                                \
            BINARY_LOAD64_A_##T1(vsrc1, k, s1_0, s1_1);                                                \
            vdst[2 * k + 0] = OP(s0_0, s1_0, nan_inf_mask);                                            \
            vdst[2 * k + 1] = OP(s0_1, s1_1, nan_inf_mask);                                            \
        }                                                                                              \
        if (nloe) {                                                                                    \
            BINARY_TAIL_LO_A_##T0(vsrc0, nvec, s0_0, s0_1);                                            \
            BINARY_TAIL_LO_A_##T1(vsrc1, nvec, s1_0, s1_1);                                            \
            HVX_Vector out0 = OP(s0_0, s1_0, nan_inf_mask);                                            \
            if (nloe <= 32) {                                                                          \
                hvx_vec_store_a(&vdst[2 * nvec], nloe * sizeof(float), out0);                          \
            } else {                                                                                   \
                vdst[2 * nvec] = out0;                                                                 \
                BINARY_TAIL_HI_A_##T0(vsrc0, nvec, s0_1);                                              \
                BINARY_TAIL_HI_A_##T1(vsrc1, nvec, s1_1);                                              \
                HVX_Vector out1 = OP(s0_1, s1_1, nan_inf_mask);                                        \
                hvx_vec_store_a(&vdst[2 * nvec + 1], (nloe - 32) * sizeof(float), out1);               \
            }                                                                                          \
        }                                                                                              \
        return;                                                                                        \
    }                                                                                                  \
    for (uint32_t k = 0; k < nvec; k++) {                                                              \
        BINARY_LOAD64_U_##T0(src0, k, s0_0, s0_1);                                                     \
        BINARY_LOAD64_U_##T1(src1, k, s1_0, s1_1);                                                     \
        *(HVX_UVector *) (dst + (size_t) (2 * k + 0) * 128) = OP(s0_0, s1_0, nan_inf_mask);            \
        *(HVX_UVector *) (dst + (size_t) (2 * k + 1) * 128) = OP(s0_1, s1_1, nan_inf_mask);            \
    }                                                                                                  \
    if (nloe) {                                                                                        \
        BINARY_TAIL_LO_U_##T0(src0, nvec, s0_0, s0_1);                                                 \
        BINARY_TAIL_LO_U_##T1(src1, nvec, s1_0, s1_1);                                                 \
        HVX_Vector out0 = OP(s0_0, s1_0, nan_inf_mask);                                                \
        if (nloe <= 32) {                                                                              \
            hvx_vec_store_u(dst + (size_t) (2 * nvec) * 128, nloe * sizeof(float), out0);              \
        } else {                                                                                       \
            *(HVX_UVector *) (dst + (size_t) (2 * nvec) * 128) = out0;                                 \
            BINARY_TAIL_HI_U_##T0(src0, nvec, s0_1);                                                   \
            BINARY_TAIL_HI_U_##T1(src1, nvec, s1_1);                                                   \
            HVX_Vector out1 = OP(s0_1, s1_1, nan_inf_mask);                                            \
            hvx_vec_store_u(dst + (size_t) (2 * nvec + 1) * 128, (nloe - 32) * sizeof(float), out1);   \
        }                                                                                              \
    }                                                                                                  \
}

#define DEFINE_BINARY_MIXED_F16(NAME, T0, T1, OP, UNROLL)                                              \
static void NAME(uint8_t * dst, const uint8_t * src0, const uint8_t * src1, uint32_t n) {              \
    const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                                         \
    const HVX_Vector max_f16      = hvx_vec_splat_f32(65504.0f);                                       \
    const HVX_Vector min_f16      = hvx_vec_splat_f32(-65504.0f);                                      \
    const HVX_Vector inf          = hvx_vec_splat_f32(INFINITY);                                       \
    const uint32_t nvec = n / 64;                                                                      \
    const uint32_t nloe = n % 64;                                                                      \
    if (hex_is_aligned(dst, 128) && hex_is_aligned(src0, 128) && hex_is_aligned(src1, 128)) {          \
        const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;                                 \
        const HVX_Vector * restrict vsrc1 = (const HVX_Vector *) src1;                                 \
        HVX_Vector * restrict vdst        = (HVX_Vector *) dst;                                        \
        UNROLL                                                                                         \
        for (uint32_t k = 0; k < nvec; k++) {                                                          \
            BINARY_LOAD64_A_##T0(vsrc0, k, s0_0, s0_1);                                                \
            BINARY_LOAD64_A_##T1(vsrc1, k, s1_0, s1_1);                                                \
            HVX_Vector out0 = OP(s0_0, s1_0, nan_inf_mask);                                            \
            HVX_Vector out1 = OP(s0_1, s1_1, nan_inf_mask);                                            \
            out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);                           \
            out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);                           \
            vdst[k] = hvx_vec_f32_to_f16(out0, out1);                                                  \
        }                                                                                              \
        if (nloe) {                                                                                    \
            BINARY_TAIL_LO_A_##T0(vsrc0, nvec, s0_0, s0_1);                                            \
            BINARY_TAIL_LO_A_##T1(vsrc1, nvec, s1_0, s1_1);                                            \
            HVX_Vector out0 = OP(s0_0, s1_0, nan_inf_mask);                                            \
            out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);                           \
            HVX_Vector out1 = Q6_V_vzero();                                                            \
            if (nloe > 32) {                                                                           \
                BINARY_TAIL_HI_A_##T0(vsrc0, nvec, s0_1);                                              \
                BINARY_TAIL_HI_A_##T1(vsrc1, nvec, s1_1);                                              \
                out1 = OP(s0_1, s1_1, nan_inf_mask);                                                   \
                out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);                       \
            }                                                                                          \
            HVX_Vector packed = hvx_vec_f32_to_f16(out0, out1);                                        \
            hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), packed);                             \
        }                                                                                              \
        return;                                                                                        \
    }                                                                                                  \
    for (uint32_t k = 0; k < nvec; k++) {                                                              \
        BINARY_LOAD64_U_##T0(src0, k, s0_0, s0_1);                                                     \
        BINARY_LOAD64_U_##T1(src1, k, s1_0, s1_1);                                                     \
        HVX_Vector out0 = OP(s0_0, s1_0, nan_inf_mask);                                                \
        HVX_Vector out1 = OP(s0_1, s1_1, nan_inf_mask);                                                \
        out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);                               \
        out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);                               \
        *(HVX_UVector *) (dst + (size_t) k * 128) = hvx_vec_f32_to_f16(out0, out1);                    \
    }                                                                                                  \
    if (nloe) {                                                                                        \
        BINARY_TAIL_LO_U_##T0(src0, nvec, s0_0, s0_1);                                                 \
        BINARY_TAIL_LO_U_##T1(src1, nvec, s1_0, s1_1);                                                 \
        HVX_Vector out0 = OP(s0_0, s1_0, nan_inf_mask);                                                \
        out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);                               \
        HVX_Vector out1 = Q6_V_vzero();                                                                \
        if (nloe > 32) {                                                                               \
            BINARY_TAIL_HI_U_##T0(src0, nvec, s0_1);                                                   \
            BINARY_TAIL_HI_U_##T1(src1, nvec, s1_1);                                                   \
            out1 = OP(s0_1, s1_1, nan_inf_mask);                                                       \
            out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);                           \
        }                                                                                              \
        HVX_Vector packed = hvx_vec_f32_to_f16(out0, out1);                                            \
        hvx_vec_store_u(dst + (size_t) nvec * 128, nloe * sizeof(_Float16), packed);                   \
    }                                                                                                  \
}

#define DEFINE_BINARY_MIXED_F16_F16_F16(NAME, OP)                                                      \
static void NAME(uint8_t * dst, const uint8_t * src0, const uint8_t * src1, uint32_t n) {              \
    const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                                         \
    const HVX_Vector max_f16      = hvx_vec_splat_f32(65504.0f);                                       \
    const HVX_Vector min_f16      = hvx_vec_splat_f32(-65504.0f);                                      \
    const HVX_Vector inf          = hvx_vec_splat_f32(INFINITY);                                       \
    const uint32_t nvec = n / 64;                                                                      \
    const uint32_t nloe = n % 64;                                                                      \
    if (hex_is_aligned(dst, 128) && hex_is_aligned(src0, 128) && hex_is_aligned(src1, 128)) {          \
        const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;                                 \
        const HVX_Vector * restrict vsrc1 = (const HVX_Vector *) src1;                                 \
        HVX_Vector * restrict vdst        = (HVX_Vector *) dst;                                        \
        _Pragma("unroll(2)")                                                                           \
        for (uint32_t k = 0; k < nvec; k++) {                                                          \
            HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(vsrc0[k]);                                    \
            HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(vsrc1[k]);                                    \
            HVX_Vector out0 = OP(Q6_V_lo_W(p0), Q6_V_lo_W(p1), nan_inf_mask);                          \
            HVX_Vector out1 = OP(Q6_V_hi_W(p0), Q6_V_hi_W(p1), nan_inf_mask);                          \
            out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);                           \
            out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);                           \
            vdst[k] = hvx_vec_f32_to_f16_shuff(out0, out1);                                            \
        }                                                                                              \
        if (nloe) {                                                                                    \
            HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(vsrc0[nvec]);                                 \
            HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(vsrc1[nvec]);                                 \
            HVX_Vector out0 = OP(Q6_V_lo_W(p0), Q6_V_lo_W(p1), nan_inf_mask);                          \
            HVX_Vector out1 = OP(Q6_V_hi_W(p0), Q6_V_hi_W(p1), nan_inf_mask);                          \
            out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);                           \
            out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);                           \
            HVX_Vector packed = hvx_vec_f32_to_f16_shuff(out0, out1);                                  \
            hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), packed);                             \
        }                                                                                              \
        return;                                                                                        \
    }                                                                                                  \
    for (uint32_t k = 0; k < nvec; k++) {                                                              \
        HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(hvx_vmemu(src0 + (size_t) k * 128));              \
        HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(hvx_vmemu(src1 + (size_t) k * 128));              \
        HVX_Vector out0 = OP(Q6_V_lo_W(p0), Q6_V_lo_W(p1), nan_inf_mask);                              \
        HVX_Vector out1 = OP(Q6_V_hi_W(p0), Q6_V_hi_W(p1), nan_inf_mask);                              \
        out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);                               \
        out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);                               \
        *(HVX_UVector *) (dst + (size_t) k * 128) = hvx_vec_f32_to_f16_shuff(out0, out1);              \
    }                                                                                                  \
    if (nloe) {                                                                                        \
        HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(hvx_vmemu(src0 + (size_t) nvec * 128));           \
        HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(hvx_vmemu(src1 + (size_t) nvec * 128));           \
        HVX_Vector out0 = OP(Q6_V_lo_W(p0), Q6_V_lo_W(p1), nan_inf_mask);                              \
        HVX_Vector out1 = OP(Q6_V_hi_W(p0), Q6_V_hi_W(p1), nan_inf_mask);                              \
        out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);                               \
        out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);                               \
        HVX_Vector packed = hvx_vec_f32_to_f16_shuff(out0, out1);                                      \
        hvx_vec_store_u(dst + (size_t) nvec * 128, nloe * sizeof(_Float16), packed);                   \
    }                                                                                                  \
}

#define DEFINE_BINARY_MIXED_OP(OP_NAME, OP, UNROLL)                                 \
    DEFINE_BINARY_MIXED_F16(binary_##OP_NAME##_f16_f32_f16, F16, F32, OP, UNROLL)   \
    DEFINE_BINARY_MIXED_F32(binary_##OP_NAME##_f16_f32_f32, F16, F32, OP, UNROLL)   \
    DEFINE_BINARY_MIXED_F16(binary_##OP_NAME##_f32_f16_f16, F32, F16, OP, UNROLL)   \
    DEFINE_BINARY_MIXED_F32(binary_##OP_NAME##_f32_f16_f32, F32, F16, OP, UNROLL)   \
    DEFINE_BINARY_MIXED_F32(binary_##OP_NAME##_f16_f16_f32, F16, F16, OP, UNROLL)   \
    DEFINE_BINARY_MIXED_F16(binary_##OP_NAME##_f32_f32_f16, F32, F32, OP, UNROLL)

DEFINE_BINARY_MIXED_OP(add, BINARY_OP_ADD, _Pragma("unroll(2)"))
DEFINE_BINARY_MIXED_OP(sub, BINARY_OP_SUB, _Pragma("unroll(2)"))
DEFINE_BINARY_MIXED_OP(mul, BINARY_OP_MUL, _Pragma("unroll(2)"))
DEFINE_BINARY_MIXED_OP(div, BINARY_OP_DIV, )
DEFINE_BINARY_MIXED_F16_F16_F16(binary_add_f16_f16_f16, BINARY_OP_ADD)

#define DEFINE_COMPUTE_MIXED_SHAPES(OP_NAME, TYPES)                                                     \
    DEFINE_COMPUTE_SAME_SHAPE(OP_NAME##_##TYPES, binary_##OP_NAME##_##TYPES)                            \
    DEFINE_COMPUTE_ROW_BCAST(OP_NAME##_##TYPES, binary_##OP_NAME##_##TYPES)                             \
    DEFINE_COMPUTE_COMPLEX(OP_NAME##_##TYPES, binary_##OP_NAME##_##TYPES)

#define DEFINE_COMPUTE_MIXED_REPEAT(OP_NAME, TYPES, DST_TYPE, SRC0_TYPE)                               \
static void compute_repeat_##OP_NAME##_##TYPES(                                                        \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,                                   \
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t src1_stride,            \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {            \
    for (uint32_t r = 0; r < n_rows; r++) {                                                            \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                               \
        const uint8_t * r_src1 = src1_plane + i11 * src1_stride;                                       \
        for (uint32_t c = 0; c < ne00; c += ne10) {                                                    \
            binary_##OP_NAME##_##TYPES(dst + r * dst_stride + (size_t) c * sizeof(DST_TYPE),           \
                src0 + r * src0_stride + (size_t) c * sizeof(SRC0_TYPE), r_src1, MIN(ne10, ne00 - c)); \
        }                                                                                              \
    }                                                                                                  \
}

#define DEFINE_COMPUTE_MIXED_ALL(OP_NAME)                                 \
    DEFINE_COMPUTE_MIXED_SHAPES(OP_NAME, f16_f32_f16)                     \
    DEFINE_COMPUTE_MIXED_SHAPES(OP_NAME, f16_f32_f32)                     \
    DEFINE_COMPUTE_MIXED_SHAPES(OP_NAME, f32_f16_f16)                     \
    DEFINE_COMPUTE_MIXED_SHAPES(OP_NAME, f32_f16_f32)                     \
    DEFINE_COMPUTE_MIXED_SHAPES(OP_NAME, f16_f16_f32)                     \
    DEFINE_COMPUTE_MIXED_SHAPES(OP_NAME, f32_f32_f16)                     \
    DEFINE_COMPUTE_MIXED_REPEAT(OP_NAME, f16_f32_f16, _Float16, _Float16) \
    DEFINE_COMPUTE_MIXED_REPEAT(OP_NAME, f16_f32_f32, float, _Float16)    \
    DEFINE_COMPUTE_MIXED_REPEAT(OP_NAME, f32_f16_f16, _Float16, float)    \
    DEFINE_COMPUTE_MIXED_REPEAT(OP_NAME, f32_f16_f32, float, float)       \
    DEFINE_COMPUTE_MIXED_REPEAT(OP_NAME, f16_f16_f32, float, _Float16)    \
    DEFINE_COMPUTE_MIXED_REPEAT(OP_NAME, f32_f32_f16, _Float16, float)

DEFINE_COMPUTE_MIXED_ALL(add)
DEFINE_COMPUTE_MIXED_ALL(sub)
DEFINE_COMPUTE_MIXED_ALL(mul)
DEFINE_COMPUTE_MIXED_ALL(div)

#define DEFINE_BINARY_MIXED_SCALAR_F16_F16(NAME, OP)                                  \
static void NAME(uint8_t * dst, const uint8_t * src0, float val, uint32_t n) {        \
    const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                        \
    const HVX_Vector val_vec      = hvx_vec_splat_f32(val);                           \
    const HVX_Vector max_f16      = hvx_vec_splat_f32(65504.0f);                      \
    const HVX_Vector min_f16      = hvx_vec_splat_f32(-65504.0f);                     \
    const HVX_Vector inf          = hvx_vec_splat_f32(INFINITY);                      \
    const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;                    \
    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;                           \
    const uint32_t nvec = n / 64;                                                     \
    const uint32_t nloe = n % 64;                                                     \
    _Pragma("unroll(2)")                                                              \
    for (uint32_t k = 0; k < nvec; k++) {                                             \
        HVX_VectorPair pair = hvx_vec_f16_to_f32_shuff(vsrc0[k]);                     \
        HVX_Vector out0 = OP(Q6_V_lo_W(pair), val_vec, nan_inf_mask);                 \
        HVX_Vector out1 = OP(Q6_V_hi_W(pair), val_vec, nan_inf_mask);                 \
        out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);              \
        out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);              \
        vdst[k] = hvx_vec_f32_to_f16_shuff(out0, out1);                               \
    }                                                                                 \
    if (nloe) {                                                                       \
        HVX_VectorPair pair = hvx_vec_f16_to_f32_shuff(vsrc0[nvec]);                  \
        HVX_Vector out0 = OP(Q6_V_lo_W(pair), val_vec, nan_inf_mask);                 \
        HVX_Vector out1 = OP(Q6_V_hi_W(pair), val_vec, nan_inf_mask);                 \
        out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);              \
        out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);              \
        HVX_Vector packed = hvx_vec_f32_to_f16_shuff(out0, out1);                     \
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), packed);                \
    }                                                                                 \
}

#define DEFINE_BINARY_MIXED_SCALAR_F16_F32(NAME, OP)                                  \
static void NAME(uint8_t * dst, const uint8_t * src0, float val, uint32_t n) {        \
    const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                        \
    const HVX_Vector val_vec      = hvx_vec_splat_f32(val);                           \
    const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;                    \
    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;                           \
    const uint32_t nvec = n / 64;                                                     \
    const uint32_t nloe = n % 64;                                                     \
    _Pragma("unroll(2)")                                                              \
    for (uint32_t k = 0; k < nvec; k++) {                                             \
        HVX_VectorPair pair = hvx_vec_f16_to_f32(vsrc0[k]);                           \
        vdst[2 * k + 0] = OP(Q6_V_lo_W(pair), val_vec, nan_inf_mask);                 \
        vdst[2 * k + 1] = OP(Q6_V_hi_W(pair), val_vec, nan_inf_mask);                 \
    }                                                                                 \
    if (nloe) {                                                                       \
        HVX_VectorPair pair = hvx_vec_f16_to_f32(vsrc0[nvec]);                        \
        HVX_Vector out0 = OP(Q6_V_lo_W(pair), val_vec, nan_inf_mask);                 \
        if (nloe <= 32) {                                                             \
            hvx_vec_store_a(&vdst[2 * nvec], nloe * sizeof(float), out0);             \
        } else {                                                                      \
            vdst[2 * nvec] = out0;                                                    \
            HVX_Vector out1 = OP(Q6_V_hi_W(pair), val_vec, nan_inf_mask);             \
            hvx_vec_store_a(&vdst[2 * nvec + 1], (nloe - 32) * sizeof(float), out1);  \
        }                                                                             \
    }                                                                                 \
}

#define DEFINE_BINARY_MIXED_SCALAR_F32_F16(NAME, OP)                                  \
static void NAME(uint8_t * dst, const uint8_t * src0, float val, uint32_t n) {        \
    const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                        \
    const HVX_Vector val_vec      = hvx_vec_splat_f32(val);                           \
    const HVX_Vector max_f16      = hvx_vec_splat_f32(65504.0f);                      \
    const HVX_Vector min_f16      = hvx_vec_splat_f32(-65504.0f);                     \
    const HVX_Vector inf          = hvx_vec_splat_f32(INFINITY);                      \
    const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;                    \
    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;                           \
    const uint32_t nvec = n / 64;                                                     \
    const uint32_t nloe = n % 64;                                                     \
    _Pragma("unroll(2)")                                                              \
    for (uint32_t k = 0; k < nvec; k++) {                                             \
        HVX_Vector out0 = OP(vsrc0[2 * k + 0], val_vec, nan_inf_mask);                \
        HVX_Vector out1 = OP(vsrc0[2 * k + 1], val_vec, nan_inf_mask);                \
        out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);              \
        out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);              \
        vdst[k] = hvx_vec_f32_to_f16(out0, out1);                                     \
    }                                                                                 \
    if (nloe) {                                                                       \
        HVX_Vector out0 = OP(vsrc0[2 * nvec], val_vec, nan_inf_mask);                 \
        out0 = binary_clamp_f16_finite_vec(out0, max_f16, min_f16, inf);              \
        HVX_Vector out1 = Q6_V_vzero();                                               \
        if (nloe > 32) {                                                              \
            out1 = OP(vsrc0[2 * nvec + 1], val_vec, nan_inf_mask);                    \
            out1 = binary_clamp_f16_finite_vec(out1, max_f16, min_f16, inf);          \
        }                                                                             \
        HVX_Vector packed = hvx_vec_f32_to_f16(out0, out1);                           \
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), packed);                \
    }                                                                                 \
}

#define DEFINE_BINARY_MIXED_SCALAR_F32_F32(NAME, OP)                                  \
static void NAME(uint8_t * dst, const uint8_t * src0, float val, uint32_t n) {        \
    const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                        \
    const HVX_Vector val_vec      = hvx_vec_splat_f32(val);                           \
    const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;                    \
    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;                           \
    const uint32_t nvec = n / 32;                                                     \
    const uint32_t nloe = n % 32;                                                     \
    _Pragma("unroll(2)")                                                              \
    for (uint32_t k = 0; k < nvec; k++) {                                             \
        vdst[k] = OP(vsrc0[k], val_vec, nan_inf_mask);                                \
    }                                                                                 \
    if (nloe) {                                                                       \
        HVX_Vector out = OP(vsrc0[nvec], val_vec, nan_inf_mask);                      \
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(float), out);                      \
    }                                                                                 \
}

#define DEFINE_COMPUTE_MIXED_SCALAR(NAME, SRC0_TYPE, DST_TYPE, FN)                                  \
static void compute_scalar_##NAME(                                                                  \
    uint8_t * dst, const uint8_t * src0, const void * s1_table,                                     \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,                                               \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {                                         \
    const float * table = (const float *) s1_table;                                                 \
    for (uint32_t r = 0; r < n_rows; r++) {                                                         \
        FN(dst + r * dst_stride, src0 + r * src0_stride, table[cur_i11], ne00);                     \
        if (ne11 > 1 && ++cur_i11 == ne11) cur_i11 = 0;                                             \
    }                                                                                               \
}

#define DEFINE_COMPUTE_MIXED_SCALAR_OP(OP_NAME, OP)                                                           \
    DEFINE_BINARY_MIXED_SCALAR_F16_F16(binary_scalar_##OP_NAME##_f16_f16, OP)                                 \
    DEFINE_BINARY_MIXED_SCALAR_F16_F32(binary_scalar_##OP_NAME##_f16_f32, OP)                                 \
    DEFINE_BINARY_MIXED_SCALAR_F32_F16(binary_scalar_##OP_NAME##_f32_f16, OP)                                 \
    DEFINE_BINARY_MIXED_SCALAR_F32_F32(binary_scalar_##OP_NAME##_f32_f32, OP)                                 \
    DEFINE_COMPUTE_MIXED_SCALAR(OP_NAME##_f16_f32_f16, _Float16, _Float16, binary_scalar_##OP_NAME##_f16_f16) \
    DEFINE_COMPUTE_MIXED_SCALAR(OP_NAME##_f16_f32_f32, _Float16, float, binary_scalar_##OP_NAME##_f16_f32)    \
    DEFINE_COMPUTE_MIXED_SCALAR(OP_NAME##_f32_f16_f16, float, _Float16, binary_scalar_##OP_NAME##_f32_f16)    \
    DEFINE_COMPUTE_MIXED_SCALAR(OP_NAME##_f32_f16_f32, float, float, binary_scalar_##OP_NAME##_f32_f32)       \
    DEFINE_COMPUTE_MIXED_SCALAR(OP_NAME##_f16_f16_f32, _Float16, float, binary_scalar_##OP_NAME##_f16_f32)    \
    DEFINE_COMPUTE_MIXED_SCALAR(OP_NAME##_f32_f32_f16, float, _Float16, binary_scalar_##OP_NAME##_f32_f16)

DEFINE_COMPUTE_MIXED_SCALAR_OP(add, BINARY_OP_ADD)
DEFINE_COMPUTE_MIXED_SCALAR_OP(sub, BINARY_OP_SUB)
DEFINE_COMPUTE_MIXED_SCALAR_OP(mul, BINARY_OP_MUL)
DEFINE_COMPUTE_MIXED_SCALAR_OP(div, BINARY_OP_DIV)

typedef void (*compute_add_id_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_data, const int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_ADD_ID_COPY(NAME, STMT)                                           \
static void add_id_copy_##NAME(uint8_t * dst, const uint8_t * src, uint32_t n) { \
    STMT;                                                                        \
}

DEFINE_ADD_ID_COPY(f16_f16, memcpy(dst, src, n * sizeof(_Float16)))
DEFINE_ADD_ID_COPY(f32_f32, memcpy(dst, src, n * sizeof(float)))
DEFINE_ADD_ID_COPY(f16_f32, binary_scalar_add_f16_f32(dst, src, 0.0f, n))
DEFINE_ADD_ID_COPY(f32_f16, binary_scalar_add_f32_f16(dst, src, 0.0f, n))

#define DEFINE_COMPUTE_ADD_ID(NAME, ADD_FN, COPY_FN)                                          \
static void compute_add_id_##NAME(                                                            \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_data, const int32_t * src2_ids, \
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,                          \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                  \
    for (uint32_t r = 0; r < n_rows; r++) {                                                   \
        const uint32_t r_i01 = i01 + r;                                                       \
        const int32_t idx = src2_ids[i02 * ne01 + r_i01];                                     \
        const uint8_t * r_src0 = src0 + r * src0_stride;                                      \
        uint8_t * r_dst = dst + r * dst_stride;                                               \
        if (idx < 0) {                                                                        \
            COPY_FN(r_dst, r_src0, ne00);                                                     \
            continue;                                                                         \
        }                                                                                     \
        const uint8_t * r_src1 = src1_data + idx * src1_stride;                               \
        ADD_FN(r_dst, r_src0, r_src1, ne00);                                                  \
    }                                                                                         \
}

DEFINE_COMPUTE_ADD_ID(f16_f16_f16, binary_add_f16_f16_f16,     add_id_copy_f16_f16)
DEFINE_COMPUTE_ADD_ID(f32_f32_f32, hvx_add_f32_aaa,             add_id_copy_f32_f32)
DEFINE_COMPUTE_ADD_ID(f16_f16_f32, binary_add_f16_f16_f32,     add_id_copy_f16_f32)
DEFINE_COMPUTE_ADD_ID(f16_f32_f16, binary_add_f16_f32_f16,     add_id_copy_f16_f16)
DEFINE_COMPUTE_ADD_ID(f16_f32_f32, binary_add_f16_f32_f32,     add_id_copy_f16_f32)
DEFINE_COMPUTE_ADD_ID(f32_f16_f16, binary_add_f32_f16_f16,     add_id_copy_f32_f16)
DEFINE_COMPUTE_ADD_ID(f32_f16_f32, binary_add_f32_f16_f32,     add_id_copy_f32_f32)
DEFINE_COMPUTE_ADD_ID(f32_f32_f16, binary_add_f32_f32_f16,     add_id_copy_f32_f16)

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
static void binary_thread_scalar(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const size_t plane_stride = (size_t) ne12 * ne11 * bctx->src1_table_elem_size;
    const size_t row_stride   = (size_t) ne11 * bctx->src1_table_elem_size;

    const uint32_t src0_row_size = bctx->src0_row_size;
    const uint32_t dst_row_size  = bctx->dst_row_size;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_hvx * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_hvx, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_hvx);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_hvx);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    const void * s1_table    = VTCM_LAYOUT_PTR(const void, bctx->vtcm_base, layout->off_src1);

    dma_queue * dma_q = th->dma_q;
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, src0_row_size,
                                                             current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = th->trace;
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        const uint8_t * scalar_table = (const uint8_t *) s1_table +
            i13 * plane_stride + i12 * row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_table, i11, ne11, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size,
                                                                current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, src0_row_size,
                                                               next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
static void binary_thread_vector_same_shape(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t src0_row_size = bctx->src0_row_size;
    const uint32_t src1_row_size = bctx->src1_row_size;
    const uint32_t dst_row_size  = bctx->dst_row_size;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_hvx * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_hvx, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_hvx);
    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_hvx);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_hvx);

    size_t src0_spad_half = layout->src0_spad_half_size;
    size_t src1_spad_half = layout->src1_spad_half_size;
    size_t dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = th->dma_q;
    const uint32_t n_srcs = bctx->n_srcs;
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, src0_row_size, current_block_size);
        for (uint32_t s = 1; s < n_srcs; s++) {
            const struct htp_tensor * src_s = octx->src[s];
            uint32_t is3 = (src_s->ne[3] == 1) ? 0 : i03;
            uint32_t is2 = (src_s->ne[2] == 1) ? 0 : i02;
            uint32_t is1 = (src_s->ne[1] == 1) ? 0 : i01;
            dma_addr_t s_curr = src_s->data + is3 * src_s->nb[3] + is2 * src_s->nb[2] + is1 * src_s->nb[1];
            uint8_t * s_spad = s1_spad + (s - 1) * 2 * src1_spad_half;
            dma_queue_push(dma_q, dma_make_data(s_spad, s_curr), bctx->src1_row_size_aligned, src_s->nb[1], src1_row_size, current_block_size);
        }
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = th->trace;
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, bctx->src1_row_size_aligned, ne00);
        for (uint32_t s = 2; s < n_srcs; s++) {
            uint8_t * ss_spad = (uint8_t *) dma_queue_pop(dma_q).dst;
            compute(d_spad, d_spad, ss_spad, current_block_size,
                    bctx->dst_row_size_aligned, bctx->dst_row_size_aligned, bctx->src1_row_size_aligned, ne00);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, src0_row_size, next_block_size);

            for (uint32_t s = 1; s < n_srcs; s++) {
                const struct htp_tensor * src_s = octx->src[s];
                uint32_t ps3 = (src_s->ne[3] == 1) ? 0 : p03;
                uint32_t ps2 = (src_s->ne[2] == 1) ? 0 : p02;
                uint32_t ps1 = (src_s->ne[1] == 1) ? 0 : p01;
                dma_addr_t s_next = src_s->data + ps3 * src_s->nb[3] + ps2 * src_s->nb[2] + ps1 * src_s->nb[1];
                uint8_t * s_spad = s1_spad + (s - 1) * 2 * src1_spad_half;
                dma_queue_push(dma_q, dma_make_data(s_spad, s_next), bctx->src1_row_size_aligned, src_s->nb[1], src1_row_size, next_block_size);
            }

            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
static void binary_thread_vector_row_broadcast(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t src0_row_size = bctx->src0_row_size;
    const uint32_t dst_row_size  = bctx->dst_row_size;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_hvx * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_hvx, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_hvx);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_hvx);

    size_t src0_spad_half = layout->src0_spad_half_size;
    size_t dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = th->dma_q;
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    void * s1_ptr = VTCM_LAYOUT_PTR(void, bctx->vtcm_base, layout->off_src1);

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, src0_row_size, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = th->trace;
    compute_row_bcast_t compute = (compute_row_bcast_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (const uint8_t *)s1_ptr, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, src0_row_size, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
static void binary_thread_vector_complex(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t src0_row_size = bctx->src0_row_size;
    const uint32_t dst_row_size  = bctx->dst_row_size;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_hvx * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_hvx, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_hvx);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_hvx);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    const uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(const uint8_t, bctx->vtcm_base, layout->off_src1);

    const size_t vtcm_row_stride   = (size_t) ne10 * bctx->src1_elem_size;
    const size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    const size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_queue * dma_q = th->dma_q;
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, src0_row_size, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = th->trace;
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        const uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, src0_row_size, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
static void binary_thread_element_repeat(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t src0_row_size = bctx->src0_row_size;
    const uint32_t dst_row_size  = bctx->dst_row_size;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_hvx * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_hvx, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_hvx);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_hvx);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    const uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(const uint8_t, bctx->vtcm_base, layout->off_src1);

    const size_t vtcm_row_stride   = (size_t) ne10 * bctx->src1_elem_size;
    const size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    const size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(HIGH, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    dma_queue * dma_q = th->dma_q;
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, src0_row_size, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = th->trace;
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        const uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, src0_row_size, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
static void binary_thread_add_id(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t nb01 = src0->nb[1];
    const uint32_t nb02 = src0->nb[2];
    const uint32_t nb03 = src0->nb[3];
    const uint32_t src1_stride = bctx->src1_row_size_aligned;

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t src0_row_size = bctx->src0_row_size;
    const uint32_t dst_row_size  = bctx->dst_row_size;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_hvx * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_hvx, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_hvx);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_hvx);
    const uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(const uint8_t, bctx->vtcm_base, layout->off_src1);
    const int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(const int32_t, bctx->vtcm_base, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;

    dma_queue * dma_q = th->dma_q;
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, src0_row_size, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = th->trace;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_add_id_t compute = (compute_add_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, dst_row_size, current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, src0_row_size, next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

static inline void hvx_div_scalar_f32(uint8_t * restrict dst, const uint8_t * restrict src, const float val, const uint32_t num_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, num_elems);
}

static inline void hvx_div_scalar_f16(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, const uint32_t num_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, num_elems);
}

static void binary_expand_f16_to_f32(uint8_t * data, uint32_t n) {
    uint32_t end = n;
    while (end > 0) {
        const uint32_t start = (end - 1) & ~63u;
        const uint32_t count = end - start;
        HVX_VectorPair pair = hvx_vec_f16_to_f32(hvx_vmemu(data + (size_t) start * sizeof(_Float16)));
        hvx_vec_store_u(data + (size_t) start * sizeof(float), MIN(count, 32) * sizeof(float), Q6_V_lo_W(pair));
        if (count > 32) {
            hvx_vec_store_u(data + (size_t) (start + 32) * sizeof(float), (count - 32) * sizeof(float), Q6_V_hi_W(pair));
        }
        end = start;
    }
}

#define SELECT_MIXED_COMPUTE(PREFIX, type_key, out)                                   \
    do {                                                                              \
        switch (type_key) {                                                           \
            case 1:                                                                   \
                switch (octx->op) {                                                   \
                    case HTP_OP_ADD: out = PREFIX##_add_f16_f16_f32; break;           \
                    case HTP_OP_SUB: out = PREFIX##_sub_f16_f16_f32; break;           \
                    case HTP_OP_MUL: out = PREFIX##_mul_f16_f16_f32; break;           \
                    case HTP_OP_DIV: out = PREFIX##_div_f16_f16_f32; break;           \
                    default: break;                                                   \
                }                                                                     \
                break;                                                                \
            case 2:                                                                   \
                switch (octx->op) {                                                   \
                    case HTP_OP_ADD: out = PREFIX##_add_f16_f32_f16; break;           \
                    case HTP_OP_SUB: out = PREFIX##_sub_f16_f32_f16; break;           \
                    case HTP_OP_MUL: out = PREFIX##_mul_f16_f32_f16; break;           \
                    case HTP_OP_DIV: out = PREFIX##_div_f16_f32_f16; break;           \
                    default: break;                                                   \
                }                                                                     \
                break;                                                                \
            case 3:                                                                   \
                switch (octx->op) {                                                   \
                    case HTP_OP_ADD: out = PREFIX##_add_f16_f32_f32; break;           \
                    case HTP_OP_SUB: out = PREFIX##_sub_f16_f32_f32; break;           \
                    case HTP_OP_MUL: out = PREFIX##_mul_f16_f32_f32; break;           \
                    case HTP_OP_DIV: out = PREFIX##_div_f16_f32_f32; break;           \
                    default: break;                                                   \
                }                                                                     \
                break;                                                                \
            case 4:                                                                   \
                switch (octx->op) {                                                   \
                    case HTP_OP_ADD: out = PREFIX##_add_f32_f16_f16; break;           \
                    case HTP_OP_SUB: out = PREFIX##_sub_f32_f16_f16; break;           \
                    case HTP_OP_MUL: out = PREFIX##_mul_f32_f16_f16; break;           \
                    case HTP_OP_DIV: out = PREFIX##_div_f32_f16_f16; break;           \
                    default: break;                                                   \
                }                                                                     \
                break;                                                                \
            case 5:                                                                   \
                switch (octx->op) {                                                   \
                    case HTP_OP_ADD: out = PREFIX##_add_f32_f16_f32; break;           \
                    case HTP_OP_SUB: out = PREFIX##_sub_f32_f16_f32; break;           \
                    case HTP_OP_MUL: out = PREFIX##_mul_f32_f16_f32; break;           \
                    case HTP_OP_DIV: out = PREFIX##_div_f32_f16_f32; break;           \
                    default: break;                                                   \
                }                                                                     \
                break;                                                                \
            case 6:                                                                   \
                switch (octx->op) {                                                   \
                    case HTP_OP_ADD: out = PREFIX##_add_f32_f32_f16; break;           \
                    case HTP_OP_SUB: out = PREFIX##_sub_f32_f32_f16; break;           \
                    case HTP_OP_MUL: out = PREFIX##_mul_f32_f32_f16; break;           \
                    case HTP_OP_DIV: out = PREFIX##_div_f32_f32_f16; break;           \
                    default: break;                                                   \
                }                                                                     \
                break;                                                                \
            default: break;                                                           \
        }                                                                             \
    } while (0)

typedef void (*compute_binary_chunked_t)(
    uint8_t * restrict dst,
    const uint8_t * restrict src0,
    const uint8_t * restrict src1,
    const uint32_t num_elems
);

typedef void (*compute_binary_scalar_chunked_f32_t)(
    uint8_t * restrict dst,
    const uint8_t * restrict src0,
    const float val,
    const uint32_t num_elems
);

typedef void (*compute_binary_scalar_chunked_f16_t)(
    uint8_t * restrict dst,
    const uint8_t * restrict src0,
    const _Float16 val,
    const uint32_t num_elems
);

struct binary_chunked_context {
    struct htp_ops_context * octx;
    struct htp_binary_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_base;
    uint32_t                 elem_start;
    uint32_t                 nelem;
    uint32_t                 chunk_size;
    uint32_t                 chunks_per_hvx;
    uint32_t                 total_chunks;
    uint32_t                 n_srcs;
    uint32_t                 src0_elem_size;
    uint32_t                 src1_elem_size;
    uint32_t                 dst_elem_size;
    bool                     is_scalar;
    float                    scalar_f32;
    _Float16                 scalar_f16;
    compute_binary_chunked_t compute;
    compute_binary_scalar_chunked_f32_t compute_scalar_f32;
    compute_binary_scalar_chunked_f16_t compute_scalar_f16;
};

static void binary_thread_chunked(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct binary_chunked_context * ctx = (struct binary_chunked_context *) data;
    struct htp_ops_context * octx = ctx->octx;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t start_chunk = ctx->chunks_per_hvx * ith;
    const uint32_t end_chunk   = MIN(start_chunk + ctx->chunks_per_hvx, ctx->total_chunks);
    if (start_chunk >= end_chunk) {
        return;
    }

    const uint32_t chunk_size  = ctx->chunk_size;
    const size_t src0_chunk_bytes = ctx->vtcm_layout.src0_spad_half_size;
    const size_t src1_chunk_bytes = ctx->vtcm_layout.src1_spad_half_size;
    const size_t dst_chunk_bytes  = ctx->vtcm_layout.dst_spad_half_size;

    FARF(HIGH, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, start_chunk, end_chunk, ctx->total_chunks, ctx->nelem);

    const struct htp_binary_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_hvx);
    uint8_t * src1_spad_base = ctx->is_scalar ? NULL : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_hvx));
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_hvx);

    dma_queue * dma_q = th->dma_q;
    const uint32_t n_srcs = ctx->n_srcs;
    uint32_t prefetch_chunk = start_chunk;
    int spad_idx = 0;

    for (int k = 0; k < 2 && prefetch_chunk < end_chunk; k++) {
        const uint32_t c_start   = ctx->elem_start + prefetch_chunk * chunk_size;
        const uint32_t c_end     = MIN(c_start + chunk_size, ctx->elem_start + ctx->nelem);
        const uint32_t cur_elems = c_end - c_start;
        const uint32_t src0_bytes = cur_elems * ctx->src0_elem_size;
        const uint32_t src1_bytes = cur_elems * ctx->src1_elem_size;
        const uint32_t dst_bytes  = cur_elems * ctx->dst_elem_size;

        dma_addr_t s0_curr = src0->data + (size_t) c_start * ctx->src0_elem_size;
        dma_addr_t d_curr  = dst->data  + (size_t) c_start * ctx->dst_elem_size;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_chunk_bytes;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_chunk_bytes;

        dma_queue_push(dma_q, dma_make_data(d_curr, d_spad), dst_chunk_bytes, dst_chunk_bytes, dst_bytes, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, s0_curr), src0_chunk_bytes, src0_chunk_bytes, src0_bytes, 1);
        if (!ctx->is_scalar) {
            uint8_t * s1_spad = src1_spad_base + spad_idx * src1_chunk_bytes;
            for (uint32_t s = 1; s < n_srcs; s++) {
                dma_addr_t s_curr = octx->src[s]->data + (size_t) c_start * ctx->src1_elem_size;
                uint8_t * s_spad  = s1_spad + (s - 1) * 2 * src1_chunk_bytes;
                dma_queue_push(dma_q, dma_make_data(s_spad, s_curr), src1_chunk_bytes, src1_chunk_bytes, src1_bytes, 1);
            }
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = th->trace;

    for (uint32_t c = start_chunk; c < end_chunk; c++) {
        const uint32_t c_start   = ctx->elem_start + c * chunk_size;
        const uint32_t c_end     = MIN(c_start + chunk_size, ctx->elem_start + ctx->nelem);
        const uint32_t cur_elems = c_end - c_start;
        const uint32_t dst_bytes = cur_elems * ctx->dst_elem_size;

        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? NULL : (uint8_t *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        if (ctx->is_scalar) {
            if (ctx->compute_scalar_f32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } else {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } else {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
            for (uint32_t s = 2; s < n_srcs; s++) {
                uint8_t * s_spad = (uint8_t *) dma_queue_pop(dma_q).dst;
                ctx->compute(d_spad, d_spad, s_spad, cur_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->data + (size_t) c_start * ctx->dst_elem_size;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), dst_chunk_bytes, dst_chunk_bytes, dst_bytes, 1);

        if (prefetch_chunk < end_chunk) {
            const uint32_t pc_start  = ctx->elem_start + prefetch_chunk * chunk_size;
            const uint32_t pc_end    = MIN(pc_start + chunk_size, ctx->elem_start + ctx->nelem);
            const uint32_t p_elems   = pc_end - pc_start;
            const uint32_t p_src0_bytes = p_elems * ctx->src0_elem_size;
            const uint32_t p_src1_bytes = p_elems * ctx->src1_elem_size;

            dma_addr_t s0_next = src0->data + (size_t) pc_start * ctx->src0_elem_size;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), src0_chunk_bytes, src0_chunk_bytes, p_src0_bytes, 1);
            if (!ctx->is_scalar) {
                for (uint32_t s = 1; s < n_srcs; s++) {
                    dma_addr_t s_next = octx->src[s]->data + (size_t) pc_start * ctx->src1_elem_size;
                    uint8_t * s_spad  = s1_spad + (s - 1) * 2 * src1_chunk_bytes;
                    dma_queue_push(dma_q, dma_make_data(s_spad, s_next), src1_chunk_bytes, src1_chunk_bytes, p_src1_bytes, 1);
                }
            }

            prefetch_chunk++;
        }
    }

    dma_queue_flush(dma_q);
}

static int execute_op_binary(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;
    const struct htp_binary_kernel_params * kparams = (const struct htp_binary_kernel_params *) octx->kernel_params;
    const uint32_t n_srcs = kparams->n_srcs > 2 ? kparams->n_srcs : 2;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    const uint32_t src0_type = src0->type;
    const uint32_t src1_type = src1->type;
    const uint32_t dst_type  = dst->type;
    const uint32_t src0_elem_size = kparams->src0_elem_size;
    const uint32_t src1_elem_size = kparams->src1_elem_size;
    const uint32_t dst_elem_size  = kparams->dst_elem_size;
    const size_t src0_row_size = src0->ne[0] * src0_elem_size;
    const size_t src1_row_size = src1->ne[0] * src1_elem_size;
    const size_t dst_row_size  = dst->ne[0]  * dst_elem_size;
    const uint32_t type_key = (src0_type == HTP_TYPE_F32 ? 4 : 0) |
                              (src1_type == HTP_TYPE_F32 ? 2 : 0) |
                              (dst_type  == HTP_TYPE_F32 ? 1 : 0);
    const uint32_t table_type_key = type_key == 0 ? 0 : (type_key | 2);

    if (kparams->kernel_type == HTP_BINARY_KERNEL_CHUNKED) {
        const bool is_scalar = kparams->is_scalar;

        compute_binary_chunked_t compute = NULL;
        compute_binary_scalar_chunked_f32_t compute_scalar_f32 = NULL;
        compute_binary_scalar_chunked_f16_t compute_scalar_f16 = NULL;

        if (is_scalar) {
            if (type_key == 7) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_scalar_f32 = hvx_add_scalar_f32; break;
                    case HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; break;
                    case HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; break;
                    case HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; break;
                    default: break;
                }
            } else if (type_key == 0) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_scalar_f16 = hvx_add_scalar_f16; break;
                    case HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; break;
                    case HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; break;
                    case HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; break;
                    default: break;
                }
            } else if (src0_type == HTP_TYPE_F16 && dst_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_scalar_f32 = binary_scalar_add_f16_f16; break;
                    case HTP_OP_SUB: compute_scalar_f32 = binary_scalar_sub_f16_f16; break;
                    case HTP_OP_MUL: compute_scalar_f32 = binary_scalar_mul_f16_f16; break;
                    case HTP_OP_DIV: compute_scalar_f32 = binary_scalar_div_f16_f16; break;
                    default: break;
                }
            } else if (src0_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_scalar_f32 = binary_scalar_add_f16_f32; break;
                    case HTP_OP_SUB: compute_scalar_f32 = binary_scalar_sub_f16_f32; break;
                    case HTP_OP_MUL: compute_scalar_f32 = binary_scalar_mul_f16_f32; break;
                    case HTP_OP_DIV: compute_scalar_f32 = binary_scalar_div_f16_f32; break;
                    default: break;
                }
            } else if (dst_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_scalar_f32 = binary_scalar_add_f32_f16; break;
                    case HTP_OP_SUB: compute_scalar_f32 = binary_scalar_sub_f32_f16; break;
                    case HTP_OP_MUL: compute_scalar_f32 = binary_scalar_mul_f32_f16; break;
                    case HTP_OP_DIV: compute_scalar_f32 = binary_scalar_div_f32_f16; break;
                    default: break;
                }
            } else {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_scalar_f32 = binary_scalar_add_f32_f32; break;
                    case HTP_OP_SUB: compute_scalar_f32 = binary_scalar_sub_f32_f32; break;
                    case HTP_OP_MUL: compute_scalar_f32 = binary_scalar_mul_f32_f32; break;
                    case HTP_OP_DIV: compute_scalar_f32 = binary_scalar_div_f32_f32; break;
                    default: break;
                }
            }
            if (!compute_scalar_f32 && !compute_scalar_f16) {
                return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            if (type_key == 7) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute = hvx_add_f32; break;
                    case HTP_OP_SUB: compute = hvx_sub_f32; break;
                    case HTP_OP_MUL: compute = hvx_mul_f32; break;
                    case HTP_OP_DIV: compute = hvx_div_f32; break;
                    default: break;
                }
            } else if (type_key == 0) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute = hvx_add_f16; break;
                    case HTP_OP_SUB: compute = hvx_sub_f16; break;
                    case HTP_OP_MUL: compute = hvx_mul_f16; break;
                    case HTP_OP_DIV: compute = hvx_div_f16; break;
                    default: break;
                }
            } else {
                SELECT_MIXED_COMPUTE(binary, type_key, compute);
            }
            if (!compute) {
                return HTP_STATUS_NO_SUPPORT;
            }
        }

        const uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        if (total_elems == 0) {
            return HTP_STATUS_OK;
        }

        uint32_t elem_start = 0;
        uint32_t nelem      = total_elems;
        const uint32_t elems_per_line = 128 / dst_elem_size;

        if (octx->ctx->mdev.count > 1) {
            bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                             htp_tensor_mdev_data_aligned(src0) &&
                             (is_scalar || htp_tensor_mdev_data_aligned(src1)) &&
                             htp_tensor_is_contiguous(dst, dst_elem_size) &&
                             htp_tensor_is_contiguous(src0, src0_elem_size) &&
                             (is_scalar || htp_tensor_is_contiguous(src1, src1_elem_size));
            for (uint32_t s = 2; can_split && s < n_srcs; s++) {
                can_split = htp_tensor_mdev_data_aligned(octx->src[s]) &&
                            htp_tensor_is_contiguous(octx->src[s], src1_elem_size);
            }
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
                total_elems, can_split ? elems_per_line : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            elem_start = range.start;
            nelem      = range.count;
        }

        if (nelem == 0) {
            return HTP_STATUS_OK;
        }

        if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
            return HTP_STATUS_INVAL_PARAMS;
        }

        struct htp_binary_vtcm_layout vtcm_layout;
        htp_binary_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_size);
        if (vtcm_layout.total_bytes == 0 || vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        const uint32_t chunk_size = kparams->chunk_size;
        const uint32_t total_chunks = (nelem + chunk_size - 1) / chunk_size;
        const uint32_t n_hvx = (total_chunks >= 2) ? MIN(octx->n_hvx, total_chunks) : 1;
        const uint32_t chunks_per_hvx = (total_chunks + n_hvx - 1) / n_hvx;

        float scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        if (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, vtcm_layout.off_src1);
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_push(dma_q, dma_make_data(vtcm_src1, src1->data), 128, 0, src1_elem_size, 1);
            dma_queue_pop(dma_q);

            if (type_key != 0 && src1_type == HTP_TYPE_F16) {
                binary_expand_f16_to_f32(vtcm_src1, 1);
            }
            if (type_key != 0) {
                scalar_f32 = ((const float *) vtcm_src1)[0];
            } else {
                scalar_f16 = ((const _Float16 *) vtcm_src1)[0];
            }
        }

        struct binary_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_base           = (uint8_t *) octx->ctx->vtcm_base,
            .elem_start          = elem_start,
            .nelem               = nelem,
            .chunk_size          = chunk_size,
            .chunks_per_hvx      = chunks_per_hvx,
            .total_chunks        = total_chunks,
            .n_srcs              = n_srcs,
            .src0_elem_size      = src0_elem_size,
            .src1_elem_size      = src1_elem_size,
            .dst_elem_size       = dst_elem_size,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_queue_run(octx->ctx->hvx_queue, binary_thread_chunked, &cctx, n_hvx);
        return HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, dst_elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx = octx->n_hvx;
    const size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    const size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    const size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    struct htp_binary_context bctx;
    bctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    htp_binary_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_size);

    if (bctx.vtcm_layout.rows_per_buffer == 0 || bctx.vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    work_queue_func_t worker_func = NULL;
    void * compute_func = NULL;

    switch (kparams->kernel_type) {
        case HTP_BINARY_KERNEL_SAME_SHAPE:
            worker_func = binary_thread_vector_same_shape;
            if (type_key == 7) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_same_shape_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_same_shape_div_f32; break;
                    default: break;
                }
            } else if (type_key == 0) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_same_shape_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_same_shape_div_f16; break;
                    default: break;
                }
            } else {
                SELECT_MIXED_COMPUTE(compute_same_shape, type_key, compute_func);
            }
            break;
        case HTP_BINARY_KERNEL_ROW_BCAST:
            worker_func = binary_thread_vector_row_broadcast;
            if (table_type_key == 7) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_row_bcast_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_row_bcast_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_row_bcast_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_row_bcast_div_f32; break;
                    default: break;
                }
            } else if (type_key == 0) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_row_bcast_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_row_bcast_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_row_bcast_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_row_bcast_div_f16; break;
                    default: break;
                }
            } else {
                SELECT_MIXED_COMPUTE(compute_row_bcast, table_type_key, compute_func);
            }
            break;
        case HTP_BINARY_KERNEL_SCALAR:
            worker_func = binary_thread_scalar;
            if (table_type_key == 7) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_scalar_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_scalar_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_scalar_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_scalar_div_f32; break;
                    default: break;
                }
            } else if (type_key == 0) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_scalar_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_scalar_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_scalar_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_scalar_div_f16; break;
                    default: break;
                }
            } else {
                SELECT_MIXED_COMPUTE(compute_scalar, table_type_key, compute_func);
            }
            break;
        case HTP_BINARY_KERNEL_COMPLEX:
            worker_func = binary_thread_vector_complex;
            if (type_key == 7) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_complex_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_complex_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_complex_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_complex_div_f32; break;
                    default: break;
                }
            } else if (type_key == 0) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_complex_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_complex_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_complex_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_complex_div_f16; break;
                    default: break;
                }
            } else {
                SELECT_MIXED_COMPUTE(compute_complex, type_key, compute_func);
            }
            break;
        case HTP_BINARY_KERNEL_REPEAT:
            worker_func = binary_thread_element_repeat;
            if (type_key == 7) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_repeat_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_repeat_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_repeat_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_repeat_div_f32; break;
                    default: break;
                }
            } else if (type_key == 0) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_repeat_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_repeat_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_repeat_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_repeat_div_f16; break;
                    default: break;
                }
            } else {
                SELECT_MIXED_COMPUTE(compute_repeat, type_key, compute_func);
            }
            break;
        case HTP_BINARY_KERNEL_ADD_ID:
            if (octx->op == HTP_OP_ADD_ID) {
                worker_func = binary_thread_add_id;
                switch (type_key) {
                    case 0: compute_func = (void *) compute_add_id_f16_f16_f16; break;
                    case 1: compute_func = (void *) compute_add_id_f16_f16_f32; break;
                    case 2: compute_func = (void *) compute_add_id_f16_f32_f16; break;
                    case 3: compute_func = (void *) compute_add_id_f16_f32_f32; break;
                    case 4: compute_func = (void *) compute_add_id_f32_f16_f16; break;
                    case 5: compute_func = (void *) compute_add_id_f32_f16_f32; break;
                    case 6: compute_func = (void *) compute_add_id_f32_f32_f16; break;
                    case 7: compute_func = (void *) compute_add_id_f32_f32_f32; break;
                    default: break;
                }
            }
            break;
        default: break;
    }

    if (!worker_func || !compute_func) {
        return HTP_STATUS_NO_SUPPORT;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src1);
    if (kparams->kernel_type == HTP_BINARY_KERNEL_ROW_BCAST) {
        if (!kparams->src1_strided) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1->ne[0] * src1_elem_size, 1);
        } else {
            dma_queue_push(dma_q, dma_make_data(vtcm_src1, src1->data), src1_elem_size, src1->nb[0], src1_elem_size, src1->ne[0]);
        }
    } else if (kparams->kernel_type == HTP_BINARY_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_BINARY_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_BINARY_KERNEL_COMPLEX) {
        if (!kparams->src1_strided) {
            const size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1_elem_size;
            dma_queue_push(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1_bytes, 1);
            if (kparams->kernel_type == HTP_BINARY_KERNEL_SCALAR && octx->src[2] != NULL) {
                uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src2);
                dma_queue_push(dma_q, dma_make_data(vtcm_src2, octx->src[2]->data), bctx.vtcm_layout.src2_size, 0, src1_bytes, 1);
            }
        } else {
            const uint32_t ne11 = src1->ne[1];
            const uint32_t ne12 = src1->ne[2];
            const uint32_t ne13 = src1->ne[3];
            const size_t row_bytes = src1->ne[0] * src1_elem_size;
            if (src1->nb[0] == src1_elem_size) {
                for (uint32_t i13 = 0; i13 < ne13; i13++) {
                    for (uint32_t i12 = 0; i12 < ne12; i12++) {
                        dma_addr_t src_addr = src1->data + i13 * src1->nb[3] + i12 * src1->nb[2];
                        uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * row_bytes;
                        while (!dma_queue_push(dma_q, dma_make_data(dst_ptr, src_addr),
                                               row_bytes, src1->nb[1], row_bytes, ne11)) {
                            dma_queue_pop(dma_q);
                        }
                    }
                }
            } else {
                for (uint32_t i13 = 0; i13 < ne13; i13++) {
                    for (uint32_t i12 = 0; i12 < ne12; i12++) {
                        for (uint32_t i11 = 0; i11 < ne11; i11++) {
                            dma_addr_t src_addr = src1->data + i13 * src1->nb[3] + i12 * src1->nb[2] + i11 * src1->nb[1];
                            uint8_t * dst_ptr = vtcm_src1 + ((i13 * ne12 * ne11 + i12 * ne11 + i11) * src1->ne[0]) * src1_elem_size;
                            while (!dma_queue_push(dma_q, dma_make_data(dst_ptr, src_addr),
                                                   src1_elem_size, src1->nb[0], src1_elem_size, src1->ne[0])) {
                                dma_queue_pop(dma_q);
                            }
                        }
                    }
                }
            }
        }
    } else if (kparams->kernel_type == HTP_BINARY_KERNEL_ADD_ID) {
        const struct htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src2);
        dma_queue_push(dma_q, dma_make_data(vtcm_src1, src1->data),
                       kparams->src1_row_size_aligned, src1->nb[1],
                       src1->ne[0] * src1_elem_size, src1->ne[1]);
        const size_t src2_row_bytes = src2->ne[0] * sizeof(int32_t);
        dma_queue_push(dma_q, dma_make_data(vtcm_src2, src2->data),
                       src2_row_bytes, src2->nb[1],
                       src2_row_bytes, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_per_hvx         = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);
    bctx.total_rows            = nrows;
    bctx.row_start             = row_start;
    bctx.n_srcs                = n_srcs;
    bctx.block_max             = bctx.vtcm_layout.rows_per_buffer;
    bctx.src0_row_size_aligned = src0_row_size_aligned;
    bctx.src1_row_size_aligned = src1_row_size_aligned;
    bctx.dst_row_size_aligned  = dst_row_size_aligned;
    bctx.src0_row_size         = src0_row_size;
    bctx.src1_row_size         = src1_row_size;
    bctx.dst_row_size          = dst_row_size;
    bctx.src0_elem_size        = src0_elem_size;
    bctx.src1_elem_size        = src1_elem_size;
    bctx.dst_elem_size         = dst_elem_size;
    bctx.src1_table_elem_size  = table_type_key == 0 ? sizeof(_Float16) : sizeof(float);

    bctx.src0_dim1_div  = kparams->div_ne01;
    bctx.src0_dim2_div  = kparams->div_ne02;
    bctx.src0_dim12_div = kparams->div_ne012;

    bctx.src1_dim1_div  = kparams->div_ne11;
    bctx.src1_dim2_div  = kparams->div_ne12;
    bctx.src1_dim3_div  = kparams->div_ne13;

    bctx.split_at_ne01  = kparams->split_at_ne01;
    bctx.split_at_ne02  = kparams->split_at_ne02;
    bctx.compute        = compute_func;

    if (kparams->kernel_type == HTP_BINARY_KERNEL_ROW_BCAST ||
        kparams->kernel_type == HTP_BINARY_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_BINARY_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_BINARY_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_BINARY_KERNEL_ADD_ID) {
        dma_queue_flush(dma_q);
        if ((kparams->kernel_type == HTP_BINARY_KERNEL_ROW_BCAST || kparams->kernel_type == HTP_BINARY_KERNEL_SCALAR) &&
            type_key != 0 && src1_type == HTP_TYPE_F16) {
            binary_expand_f16_to_f32(vtcm_src1, (uint32_t) (src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3]));
        }
        if (kparams->kernel_type == HTP_BINARY_KERNEL_SCALAR && octx->src[2] != NULL && octx->op == HTP_OP_MUL) {
            uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src2);
            const uint32_t n_scalars = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
            if (src0_type == HTP_TYPE_F32) {
                hvx_mul_f32_aaa(vtcm_src1, vtcm_src1, vtcm_src2, n_scalars);
            } else if (src0_type == HTP_TYPE_F16) {
                hvx_mul_f16_aaa(vtcm_src1, vtcm_src1, vtcm_src2, n_scalars);
            }
        }
    }

    work_queue_run(octx->ctx->hvx_queue, worker_func, &bctx, n_hvx);

    return HTP_STATUS_OK;
}

int op_binary(struct htp_ops_context * octx) {
    const uint32_t src0_type = octx->src[0]->type;
    if ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        return execute_op_binary(octx);
    }

    return HTP_STATUS_NO_SUPPORT;
}
