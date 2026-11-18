#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "work-queue.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-trace.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "htp/softmax-ops.h"
#include "hvx-flash-attn.h"

struct htp_softmax_context {
    struct htp_ops_context * octx;
    const struct htp_softmax_kernel_params * kparams;

    void *   prep;
    void *   softmax;

    dma_addr_t data_src0;
    dma_addr_t data_src1;
    dma_addr_t data_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;
    uint8_t * vtcm_work;
    float *   vtcm_src2;

    uint32_t  vtcm_src0_size_per_hvx;
    uint32_t  vtcm_src1_size_per_hvx;
    uint32_t  vtcm_dst_size_per_hvx;
    uint32_t  vtcm_work_size_per_hvx;

    uint32_t  src0_spad_half_size;
    uint32_t  src1_spad_half_size;
    uint32_t  dst_spad_half_size;

    uint32_t  src0_row_size_aligned;
    uint32_t  src1_row_size_aligned;
    uint32_t  dst_row_size_aligned;

    bool     src0_f16;
    bool     dst_f16;
    bool     use_f16;
    bool     use_src2;

    uint32_t n_head;
    uint32_t n_head_log2;

    float    scale;
    float    max_bias;
    float    m0;
    float    m1;

    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne12;
    struct fastdiv_values div_ne13;

    uint32_t src0_nrows_per_hvx;
    uint32_t row_start;
    uint32_t nrows;

    float    slopes[512] __attribute__((aligned(128)));
};

typedef void (*softmax_prep_fn_t)(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
);

static __attribute__((noinline)) void hvx_fast_softmax_prep_nomask(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    (void) mask;
    (void) slope;
    hvx_scale_f32((uint8_t *) dst, (const uint8_t *) src0, ne00, scale);
}

static __attribute__((noinline)) void hvx_fast_softmax_prep_src16_nomask(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    (void) mask;
    (void) slope;
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src0;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);

    const int nvec_64 = (int) ne00 / VLEN_FP16;
    const int nloe_64 = (int) ne00 % VLEN_FP16;

    #pragma unroll(2)
    for (int i = 0; i < nvec_64; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector s0 = Q6_V_lo_W(p);
        HVX_Vector s1 = Q6_V_hi_W(p);

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec));
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec));
    }

    if (nloe_64 > 0) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[nvec_64]);
        HVX_Vector s0 = Q6_V_lo_W(p);
        HVX_Vector v0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec));

        if (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * sizeof(float), v0);
        } else {
            v_dst[2 * nvec_64] = v0;

            HVX_Vector s1 = Q6_V_hi_W(p);
            HVX_Vector v1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * sizeof(float), v1);
        }
    }
}

static __attribute__((noinline)) void hvx_fast_softmax_prep_f16(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    const HVX_Vector * restrict v_src  = (const HVX_Vector *) src0;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    const HVX_Vector * restrict v_mask = (const HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    const int nvec_64 = (int) ne00 / VLEN_FP16;
    const int nloe_64 = (int) ne00 % VLEN_FP16;

    #pragma unroll(2)
    for (int i = 0; i < nvec_64; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[i]);
        HVX_Vector m0 = Q6_V_lo_W(p);
        HVX_Vector m1 = Q6_V_hi_W(p);

        HVX_Vector s0 = v_src[2 * i];
        HVX_Vector s1 = v_src[2 * i + 1];

        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));
        HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    if (nloe_64 > 0) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[nvec_64]);
        HVX_Vector m0 = Q6_V_lo_W(p);

        HVX_Vector s0 = v_src[2 * nvec_64];
        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        if (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * sizeof(float), Q6_Vsf_equals_Vqf32(v0));
        } else {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector m1 = Q6_V_hi_W(p);
            HVX_Vector s1 = v_src[2 * nvec_64 + 1];
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * sizeof(float), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

static __attribute__((noinline)) void hvx_fast_softmax_prep_f32(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    const HVX_Vector * restrict v_src  = (const HVX_Vector *) src0;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    const HVX_Vector * restrict v_mask = (const HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    const int nvec = (int) ne00 / VLEN_FP32;
    const int nloe = (int) ne00 % VLEN_FP32;

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v3 = v_mask[i];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    if (nloe > 0) {
        HVX_Vector v1 = v_src[nvec];
        HVX_Vector v3 = v_mask[nvec];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), Q6_Vsf_equals_Vqf32(v5));
    }
}

static __attribute__((noinline)) void hvx_fast_softmax_prep_src16_mask_f16(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    const HVX_Vector * restrict v_src  = (const HVX_Vector *) src0;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    const HVX_Vector * restrict v_mask = (const HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    const int nvec_64 = (int) ne00 / VLEN_FP16;
    const int nloe_64 = (int) ne00 % VLEN_FP16;

    #pragma unroll(2)
    for (int i = 0; i < nvec_64; i++) {
        HVX_VectorPair ps = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector s0 = Q6_V_lo_W(ps);
        HVX_Vector s1 = Q6_V_hi_W(ps);

        HVX_VectorPair pm = hvx_vec_f16_to_f32(v_mask[i]);
        HVX_Vector m0 = Q6_V_lo_W(pm);
        HVX_Vector m1 = Q6_V_hi_W(pm);

        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));
        HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    if (nloe_64 > 0) {
        HVX_VectorPair ps = hvx_vec_f16_to_f32(v_src[nvec_64]);
        HVX_Vector s0 = Q6_V_lo_W(ps);

        HVX_VectorPair pm = hvx_vec_f16_to_f32(v_mask[nvec_64]);
        HVX_Vector m0 = Q6_V_lo_W(pm);

        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        if (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * sizeof(float), Q6_Vsf_equals_Vqf32(v0));
        } else {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector s1 = Q6_V_hi_W(ps);
            HVX_Vector m1 = Q6_V_hi_W(pm);
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * sizeof(float), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

static __attribute__((noinline)) void hvx_fast_softmax_prep_src16_mask_f32(
    void * restrict dst,
    const void * restrict src0,
    const void * restrict mask,
    uint32_t ne00,
    float scale,
    float slope
) {
    const HVX_Vector * restrict v_src  = (const HVX_Vector *) src0;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    const HVX_Vector * restrict v_mask = (const HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    const int nvec_64 = (int) ne00 / VLEN_FP16;
    const int nloe_64 = (int) ne00 % VLEN_FP16;

    #pragma unroll(2)
    for (int i = 0; i < nvec_64; i++) {
        HVX_VectorPair ps = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector s0 = Q6_V_lo_W(ps);
        HVX_Vector s1 = Q6_V_hi_W(ps);

        HVX_Vector m0 = v_mask[2 * i];
        HVX_Vector m1 = v_mask[2 * i + 1];

        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));
        HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    if (nloe_64 > 0) {
        HVX_VectorPair ps = hvx_vec_f16_to_f32(v_src[nvec_64]);
        HVX_Vector s0 = Q6_V_lo_W(ps);
        HVX_Vector m0 = v_mask[2 * nvec_64];
        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        if (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * sizeof(float), Q6_Vsf_equals_Vqf32(v0));
        } else {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector s1 = Q6_V_hi_W(ps);
            HVX_Vector m1 = v_mask[2 * nvec_64 + 1];
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * sizeof(float), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

typedef void (*softmax_fn_t)(
    uint8_t * restrict work,
    uint8_t * restrict dst,
    const int num_elems,
    const float * restrict sink_ptr
);

#define DEFINE_HVX_FAST_SOFTMAX(SFX, HAS_SINK, DST_F16)                                                       \
static __attribute__((noinline)) void hvx_fast_softmax_##SFX(                                                 \
    uint8_t * restrict work,                                                                                  \
    uint8_t * restrict dst,                                                                                   \
    const int num_elems,                                                                                      \
    const float * restrict sink_ptr                                                                           \
) {                                                                                                           \
    HVX_Vector * restrict v_work = (HVX_Vector *) work;                                                       \
    HVX_Vector * restrict v_dst  = (HVX_Vector *) dst;                                                        \
    (void) sink_ptr;                                                                                          \
    (void) v_dst;                                                                                             \
                                                                                                              \
    const int nvec = num_elems / VLEN_FP32;                                                                   \
    const int nloe = num_elems % VLEN_FP32;                                                                   \
                                                                                                              \
    HVX_Vector max_vec = hvx_vec_repl4(v_work[0]);                                                            \
                                                                                                              \
    _Pragma("unroll(2)")                                                                                      \
    for (int i = 0; i < nvec; i++) {                                                                          \
        HVX_Vector v1 = v_work[i];                                                                            \
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);                                                      \
    }                                                                                                         \
                                                                                                              \
    if (nloe > 0) {                                                                                           \
        HVX_VectorPred q_mask  = Q6_Q_vsetq_R(nloe * sizeof(float));                                          \
        HVX_Vector     neg_inf = hvx_vec_splat_f32(-INFINITY);                                                \
        HVX_Vector     v_tail  = Q6_V_vmux_QVV(q_mask, v_work[nvec], neg_inf);                                \
        max_vec                = Q6_Vsf_vmax_VsfVsf(max_vec, v_tail);                                         \
    }                                                                                                         \
                                                                                                              \
    max_vec = hvx_vec_reduce_max_f32(max_vec);                                                                \
    HVX_Vector sink_vec;                                                                                      \
    if (HAS_SINK) {                                                                                           \
        sink_vec = hvx_vec_repl4(hvx_vmemu(sink_ptr));                                                        \
        max_vec  = Q6_Vsf_vmax_VsfVsf(max_vec, sink_vec);                                                     \
    }                                                                                                         \
                                                                                                              \
    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);                                                           \
                                                                                                              \
    _Pragma("nounroll")                                                                                       \
    _Pragma("clang loop pipeline(disable)")                                                                   \
    for (int i = 0; i < nvec; i++) {                                                                          \
        HVX_Vector v1 = v_work[i];                                                                            \
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);                                                    \
        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));                                             \
                                                                                                              \
        sum_vec   = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);                                   \
        v_work[i] = v3;                                                                                       \
    }                                                                                                         \
                                                                                                              \
    if (nloe > 0) {                                                                                           \
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * sizeof(float));                                           \
        HVX_Vector     v1     = v_work[nvec];                                                                 \
        HVX_Vector     v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);                                            \
        HVX_Vector     v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));                                     \
        HVX_Vector     v3_pad = Q6_V_vmux_QVV(q_mask, v3, Q6_V_vzero());                                      \
                                                                                                              \
        sum_vec      = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3_pad);                            \
        v_work[nvec] = v3_pad;                                                                                \
    }                                                                                                         \
                                                                                                              \
    sum_vec = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_vec));                                           \
    if (HAS_SINK) {                                                                                           \
        HVX_Vector sink_sub = Q6_Vqf32_vsub_VsfVsf(sink_vec, max_vec);                                        \
        HVX_Vector sink_exp = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(sink_sub));                                 \
        sum_vec             = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum_vec, sink_exp));                   \
    }                                                                                                         \
                                                                                                              \
    HVX_VectorPred pos_sum   = Q6_Q_vcmp_gt_VwVw(sum_vec, Q6_V_vzero());                                      \
    HVX_Vector     v4        = hvx_vec_inverse_f32(sum_vec);                                                  \
    HVX_Vector     scale_vec = Q6_V_vmux_QVV(pos_sum, v4, hvx_vec_splat_f32(1.0f));                           \
                                                                                                              \
    if (DST_F16) {                                                                                            \
        const int nvec_64 = num_elems / VLEN_FP16;                                                            \
        const int nloe_64 = num_elems % VLEN_FP16;                                                            \
                                                                                                              \
        _Pragma("unroll(2)")                                                                                  \
        for (int i = 0; i < nvec_64; i++) {                                                                   \
            HVX_Vector s0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_work[2 * i],     scale_vec));          \
            HVX_Vector s1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_work[2 * i + 1], scale_vec));          \
            v_dst[i]      = hvx_vec_f32_to_f16(s0, s1);                                                       \
        }                                                                                                     \
                                                                                                              \
        if (nloe_64 > 0) {                                                                                    \
            HVX_Vector s0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_work[2 * nvec_64], scale_vec));        \
            HVX_Vector s1 = (nloe_64 > VLEN_FP32)                                                             \
                ? Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_work[2 * nvec_64 + 1], scale_vec))               \
                : Q6_V_vzero();                                                                               \
            HVX_Vector d  = hvx_vec_f32_to_f16(s0, s1);                                                       \
            hvx_vec_store_a(&v_dst[nvec_64], nloe_64 * sizeof(__fp16), d);                                    \
        }                                                                                                     \
    } else {                                                                                                  \
        _Pragma("unroll(2)")                                                                                  \
        for (int i = 0; i < nvec; i++) {                                                                      \
            HVX_Vector v1 = v_work[i];                                                                        \
            HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);                                              \
            v_work[i]     = Q6_Vsf_equals_Vqf32(v2);                                                          \
        }                                                                                                     \
                                                                                                              \
        if (nloe > 0) {                                                                                       \
            HVX_Vector v1 = v_work[nvec];                                                                     \
            HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);                                              \
            hvx_vec_store_a(&v_work[nvec], nloe * sizeof(float), Q6_Vsf_equals_Vqf32(v2));                    \
        }                                                                                                     \
    }                                                                                                         \
}

DEFINE_HVX_FAST_SOFTMAX(f32_nosink, 0, 0)
DEFINE_HVX_FAST_SOFTMAX(f32_sink,   1, 0)
DEFINE_HVX_FAST_SOFTMAX(f16_nosink, 0, 1)
DEFINE_HVX_FAST_SOFTMAX(f16_sink,   1, 1)

#undef DEFINE_HVX_FAST_SOFTMAX

static __attribute__((noinline)) void softmax_init_alibi_slopes(
    float * restrict slopes,
    uint32_t n_head,
    uint32_t n_head_log2,
    float m0,
    float m1
) {
    for (uint32_t h = 0; h < n_head; h += 32) {
        HVX_Vector v_slopes = hvx_alibi_slopes(h, 1, n_head_log2, m0, m1);
        hvx_vmem(&slopes[h]) = v_slopes;
    }
}

static const softmax_prep_fn_t softmax_prep_kernels[HTP_SOFTMAX_KERNEL_COUNT] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]         = hvx_fast_softmax_prep_nomask,
    [HTP_SOFTMAX_KERNEL_MASK_F32]       = hvx_fast_softmax_prep_f32,
    [HTP_SOFTMAX_KERNEL_MASK_F16]       = hvx_fast_softmax_prep_f16,
    [HTP_SOFTMAX_KERNEL_SRC16_NOMASK]   = hvx_fast_softmax_prep_src16_nomask,
    [HTP_SOFTMAX_KERNEL_SRC16_MASK_F32] = hvx_fast_softmax_prep_src16_mask_f32,
    [HTP_SOFTMAX_KERNEL_SRC16_MASK_F16] = hvx_fast_softmax_prep_src16_mask_f16,
};

static const softmax_fn_t softmax_kernels[2][2] = {
    { hvx_fast_softmax_f32_nosink, hvx_fast_softmax_f32_sink },
    { hvx_fast_softmax_f16_nosink, hvx_fast_softmax_f16_sink },
};

static void softmax_thread_dma(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_softmax_context * smctx = (const struct htp_softmax_context *) data;
    struct htp_ops_context * octx = smctx->octx;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    const uint32_t src0_nrows            = smctx->nrows;
    const uint32_t src0_nrows_per_hvx    = smctx->src0_nrows_per_hvx;

    const uint32_t src0_start_row = smctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, smctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src0 = smctx->data_src0;
    const dma_addr_t data_dst  = smctx->data_dst;

    const size_t src0_row_size = src0->ne[0] * (smctx->src0_f16 ? sizeof(__fp16) : sizeof(float));
    const size_t dst_row_size  = dst->ne[0]  * (smctx->dst_f16  ? sizeof(__fp16) : sizeof(float));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_hvx);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_hvx);
    uint8_t * work_vtcm_base = smctx->dst_f16 ? (smctx->vtcm_work + ith * smctx->vtcm_work_size_per_hvx) : NULL;

    const size_t src0_vtcm_half = smctx->src0_spad_half_size;
    const size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne02 = src0->ne[2];
    const struct fastdiv_values * div_ne01 = &smctx->div_ne01;
    const struct fastdiv_values * div_ne02 = &smctx->div_ne02;

    dma_queue * dma_q = th->dma_q;

    for (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;
        void * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        void * s_spad = src0_vtcm_base + idx * src0_vtcm_half;

        dma_queue_push(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
    }

    softmax_prep_fn_t prep   = (softmax_prep_fn_t) smctx->prep;
    softmax_fn_t softmax_fn  = (softmax_fn_t) smctx->softmax;
    const bool has_sinks     = smctx->use_src2;
    const bool dst_f16       = smctx->dst_f16;
    uint32_t prev_i2         = (uint32_t) -1;
    const float * sink_ptr   = NULL;

    for (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        void * d_spad = (void *) dma_queue_pop(dma_q).src;
        void * s_spad = (void *) dma_queue_pop(dma_q).dst;
        void * w_spad = dst_f16 ? (void *) work_vtcm_base : d_spad;

        if (has_sinks) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            if (i2 != prev_i2) {
                sink_ptr = smctx->vtcm_src2 + i2;
                prev_i2  = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        prep(w_spad, s_spad, NULL, ne00, smctx->scale, 1.0f);
        softmax_fn((uint8_t *) w_spad, (uint8_t *) d_spad, ne00, sink_ptr);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_push(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        const uint32_t next_r = r + 2;
        if (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;
            dma_queue_push(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

static void softmax_thread_mask_dma(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_softmax_context * smctx = (const struct htp_softmax_context *) data;
    struct htp_ops_context * octx = smctx->octx;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    const uint32_t src0_nrows            = smctx->nrows;
    const uint32_t src0_nrows_per_hvx    = smctx->src0_nrows_per_hvx;

    const uint32_t src0_start_row = smctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, smctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src0 = smctx->data_src0;
    const dma_addr_t data_src1 = smctx->data_src1;
    const dma_addr_t data_dst  = smctx->data_dst;

    const size_t src0_row_size = src0->ne[0] * (smctx->src0_f16 ? sizeof(__fp16) : sizeof(float));
    const size_t dst_row_size  = dst->ne[0]  * (smctx->dst_f16  ? sizeof(__fp16) : sizeof(float));
    const size_t mask_row_size = smctx->use_f16 ? (src1->ne[0] * sizeof(__fp16)) : (src1->ne[0] * sizeof(float));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_hvx);
    uint8_t * src1_vtcm_base = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_size_per_hvx);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_hvx);
    uint8_t * work_vtcm_base = smctx->dst_f16 ? (smctx->vtcm_work + ith * smctx->vtcm_work_size_per_hvx) : NULL;

    const size_t src0_vtcm_half = smctx->src0_spad_half_size;
    const size_t src1_vtcm_half = smctx->src1_spad_half_size;
    const size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    const uint32_t nb11 = src1->nb[1];
    const uint32_t nb12 = src1->nb[2];
    const uint32_t nb13 = src1->nb[3];

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t ne13 = src1->ne[3];

    const struct fastdiv_values * div_ne01 = &smctx->div_ne01;
    const struct fastdiv_values * div_ne02 = &smctx->div_ne02;
    const struct fastdiv_values * div_ne12 = &smctx->div_ne12;
    const struct fastdiv_values * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = th->dma_q;

    for (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = data_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        void * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        void * s_spad = src0_vtcm_base + idx * src0_vtcm_half;
        void * m_spad = src1_vtcm_base + idx * src1_vtcm_half;

        dma_queue_push(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_push(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        dma_queue_push(dma_q, dma_make_data(m_spad, cur_src1),
                       smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
    }

    softmax_prep_fn_t prep   = (softmax_prep_fn_t) smctx->prep;
    softmax_fn_t softmax_fn  = (softmax_fn_t) smctx->softmax;
    const bool has_bias      = smctx->max_bias > 0.0f;
    const bool has_sinks     = smctx->use_src2;
    const bool dst_f16       = smctx->dst_f16;
    uint32_t prev_i2         = (uint32_t) -1;
    float slope              = 1.0f;
    const float * sink_ptr   = NULL;

    for (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        void * d_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * s_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * m_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * w_spad = dst_f16 ? (void *) work_vtcm_base : d_spad;

        if (has_bias || has_sinks) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            if (i2 != prev_i2) {
                if (has_bias) {
                    slope = smctx->slopes[i2];
                }
                if (has_sinks) {
                    sink_ptr = smctx->vtcm_src2 + i2;
                }
                prev_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        prep(w_spad, s_spad, m_spad, ne00, smctx->scale, slope);
        softmax_fn((uint8_t *) w_spad, (uint8_t *) d_spad, ne00, sink_ptr);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_push(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        const uint32_t next_r = r + 2;
        if (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;

            uint32_t ni1 = fastmodulo(next_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(next_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t next_src1 = data_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_queue_push(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
            dma_queue_push(dma_q, dma_make_data(m_spad, next_src1),
                           smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

static int execute_op_softmax_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const char * op_type = "softmax-f32";

    const struct htp_softmax_kernel_params * kparams =
        (const struct htp_softmax_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (kparams->kernel_type >= HTP_SOFTMAX_KERNEL_COUNT) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_size, kparams->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t elem_size    = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t dst_row_size = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
        if (nrows < octx->n_hvx) {
            htp_ops_context_set_n_hvx(octx, nrows ? nrows : 1);
        }
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx = octx->n_hvx;
    uint8_t * const vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    struct htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout,
                                  src0->ne[0],
                                  kparams->use_src1 ? octx->src[1]->ne[0] : 0,
                                  kparams->n_head,
                                  kparams->src0_f16 != 0,
                                  kparams->dst_f16 != 0,
                                  kparams->use_src1 != 0,
                                  kparams->use_f16 != 0,
                                  kparams->use_src2 != 0,
                                  kparams->n_hvx);

    struct htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .prep                  = (void *) softmax_prep_kernels[kparams->kernel_type],
        .softmax               = (void *) softmax_kernels[kparams->dst_f16 != 0][kparams->use_src2 != 0],

        .data_src0             = src0->data,
        .data_src1             = kparams->use_src1 ? octx->src[1]->data : 0,
        .data_dst              = dst->data,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_base, layout.off_src1, kparams->use_src1),
        .vtcm_work             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_base, layout.off_work, kparams->dst_f16),
        .vtcm_src2             = VTCM_LAYOUT_PTR_OPTIONAL(float,   vtcm_base, layout.off_src2, kparams->use_src2),

        .vtcm_src0_size_per_hvx = (uint32_t) layout.src0_bytes_per_hvx,
        .vtcm_src1_size_per_hvx = (uint32_t) layout.src1_bytes_per_hvx,
        .vtcm_dst_size_per_hvx  = (uint32_t) layout.dst_bytes_per_hvx,
        .vtcm_work_size_per_hvx = (uint32_t) layout.work_bytes_per_hvx,

        .src0_spad_half_size   = (uint32_t) layout.src0_spad_half_size,
        .src1_spad_half_size   = (uint32_t) layout.src1_spad_half_size,
        .dst_spad_half_size    = (uint32_t) layout.dst_spad_half_size,

        .src0_row_size_aligned = (uint32_t) layout.src0_spad_half_size,
        .src1_row_size_aligned = (uint32_t) layout.src1_spad_half_size,
        .dst_row_size_aligned  = (uint32_t) layout.dst_spad_half_size,

        .src0_f16              = kparams->src0_f16 != 0,
        .dst_f16               = kparams->dst_f16 != 0,
        .use_f16               = kparams->use_f16 != 0,
        .use_src2              = kparams->use_src2 != 0,

        .n_head                = kparams->n_head,
        .n_head_log2           = kparams->n_head_log2,

        .scale                 = kparams->scale,
        .max_bias              = kparams->max_bias,
        .m0                    = kparams->m0,
        .m1                    = kparams->m1,

        .div_ne01              = kparams->div_ne01,
        .div_ne02              = kparams->div_ne02,
        .div_ne12              = kparams->div_ne12,
        .div_ne13              = kparams->div_ne13,

        .src0_nrows_per_hvx    = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .row_start             = row_start,
        .nrows                 = nrows,
    };

    if (kparams->max_bias > 0.0f && kparams->use_src1) {
        if (kparams->n_head > 512) {
            return HTP_STATUS_INVAL_PARAMS;
        }
        softmax_init_alibi_slopes(smctx.slopes, kparams->n_head, kparams->n_head_log2, kparams->m0, kparams->m1);
    }

    if (kparams->use_src2) {
        const struct htp_tensor * src2 = octx->src[2];
        dma_queue * dma_q = octx->ctx->dma[0];
        dma_queue_push(dma_q, dma_make_data(smctx.vtcm_src2, src2->data), layout.src2_bytes, 0, src2->size, 1);
        dma_queue_pop(dma_q);
    }

    work_queue_func_t task_func = kparams->use_src1 ? softmax_thread_mask_dma : softmax_thread_dma;
    work_queue_run(octx->ctx->hvx_queue, task_func, &smctx, n_hvx);

    return HTP_STATUS_OK;
}

int op_softmax(struct htp_ops_context * octx) {
    if (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
        case HTP_TYPE_F16:
            return execute_op_softmax_f32(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
