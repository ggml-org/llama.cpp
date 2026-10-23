#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "hex-fastdiv.h"
#include "hvx-exp.h"
#include "hvx-sigmoid.h"
#include "hvx-utils.h"
#include "unary-ops.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hex-profile.h"

struct htp_unary_context;

typedef void (*unary_compute_fn_t)(const void * restrict src,
                                   void * restrict dst,
                                   uint32_t num_rows,
                                   const struct htp_unary_context * uctx);

typedef void (*unary_rms_norm_mul_compute_fn_t)(const void * restrict src,
                                                const void * restrict weight,
                                                void * restrict dst,
                                                uint32_t num_rows,
                                                const struct htp_unary_context * uctx);

typedef void (*unary_tri_compute_fn_t)(const void * restrict src,
                                       void * restrict dst,
                                       uint32_t num_rows,
                                       uint32_t ir,
                                       const struct htp_unary_context * uctx);

typedef void (*unary_tile_compute_fn_t)(void * restrict dst,
                                        const void * restrict src,
                                        uint32_t tw,
                                        const struct htp_unary_context * uctx);

typedef void (*unary_tiled_tri_compute_fn_t)(const void * restrict src,
                                             void * restrict dst,
                                             uint32_t tile_elems,
                                             uint32_t col_start,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

struct htp_unary_context {
    struct htp_ops_context * octx;
    const struct htp_unary_kernel_params * kparams;

    void *                    compute;

    dma_addr_t                data_src0;
    dma_addr_t                data_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                data_dst;

    size_t                    src0_data_row_size;   // actual data bytes per row
    size_t                    src1_data_row_size;
    size_t                    dst_data_row_size;    // actual data bytes per row

    size_t                    src0_row_size_aligned;
    size_t                    src1_row_size_aligned;
    size_t                    dst_row_size_aligned;

    size_t                    src0_vtcm_half_size;
    size_t                    src1_vtcm_half_size;
    size_t                    dst_vtcm_half_size;

    uint32_t                  block;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_per_thread;
    uint32_t                  row_start;
    uint32_t                  nc;
    uint32_t                  col_tile;             // tiled mode
    bool                      broadcast_weight;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_size_per_thread;
    size_t                    vtcm_src1_size_per_thread;
    size_t                    vtcm_dst_size_per_thread;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
static inline size_t unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      const struct fastdiv_values * div_ne1,
                                      const struct fastdiv_values * div_ne2,
                                      const struct fastdiv_values * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    const uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    const uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    const uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    const uint32_t i3 = fastdiv(ir, div_ne12);
    return i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the dim-1 slice boundary
// when non-contiguous so the nb1 stride stays valid for all rows.
static inline uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool contig,
                                        uint32_t ne1,
                                        const struct fastdiv_values * div_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    if (!contig) {
        const uint32_t slice_end = (fastdiv(ir, div_ne1) + 1) * ne1;
        limit = MIN(limit, slice_end - ir);
    }

    return limit;
}

#define htp_unary_preamble            \
    const uint32_t ne00 = src->ne[0]; \
    const uint32_t ne01 = src->ne[1]; \
    const uint32_t ne02 = src->ne[2]; \
    const uint32_t ne03 = src->ne[3]; \
                                      \
    const uint32_t ne0 = dst->ne[0];  \
    const uint32_t ne1 = dst->ne[1];  \
    const uint32_t ne2 = dst->ne[2];  \
    const uint32_t ne3 = dst->ne[3];  \
                                      \
    const uint32_t nb00 = src->nb[0]; \
    const uint32_t nb01 = src->nb[1]; \
    const uint32_t nb02 = src->nb[2]; \
    const uint32_t nb03 = src->nb[3]; \
                                      \
    const uint32_t nb0 = dst->nb[0];  \
    const uint32_t nb1 = dst->nb[1];  \
    const uint32_t nb2 = dst->nb[2];  \
    const uint32_t nb3 = dst->nb[3];

#define htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    const uint32_t ne0 = uctx->nc;                                    \
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned; \
    const size_t dst_row_size_aligned = uctx->dst_row_size_aligned;

static void scale_f32(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0, scale, bias);
    }
}

static void clamp_f32(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float min = 0.f;
    float max = 0.f;
    memcpy(&min, &op_params[0], sizeof(float));
    memcpy(&max, &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

static void leaky_relu_f32(const void * restrict src,
                           void * restrict dst,
                           const uint32_t num_rows,
                           const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float negative_slope = 0.f;
    memcpy(&negative_slope, &op_params[0], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_slope, ne0);
    }
}

static void rms_norm_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    float scale   = 0.f;
    memcpy(&epsilon, &op_params[0], sizeof(float));
    memcpy(&scale,   &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon, scale);
    }
}

static void rms_norm_mul_f32(const void * restrict src,
                             const void * restrict weight,
                             void * restrict dst,
                             const uint32_t num_rows,
                             const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    float scale   = 0.f;
    memcpy(&epsilon, &op_params[0], sizeof(float));
    memcpy(&scale,   &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        const uint8_t * restrict w_local   = (const uint8_t *)weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon, scale);
    }
}

static void norm_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void sqr_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sqrt_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void scale_f16(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0, scale, bias);
    }
}

static void clamp_f16(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float min = 0.f;
    float max = 0.f;
    memcpy(&min, &op_params[0], sizeof(float));
    memcpy(&max, &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

static void rms_norm_f16(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f16((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void norm_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f16((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void sqr_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sqrt_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void abs_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void log_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void step_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void ceil_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_ceil_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void l2_norm_f16(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_f = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f16((const uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

static void neg_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

static void exp_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

static void sigmoid_f32(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
static void silu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
static __attribute__((noinline)) HVX_Vector hvx_vec_gelu_f32(HVX_Vector x) {
    const HVX_Vector half = hvx_vec_splat_f32(0.5f);
    const HVX_Vector one  = hvx_vec_splat_f32(1.0f);

    HVX_Vector inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_add_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    return hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_add_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

static inline void hvx_gelu_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
static __attribute__((noinline)) HVX_Vector hvx_vec_gelu_quick_f32(HVX_Vector x) {
    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);
    const HVX_Vector scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    const HVX_Vector sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    return hvx_vec_mul_f32_f32(x, sigmoid);
}

static inline void hvx_gelu_quick_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_quick_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_quick_f32(vsrc[i]));
    }
}

static void gelu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

static void gelu_quick_f32(const void * restrict src,
                           void * restrict dst,
                           const uint32_t num_rows,
                           const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_quick_f32_aa(dst_local, src_local, ne0);
    }
}

static void gelu_erf_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

static void tri_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ir,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const int32_t ttype = op_params[0];
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    const uint32_t nvec  = ne0 / VLEN_FP32;
    const uint32_t nloe  = ne0 % VLEN_FP32;

    const uint32_t ne01 = uctx->octx->src[0]->ne[1];
    const struct fastdiv_values * div_ne01 = &uctx->kparams->div_ne01;

    for (uint32_t b = 0; b < num_rows; b++) {
        const uint32_t abs_row = ir + b;
        const uint32_t i01     = fastmodulo(abs_row, ne01, div_ne01);

        const HVX_Vector * restrict v_src = (const HVX_Vector *) ((const uint8_t *) src + b * src0_row_size_aligned);
        HVX_Vector * restrict v_dst       = (HVX_Vector *) ((uint8_t *) dst + b * dst_row_size_aligned);

        uint32_t boundary;
        int      keep_left;
        switch (ttype) {
            case 0: boundary = i01;     keep_left = 0; break;  // keep col >= row
            case 1: boundary = i01 + 1; keep_left = 0; break;  // keep col > row
            case 2: boundary = i01 + 1; keep_left = 1; break;  // keep col <= row
            case 3: boundary = i01;     keep_left = 1; break;  // keep col < row
            default: boundary = 0; keep_left = 0; break;
        }
        if (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        for (uint32_t i = 0; i < nvec; i++) {
            const uint32_t vec_start = i * VLEN_FP32;
            const uint32_t vec_end   = vec_start + VLEN_FP32;
            if (keep_left) {
                if (vec_end <= boundary) {
                    v_dst[i] = v_src[i];
                } else if (vec_start >= boundary) {
                    v_dst[i] = zero;
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * sizeof(float));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
                }
            } else {
                if (vec_end <= boundary) {
                    v_dst[i] = zero;
                } else if (vec_start >= boundary) {
                    v_dst[i] = v_src[i];
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * sizeof(float));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        if (nloe > 0) {
            const uint32_t abs_start = nvec * VLEN_FP32;
            const uint32_t abs_end   = abs_start + nloe;
            HVX_Vector     tail_val;
            if (keep_left) {
                if (abs_end <= boundary) {
                    tail_val = v_src[nvec];
                } else if (abs_start >= boundary) {
                    tail_val = zero;
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                    tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
                }
            } else {
                if (abs_end <= boundary) {
                    tail_val = zero;
                } else if (abs_start >= boundary) {
                    tail_val = v_src[nvec];
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                    tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), tail_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
static inline void hvx_softplus_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const HVX_Vector twenty = hvx_vec_splat_f32(20.0f);
    const HVX_Vector one    = hvx_vec_splat_f32(1.0f);

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    #pragma unroll(4)
    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector x              = v_src[i];
        HVX_VectorPred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx           = hvx_vec_exp_f32(x);
        HVX_Vector logv           = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    if (nloe > 0) {
        HVX_Vector x             = v_src[nvec];
        HVX_VectorPred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx          = hvx_vec_exp_f32(x);
        HVX_Vector logv          = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        HVX_Vector sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((void *) &v_dst[nvec], nloe * sizeof(float), sp);
    }
}

static void softplus_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

static void l2_norm_f32(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_f = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f32((const uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

static inline HVX_VectorPair hvx_fwht_stage_pair(HVX_Vector v0, HVX_Vector v1, int32_t b) {
    HVX_VectorPair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_Vector d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    return Q6_W_vshuff_VVR(d, s, b);
}

static inline HVX_VectorPair hvx_fwht32_pair(HVX_Vector v0_sf, HVX_Vector v1_sf, HVX_Vector v_scale) {
    HVX_Vector v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_VectorPair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    return p;
}

static inline void hvx_fwht_store32(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

static inline void hvx_fwht_store64(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    HVX_Vector u = Q6_V_lo_W(p);
    HVX_Vector v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

static inline void hvx_fwht_tail_f32(HVX_Vector * restrict v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    for (; len_vec * 2 < nvec; len_vec <<= 2) {
        const uint32_t step = len_vec * 4;
        for (uint32_t i = 0; i < nvec; i += step) {
            for (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector a0 = v_dst[i + j];
                HVX_Vector a1 = v_dst[i + j + len_vec];
                HVX_Vector a2 = v_dst[i + j + 2 * len_vec];
                HVX_Vector a3 = v_dst[i + j + 3 * len_vec];

                HVX_Vector t0 = Q6_Vqf32_vadd_VsfVsf(a0, a1);
                HVX_Vector t1 = Q6_Vqf32_vsub_VsfVsf(a0, a1);
                HVX_Vector t2 = Q6_Vqf32_vadd_VsfVsf(a2, a3);
                HVX_Vector t3 = Q6_Vqf32_vsub_VsfVsf(a2, a3);

                v_dst[i + j]               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t0, t2));
                v_dst[i + j + len_vec]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t1, t3));
                v_dst[i + j + 2 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t0, t2));
                v_dst[i + j + 3 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t1, t3));
            }
        }
    }

    if (len_vec < nvec) {
        for (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            for (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector u = v_dst[i + j];
                HVX_Vector v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_add_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

static inline void hvx_fwht_row_f32(const uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    const uint32_t nvec               = n / VLEN_FP32;

    if (nvec == 1) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        return;
    }

    for (uint32_t i = 0; i < nvec; i += 2) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

static inline void hvx_fwht_row_f16(const uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    const uint32_t nvec               = n / VLEN_FP32;

    if (nvec == 1) {
        HVX_Vector v0 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src[0]));
        HVX_VectorPair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        return;
    }

    const uint32_t nvec_f16 = n / VLEN_FP16;
    for (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_VectorPair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

static void hadamard_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((float) ne0));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_row = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f32(src_row, dst_row, ne0, v_scale);
    }
}

static void hadamard_f16(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((float) ne0));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_row = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f16(src_row, dst_row, ne0, v_scale);
    }
}

static void tanh_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

static void abs_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

static void relu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

static void step_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f32_aa(dst_local, src_local, ne0);
    }
}

static void ceil_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_ceil_f32_aa(dst_local, src_local, ne0);
    }
}

static void log_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
static void tile_scale_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &uctx->octx->op_params[0], sizeof(float));
    memcpy(&bias,  &uctx->octx->op_params[1], sizeof(float));
    hvx_scale_offset_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw, scale, bias);
}

static void tile_clamp_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    float min = 0.f;
    float max = 0.f;
    memcpy(&min, &uctx->octx->op_params[0], sizeof(float));
    memcpy(&max, &uctx->octx->op_params[1], sizeof(float));
    hvx_clamp_scalar_f32((uint8_t *) dst, (const uint8_t *) src, min, max, tw);
}

static void tile_leaky_relu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    float negative_slope = 0.f;
    memcpy(&negative_slope, &uctx->octx->op_params[0], sizeof(float));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (const uint8_t *) src, negative_slope, tw);
}

static void tile_sqr_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_sqrt_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_neg_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw, -1.0f);
}

static void tile_exp_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_exp_f32((uint8_t *) dst, (const uint8_t *) src, tw, false);
}

static void tile_sigmoid_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_silu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (const uint8_t *) src, (uint8_t *) dst, tw);
}

static void tile_gelu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_gelu_quick_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_gelu_quick_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_gelu_erf_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_softplus_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_tanh_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_abs_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_log_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_relu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (const uint8_t *) src, 0.0f, tw);
}

static void tile_step_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_step_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_ceil_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_ceil_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tri_apply_tile_f32(const void * restrict src, void * restrict dst,
                               uint32_t tile_elems, uint32_t col_start, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    int      keep_left;
    switch (ttype) {
        case 0: boundary = i01;     keep_left = 0; break;
        case 1: boundary = i01 + 1; keep_left = 0; break;
        case 2: boundary = i01 + 1; keep_left = 1; break;
        case 3: boundary = i01;     keep_left = 1; break;
        default: boundary = 0; keep_left = 0; break;
    }
    if (boundary > ne0) boundary = ne0;

    const uint32_t nvec = tile_elems / VLEN_FP32;
    const uint32_t nloe = tile_elems % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        const uint32_t abs_start = col_start + i * VLEN_FP32;
        const uint32_t abs_end   = abs_start + VLEN_FP32;
        if (keep_left) {
            if (abs_end <= boundary) {
                v_dst[i] = v_src[i];
            } else if (abs_start >= boundary) {
                v_dst[i] = zero;
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
            }
        } else {
            if (abs_end <= boundary) {
                v_dst[i] = zero;
            } else if (abs_start >= boundary) {
                v_dst[i] = v_src[i];
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
            }
        }
    }

    if (nloe > 0) {
        const uint32_t abs_start = col_start + nvec * VLEN_FP32;
        const uint32_t abs_end   = abs_start + nloe;
        HVX_Vector     tail_val;
        if (keep_left) {
            if (abs_end <= boundary) {
                tail_val = v_src[nvec];
            } else if (abs_start >= boundary) {
                tail_val = zero;
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
            }
        } else {
            if (abs_end <= boundary) {
                tail_val = zero;
            } else if (abs_start >= boundary) {
                tail_val = v_src[nvec];
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), tail_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
static void unary_thread_row_block(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    const bool src0_contig = uctx->kparams->src0_contig;
    const uint32_t BLOCK   = uctx->block;

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    dma_queue * dma_q = octx->ctx->dma[ith];

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = src0_contig ? ((size_t) ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, ne01, div_ne01);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = (size_t) ir * nb1;
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? ((size_t) pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
static void unary_thread_rms_norm_mul_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src  = uctx->data_src0;
    const dma_addr_t data_src1 = uctx->data_src1;
    const dma_addr_t data_dst  = uctx->data_dst;

    const struct htp_tensor * src1 = octx->src[1];
    const uint32_t nb11 = src1->nb[1];
    const uint32_t nb12 = src1->nb[2];
    const uint32_t nb13 = src1->nb[3];

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_thread)) : NULL;
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t src1_vtcm_half_size = uctx->src1_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    const bool src0_contig  = uctx->kparams->src0_contig;
    const bool src1_contig  = uctx->kparams->src1_contig;
    const bool block_contig = src0_contig && src1_contig;
    const uint32_t BLOCK    = uctx->block;

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    dma_queue * dma_q = octx->ctx->dma[ith];

    if (uctx->broadcast_weight) {
        dma_queue_push(dma_q, dma_make_data(src1_vtcm_data, data_src1),
                       uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
        dma_queue_flush(dma_q);
    }

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_contig, ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = src0_contig ? ((size_t) ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        if (!uctx->broadcast_weight) {
            const size_t src1_off = src1_contig ? ((size_t) ir * nb11) :
                unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11, nb12, nb13);
            dma_queue_push(dma_q,
                dma_make_data(src1_vtcm_data + (vtcm_idx * src1_vtcm_half_size), data_src1 + src1_off),
                uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, block_size);
        }

        ir += block_size;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_contig, ne01, div_ne01);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * src1_vtcm = NULL;
        if (!uctx->broadcast_weight) {
            src1_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        const void * w = uctx->broadcast_weight ? (const void *) src1_vtcm_data : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = (size_t) ir * nb1;
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, block_contig,
                                                              ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, block_contig,
                                                                  ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? ((size_t) pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);

                if (!uctx->broadcast_weight) {
                    const size_t src1_pref_off = src1_contig ? ((size_t) pref_ir * nb11) :
                        unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11, nb12, nb13);
                    dma_queue_push(dma_q,
                        dma_make_data(src1_vtcm, data_src1 + src1_pref_off),
                        uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, pref_block_size);
                }
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 3. TRI row-block task with row index ir.
static void unary_thread_tri_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;
    const uint32_t BLOCK             = uctx->block;

    dma_queue * dma_q = octx->ctx->dma[ith];

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = (size_t) ir * nb01;
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = (size_t) ir * nb1;
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = MIN(BLOCK, src0_end_row - next_ir);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_ir);
                const size_t src0_pref_off = (size_t) pref_ir * nb01;
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Pointwise tiled unary task.
static void unary_thread_tiled(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const uint32_t col_tile              = uctx->col_tile;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_half = uctx->src0_vtcm_half_size;
    const size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;
    const struct fastdiv_values * div_tpr   = &uctx->kparams->div_tpr;

    const uint32_t tiles_per_row = uctx->kparams->tiles_per_row;
    const bool     src0_contig   = uctx->kparams->src0_contig;

    const uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    for (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        const uint32_t row  = src0_start_row + fastdiv(t, div_tpr);
        const uint32_t col  = fastmodulo(t, tiles_per_row, div_tpr) * col_tile;
        const uint32_t tw   = MIN(col_tile, ne0 - col);
        const size_t   tb   = (size_t) tw * sizeof(float);
        const size_t   soff = (src0_contig ? ((size_t) row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * sizeof(float);

        dma_queue_push(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    for (uint32_t t = 0; t < total_tiles; t++) {
        void * dst_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        const uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        const size_t doff = (size_t) row * nb1 + (size_t) col * sizeof(float);
        const size_t tb   = (size_t) tw * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        const uint32_t pt = t + 2;
        if (pt < total_tiles) {
            const uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            const size_t   ptb  = (size_t) ptw * sizeof(float);
            const size_t   psoff = (src0_contig ? ((size_t) prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        if (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
        }

        ptile_in_row++;
        pcol += col_tile;
        if (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

// 5. TRI tiled task.
static void unary_thread_tiled_tri_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const int32_t * op_params            = octx->op_params;
    const uint32_t col_tile              = uctx->col_tile;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_half = uctx->src0_vtcm_half_size;
    const size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const struct fastdiv_values * div_ne01 = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_tpr  = &uctx->kparams->div_tpr;

    const uint32_t tiles_per_row = uctx->kparams->tiles_per_row;
    const int32_t  tri_ttype     = op_params[0];

    const uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    for (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        const uint32_t row  = src0_start_row + fastdiv(t, div_tpr);
        const uint32_t col  = fastmodulo(t, tiles_per_row, div_tpr) * col_tile;
        const uint32_t tw   = MIN(col_tile, ne0 - col);
        const size_t   tb   = (size_t) tw * sizeof(float);
        const size_t   soff = (size_t) row * nb01 + (size_t) col * sizeof(float);

        dma_queue_push(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    for (uint32_t t = 0; t < total_tiles; t++) {
        void * dst_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        const uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        const size_t doff = (size_t) row * nb1 + (size_t) col * sizeof(float);
        const size_t tb   = (size_t) tw * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        const uint32_t pt = t + 2;
        if (pt < total_tiles) {
            const uint32_t ptw   = MIN(col_tile, ne0 - pcol);
            const size_t   ptb   = (size_t) ptw * sizeof(float);
            const size_t   psoff = (size_t) prow * nb01 + (size_t) pcol * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        if (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
            i01++;
            if (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_in_row++;
        pcol += col_tile;
        if (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

static int execute_op_unary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const bool is_f16 = (src0->type == HTP_TYPE_F16);

    const char * op_type = NULL;

    switch (octx->op) {
        case HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         break;
        case HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      break;
        case HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            break;
        case HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        break;
        case HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        break;
        case HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             break;
        case HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          break;
        case HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         break;
        case HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    break;
        case HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    break;
        case HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                break;
        case HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   break;
        case HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   break;
        case HTP_OP_UNARY_GELU_QUICK: op_type = "gelu-quick-f32";                             break;
        case HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               break;
        case HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               break;
        case HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   break;
        case HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          break;
        case HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          break;
        case HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   break;
        case HTP_OP_UNARY_STEP:      op_type = is_f16 ? "step-f16"     : "step-f32";         break;
        case HTP_OP_UNARY_CEIL:      op_type = is_f16 ? "ceil-f16"     : "ceil-f32";         break;
        case HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       break;
        case HTP_OP_TRI:             op_type = "tri-f32";                                    break;
        case HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     break;
        default:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_unary_kernel_params * kparams = (const struct htp_unary_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t src0_elem_size = is_f16 ? sizeof(_Float16) : sizeof(float);
    const size_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? sizeof(_Float16) : sizeof(float);
    const size_t src0_data_row_size = src0->ne[0] * src0_elem_size;
    const size_t dst_data_row_size  = dst->ne[0]  * dst_elem_size;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) dst_elem_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

    // Always 0 for F16 - htp_unary_solve_layout() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    const uint32_t col_tile = kparams->col_tile;

    size_t src1_data_row_size = 0;
    bool broadcast_weight = kparams->broadcast_weight;
    const struct htp_tensor * src1 = NULL;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    if (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_data_row_size = src1->ne[0] * sizeof(float);
    }

    if ((!col_tile && kparams->block == 0) || octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, (size_t) kparams->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    struct htp_unary_vtcm_layout L;
    htp_unary_vtcm_layout_build(&L, octx->op, src0_data_row_size, dst_data_row_size, src1_data_row_size,
                                src0_elem_size, col_tile, kparams->block, broadcast_weight, kparams->n_threads);

    FARF(HIGH, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         L.src0_bytes * n_threads, L.src1_bytes * n_threads, L.dst_bytes * n_threads);

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    struct htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .src0_nrows            = nrows,
        .row_start             = row_start,

        .data_src0             = src0->data,
        .data_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? src1->data : 0,
        .data_dst              = dst->data,

        .src0_data_row_size    = src0_data_row_size,
        .src1_data_row_size    = src1_data_row_size,
        .dst_data_row_size     = dst_data_row_size,

        .src0_row_size_aligned = L.src0_row_size_aligned,
        .src1_row_size_aligned = L.src1_row_size_aligned,
        .dst_row_size_aligned  = L.dst_row_size_aligned,

        .src0_vtcm_half_size   = L.src0_bytes >> 1,
        .src1_vtcm_half_size   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (L.src1_bytes >> (broadcast_weight ? 0 : 1)) : 0,
        .dst_vtcm_half_size    = L.dst_bytes >> 1,

        .block                 = kparams->block,
        .nc                    = src0->ne[0],
        .col_tile              = col_tile,
        .broadcast_weight      = broadcast_weight,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_src1, L.src1_bytes > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst),

        .vtcm_src0_size_per_thread = L.src0_bytes,
        .vtcm_src1_size_per_thread = L.src1_bytes,
        .vtcm_dst_size_per_thread  = L.dst_bytes,
    };

    FARF(HIGH, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "tiled" : "row-block", col_tile);

    worker_callback_t task_func = NULL;
    void * compute_func = NULL;

    if (col_tile) {
        task_func = unary_thread_tiled;
        switch (octx->op) {
            case HTP_OP_SCALE:           compute_func = (void *) tile_scale_f32;          break;
            case HTP_OP_CLAMP:           compute_func = (void *) tile_clamp_f32;          break;
            case HTP_OP_LEAKY_RELU:      compute_func = (void *) tile_leaky_relu_f32;     break;
            case HTP_OP_SQR:             compute_func = (void *) tile_sqr_f32;            break;
            case HTP_OP_SQRT:            compute_func = (void *) tile_sqrt_f32;           break;
            case HTP_OP_UNARY_NEG:       compute_func = (void *) tile_neg_f32;            break;
            case HTP_OP_UNARY_EXP:       compute_func = (void *) tile_exp_f32;            break;
            case HTP_OP_UNARY_SIGMOID:   compute_func = (void *) tile_sigmoid_f32;        break;
            case HTP_OP_UNARY_SILU:      compute_func = (void *) tile_silu_f32;           break;
            case HTP_OP_UNARY_GELU:      compute_func = (void *) tile_gelu_f32;           break;
            case HTP_OP_UNARY_GELU_QUICK: compute_func = (void *) tile_gelu_quick_f32;     break;
            case HTP_OP_UNARY_GELU_ERF:  compute_func = (void *) tile_gelu_erf_f32;       break;
            case HTP_OP_UNARY_SOFTPLUS:  compute_func = (void *) tile_softplus_f32;       break;
            case HTP_OP_UNARY_TANH:      compute_func = (void *) tile_tanh_f32;           break;
            case HTP_OP_UNARY_ABS:       compute_func = (void *) tile_abs_f32;            break;
            case HTP_OP_UNARY_LOG:       compute_func = (void *) tile_log_f32;            break;
            case HTP_OP_UNARY_RELU:      compute_func = (void *) tile_relu_f32;           break;
            case HTP_OP_UNARY_STEP:      compute_func = (void *) tile_step_f32;           break;
            case HTP_OP_UNARY_CEIL:      compute_func = (void *) tile_ceil_f32;           break;
            case HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (void *) tri_apply_tile_f32;
                break;
            default:                     break;
        }
    } else if (is_f16) {
        task_func = unary_thread_row_block;
        switch (octx->op) {
            case HTP_OP_NORM:            compute_func = (void *) norm_f16;                break;
            case HTP_OP_RMS_NORM:        compute_func = (void *) rms_norm_f16;            break;
            case HTP_OP_SCALE:           compute_func = (void *) scale_f16;               break;
            case HTP_OP_CLAMP:           compute_func = (void *) clamp_f16;               break;
            case HTP_OP_SQR:             compute_func = (void *) sqr_f16;                 break;
            case HTP_OP_SQRT:            compute_func = (void *) sqrt_f16;                break;
            case HTP_OP_L2_NORM:         compute_func = (void *) l2_norm_f16;             break;
            case HTP_OP_UNARY_ABS:       compute_func = (void *) abs_f16;                 break;
            case HTP_OP_UNARY_LOG:       compute_func = (void *) log_f16;                 break;
            case HTP_OP_UNARY_STEP:      compute_func = (void *) step_f16;                break;
            case HTP_OP_UNARY_CEIL:      compute_func = (void *) ceil_f16;                break;
            case HTP_OP_HADAMARD:        compute_func = (void *) hadamard_f16;            break;
            default:                     break;
        }
    } else {
        task_func = unary_thread_row_block;
        switch (octx->op) {
            case HTP_OP_NORM:            compute_func = (void *) norm_f32;                break;
            case HTP_OP_RMS_NORM:        compute_func = (void *) rms_norm_f32;            break;
            case HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (void *) rms_norm_mul_f32;
                break;
            case HTP_OP_SCALE:           compute_func = (void *) scale_f32;               break;
            case HTP_OP_CLAMP:           compute_func = (void *) clamp_f32;               break;
            case HTP_OP_LEAKY_RELU:      compute_func = (void *) leaky_relu_f32;          break;
            case HTP_OP_SQR:             compute_func = (void *) sqr_f32;                 break;
            case HTP_OP_SQRT:            compute_func = (void *) sqrt_f32;                break;
            case HTP_OP_UNARY_NEG:       compute_func = (void *) neg_f32;                 break;
            case HTP_OP_UNARY_EXP:       compute_func = (void *) exp_f32;                 break;
            case HTP_OP_UNARY_SIGMOID:   compute_func = (void *) sigmoid_f32;             break;
            case HTP_OP_UNARY_SILU:      compute_func = (void *) silu_f32;                break;
            case HTP_OP_UNARY_GELU:      compute_func = (void *) gelu_f32;                break;
            case HTP_OP_UNARY_GELU_QUICK: compute_func = (void *) gelu_quick_f32;          break;
            case HTP_OP_UNARY_GELU_ERF:  compute_func = (void *) gelu_erf_f32;            break;
            case HTP_OP_UNARY_SOFTPLUS:  compute_func = (void *) softplus_f32;            break;
            case HTP_OP_UNARY_TANH:      compute_func = (void *) tanh_f32;                break;
            case HTP_OP_UNARY_ABS:       compute_func = (void *) abs_f32;                 break;
            case HTP_OP_UNARY_LOG:       compute_func = (void *) log_f32;                 break;
            case HTP_OP_UNARY_RELU:      compute_func = (void *) relu_f32;                break;
            case HTP_OP_UNARY_STEP:      compute_func = (void *) step_f32;                break;
            case HTP_OP_UNARY_CEIL:      compute_func = (void *) ceil_f32;                break;
            case HTP_OP_L2_NORM:         compute_func = (void *) l2_norm_f32;             break;
            case HTP_OP_HADAMARD:        compute_func = (void *) hadamard_f32;            break;
            case HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (void *) tri_f32;
                break;
            default:                     break;
        }
    }

    if (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        return HTP_STATUS_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_queue_run(octx->ctx->work_queue, task_func, &uctx, n_threads);

    return err;
}

int op_unary(struct htp_ops_context * octx) {
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
        case HTP_TYPE_F16: {
            if (!htp_unary_is_elementwise(octx->op)) {
                return execute_op_unary(octx);
            }

            struct htp_tensor src0_c = *octx->src[0];
            struct htp_tensor dst_c  = *octx->dst;
            const size_t elem_size = (src0_c.type == HTP_TYPE_F16) ? sizeof(_Float16) : sizeof(float);
            HTP_UNARY_COLLAPSE_DIMS(src0_c.ne, src0_c.nb, dst_c.ne, dst_c.nb, elem_size);

            const struct htp_tensor * orig_src0 = octx->src[0];
            const struct htp_tensor * orig_dst  = octx->dst;
            octx->src[0] = &src0_c;
            octx->dst    = &dst_c;
            const int status = execute_op_unary(octx);
            octx->src[0] = orig_src0;
            octx->dst    = orig_dst;
            return status;
        }

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
