#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
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

struct htp_unary_context {
    struct htp_ops_context * octx;
    const struct htp_unary_kernel_params * kparams;

    const uint8_t *           data_src0;
    const uint8_t *           data_src1;            // weight/scale tensor for RMS_NORM_MUL
    uint8_t *                 data_dst;

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

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
static inline uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        const struct fastdiv_values * div_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    if (!src_contig || !dst_contig) {
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

static void scale_f32(const float * restrict src,
                      float * restrict dst,
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

static void rms_norm_f32(const float * restrict src,
                         float * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void rms_norm_mul_f32(const float * restrict src,
                             const float * restrict weight,
                             float * restrict dst,
                             const uint32_t num_rows,
                             const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        const uint8_t * restrict w_local   = (const uint8_t *)weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

static void norm_f32(const float * restrict src,
                     float * restrict dst,
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

static void sqr_f32(const float * restrict src,
                    float * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sqrt_f32(const float * restrict src,
                     float * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void neg_f32(const float * restrict src,
                    float * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

static void exp_f32(const float * restrict src,
                    float * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

static void sigmoid_f32(const float * restrict src,
                        float * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

static void tri_f32(const float * restrict src,
                    float * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ir,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const int32_t ttype = op_params[0];
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    const uint32_t nvec  = ne0 / VLEN_FP32;
    const uint32_t nloe  = ne0 % VLEN_FP32;

    const uint32_t ne01 = uctx->octx->src[0]->ne[1];

    for (uint32_t b = 0; b < num_rows; b++) {
        const uint32_t abs_row = ir + b;
        const uint32_t i01     = abs_row % ne01;

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

        // Full HVX vectors — each starts at a 128-byte aligned offset
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

static void softplus_f32(const float * restrict src,
                         float * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    // softplus(x) = log(1 + exp(x))
    // Match CPU reference: ggml_compute_softplus_f32() in ggml-impl.h
    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const float * restrict src_f = (const float *)((const uint8_t *)src + (ir * src0_row_size_aligned));
        float * restrict dst_f       = (float *)((uint8_t *)dst + (ir * dst_row_size_aligned));

        for (uint32_t i = 0; i < ne0; i++) {
            float x = src_f[i];
            // For x > 20: softplus(x) ≈ x (avoids exp overflow)
            dst_f[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
        }
    }
}

static void l2_norm_f32(const float * restrict src,
                        float * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const float * restrict src_f = (const float *)((const uint8_t *)src + (ir * src0_row_size_aligned));
        float * restrict dst_f       = (float *)((uint8_t *)dst + (ir * dst_row_size_aligned));

        hvx_fast_l2_norm_f32((const uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

static void tanh_f32(const float * restrict src,
                     float * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

#define DEFINE_UNARY_TASK(NAME, IS_RMS_NORM_MUL, IS_TRI, CORE_EXPR)                                                 \
static void unary_task_f32_##NAME(unsigned int nth, unsigned int ith, void * data) {                                \
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;                                \
    struct htp_ops_context * octx = uctx->octx;                                                                     \
    const struct htp_tensor * src = octx->src[0];                                                                   \
    const struct htp_tensor * dst = octx->dst;                                                                      \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                          \
                                                                                                                    \
    htp_unary_preamble;                                                                                             \
                                                                                                                    \
    int32_t *    op_params = octx->op_params;                                                                       \
    uint32_t     src0_nrows_per_thread = uctx->src0_nrows_per_thread;                                               \
                                                                                                                    \
    const size_t src0_data_row_size = uctx->src0_data_row_size;                                                     \
    const size_t dst_data_row_size  = uctx->dst_data_row_size;                                                      \
                                                                                                                    \
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;                                               \
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;                                                \
                                                                                                                    \
    const uint32_t src0_nrows = uctx->src0_nrows;                                                                   \
    const uint32_t src0_start_row = src0_nrows_per_thread * ith;                                                    \
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                        \
                                                                                                                    \
    if (src0_start_row >= src0_end_row) {                                                                           \
        return;                                                                                                     \
    }                                                                                                               \
                                                                                                                    \
    const uint8_t * restrict data_src = uctx->data_src0;                                                            \
    const uint8_t * restrict data_src1 = uctx->data_src1;                                                           \
    uint8_t * restrict       data_dst = uctx->data_dst;                                                             \
                                                                                                                    \
    const struct htp_tensor * src1 = (IS_RMS_NORM_MUL) ? octx->src[1] : NULL;                                       \
    const uint32_t nb11 = src1 ? src1->nb[1] : 0;                                                                   \
    const uint32_t nb12 = src1 ? src1->nb[2] : 0;                                                                   \
    const uint32_t nb13 = src1 ? src1->nb[3] : 0;                                                                   \
    const bool src1_contig = src1 ? ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12)) : false;       \
                                                                                                                    \
    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);                           \
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_thread)) : NULL;\
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);                             \
                                                                                                                    \
    size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;                                                         \
    size_t src1_vtcm_half_size = uctx->src1_vtcm_half_size;                                                         \
    size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;                                                          \
                                                                                                                    \
    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&                                                       \
                             (nb03 == (size_t)ne02 * nb02);                                                         \
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&                                                       \
                             (nb3  == (size_t)ne2  * nb2);                                                          \
                                                                                                                    \
    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;                                             \
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;                                             \
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;                                            \
                                                                                                                    \
    const uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);                   \
    const uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);                    \
    const uint32_t BLOCK = MIN(src0_max_block, dst_max_block);                                                      \
    if (BLOCK == 0) {                                                                                               \
        FARF(ERROR, "unary-f32 : current VTCM reservation %zu is too small, needed at least %zu\n",                 \
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);                                               \
        return;                                                                                                     \
    }                                                                                                               \
                                                                                                                    \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                    \
                                                                                                                    \
    if ((IS_RMS_NORM_MUL) && uctx->broadcast_weight) {                                                              \
        dma_queue_push(dma_queue, dma_make_ptr(src1_vtcm_data, data_src1),                                          \
                       uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);                                \
        dma_queue_flush(dma_queue);                                                                                 \
    }                                                                                                               \
                                                                                                                    \
    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {               \
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01,        \
                                                     div_ne01);                                                     \
                                                                                                                    \
        dma_queue_push(dma_queue,                                                                                   \
            dma_make_ptr(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),                                \
            nb1, dst_row_size_aligned, dst_data_row_size, 0);                                                       \
                                                                                                                    \
        const size_t src0_off = src0_contig ? (ir * nb01) :                                                         \
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);                      \
        dma_queue_push(dma_queue,                                                                                   \
            dma_make_ptr(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),                   \
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);                                           \
                                                                                                                    \
        if ((IS_RMS_NORM_MUL) && !uctx->broadcast_weight) {                                                         \
            const size_t src1_off = src1_contig ? (ir * nb11) :                                                     \
                unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11, nb12, nb13);                  \
            dma_queue_push(dma_queue,                                                                               \
                dma_make_ptr(src1_vtcm_data + (vtcm_idx * src1_vtcm_half_size), data_src1 + src1_off),              \
                uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, block_size);                           \
        }                                                                                                           \
                                                                                                                    \
        ir += block_size;                                                                                           \
    }                                                                                                               \
                                                                                                                    \
    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {                                                       \
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01,        \
                                                     div_ne01);                                                     \
                                                                                                                    \
        float * dst_vtcm  = (float *) dma_queue_pop(dma_queue).src;                                                 \
        float * src0_vtcm = (float *) dma_queue_pop(dma_queue).dst;                                                 \
        float * src1_vtcm = NULL;                                                                                   \
        if ((IS_RMS_NORM_MUL) && !uctx->broadcast_weight) {                                                         \
            src1_vtcm = (float *) dma_queue_pop(dma_queue).dst;                                                     \
        }                                                                                                           \
                                                                                                                    \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
        CORE_EXPR;                                                                                                  \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                       \
                                                                                                                    \
        const size_t dst_off = dst_contig ? (ir * nb1) :                                                            \
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);                           \
        dma_queue_push(dma_queue,                                                                                   \
            dma_make_ptr(data_dst + dst_off, dst_vtcm),                                                             \
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);                                              \
                                                                                                                    \
        const uint32_t next_ir = ir + block_size;                                                                   \
        if (next_ir < src0_end_row) {                                                                               \
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig, dst_contig,\
                                                              ne01, div_ne01);                                      \
            const uint32_t pref_ir = next_ir + next_block_size;                                                     \
            if (pref_ir < src0_end_row) {                                                                           \
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,        \
                                                                  dst_contig, ne01, div_ne01);                      \
                const size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :                                       \
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);         \
                dma_queue_push(dma_queue,                                                                           \
                    dma_make_ptr(src0_vtcm, data_src + src0_pref_off),                                              \
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);                              \
                                                                                                                    \
                if ((IS_RMS_NORM_MUL) && !uctx->broadcast_weight) {                                                 \
                    const size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :                                   \
                        unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11, nb12, nb13);     \
                    dma_queue_push(dma_queue,                                                                       \
                        dma_make_ptr(src1_vtcm, data_src1 + src1_pref_off),                                         \
                        uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, pref_block_size);              \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        ir += block_size;                                                                                           \
    }                                                                                                               \
                                                                                                                    \
    dma_queue_flush(dma_queue);                                                                                     \
}

DEFINE_UNARY_TASK(norm,           false, false, norm_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(rms_norm,       false, false, rms_norm_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(rms_norm_mul,   true,  false, rms_norm_mul_f32(src0_vtcm, uctx->broadcast_weight ? (const float *) src1_vtcm_data : src1_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(scale,          false, false, scale_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(sqr,            false, false, sqr_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(sqrt,           false, false, sqrt_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_neg,      false, false, neg_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_exp,      false, false, exp_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_sigmoid,  false, false, sigmoid_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_softplus, false, false, softplus_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_tanh,     false, false, tanh_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(l2_norm,        false, false, l2_norm_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(tri,            false, true,  tri_f32(src0_vtcm, dst_vtcm, block_size, ir, uctx))

// Apply a pointwise unary op to one column tile that is already in VTCM.
#define DEFINE_UNARY_TILED_TASK(NAME, IS_TRI, CORE_TILE_EXPR)                                                       \
static void unary_task_f32_tiled_##NAME(unsigned int nth, unsigned int ith, void * data) {                          \
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;                                \
    struct htp_ops_context * octx = uctx->octx;                                                                     \
    const struct htp_tensor * src = octx->src[0];                                                                   \
    const struct htp_tensor * dst = octx->dst;                                                                      \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                          \
                                                                                                                    \
    htp_unary_preamble;                                                                                             \
                                                                                                                    \
    int32_t *      op_params = octx->op_params;                                                                     \
    const uint32_t col_tile  = uctx->col_tile;                                                                      \
                                                                                                                    \
    const uint32_t src0_nrows     = uctx->src0_nrows;                                                               \
    const uint32_t src0_start_row = uctx->src0_nrows_per_thread * ith;                                              \
    const uint32_t src0_end_row   = MIN(src0_start_row + uctx->src0_nrows_per_thread, src0_nrows);                  \
                                                                                                                    \
    if (src0_start_row >= src0_end_row) {                                                                           \
        return;                                                                                                     \
    }                                                                                                               \
                                                                                                                    \
    const uint8_t * restrict data_src = uctx->data_src0;                                                            \
    uint8_t * restrict       data_dst = uctx->data_dst;                                                             \
                                                                                                                    \
    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);                           \
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);                             \
                                                                                                                    \
    const size_t src0_half = uctx->src0_vtcm_half_size;                                                             \
    const size_t dst_half  = uctx->dst_vtcm_half_size;                                                              \
                                                                                                                    \
    dma_queue * dmaq = octx->ctx->dma[ith];                                                                         \
                                                                                                                    \
    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;                                             \
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;                                             \
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;                                            \
    const struct fastdiv_values * div_tpr   = &uctx->kparams->div_tpr;                                              \
                                                                                                                    \
    const uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;                                                 \
    const int32_t  tri_ttype     = (IS_TRI) ? op_params[0] : 0;                                                     \
                                                                                                                    \
    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&                                                       \
                             (nb03 == (size_t)ne02 * nb02);                                                         \
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&                                                       \
                             (nb3  == (size_t)ne2  * nb2);                                                          \
                                                                                                                    \
    const uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;                                   \
                                                                                                                    \
    for (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {                          \
        const uint32_t row  = src0_start_row + t / tiles_per_row;                                                   \
        const uint32_t col  = (t % tiles_per_row) * col_tile;                                                       \
        const uint32_t tw   = MIN(col_tile, ne0 - col);                                                             \
        const size_t   tb   = (size_t) tw * sizeof(float);                                                          \
        const size_t   soff = (src0_contig ? (row * nb01) :                                                         \
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +\
                               (size_t) col * sizeof(float);                                                        \
                                                                                                                    \
        dma_queue_push(dmaq, dma_make_ptr(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);            \
        dma_queue_push(dmaq, dma_make_ptr(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);\
    }                                                                                                               \
                                                                                                                    \
    uint32_t row = src0_start_row;                                                                                  \
    uint32_t col = 0;                                                                                               \
    uint32_t tile_in_row = 0;                                                                                       \
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);                                                                 \
                                                                                                                    \
    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);                                                           \
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;                                               \
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);                                                  \
                                                                                                                    \
    for (uint32_t t = 0; t < total_tiles; t++) {                                                                    \
        uint8_t * dst_vtcm = (uint8_t *) dma_queue_pop(dmaq).src;                                                   \
        uint8_t * src_vtcm = (uint8_t *) dma_queue_pop(dmaq).dst;                                                   \
                                                                                                                    \
        const uint32_t tw  = MIN(col_tile, ne0 - col);                                                              \
                                                                                                                    \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);                                                       \
        CORE_TILE_EXPR;                                                                                             \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);                                                        \
                                                                                                                    \
        const size_t doff = (dst_contig ? (row * nb1) :                                                             \
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +       \
                             (size_t) col * sizeof(float);                                                          \
        const size_t tb   = (size_t) tw * sizeof(float);                                                            \
        dma_queue_push(dmaq, dma_make_ptr(data_dst + doff, dst_vtcm), tb, tb, tb, 1);                               \
                                                                                                                    \
        const uint32_t pt = t + 2;                                                                                  \
        if (pt < total_tiles) {                                                                                     \
            const uint32_t ptw  = MIN(col_tile, ne0 - pcol);                                                        \
            const size_t   ptb  = (size_t) ptw * sizeof(float);                                                     \
            const size_t   psoff = (src0_contig ? (prow * nb01) :                                                   \
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,   \
                                                     nb03)) +                                                       \
                                   (size_t) pcol * sizeof(float);                                                   \
            dma_queue_push(dmaq, dma_make_ptr(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);                       \
        }                                                                                                           \
                                                                                                                    \
        tile_in_row++;                                                                                              \
        col += col_tile;                                                                                            \
        if (tile_in_row == tiles_per_row) {                                                                         \
            tile_in_row = 0;                                                                                        \
            col = 0;                                                                                                \
            row++;                                                                                                  \
            i01++;                                                                                                  \
            if (i01 == ne01) {                                                                                      \
                i01 = 0;                                                                                            \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        ptile_in_row++;                                                                                             \
        pcol += col_tile;                                                                                           \
        if (ptile_in_row == tiles_per_row) {                                                                        \
            ptile_in_row = 0;                                                                                       \
            pcol = 0;                                                                                               \
            prow++;                                                                                                 \
        }                                                                                                           \
    }                                                                                                               \
                                                                                                                    \
    dma_queue_flush(dmaq);                                                                                          \
}

static inline void tile_scale_f32(uint8_t * dst_vtcm, const uint8_t * src_vtcm, uint32_t tw, const int32_t * op_params) {
    float scale = 0.f;
    float bias = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));
    hvx_scale_offset_f32_aa(dst_vtcm, src_vtcm, tw, scale, bias);
}

static inline void tile_unary_softplus_f32(uint8_t * dst_vtcm, const uint8_t * src_vtcm, uint32_t tw) {
    const float * restrict sf = (const float *) src_vtcm;
    float * restrict df       = (float *) dst_vtcm;
    for (uint32_t i = 0; i < tw; i++) {
        float x = sf[i];
        df[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
    }
}

// Triangular mask applied to one column tile. Boundary is an absolute column index, so
// each vector compares against its absolute column position (col_start + i*VLEN_FP32).
static inline void tri_apply_tile_f32(const uint8_t * restrict src, uint8_t * restrict dst,
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

DEFINE_UNARY_TILED_TASK(scale,          false, tile_scale_f32(dst_vtcm, src_vtcm, tw, op_params))
DEFINE_UNARY_TILED_TASK(sqr,            false, hvx_sqr_f32_aa(dst_vtcm, src_vtcm, tw))
DEFINE_UNARY_TILED_TASK(sqrt,           false, hvx_sqrt_f32_aa(dst_vtcm, src_vtcm, tw))
DEFINE_UNARY_TILED_TASK(unary_neg,      false, hvx_scale_f32_aa(dst_vtcm, src_vtcm, tw, -1.0f))
DEFINE_UNARY_TILED_TASK(unary_exp,      false, hvx_exp_f32(dst_vtcm, src_vtcm, tw, false))
DEFINE_UNARY_TILED_TASK(unary_sigmoid,  false, hvx_sigmoid_f32_aa(dst_vtcm, src_vtcm, tw))
DEFINE_UNARY_TILED_TASK(unary_softplus, false, tile_unary_softplus_f32(dst_vtcm, src_vtcm, tw))
DEFINE_UNARY_TILED_TASK(unary_tanh,     false, hvx_tanh_f32_aa(dst_vtcm, src_vtcm, tw))
DEFINE_UNARY_TILED_TASK(tri,            true,  tri_apply_tile_f32(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype))

static int execute_op_unary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const char * op_type = NULL;

    switch (octx->op) {
        case HTP_OP_NORM:            op_type = "norm-f32";         break;
        case HTP_OP_RMS_NORM:        op_type = "rmsnorm-f32";      break;
        case HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";  break;
        case HTP_OP_SCALE:           op_type = "scale-f32";        break;
        case HTP_OP_SQR:             op_type = "sqr-f32";          break;
        case HTP_OP_SQRT:            op_type = "sqrt-f32";         break;
        case HTP_OP_UNARY_NEG:       op_type = "neg-f32";          break;
        case HTP_OP_UNARY_EXP:       op_type = "exp-f32";          break;
        case HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";      break;
        case HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";     break;
        case HTP_OP_UNARY_TANH:      op_type = "tanh-f32";         break;
        case HTP_OP_L2_NORM:         op_type = "l2norm-f32";       break;
        case HTP_OP_TRI:             op_type = "tri-f32";          break;

        default:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_unary_kernel_params * kparams = (const struct htp_unary_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = kparams->n_threads;

    const size_t src0_data_row_size = src0->ne[0] * sizeof(float);
    const size_t dst_data_row_size  = dst->ne[0]  * sizeof(float);

    const size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    const uint32_t col_tile = kparams->col_tile;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    bool broadcast_weight = kparams->broadcast_weight;
    const struct htp_tensor * src1 = NULL;

    if (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_data_row_size = src1->ne[0] * sizeof(float);
    }

    if (octx->ctx->vtcm_size < (size_t)kparams->vtcm_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, (size_t)kparams->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    FARF(HIGH, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_size, kparams->vtcm_src1_size, kparams->vtcm_dst_size);

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
        struct htp_unary_context uctx = {
            .octx                  = octx,
            .kparams               = kparams,
            .src0_nrows_per_thread = (src0_nrows + n_threads - 1) / n_threads,
            .src0_nrows            = src0_nrows,

            .data_src0             = (const uint8_t *)src0->data,
            .data_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? (const uint8_t *)src1->data : NULL,
            .data_dst              = (uint8_t *)dst->data,

            .src0_data_row_size    = src0_data_row_size,
            .src1_data_row_size    = src1_data_row_size,
            .dst_data_row_size     = dst_data_row_size,

            .src0_row_size_aligned = src0_row_size_aligned,
            .src1_row_size_aligned = src1_row_size_aligned,
            .dst_row_size_aligned  = dst_row_size_aligned,

            .src0_vtcm_half_size   = kparams->vtcm_src0_size_per_thread / 2,
            .src1_vtcm_half_size   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_size_per_thread / (broadcast_weight ? 1 : 2)) : 0,
            .dst_vtcm_half_size    = kparams->vtcm_dst_size_per_thread / 2,

            .block                 = kparams->block,
            .nc                    = src0->ne[0],
            .col_tile              = (uint32_t) kparams->col_tile,
            .broadcast_weight      = broadcast_weight,

            .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, base, 0),
            .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, kparams->vtcm_src0_size, kparams->vtcm_src1_size > 0),
            .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, base, kparams->vtcm_src0_size + kparams->vtcm_src1_size),

            .vtcm_src0_size_per_thread = kparams->vtcm_src0_size_per_thread,
            .vtcm_src1_size_per_thread = kparams->vtcm_src1_size_per_thread,
            .vtcm_dst_size_per_thread  = kparams->vtcm_dst_size_per_thread,
        };

        FARF(HIGH, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "tiled" : "row-block", col_tile);

        worker_callback_t task_func = NULL;
        if (col_tile) {
            switch (octx->op) {
                case HTP_OP_SCALE:           task_func = unary_task_f32_tiled_scale;          break;
                case HTP_OP_SQR:             task_func = unary_task_f32_tiled_sqr;            break;
                case HTP_OP_SQRT:            task_func = unary_task_f32_tiled_sqrt;           break;
                case HTP_OP_UNARY_NEG:       task_func = unary_task_f32_tiled_unary_neg;      break;
                case HTP_OP_UNARY_EXP:       task_func = unary_task_f32_tiled_unary_exp;      break;
                case HTP_OP_UNARY_SIGMOID:   task_func = unary_task_f32_tiled_unary_sigmoid;  break;
                case HTP_OP_UNARY_SOFTPLUS:  task_func = unary_task_f32_tiled_unary_softplus; break;
                case HTP_OP_UNARY_TANH:      task_func = unary_task_f32_tiled_unary_tanh;     break;
                case HTP_OP_TRI:             task_func = unary_task_f32_tiled_tri;            break;
                default:                     break;
            }
        } else {
            switch (octx->op) {
                case HTP_OP_NORM:            task_func = unary_task_f32_norm;                 break;
                case HTP_OP_RMS_NORM:        task_func = unary_task_f32_rms_norm;             break;
                case HTP_OP_RMS_NORM_MUL:    task_func = unary_task_f32_rms_norm_mul;         break;
                case HTP_OP_SCALE:           task_func = unary_task_f32_scale;                break;
                case HTP_OP_SQR:             task_func = unary_task_f32_sqr;                  break;
                case HTP_OP_SQRT:            task_func = unary_task_f32_sqrt;                 break;
                case HTP_OP_UNARY_NEG:       task_func = unary_task_f32_unary_neg;            break;
                case HTP_OP_UNARY_EXP:       task_func = unary_task_f32_unary_exp;            break;
                case HTP_OP_UNARY_SIGMOID:   task_func = unary_task_f32_unary_sigmoid;        break;
                case HTP_OP_UNARY_SOFTPLUS:  task_func = unary_task_f32_unary_softplus;       break;
                case HTP_OP_UNARY_TANH:      task_func = unary_task_f32_unary_tanh;           break;
                case HTP_OP_L2_NORM:         task_func = unary_task_f32_l2_norm;              break;
                case HTP_OP_TRI:             task_func = unary_task_f32_tri;                  break;
                default:                     break;
            }
        }

        if (task_func) {
            worker_pool_run_func(octx->ctx->worker_pool, task_func, &uctx, n_threads);
        } else {
            FARF(ERROR, "execute_op_unary_f32: task function is NULL for op %d\n", octx->op);
            err = HTP_STATUS_NO_SUPPORT;
        }
    }

    return err;
}

int op_unary(struct htp_ops_context * octx) {
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            return execute_op_unary_f32(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
