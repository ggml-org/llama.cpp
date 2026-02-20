#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_ps.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <qurt_thread.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "hex-dma.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"

#include "hvx-dump.h"

#define htp_ssm_conv_tensors_preamble    \
    struct htp_tensor * restrict src0    = &octx->src0;      \
    struct htp_tensor * restrict src1    = &octx->src1;      \
    struct htp_tensor * restrict dst     = &octx->dst;       \
    struct htp_spad * restrict src0_spad = &octx->src0_spad; \
    struct htp_spad * restrict src1_spad = &octx->src1_spad; \
    struct htp_spad * restrict dst_spad  = &octx->dst_spad;  \
                                                             \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne10 = src1->ne[0]; \
    const uint32_t ne11 = src1->ne[1]; \
    const uint32_t ne12 = src1->ne[2]; \
    const uint32_t ne13 = src1->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb10 = src1->nb[0]; \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

#define htp_ssm_conv_preamble            \
    htp_ssm_conv_tensors_preamble;       \
    dma_queue *dma_queue           = octx->ctx->dma[ith];         \
    uint32_t src0_nrows_per_thread = octx->src0_nrows_per_thread;

// Scalar FP32 SSM_CONV implementation
static void ssm_conv_thread_f32_f32(struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    const struct htp_tensor * src0 = &octx->src0;  // conv_x input   -> {d_conv - 1 + n_t, d_inner, n_seqs}
    const struct htp_tensor * src1 = &octx->src1;  // conv1d weights -> {d_conv, d_inner}
    struct htp_tensor *       dst  = &octx->dst;   // output -> {d_inner, n_t, n_seqs}

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];
    const uint32_t n_s     = dst->ne[2];

    const uint32_t src0_stride_inner = src0->nb[1] / sizeof(float); // stride for inner dimension
    const uint32_t src0_stride_seq = src0->nb[2] / sizeof(float);   // stride for sequence dimension
    const uint32_t src1_stride_inner = src1->nb[1] / sizeof(float); // stride for inner dimension
    const uint32_t dst_stride_token = dst->nb[1] / sizeof(float);   // stride for token dimension
    const uint32_t dst_stride_seq = dst->nb[2] / sizeof(float);     // stride for sequence dimension

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float *       dst_data  = (float *) dst->data;

    // Calculate row range for this thread
    const uint32_t d_inner_per_thread = (d_inner + nth - 1) / nth;
    const uint32_t d_inner_start = d_inner_per_thread * ith;
    const uint32_t d_inner_end   = MIN(d_inner_start + d_inner_per_thread, d_inner);

    // No work for this thread
    if (d_inner_start >= d_inner_end) {
        return;
    }

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        for (uint32_t i2 = 0; i2 < n_t; ++i2) {
            for (uint32_t i1 = d_inner_start; i1 < d_inner_end; ++i1) {
                float sumf = 0.0f;

                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    const uint32_t src0_idx = (i2 + i0) + i1 * src0_stride_inner + i3 * src0_stride_seq;
                    const uint32_t src1_idx = i0 + i1 * src1_stride_inner;

                    sumf += src0_data[src0_idx] * src1_data[src1_idx];
                }

                const uint32_t dst_idx = i1 + i2 * dst_stride_token + i3 * dst_stride_seq;
                dst_data[dst_idx] = sumf;
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "ssm-conv-f32 %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], d_inner_start, d_inner_end,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// HVX FP32 SSM_CONV implementation - vectorizes across d_inner dimension
static void ssm_conv_thread_f32_f32_hvx(struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    htp_ssm_conv_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const int nc  = src1->ne[0]; // d_conv
    const int ncs = src0->ne[0]; // d_conv - 1 + n_t

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];
    const uint32_t n_s     = dst->ne[2];

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float *       dst_data  = (float *) dst->data;

    // Calculate row range for this thread
    const int dr = (d_inner + nth - 1) / nth;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, d_inner);
    const int      ir  = ir1 - ir0;

    if (ir0 >= ir1) {
        return;  // No work for this thread
    }

    // src0 gather offsets
    uint32_t src0_offsets[VLEN_FP32] = { 0 };
    for (uint32_t i = 0; i < VLEN_FP32; ++i) {
        src0_offsets[i] = i * (ncs) * sizeof(float);
    }
    uint32_t src0_gather_len = VLEN * ncs;

    // src1 gather offsets
    uint32_t src1_offsets[VLEN_FP32] = { 0 };
    for (uint32_t i = 0; i < VLEN_FP32; ++i) {
        src1_offsets[i] = i * (d_conv) * sizeof(float);
    }
    uint32_t src1_gather_len = VLEN * d_conv;

    HVX_Vector * src0_vec = (HVX_Vector *) (octx->ctx->vtcm_base + ith * VLEN);
    HVX_Vector * src1_vec = (HVX_Vector *) (octx->ctx->vtcm_base + 1024 + ith * VLEN);

    float * data_src0 = (float *) ((char *) src0->data + ir0*(src0->nb[1]));
    float * data_src1 = (float *) ((char *) src1->data + ir0*(src1->nb[1]));

    uint8_t * spad_src0 = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad_src1 = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    memcpy(spad_src1, data_src1, octx->src1_spad.size_per_thread);

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        float * src0_data_ptr = (float *) ((char *) data_src0 + i3 * (src0->nb[2]));

        memcpy(spad_src0, src0_data_ptr, octx->src0_spad.size_per_thread);

        for (uint32_t i2 = 0; i2 < n_t; ++i2) {
            for (uint32_t i1 = 0; i1 < ir; i1 += VLEN_FP32) {
                HVX_Vector acc_vec = Q6_V_vzero();

                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    Q6_vgather_ARMVw(src0_vec,
                                     SCATTER_TYPE(spad_src0 + (i0 + i1 * ncs) * sizeof(float) + i2 * (src0->nb[0])),
                                     src0_gather_len, (*(const HVX_Vector *) src0_offsets));
                    Q6_vgather_ARMVw(src1_vec,
                                     SCATTER_TYPE(spad_src1 + (i0 + i1 * nc) * sizeof(float)),
                                     src1_gather_len, (*(const HVX_Vector *) src1_offsets));

                    HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *) src0_vec, *(const HVX_Vector *) src1_vec);
                    acc_vec         = Q6_Vqf32_vadd_Vqf32Vqf32(acc_vec, prod);
                }

                float * dst_ptr = (float *) ((char *) dst->data  + i1*sizeof(float) + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2]));
                HVX_Vector result_vec   = Q6_Vsf_equals_Vqf32(acc_vec);
                *(HVX_Vector *) dst_ptr = result_vec;
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "ssm-conv-f32-hvx %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void ssm_conv_work_f32_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;
    ssm_conv_thread_f32_f32(octx, nth, ith);
}

static void ssm_conv_work_f32_f32_hvx(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;
    ssm_conv_thread_f32_f32_hvx(octx, nth, ith);
}

int op_ssm_conv_f32(struct htp_ops_context * octx) {
    htp_ssm_conv_tensors_preamble;

    assert(sizeof(float) == SIZEOF_FP32);

    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "ssm_conv: only (F32 x F32 -> F32) OPs supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];  // tokens per sequence
    const uint32_t n_s     = dst->ne[2];  // number of sequences in the batch

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        const uint32_t n_jobs = MIN(octx->n_threads, d_inner);

        int use_hvx = 0;
        if (d_inner >= VLEN_FP32 && d_inner % VLEN_FP32 == 0) {
            int is_aligned = hex_is_aligned((void *) src0->data, VLEN) &&
                             hex_is_aligned((void *) src1->data, VLEN) &&
                             hex_is_aligned((void *) dst->data, VLEN);

            if (is_aligned && n_t > 3) {
                use_hvx = 1;
            }
        }

        FARF(HIGH, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : use_hvx %d\n",
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], use_hvx);

        // chunks per thread
        const int dr = (d_inner + n_jobs - 1) / n_jobs;

        octx->dst_spad.size_per_thread  = hex_round_up(dr * ne1 * ne2 * ne3 * sizeof(float), 256);
        octx->src0_spad.size_per_thread = hex_round_up(ne00 * dr * sizeof(float), 256);
        octx->src1_spad.size_per_thread = hex_round_up(ne10 * dr * sizeof(float), 256);

        octx->src1_spad.size = octx->src1_spad.size_per_thread * octx->n_threads;
        octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
        octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;

        octx->src0_spad.data = octx->ctx->vtcm_base + 2048;
        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
        octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

        FARF(HIGH, "ssm_conv-f32: spad-per-thread:(%u:%u:%u) spad-sizes:(%u:%u:%u) spad-data:(%p:%p:%p)\n",
             octx->src0_spad.size_per_thread, octx->src1_spad.size_per_thread, octx->dst_spad.size_per_thread,
             octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size, octx->src0_spad.data,
             octx->src1_spad.data, octx->dst_spad.data);

        if (use_hvx) {
            worker_pool_run_func(octx->ctx->worker_pool, ssm_conv_work_f32_f32_hvx, octx, n_jobs);
        } else {
            worker_pool_run_func(octx->ctx->worker_pool, ssm_conv_work_f32_f32, octx, n_jobs);
        }
    }

    return HTP_STATUS_OK;
}

int op_ssm_conv(struct htp_ops_context * octx) {
    int                 err = HTP_STATUS_OK;
    struct htp_tensor * dst = &octx->dst;

    switch (dst->type) {
        case HTP_TYPE_F32:
            err = op_ssm_conv_f32(octx);
            break;
        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
