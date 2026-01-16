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
                    // src0: window starting at position i2, element at window offset i0
                    // src0 layout: {d_conv - 1 + n_t, d_inner, n_seqs}
                    const uint32_t src0_idx = (i2 + i0) + i1 * src0_stride_inner + i3 * src0_stride_seq;
                    // src1: conv weight at position i0, inner dim i1
                    // src1 layout: {d_conv, d_inner}
                    const uint32_t src1_idx = i0 + i1 * src1_stride_inner;

                    sumf += src0_data[src0_idx] * src1_data[src1_idx];
                }

                // dst: inner dim i1, token i2, sequence i3
                // dst layout: {d_inner, n_t, n_seqs}
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

// HVX FP32 SSM_CONV implementation
// Vectorizes across d_inner dimension, processing 32 inner dims at once
static void ssm_conv_thread_f32_f32_hvx(struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];
    const uint32_t n_s     = dst->ne[2];

    const uint32_t src0_stride_inner = src0->nb[1] / sizeof(float);  // stride for inner dimension
    const uint32_t src0_stride_seq   = src0->nb[2] / sizeof(float);  // stride for sequence dimension
    const uint32_t src1_stride_inner = src1->nb[1] / sizeof(float);  // stride for inner dimension
    const uint32_t dst_stride_token  = dst->nb[1] / sizeof(float);   // stride for token dimension
    const uint32_t dst_stride_seq    = dst->nb[2] / sizeof(float);   // stride for sequence dimension

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float *       dst_data  = (float *) dst->data;

    // Calculate row range for this thread
    const uint32_t d_inner_per_thread = (d_inner + nth - 1) / nth;
    const uint32_t d_inner_start      = d_inner_per_thread * ith;
    const uint32_t d_inner_end        = MIN(d_inner_start + d_inner_per_thread, d_inner);

    if (d_inner_start >= d_inner_end) {
        return;  // No work for this thread
    }

    // Align start to VLEN_FP32 boundary
    const uint32_t d_inner_vec_start = (d_inner_start + VLEN_FP32 - 1) & ~(VLEN_FP32 - 1);
    const uint32_t d_inner_vec_end   = d_inner_end & ~(VLEN_FP32 - 1);

    // Per sequence
    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        // Per token
        for (uint32_t i2 = 0; i2 < n_t; ++i2) {
            // Handle scalar remainder at the beginning (when start is not aligned)
            for (uint32_t i1 = d_inner_start; i1 < MIN(d_inner_vec_start, d_inner_end); ++i1) {
                float sumf = 0.0f;
                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    const uint32_t src0_idx = (i2 + i0) + i1 * src0_stride_inner + i3 * src0_stride_seq;
                    const uint32_t src1_idx = i0 + i1 * src1_stride_inner;
                    sumf += src0_data[src0_idx] * src1_data[src1_idx];
                }
                const uint32_t dst_idx = i1 + i2 * dst_stride_token + i3 * dst_stride_seq;
                dst_data[dst_idx]      = sumf;
            }

            for (uint32_t i1_vec = d_inner_vec_start; i1_vec < d_inner_vec_end; i1_vec += VLEN_FP32) {
                HVX_Vector acc_vec = Q6_V_vzero();

                // Per kernel element
                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    // Load 32 elements from src0: window at position (i2+i0), inner dims [i1_vec, i1_vec+32)
                    const float * src0_ptr = src0_data + (i2 + i0) + i1_vec * src0_stride_inner + i3 * src0_stride_seq;
                    HVX_Vector    src0_vec = *(const HVX_Vector *) src0_ptr;

                    // Load 32 elements from src1: kernel at position i0, inner dims [i1_vec, i1_vec+32)
                    const float * src1_ptr = src1_data + i0 + i1_vec * src1_stride_inner;
                    HVX_Vector    src1_vec = *(const HVX_Vector *) src1_ptr;

                    HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(src0_vec, src1_vec);
                    acc_vec         = Q6_Vqf32_vadd_Vqf32Vqf32(acc_vec, prod);
                }

                HVX_Vector result_vec   = Q6_Vsf_equals_Vqf32(acc_vec);
                float *    dst_ptr      = dst_data + i1_vec + i2 * dst_stride_token + i3 * dst_stride_seq;
                *(HVX_Vector *) dst_ptr = result_vec;
            }

            // Handle scalar remainder at the end (if end is not aligned)
            for (uint32_t i1 = d_inner_vec_end; i1 < d_inner_end; ++i1) {
                float sumf = 0.0f;
                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    const uint32_t src0_idx = (i2 + i0) + i1 * src0_stride_inner + i3 * src0_stride_seq;
                    const uint32_t src1_idx = i0 + i1 * src1_stride_inner;
                    sumf += src0_data[src0_idx] * src1_data[src1_idx];
                }
                const uint32_t dst_idx = i1 + i2 * dst_stride_token + i3 * dst_stride_seq;
                dst_data[dst_idx]      = sumf;
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "ssm-conv-f32-hvx %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], d_inner_start, d_inner_end,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void ssm_conv_work_f32_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;
    ssm_conv_thread_f32_f32(octx, n, i);
}

static void ssm_conv_work_f32_f32_hvx(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;
    ssm_conv_thread_f32_f32_hvx(octx, n, i);
}

int op_ssm_conv_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "ssm_conv: only (F32 x F32 -> F32) OPs supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t nc  = src1->ne[0];   // d_conv
    const uint32_t ncs = src0->ne[0];   // d_conv - 1 + n_t
    const int nr  = src0->ne[1];        // d_inner
    const int n_t =  dst->ne[1];        // tokens per sequence
    const int n_s =  dst->ne[2];        // number of sequences in the batch

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        const uint32_t n_jobs = MIN(octx->n_threads, nr);

        const uint32_t src0_stride_inner = src0->nb[1] / sizeof(float);
        const uint32_t src1_stride_inner = src1->nb[1] / sizeof(float);

        int use_hvx = 0;
        if (nr >= VLEN_FP32) {
            int is_aligned = hex_is_aligned((void *) src0->data, VLEN) &&
                             hex_is_aligned((void *) src1->data, VLEN) &&
                             hex_is_aligned((void *) dst->data, VLEN);

            int strides_aligned = !(src0_stride_inner & (VLEN_FP32 - 1)) && !(src1_stride_inner & (VLEN_FP32 - 1));

            if (is_aligned && strides_aligned) {
                use_hvx = 1;
            }

        FARF(HIGH, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : use_hvx %d is_aligned %d strides_aligned %d\n",
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], use_hvx, is_aligned, strides_aligned);
        }

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
