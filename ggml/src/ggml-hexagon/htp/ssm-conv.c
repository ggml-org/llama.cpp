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

#define htp_ssm_conv_tensors_preamble                        \
    struct htp_tensor * restrict src0    = &octx->src0;      \
    struct htp_tensor * restrict src1    = &octx->src1;      \
    struct htp_tensor * restrict dst     = &octx->dst;       \
    struct htp_spad * restrict src0_spad = &octx->src0_spad; \
    struct htp_spad * restrict src1_spad = &octx->src1_spad; \
    struct htp_spad * restrict dst_spad  = &octx->dst_spad;  \
                                                             \
    const uint32_t ne00 = src0->ne[0];                       \
    const uint32_t ne01 = src0->ne[1];                       \
    const uint32_t ne02 = src0->ne[2];                       \
    const uint32_t ne03 = src0->ne[3];                       \
                                                             \
    const uint32_t ne10 = src1->ne[0];                       \
    const uint32_t ne11 = src1->ne[1];                       \
    const uint32_t ne12 = src1->ne[2];                       \
    const uint32_t ne13 = src1->ne[3];                       \
                                                             \
    const uint32_t ne0 = dst->ne[0];                         \
    const uint32_t ne1 = dst->ne[1];                         \
    const uint32_t ne2 = dst->ne[2];                         \
    const uint32_t ne3 = dst->ne[3];                         \
                                                             \
    const uint32_t nb00 = src0->nb[0];                       \
    const uint32_t nb01 = src0->nb[1];                       \
    const uint32_t nb02 = src0->nb[2];                       \
    const uint32_t nb03 = src0->nb[3];                       \
                                                             \
    const uint32_t nb10 = src1->nb[0];                       \
    const uint32_t nb11 = src1->nb[1];                       \
    const uint32_t nb12 = src1->nb[2];                       \
    const uint32_t nb13 = src1->nb[3];                       \
                                                             \
    const uint32_t nb0 = dst->nb[0];                         \
    const uint32_t nb1 = dst->nb[1];                         \
    const uint32_t nb2 = dst->nb[2];                         \
    const uint32_t nb3 = dst->nb[3];

struct htp_ssm_conv_context {
    struct htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint64_t t_start;
};

#define htp_ssm_conv_preamble                            \
    struct htp_ssm_conv_context * scctx = (struct htp_ssm_conv_context *) data; \
    struct htp_ops_context * octx = scctx->octx;         \
    htp_ssm_conv_tensors_preamble;                       \
    dma_queue * dma_queue         = octx->ctx->dma[ith];

// Scalar FP32 SSM_CONV implementation
static void ssm_conv_thread_f32_f32(unsigned int nth, unsigned int ith, void *data) {
    htp_ssm_conv_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];
    const uint32_t n_s     = dst->ne[2];

    const uint32_t src0_stride_inner = src0->nb[1] / sizeof(float); // stride for inner dimension
    const uint32_t src0_stride_seq   = src0->nb[2] / sizeof(float); // stride for sequence dimension
    const uint32_t src1_stride_inner = src1->nb[1] / sizeof(float); // stride for inner dimension
    const uint32_t dst_stride_token  = dst->nb[1]  / sizeof(float); // stride for token dimension
    const uint32_t dst_stride_seq    = dst->nb[2]  / sizeof(float); // stride for sequence dimension

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float *       dst_data  = (float *) dst->data;

    // Calculate row range for this thread
    const uint32_t d_inner_per_thread = scctx->nrows_per_thread;
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
static void ssm_conv_thread_f32_f32_hvx(unsigned int nth, unsigned int ith, void *data) {
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
    const int dr = scctx->nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, d_inner);
    const uint32_t ir  = ir1 - ir0;

    if (ir0 >= ir1) {
        return;  // No work for this thread
    }

    // src0 and src1 gather offsets
    uint32_t __attribute__((aligned(VLEN))) src0_offsets[VLEN_FP32] = { 0 };
    uint32_t __attribute__((aligned(VLEN))) src1_offsets[VLEN_FP32] = { 0 };

    for (uint32_t i = 0; i < VLEN_FP32; ++i) {
        src0_offsets[i] = i * (ncs)    * sizeof(float);
        src1_offsets[i] = i * (d_conv) * sizeof(float);
    }

    const uint32_t src0_gather_len = VLEN * ncs;
    const uint32_t src1_gather_len = VLEN * d_conv;

    // gather scratchpads
    HVX_Vector * src0_vec = (HVX_Vector *) (octx->ctx->vtcm_base + ith * VLEN*2 + 0);
    HVX_Vector * src1_vec = (HVX_Vector *) (octx->ctx->vtcm_base + ith * VLEN*2 + VLEN);

    float * data_src0 = (float *) ((char *) src0->data + ir0 * src0->nb[1]);
    float * data_src1 = (float *) ((char *) src1->data + ir0 * src1->nb[1]);

    uint8_t * spad_src0 = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad_src1 = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    // copy src1 workload to VTCM
    dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src1, data_src1), nb11, nb11, ir);

    // FARF(HIGH, "ssm-conv-src1-fetch %d: ir0 %u size %u\n", ith, ir0, nb11 * ir);

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        float * src0_data_ptr = (float *) ((char *) data_src0 + i3 * (src0->nb[2]));

        // copy src0 workload to VTCM
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0, src0_data_ptr), nb01, nb01, ir);

        // FARF(HIGH, "ssm-conv-src0-fetch %d: ir0 %u i3 %u size %u\n", ith, ir0, i3, nb01 * ir);

        dma_queue_flush(dma_queue);

        for (uint32_t i2 = 0; i2 < n_t; ++i2) {
            float * dst_ptr = (float *) ((char *) dst->data + ir0 * (dst->nb[0]) + i2 * (dst->nb[1]) + i3 * (dst->nb[2]));

            const uint32_t nvec = ir / VLEN_FP32;
            const uint32_t nloe = ir % VLEN_FP32;
            uint32_t i1 = 0;

            for (uint32_t vi1 = 0; vi1 < nvec; vi1++) {
                HVX_Vector acc_vec = Q6_V_vsplat_R(0);

                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    uint32_t src0_base = (uint32_t) spad_src0 + (i0 + i1 * ncs) * sizeof(float) + i2 * (src0->nb[0]);
                    uint32_t src1_base = (uint32_t) spad_src1 + (i0 + i1 * nc)  * sizeof(float);
                    Q6_vgather_ARMVw(src0_vec, src0_base, src0_gather_len, (*(const HVX_Vector *) src0_offsets));
                    Q6_vgather_ARMVw(src1_vec, src1_base, src1_gather_len, (*(const HVX_Vector *) src1_offsets));

                    HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *) src0_vec, *(const HVX_Vector *) src1_vec);
                    acc_vec = Q6_Vqf32_vadd_Vqf32Vqf32(acc_vec, prod);
                }

                *(HVX_UVector *) (dst_ptr + i1) = Q6_Vsf_equals_Vqf32(acc_vec);
                i1 += VLEN_FP32;
            }

            if (nloe) {
                HVX_Vector acc_vec = Q6_V_vsplat_R(0);

                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    uint32_t src0_base = (uint32_t) spad_src0 + (i0 + i1 * ncs) * sizeof(float) + i2 * (src0->nb[0]);
                    uint32_t src1_base = (uint32_t) spad_src1 + (i0 + i1 * nc)  * sizeof(float);
                    Q6_vgather_ARMVw(src0_vec, src0_base, src0_gather_len, (*(const HVX_Vector *) src0_offsets));
                    Q6_vgather_ARMVw(src1_vec, src1_base, src1_gather_len, (*(const HVX_Vector *) src1_offsets));

                    HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *) src0_vec, *(const HVX_Vector *) src1_vec);
                    acc_vec = Q6_Vqf32_vadd_Vqf32Vqf32(acc_vec, prod);
                }

                hvx_vec_store_u(dst_ptr + i1, (ir - i1) * 4, Q6_Vsf_equals_Vqf32(acc_vec));
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "ssm-conv-f32-hvx %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

int op_ssm_conv_f32(struct htp_ops_context * octx) {
    htp_ssm_conv_tensors_preamble;

    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "ssm_conv: only (F32 x F32 -> F32) OPs supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_ssm_conv_context scctx = { 0 };
    scctx.octx = octx;

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];  // tokens per sequence
    const uint32_t n_s     = dst->ne[2];  // number of sequences in the batch

    const uint32_t n_threads = MIN(octx->n_threads, d_inner);

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t use_hvx = 0;
        if (d_inner >= VLEN_FP32 && d_inner % VLEN_FP32 == 0) {
            int is_aligned = hex_is_aligned((void *) src0->data, VLEN) &&
                             hex_is_aligned((void *) src1->data, VLEN) &&
                             hex_is_aligned((void *) dst->data, VLEN);

            if (is_aligned) {
                use_hvx = 1;
            }
        }

        if (use_hvx) {
            scctx.nrows_per_thread  = (d_inner + n_threads - 1) / n_threads; // d_inner chunks per thread
            scctx.nrows_per_thread += (scctx.nrows_per_thread & 1); // round up to even

            octx->src0_spad.size_per_thread = hex_round_up(scctx.nrows_per_thread * nb01, 256);
            octx->src1_spad.size_per_thread = hex_round_up(scctx.nrows_per_thread * nb11, 256);
            octx->dst_spad.size_per_thread  = hex_round_up(scctx.nrows_per_thread * sizeof(float), 256);

            octx->src0_spad.size = octx->src0_spad.size_per_thread * n_threads;
            octx->src1_spad.size = octx->src1_spad.size_per_thread * n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread  * n_threads;

            // Compute gather scratchpad size for src0 and src1
            const size_t gather_spad_size = n_threads * VLEN * 2;

            octx->src0_spad.data = octx->ctx->vtcm_base + gather_spad_size;
            octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
            octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

            FARF(HIGH, "ssm_conv-f32: gather-spad:%zu spad-per-thread:(%u:%u:%u) spad-sizes:(%u:%u:%u) spad-data:(%p:%p:%p)\n",
                gather_spad_size, octx->src0_spad.size_per_thread, octx->src1_spad.size_per_thread,
                octx->dst_spad.size_per_thread, octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size,
                octx->src0_spad.data, octx->src1_spad.data, octx->dst_spad.data);

            const size_t total_spad_size =
                gather_spad_size + octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size;

            if (total_spad_size > octx->ctx->vtcm_size) {
                FARF(HIGH, "ssm_conv-f32: HVX scratchpad size %zu exceeds VTCM size %zu", total_spad_size,
                     octx->ctx->vtcm_size);
                use_hvx = 0;
            }
        }

        FARF(HIGH, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : use_hvx %d\n", src0->ne[0],
             src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
             dst->ne[1], dst->ne[2], dst->ne[3], use_hvx);

        if (use_hvx) {
            worker_pool_run_func(octx->ctx->worker_pool, ssm_conv_thread_f32_f32_hvx, &scctx, n_threads);
        } else {
            worker_pool_run_func(octx->ctx->worker_pool, ssm_conv_thread_f32_f32, &scctx, n_threads);
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
