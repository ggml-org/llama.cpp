// HVX + DMA kernel for 5x5 stride-1 pad-2 dilation-1 depthwise conv (f16 weights, f32 activations).

#include "conv2d-ops.h"
#include "dma-queue.h"
#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-trace.h"
#include "hvx-base.h"
#include "hvx-copy.h"

#include <HAP_farf.h>

// Kernel parameters derived on entry from htp_conv2d_dw_kernel_params (solved on the host).
struct htp_conv_2d_dw5_kernel_params {
    uint32_t n_threads;
    uint32_t planes;
    uint32_t src_row_stride;
    uint32_t src_slot_bytes;
    uint32_t weight_slot_bytes;
    uint32_t dst_slot_bytes;
    uint32_t off_src;
    uint32_t off_weight;
    uint32_t off_dst;
};

struct conv_2d_dw5_hvx_context {
    struct htp_ops_context *                     octx;
    const struct htp_conv_2d_dw5_kernel_params * kparams;
    uint32_t                                     plane_start;
    uint32_t                                     plane_count;
    uint32_t                                     planes_per_thread;
};

// Zero both source slots once per thread; DMA only writes the interior, so the 2-pixel halo stays zero.
static __attribute__((noinline)) void conv_2d_dw5_clear(uint8_t * restrict data, uint32_t bytes) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t offset = 0; offset < bytes; offset += VLEN) {
        *(HVX_Vector *) (data + offset) = zero;
    }
}

// Expand the 25 f16 taps in weight slot b into the f32 splat table that follows the two slots.
static __attribute__((noinline)) void conv_2d_dw5_prep_weights(uint8_t * restrict wbase, uint32_t b) {
    HVX_Vector *     wsplat = (HVX_Vector *) (wbase + 2 * VLEN);
    const HVX_Vector w      = Q6_V_lo_W(hvx_vec_f16_to_f32(*(const HVX_Vector *) (wbase + b * VLEN)));
    for (uint32_t t = 0; t < 25; ++t) {
        wsplat[t] = hvx_vec_repl4(Q6_V_vror_VR(w, t * sizeof(float)));
    }
}

// One output plane. src rows are 128B aligned with the 2-pixel left halo at the row start, so the five kx taps
// of a 32-wide output vector are valigns of two aligned loads.
static __attribute__((noinline)) void conv_2d_dw5_plane(const uint8_t * restrict    src,
                                                        const HVX_Vector * restrict wsplat,
                                                        float * restrict            dst,
                                                        uint32_t                    src_row_stride,
                                                        uint32_t                    ow,
                                                        uint32_t                    oh) {
    for (uint32_t oy = 0; oy < oh; ++oy) {
        for (uint32_t ox = 0; ox < ow; ox += 2 * VLEN_FP32) {
            HVX_Vector acc0 = Q6_V_vzero();
            HVX_Vector acc1 = Q6_V_vzero();

            for (uint32_t ky = 0; ky < 5; ++ky) {
                const HVX_Vector * row = (const HVX_Vector *) (src + (oy + ky) * src_row_stride + ox * sizeof(float));
                const HVX_Vector * w   = wsplat + ky * 5;
                const HVX_Vector   v0  = row[0];
                const HVX_Vector   v1  = row[1];
                const HVX_Vector   v2  = row[2];

                const HVX_Vector a01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0, w[0]),
                                                                Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v1, v0, 4), w[1]));
                const HVX_Vector a23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v1, v0, 8), w[2]),
                                                                Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v1, v0, 12), w[3]));
                const HVX_Vector a4  = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v1, v0, 16), w[4]));
                acc0                 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a01, a23), a4);

                const HVX_Vector b01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, w[0]),
                                                                Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v2, v1, 4), w[1]));
                const HVX_Vector b23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v2, v1, 8), w[2]),
                                                                Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v2, v1, 12), w[3]));
                const HVX_Vector b4  = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v2, v1, 16), w[4]));
                acc1                 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(b01, b23), b4);
            }

            const uint32_t rem = ow - ox;
            float *        out = dst + oy * ow + ox;
            hvx_vec_store_u(out, MIN(rem, VLEN_FP32) * sizeof(float), Q6_Vsf_equals_Vqf32(acc0));
            if (rem > VLEN_FP32) {
                hvx_vec_store_u(out + VLEN_FP32, MIN(rem - VLEN_FP32, VLEN_FP32) * sizeof(float),
                                Q6_Vsf_equals_Vqf32(acc1));
            }
        }
    }
}

// Planes are processed in memory order (plane = n * C + c). Each plane is three DMAs: weights in, src in, dst out.
// Two buffers per stream: plane i+1 is in flight while plane i is computed.
static void conv_2d_dw5_hvx_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct conv_2d_dw5_hvx_context *       ctx    = (const struct conv_2d_dw5_hvx_context *) data;
    const struct htp_ops_context *               octx   = ctx->octx;
    const struct htp_conv_2d_dw5_kernel_params * p      = ctx->kparams;
    const struct htp_tensor *                    weight = octx->src[0];
    const struct htp_tensor *                    input  = octx->src[1];
    const struct htp_tensor *                    output = octx->dst;

    const uint32_t first = ctx->plane_start + ith * ctx->planes_per_thread;
    const uint32_t last  = MIN(first + ctx->planes_per_thread, ctx->plane_start + ctx->plane_count);
    if (first >= last) {
        return;
    }
    const uint32_t n = last - first;

    uint8_t *    vtcm        = octx->ctx->vtcm_base;
    uint8_t *    src_base    = vtcm + p->off_src + ith * 2 * p->src_slot_bytes;
    uint8_t *    weight_base = vtcm + p->off_weight + ith * p->weight_slot_bytes;
    uint8_t *    dst_base    = vtcm + p->off_dst + ith * 2 * p->dst_slot_bytes;
    uint8_t *    srcb[2]     = { src_base, src_base + p->src_slot_bytes };
    uint8_t *    wb[2]       = { weight_base, weight_base + VLEN };
    HVX_Vector * wsplat      = (HVX_Vector *) (weight_base + 2 * VLEN);
    uint8_t *    dstb[2]     = { dst_base, dst_base + p->dst_slot_bytes };

    dma_queue *               dma_q = th->dma_q;
    struct htp_thread_trace * tr    = th->trace;

    const uint32_t channels     = input->ne[2];
    const uint32_t ih           = input->ne[1];
    const uint32_t ow           = output->ne[0];
    const uint32_t oh           = output->ne[1];
    const uint32_t stride       = p->src_row_stride;
    const uint32_t in_row       = input->nb[1];
    const uint32_t in_plane     = input->nb[2];
    const uint32_t out_row      = output->nb[1];
    const uint32_t out_plane    = output->nb[2];
    const uint32_t w_plane      = weight->nb[3];
    const uint32_t weight_bytes = 25 * sizeof(__fp16);
    const uint32_t halo_offset  = 2 * stride + 2 * sizeof(float);

    conv_2d_dw5_clear(src_base, 2 * p->src_slot_bytes);

    // channel of the next plane to fetch
    uint32_t c_next = first % channels;

    // prologue: dummy dst completions keep the pop pattern uniform for the first two planes
    for (uint32_t i = 0; i < n && i < 2; ++i) {
        const uint32_t plane = first + i;
        dma_queue_push(dma_q, dma_make_data(output->data, dstb[i]), out_row, out_row, out_row, 0);
        dma_queue_push(dma_q, dma_make_data(wb[i], weight->data + (dma_addr_t) c_next * w_plane), VLEN, w_plane, weight_bytes, 1);
        dma_queue_push(dma_q, dma_make_data(srcb[i] + halo_offset, input->data + (dma_addr_t) plane * in_plane), stride, in_row,
                       in_row, ih);
        c_next = (c_next + 1 == channels) ? 0 : c_next + 1;
    }

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t plane = first + i;
        const uint32_t b     = i & 1;

        dma_queue_pop(dma_q);  // dst slot b free
        dma_queue_pop(dma_q);  // weights for plane
        dma_queue_pop(dma_q);  // src for plane

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        conv_2d_dw5_prep_weights(weight_base, b);
        conv_2d_dw5_plane(srcb[b], wsplat, (float *) dstb[b], stride, ow, oh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_queue_push(dma_q, dma_make_data(output->data + (dma_addr_t) plane * out_plane, dstb[b]), out_row, out_row, out_row, oh);

        if (i + 2 < n) {
            dma_queue_push(dma_q, dma_make_data(wb[b], weight->data + (dma_addr_t) c_next * w_plane), VLEN, w_plane, weight_bytes,
                           1);
            dma_queue_push(dma_q, dma_make_data(srcb[b] + halo_offset, input->data + (dma_addr_t) (plane + 2) * in_plane), stride,
                           in_row, in_row, ih);
            c_next = (c_next + 1 == channels) ? 0 : c_next + 1;
        }
    }

    // epilogue: drain the last outputs
    for (uint32_t i = 0; i < n && i < 2; ++i) {
        dma_queue_pop(dma_q);
    }

    FARF(HIGH, "conv2d-dw-hvx-dma-5x5 %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n", ith, nth, input->ne[0],
         input->ne[1], input->ne[2], input->ne[3], output->ne[0], output->ne[1], output->ne[2], output->ne[3], first,
         last);
}

static int conv_2d_dw5_run_hvx(struct htp_ops_context * octx, const struct htp_conv_2d_dw5_kernel_params * p) {
    const struct htp_tensor * output      = octx->dst;
    uint32_t                  plane_start = 0;
    uint32_t                  plane_count = p->planes;

    if (octx->ctx->mdev.count > 1) {
        // output planes are contiguous, split them on 128B line boundaries
        const uint32_t planes_per_chunk =
            HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(output->nb[2], HTP_TENSOR_MDEV_LINE_SIZE);
        const bool                         can_split = htp_tensor_mdev_data_aligned(output);
        const struct htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(plane_count, can_split ? planes_per_chunk : 0, octx->ctx->mdev.idx,
                                      octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        plane_start = range.start;
        plane_count = range.count;
    }
    if (plane_count == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t                 n_threads         = MIN(p->n_threads, plane_count);
    const uint32_t                 planes_per_thread = (plane_count + n_threads - 1) / n_threads;
    struct conv_2d_dw5_hvx_context ctx               = {
        .octx              = octx,
        .kparams           = p,
        .plane_start       = plane_start,
        .plane_count       = plane_count,
        .planes_per_thread = planes_per_thread,
    };

    work_queue_run(octx->ctx->hvx_queue, conv_2d_dw5_hvx_thread, &ctx, n_threads);
    return HTP_STATUS_OK;
}

int op_conv_2d_dw5(struct htp_ops_context * octx) {
    // Tensor shapes, types and VTCM sizing were validated by the host (htp_kparams_solve_conv2d_dw).
    const struct htp_conv2d_dw_kernel_params * kp = (const struct htp_conv2d_dw_kernel_params *) octx->kernel_params;
    const struct htp_conv_2d_dw5_kernel_params p  = {
        .n_threads         = kp->n_hvx,
        .planes            = kp->total_tasks,
        .src_row_stride    = kp->row_src_bytes,
        .src_slot_bytes    = kp->pad_top_stride,
        .weight_slot_bytes = HTP_CONV2D_DW5_W_BYTES,
        .dst_slot_bytes    = kp->row_dst_bytes,
        .off_src           = 0,
        .off_weight        = 2 * kp->n_hvx * kp->pad_top_stride,
        .off_dst           = 2 * kp->n_hvx * kp->pad_top_stride + kp->n_hvx * HTP_CONV2D_DW5_W_BYTES,
    };

    return conv_2d_dw5_run_hvx(octx, &p);
}
