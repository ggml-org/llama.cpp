#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hvx-types.h"
#include "hex-utils.h"
#include "hvx-copy.h"
#include "dma-queue.h"
#include "dma-pipeline.h"
#include "diag-ops.h"

struct htp_diag_context {
    struct htp_ops_context *              octx;
    const struct htp_diag_kernel_params * kparams;
    dma_pipe_compute_io_fn                compute;
    uint8_t *                             vtcm_src0;
    uint8_t *                             vtcm_dst;
    size_t                                src0_bytes_per_hvx;
    size_t                                dst_bytes_per_hvx;
    size_t                                elem_size;
    size_t                                dst_chunk_bytes_aligned;
    uint32_t                              chunk_elems;
    uint32_t                              batches_per_hvx;
    uint32_t                              total_batches;
    uint32_t                              batch_start;
};

struct diag_pipe_state {
    dma_addr_t                    src_data;
    dma_addr_t                    dst_data;
    size_t                        elem_size;
    size_t                        dst_chunk_bytes_aligned;
    uint32_t                      ne0;
    uint32_t                      ne1;
    uint32_t                      ne02;
    uint32_t                      nb02;
    uint32_t                      nb03;
    uint32_t                      nb1;
    uint32_t                      nb2;
    uint32_t                      nb3;
    uint32_t                      chunk_elems;
    uint32_t                      ib;
    uint32_t                      ib_end;
    uint32_t                      i1;
    uint32_t                      off;
    const struct fastdiv_values * div_ne02;
};

static void __attribute__((noinline)) diag_compute_f32(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    (void) ctx;

    uint8_t * restrict       dst = (uint8_t *) out_spad;
    const uint8_t * restrict src = (const uint8_t *) in_spad;

    hvx_splat_f32_a(dst, 0.0f, task->len);

    if (task->row >= task->off && task->row < task->off + task->len) {
        const uint32_t       local_idx = task->row - task->off;
        const uint32_t       vec_idx   = local_idx >> 5;
        const uint32_t       b_off     = (local_idx & 31u) * sizeof(float);
        const HVX_Vector     src_v     = *((const HVX_Vector *) src);
        const HVX_Vector     val_v     = Q6_V_vsplat_R(Q6_R_vextract_VR(src_v, 0));
        const HVX_VectorPred b_mask    = Q6_Q_vsetq_R(b_off);
        const HVX_VectorPred e_mask    = Q6_Q_vsetq2_R(b_off + sizeof(float));
        Q6_vmem_QnRIV(Q6_Q_or_QQn(b_mask, e_mask), &((HVX_Vector *) dst)[vec_idx], val_v);
    }
}

static void __attribute__((noinline)) diag_compute_f16(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    (void) ctx;

    uint8_t * restrict       dst = (uint8_t *) out_spad;
    const uint8_t * restrict src = (const uint8_t *) in_spad;

    hvx_splat_u16_a(dst, 0, task->len);

    if (task->row >= task->off && task->row < task->off + task->len) {
        const uint32_t       local_idx = task->row - task->off;
        const uint32_t       vec_idx   = local_idx >> 6;
        const uint32_t       b_off     = (local_idx & 63u) * sizeof(__fp16);
        const HVX_Vector     src_v     = *((const HVX_Vector *) src);
        const HVX_Vector     val_v     = Q6_Vh_vsplat_R(Q6_R_vextract_VR(src_v, 0));
        const HVX_VectorPred b_mask    = Q6_Q_vsetq_R(b_off);
        const HVX_VectorPred e_mask    = Q6_Q_vsetq2_R(b_off + sizeof(__fp16));
        Q6_vmem_QnRIV(Q6_Q_or_QQn(b_mask, e_mask), &((HVX_Vector *) dst)[vec_idx], val_v);
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA pipeline
// ---------------------------------------------------------------------------

static bool diag_next_task(void * ctx, struct dma_pipe_task * task) {
    struct diag_pipe_state * s = (struct diag_pipe_state *) ctx;
    if (s->ib >= s->ib_end) {
        return false;
    }

    const uint32_t rem = s->ne0 - s->off;
    const uint32_t len = MIN(s->chunk_elems, rem);

    task->row = s->i1;
    task->off = s->off;
    task->len = len;
    task->aux = s->ib;

    s->off += len;
    if (s->off >= s->ne0) {
        s->off = 0;
        s->i1++;
        if (s->i1 >= s->ne1) {
            s->i1 = 0;
            s->ib++;
        }
    }
    return true;
}

static void diag_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct diag_pipe_state * s = (struct diag_pipe_state *) ctx;

    if (task->row < task->off || task->row >= task->off + task->len) {
        dma_queue_push(dma_q, dma_make_data(in_spad, 0), 0, 0, 0, 0);
        return;
    }

    const uint32_t   i3       = fastdiv(task->aux, s->div_ne02);
    const uint32_t   i2       = task->aux - i3 * s->ne02;
    const dma_addr_t src_elem = s->src_data + (size_t) i3 * s->nb03 + (size_t) i2 * s->nb02 + (size_t) task->row * s->elem_size;

    dma_queue_push(dma_q, dma_make_data(in_spad, src_elem), s->elem_size, s->elem_size, s->elem_size, 1);
}

static void diag_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct diag_pipe_state * s = (struct diag_pipe_state *) ctx;

    const uint32_t   i3    = fastdiv(task->aux, s->div_ne02);
    const uint32_t   i2    = task->aux - i3 * s->ne02;
    const size_t     bytes = (size_t) task->len * s->elem_size;
    const dma_addr_t dst   = s->dst_data + (size_t) i3 * s->nb3 + (size_t) i2 * s->nb2 + (size_t) task->row * s->nb1 + (size_t) task->off * s->elem_size;

    dma_queue_push(dma_q, dma_make_data(dst, out_spad), bytes, s->dst_chunk_bytes_aligned, bytes, 1);
}

static void diag_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_diag_context * dctx = (struct htp_diag_context *) data;
    struct htp_ops_context *  octx = dctx->octx;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t ib0 = dctx->batch_start + dctx->batches_per_hvx * ith;
    const uint32_t ib1 = MIN(ib0 + dctx->batches_per_hvx, dctx->batch_start + dctx->total_batches);

    if (ib0 >= ib1) {
        return;
    }

    uint8_t * src_spad = dctx->vtcm_src0 + ith * dctx->src0_bytes_per_hvx;
    uint8_t * dst_spad = dctx->vtcm_dst  + ith * dctx->dst_bytes_per_hvx;
    void *    in_buf[2]  = { src_spad, src_spad + 128 };
    void *    out_buf[2] = { dst_spad, dst_spad + dctx->dst_chunk_bytes_aligned };

    struct diag_pipe_state s = {
        .src_data                = src0->data,
        .dst_data                = dst->data,
        .elem_size               = dctx->elem_size,
        .dst_chunk_bytes_aligned = dctx->dst_chunk_bytes_aligned,
        .ne0                     = dst->ne[0],
        .ne1                     = dst->ne[1],
        .ne02                    = src0->ne[2],
        .nb02                    = src0->nb[2],
        .nb03                    = src0->nb[3],
        .nb1                     = dst->nb[1],
        .nb2                     = dst->nb[2],
        .nb3                     = dst->nb[3],
        .chunk_elems             = dctx->chunk_elems,
        .ib                      = ib0,
        .ib_end                  = ib1,
        .i1                      = 0,
        .off                     = 0,
        .div_ne02                = &dctx->kparams->div_ne02,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf, diag_next_task, diag_push_in, dctx->compute, diag_push_out);

    FARF(HIGH, "diag %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

static int op_diag_impl(struct htp_ops_context * octx, size_t elem_size, dma_pipe_compute_io_fn compute) {
    const struct htp_diag_kernel_params * kparams =
        (const struct htp_diag_kernel_params *) octx->kernel_params;

    const uint32_t total_batches = kparams->total_batches;

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        batch_start = range.start;
        nbatches    = range.count;
    }

    if (nbatches == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, nbatches))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx = octx->n_hvx;

    struct htp_diag_vtcm_layout layout = { 0 };
    htp_diag_vtcm_layout_build(&layout, (size_t) kparams->chunk_elems * elem_size, n_hvx);

    struct htp_diag_context dctx = {
        .octx                    = octx,
        .kparams                 = kparams,
        .compute                 = compute,
        .vtcm_src0               = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, layout.off_src0),
        .vtcm_dst                = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, layout.off_dst),
        .src0_bytes_per_hvx      = layout.src0_bytes_per_hvx,
        .dst_bytes_per_hvx       = layout.dst_bytes_per_hvx,
        .elem_size               = elem_size,
        .dst_chunk_bytes_aligned = layout.dst_chunk_bytes_aligned,
        .chunk_elems             = kparams->chunk_elems,
        .batches_per_hvx         = fastdiv(nbatches + n_hvx - 1, &octx->n_hvx_div),
        .total_batches           = nbatches,
        .batch_start             = batch_start,
    };

    work_queue_run(octx->ctx->hvx_queue, diag_thread, &dctx, n_hvx);

    return HTP_STATUS_OK;
}

int op_diag(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != dst->type) {
        return HTP_STATUS_NO_SUPPORT;
    }

    switch (dst->type) {
        case HTP_TYPE_F32:
            return op_diag_impl(octx, sizeof(float), diag_compute_f32);
        case HTP_TYPE_F16:
            return op_diag_impl(octx, sizeof(__fp16), diag_compute_f16);
        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
