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
#include "hvx-utils.h"
#include "dma-queue.h"
#include "dma-pipeline.h"
#include "cumsum-ops.h"

struct htp_cumsum_context {
    struct htp_ops_context * octx;
    dma_pipe_compute_io_fn   compute;
    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_dst;
    size_t                   src0_bytes_per_hvx;
    size_t                   dst_bytes_per_hvx;
    size_t                   src_elem_size;
    size_t                   dst_elem_size;
    size_t                   src_row_size;
    size_t                   dst_row_size;
    size_t                   src_chunk_bytes_aligned;
    size_t                   dst_chunk_bytes_aligned;
    uint32_t                 chunk_elems;
    uint32_t                 rows_per_hvx;
    uint32_t                 total_rows;
    uint32_t                 row_start;
};

struct cumsum_pipe_state {
    dma_addr_t src_data;
    dma_addr_t dst_data;
    size_t     src_elem_size;
    size_t     dst_elem_size;
    size_t     src_row_size;
    size_t     dst_row_size;
    size_t     src_chunk_bytes_aligned;
    size_t     dst_chunk_bytes_aligned;
    uint32_t   ne00;
    uint32_t   chunk_elems;
    uint32_t   row;
    uint32_t   row_end;
    uint32_t   off;
    uint32_t   carry_bits;
};

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#if __HVX_ARCH__ > 75
static inline HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_vadd_VsfVsf(a, b);
}
#else
static inline HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#endif // __HVX_ARCH__ > 75

static inline HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    const HVX_Vector zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    return v;
}

static inline HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    return hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

static inline HVX_Vector hvx_cumsum_clamp_f16(HVX_Vector v) {
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    return Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v));
}

static void __attribute__((noinline)) cumsum_compute_f32_f32(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct cumsum_pipe_state * s = (struct cumsum_pipe_state *) ctx;

    const float * restrict src = (const float *) in_spad;
    float * restrict       dst = (float *)       out_spad;

    const uint32_t n    = task->len;
    const uint32_t nvec = n >> 5;
    const uint32_t nloe = n & (VLEN_FP32 - 1);

    HVX_Vector carry = (task->off == 0) ? Q6_V_vsplat_R(0) : Q6_V_vsplat_R(s->carry_bits);

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((const HVX_Vector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        *((HVX_Vector *) (dst + i * VLEN_FP32)) = v;
        carry = hvx_splat_last_f32(v);
    }

    if (nloe) {
        HVX_Vector v = *((const HVX_Vector *) (src + nvec * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_a(dst + nvec * VLEN_FP32, nloe * sizeof(float), v);
    }

    s->carry_bits = (uint32_t) Q6_R_vextract_VR(carry, 0);
}

static void __attribute__((noinline)) cumsum_compute_f16_f16(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct cumsum_pipe_state * s = (struct cumsum_pipe_state *) ctx;

    const HVX_Vector * restrict vsrc = (const HVX_Vector *) in_spad;
    HVX_Vector * restrict       vdst = (HVX_Vector *)       out_spad;

    const uint32_t n    = task->len;
    const uint32_t nvec = n >> 6;
    const uint32_t nloe = n & (VLEN_FP16 - 1);

    HVX_Vector carry = (task->off == 0) ? Q6_V_vsplat_R(0) : Q6_V_vsplat_R(s->carry_bits);

    for (uint32_t i = 0; i < nvec; i++) {
        const HVX_VectorPair p  = hvx_vec_f16_to_f32(vsrc[i]);
        const HVX_Vector     v0 = hvx_prefix_scan_f32(Q6_V_lo_W(p), carry);
        carry = hvx_splat_last_f32(v0);
        const HVX_Vector     v1 = hvx_prefix_scan_f32(Q6_V_hi_W(p), carry);
        carry = hvx_splat_last_f32(v1);
        vdst[i] = hvx_vec_f32_to_f16(hvx_cumsum_clamp_f16(v0), hvx_cumsum_clamp_f16(v1));
    }

    if (nloe) {
        const HVX_VectorPair p   = hvx_vec_f16_to_f32(vsrc[nvec]);
        const HVX_Vector     v0  = hvx_prefix_scan_f32(Q6_V_lo_W(p), carry);
        carry = hvx_splat_last_f32(v0);
        const HVX_Vector     v1  = hvx_prefix_scan_f32(Q6_V_hi_W(p), carry);
        const HVX_Vector     out = hvx_vec_f32_to_f16(hvx_cumsum_clamp_f16(v0), hvx_cumsum_clamp_f16(v1));
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(__fp16), out);
    }

    s->carry_bits = (uint32_t) Q6_R_vextract_VR(carry, 0);
}

static void __attribute__((noinline)) cumsum_compute_f16_f32(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct cumsum_pipe_state * s = (struct cumsum_pipe_state *) ctx;

    const HVX_Vector * restrict vsrc = (const HVX_Vector *) in_spad;
    HVX_Vector * restrict       vdst = (HVX_Vector *)       out_spad;

    const uint32_t n    = task->len;
    const uint32_t nvec = n >> 6;
    const uint32_t nloe = n & (VLEN_FP16 - 1);

    HVX_Vector carry = (task->off == 0) ? Q6_V_vsplat_R(0) : Q6_V_vsplat_R(s->carry_bits);

    for (uint32_t i = 0; i < nvec; i++) {
        const HVX_VectorPair p  = hvx_vec_f16_to_f32(vsrc[i]);
        const HVX_Vector     v0 = hvx_prefix_scan_f32(Q6_V_lo_W(p), carry);
        carry = hvx_splat_last_f32(v0);
        const HVX_Vector     v1 = hvx_prefix_scan_f32(Q6_V_hi_W(p), carry);
        carry = hvx_splat_last_f32(v1);
        vdst[2 * i]     = v0;
        vdst[2 * i + 1] = v1;
    }

    if (nloe) {
        const HVX_VectorPair p  = hvx_vec_f16_to_f32(vsrc[nvec]);
        const HVX_Vector     v0 = hvx_prefix_scan_f32(Q6_V_lo_W(p), carry);
        if (nloe > VLEN_FP32) {
            vdst[2 * nvec] = v0;
            carry = hvx_splat_last_f32(v0);
            const HVX_Vector v1 = hvx_prefix_scan_f32(Q6_V_hi_W(p), carry);
            hvx_vec_store_a(&vdst[2 * nvec + 1], (nloe - VLEN_FP32) * sizeof(float), v1);
        } else {
            hvx_vec_store_a(&vdst[2 * nvec], nloe * sizeof(float), v0);
        }
    }

    s->carry_bits = (uint32_t) Q6_R_vextract_VR(carry, 0);
}

static void __attribute__((noinline)) cumsum_compute_f32_f16(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct cumsum_pipe_state * s = (struct cumsum_pipe_state *) ctx;

    const HVX_Vector * restrict vsrc = (const HVX_Vector *) in_spad;
    HVX_Vector * restrict       vdst = (HVX_Vector *)       out_spad;

    const uint32_t n    = task->len;
    const uint32_t nvec = n >> 6;
    const uint32_t nloe = n & (VLEN_FP16 - 1);

    HVX_Vector carry = (task->off == 0) ? Q6_V_vsplat_R(0) : Q6_V_vsplat_R(s->carry_bits);

    for (uint32_t i = 0; i < nvec; i++) {
        const HVX_Vector v0 = hvx_prefix_scan_f32(vsrc[2 * i], carry);
        carry = hvx_splat_last_f32(v0);
        const HVX_Vector v1 = hvx_prefix_scan_f32(vsrc[2 * i + 1], carry);
        carry = hvx_splat_last_f32(v1);
        vdst[i] = hvx_vec_f32_to_f16(hvx_cumsum_clamp_f16(v0), hvx_cumsum_clamp_f16(v1));
    }

    if (nloe) {
        const HVX_Vector v0 = hvx_prefix_scan_f32(vsrc[2 * nvec], carry);
        carry = hvx_splat_last_f32(v0);
        const HVX_Vector v1  = (nloe > VLEN_FP32) ? hvx_prefix_scan_f32(vsrc[2 * nvec + 1], carry) : Q6_V_vzero();
        const HVX_Vector out = hvx_vec_f32_to_f16(hvx_cumsum_clamp_f16(v0), hvx_cumsum_clamp_f16(v1));
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(__fp16), out);
    }

    s->carry_bits = (uint32_t) Q6_R_vextract_VR(carry, 0);
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA pipeline
// ---------------------------------------------------------------------------

static bool cumsum_next_task(void * ctx, struct dma_pipe_task * task) {
    struct cumsum_pipe_state * s = (struct cumsum_pipe_state *) ctx;
    if (s->row >= s->row_end) {
        return false;
    }

    const uint32_t rem = s->ne00 - s->off;
    const uint32_t len = MIN(s->chunk_elems, rem);

    task->row = s->row;
    task->off = s->off;
    task->len = len;
    task->aux = 0;

    s->off += len;
    if (s->off >= s->ne00) {
        s->off = 0;
        s->row++;
    }
    return true;
}

static void cumsum_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct cumsum_pipe_state * s = (struct cumsum_pipe_state *) ctx;

    const size_t     bytes = (size_t) task->len * s->src_elem_size;
    const dma_addr_t src   = s->src_data + (size_t) task->row * s->src_row_size + (size_t) task->off * s->src_elem_size;

    dma_queue_push(dma_q, dma_make_data(in_spad, src), s->src_chunk_bytes_aligned, bytes, bytes, 1);
}

static void cumsum_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct cumsum_pipe_state * s = (struct cumsum_pipe_state *) ctx;

    const size_t     bytes = (size_t) task->len * s->dst_elem_size;
    const dma_addr_t dst   = s->dst_data + (size_t) task->row * s->dst_row_size + (size_t) task->off * s->dst_elem_size;

    dma_queue_push(dma_q, dma_make_data(dst, out_spad), bytes, s->dst_chunk_bytes_aligned, bytes, 1);
}

static void cumsum_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_cumsum_context * cctx = (struct htp_cumsum_context *) data;
    struct htp_ops_context *    octx = cctx->octx;
    const struct htp_tensor *   src0 = octx->src[0];
    const struct htp_tensor *   dst  = octx->dst;

    const uint32_t ir0 = cctx->row_start + cctx->rows_per_hvx * ith;
    const uint32_t ir1 = MIN(ir0 + cctx->rows_per_hvx, cctx->row_start + cctx->total_rows);

    if (ir0 >= ir1) {
        return;
    }

    uint8_t * src_spad = cctx->vtcm_src0 + ith * cctx->src0_bytes_per_hvx;
    uint8_t * dst_spad = cctx->vtcm_dst  + ith * cctx->dst_bytes_per_hvx;
    void *    in_buf[2]  = { src_spad, src_spad + cctx->src_chunk_bytes_aligned };
    void *    out_buf[2] = { dst_spad, dst_spad + cctx->dst_chunk_bytes_aligned };

    struct cumsum_pipe_state s = {
        .src_data                = src0->data,
        .dst_data                = dst->data,
        .src_elem_size           = cctx->src_elem_size,
        .dst_elem_size           = cctx->dst_elem_size,
        .src_row_size            = cctx->src_row_size,
        .dst_row_size            = cctx->dst_row_size,
        .src_chunk_bytes_aligned = cctx->src_chunk_bytes_aligned,
        .dst_chunk_bytes_aligned = cctx->dst_chunk_bytes_aligned,
        .ne00                    = src0->ne[0],
        .chunk_elems             = cctx->chunk_elems,
        .row                     = ir0,
        .row_end                 = ir1,
        .off                     = 0,
        .carry_bits              = 0,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf, cumsum_next_task, cumsum_push_in, cctx->compute, cumsum_push_out);

    FARF(HIGH, "cumsum %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

static int op_cumsum_impl(struct htp_ops_context * octx, size_t src_elem_size, size_t dst_elem_size, dma_pipe_compute_io_fn compute) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;
    const struct htp_cumsum_kernel_params * kparams =
        (const struct htp_cumsum_kernel_params *) octx->kernel_params;

    const uint32_t total_rows   = kparams->total_rows;
    const size_t   src_row_size = src0->nb[1];
    const size_t   dst_row_size = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, nrows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx = octx->n_hvx;

    struct htp_cumsum_vtcm_layout layout = { 0 };
    htp_cumsum_vtcm_layout_build(&layout, (size_t) kparams->chunk_elems * src_elem_size, (size_t) kparams->chunk_elems * dst_elem_size, n_hvx);

    struct htp_cumsum_context cctx = {
        .octx                    = octx,
        .compute                 = compute,
        .vtcm_src0               = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, layout.off_src0),
        .vtcm_dst                = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, layout.off_dst),
        .src0_bytes_per_hvx      = layout.src0_bytes_per_hvx,
        .dst_bytes_per_hvx       = layout.dst_bytes_per_hvx,
        .src_elem_size           = src_elem_size,
        .dst_elem_size           = dst_elem_size,
        .src_row_size            = src_row_size,
        .dst_row_size            = dst_row_size,
        .src_chunk_bytes_aligned = layout.src_chunk_bytes_aligned,
        .dst_chunk_bytes_aligned = layout.dst_chunk_bytes_aligned,
        .chunk_elems             = kparams->chunk_elems,
        .rows_per_hvx            = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .total_rows              = nrows,
        .row_start               = row_start,
    };

    work_queue_run(octx->ctx->hvx_queue, cumsum_thread, &cctx, n_hvx);

    return HTP_STATUS_OK;
}

int op_cumsum(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F32) {
        return op_cumsum_impl(octx, sizeof(float), sizeof(float), cumsum_compute_f32_f32);
    }
    if (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F16) {
        return op_cumsum_impl(octx, sizeof(__fp16), sizeof(__fp16), cumsum_compute_f16_f16);
    }
    if (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        return op_cumsum_impl(octx, sizeof(__fp16), sizeof(float), cumsum_compute_f16_f32);
    }
    if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        return op_cumsum_impl(octx, sizeof(float), sizeof(__fp16), cumsum_compute_f32_f16);
    }
    return HTP_STATUS_NO_SUPPORT;
}
