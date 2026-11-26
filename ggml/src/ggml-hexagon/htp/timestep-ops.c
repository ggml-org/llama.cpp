#include <HAP_farf.h>
#include <stdint.h>

#include "hex-common.h"
#include "htp-trace.h"

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "dma-queue.h"
#include "timestep-ops.h"

// ggml op_params layout for TIMESTEP_EMBEDDING:
//   op_params[0] - dim
//   op_params[1] - max_period

// f32 lanes 0..31 built in registers
static inline HVX_Vector timestep_iota_f32(void) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = Q6_V_vsplat_R(1);
    HVX_Vector       v    = one;
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 4));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 64));
    return Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(v, one));
}

struct htp_timestep_context {
    struct htp_ops_context * octx;
    const struct htp_timestep_kernel_params * kparams;
    struct htp_timestep_vtcm_layout layout;
};

static __attribute__((noinline)) void timestep_compute_freq(float * restrict freq, uint32_t jb0, uint32_t nb, float freq_scale) {
    const HVX_Vector v_iota  = timestep_iota_f32();
    const HVX_Vector v_scale = hvx_vec_splat_f32(freq_scale);
    const HVX_Vector v_step  = hvx_vec_splat_f32((float) HTP_TIMESTEP_BLOCK);

    HVX_Vector v_j = hvx_vec_add_f32_f32(hvx_vec_splat_f32((float) (jb0 * HTP_TIMESTEP_BLOCK)), v_iota);

    // exp constants fill the register file, pipelining this loop spills them
    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t b = 0; b < nb; ++b) {
        hvx_vmem(freq + b * HTP_TIMESTEP_BLOCK) = hvx_vec_exp_f32(hvx_vec_mul_f32_f32(v_j, v_scale));
        v_j = hvx_vec_add_f32_f32(v_j, v_step);
    }
}

// out row r: [cos (col_stride) | sin (col_stride)], all VTCM
static __attribute__((noinline)) void timestep_compute_rows(float * restrict out, const float * restrict ts, const float * restrict freq,
                                                            uint32_t nr, uint32_t nb, uint32_t col_stride) {
    for (uint32_t r = 0; r < nr; ++r) {
        const HVX_Vector v_t = hvx_vec_repl4(Q6_V_vror_VR(hvx_vmem(ts + (r & ~31u)), (r & 31) * sizeof(float)));
        float * out_cos = out + (size_t) r * 2 * col_stride;
        float * out_sin = out_cos + col_stride;

        // sincos constants fill the register file, pipelining this loop spills them
        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t b = 0; b < nb; ++b) {
            HVX_Vector v_cos;
            HVX_Vector v_sin;
            hvx_vec_sincos_f32(hvx_vec_mul_f32_f32(v_t, hvx_vmem(freq + b * HTP_TIMESTEP_BLOCK)), &v_cos, &v_sin);
            hvx_vmem(out_cos + b * HTP_TIMESTEP_BLOCK) = v_cos;
            hvx_vmem(out_sin + b * HTP_TIMESTEP_BLOCK) = v_sin;
        }
    }
}

// odd dim: zero sin lanes from n_cols on, this writes the pad element
static __attribute__((noinline)) void timestep_zero_pad(float * restrict out, uint32_t nr, uint32_t col_stride, uint32_t n_cols) {
    float * pad = out + col_stride + (n_cols & ~(HTP_TIMESTEP_BLOCK - 1));
    const HVX_VectorPred q_keep = Q6_Q_vsetq_R((n_cols & (HTP_TIMESTEP_BLOCK - 1)) * sizeof(float));
    const HVX_Vector     v_zero = Q6_V_vzero();
    for (uint32_t r = 0; r < nr; ++r) {
        hvx_vmem(pad) = Q6_V_vmux_QVV(q_keep, hvx_vmem(pad), v_zero);
        pad += 2 * col_stride;
    }
}

// Each thread owns a tile: rows [ir0, ir1) x freq blocks [jb0, jb1).
// Rows are processed in chunks: DMA timesteps in, compute [cos | sin] rows in VTCM, DMA out.
// Two buffers for both, so compute of chunk c overlaps the writeback of chunk c-1.
static void timestep_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_timestep_context * tctx = (const struct htp_timestep_context *) data;
    const struct htp_timestep_kernel_params * kparams = tctx->kparams;
    const struct htp_timestep_vtcm_layout * L = &tctx->layout;
    struct htp_ops_context * octx = tctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    uint32_t ir0 = 0, ir1 = kparams->n_rows;
    uint32_t jb0 = 0, jb1 = kparams->n_blocks;
    if (kparams->split_cols) {
        jb0 = kparams->blocks_per_hvx * ith;
        jb1 = MIN(jb0 + kparams->blocks_per_hvx, kparams->n_blocks);
    } else {
        ir0 = kparams->rows_per_hvx * ith;
        ir1 = MIN(ir0 + kparams->rows_per_hvx, kparams->n_rows);
    }
    if (ir0 >= ir1 || jb0 >= jb1) {
        return;
    }

    const uint32_t half       = kparams->half;
    const uint32_t chunk_rows = kparams->chunk_rows;
    const uint32_t n_chunks   = (ir1 - ir0 + chunk_rows - 1) / chunk_rows;
    const uint32_t j0         = jb0 * HTP_TIMESTEP_BLOCK;
    const uint32_t n_cols     = MIN(jb1 * HTP_TIMESTEP_BLOCK, half) - j0;
    const uint32_t n_tail     = (jb1 == kparams->n_blocks) ? (kparams->dim & 1) : 0;
    const size_t   col_stride = L->col_stride;
    const size_t   dst_nb1    = dst->nb[1];

    uint8_t * vtcm = (uint8_t *) octx->ctx->vtcm_base + ith * L->bytes_per_hvx;
    uint8_t * ts_buf[2]  = { vtcm + L->off_ts, vtcm + L->off_ts + L->ts_buf_size };
    float   * freq       = (float *) (vtcm + L->off_freq);
    uint8_t * out_buf[2] = { vtcm + L->off_out, vtcm + L->off_out + L->out_buf_size };

    const dma_addr_t src_data = src0->data;
    const dma_addr_t dst_data = dst->data;

    dma_queue * dma_q = th->dma_q;
    struct htp_thread_trace * tr = th->trace;

    {
        const uint32_t nr = MIN(chunk_rows, ir1 - ir0);
        dma_queue_push(dma_q, dma_make_data(ts_buf[0], src_data + (size_t) ir0 * sizeof(float)),
                       nr * sizeof(float), nr * sizeof(float), nr * sizeof(float), 1);
    }

    // freq table for this tile, overlaps the first timestep fetch
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) jb0);
    timestep_compute_freq(freq, jb0, jb1 - jb0, kparams->freq_scale);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) jb0);

    for (uint32_t c = 0; c < n_chunks; ++c) {
        const uint32_t r0 = ir0 + c * chunk_rows;
        const uint32_t nr = MIN(chunk_rows, ir1 - r0);

        // queue order is: wb(c-2) cos, wb(c-2) sin, ts(c), wb(c-1) cos, wb(c-1) sin
        if (c >= 2) {
            dma_queue_pop(dma_q);
            dma_queue_pop(dma_q);
        }
        dma_queue_pop(dma_q);

        if (c + 1 < n_chunks) {
            const uint32_t r0n = r0 + chunk_rows;
            const uint32_t nrn = MIN(chunk_rows, ir1 - r0n);
            dma_queue_push(dma_q, dma_make_data(ts_buf[(c + 1) & 1], src_data + (size_t) r0n * sizeof(float)),
                           nrn * sizeof(float), nrn * sizeof(float), nrn * sizeof(float), 1);
        }

        float * out = (float *) out_buf[c & 1];

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        timestep_compute_rows(out, (const float *) ts_buf[c & 1], freq, nr, jb1 - jb0, col_stride);
        if (n_tail) {
            timestep_zero_pad(out, nr, col_stride, n_cols);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        const dma_addr_t dst_row = dst_data + (size_t) r0 * dst_nb1;
        const size_t     src_nb  = 2 * col_stride * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(dst_row + (size_t) j0 * sizeof(float), out),
                       dst_nb1, src_nb, n_cols * sizeof(float), nr);
        dma_queue_push(dma_q, dma_make_data(dst_row + (size_t) (half + j0) * sizeof(float), out + col_stride),
                       dst_nb1, src_nb, (n_cols + n_tail) * sizeof(float), nr);
    }

    dma_queue_flush(dma_q);
}

int op_timestep_embedding(struct htp_ops_context * octx) {
    const struct htp_timestep_kernel_params * kparams =
        (const struct htp_timestep_kernel_params *) octx->kernel_params;

    // Only device 0 writes the output when running multi-device
    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx != 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct htp_timestep_context tctx = {
        .octx    = octx,
        .kparams = kparams,
    };

    htp_timestep_vtcm_layout_build(&tctx.layout, kparams->dim, kparams->blocks_per_hvx, kparams->chunk_rows, kparams->n_hvx);

    if (octx->ctx->vtcm_size < tctx.layout.total_bytes) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    FARF(HIGH, "timestep-embedding: %u timesteps dim %u max-period %d split-%s n-hvx %u chunk-rows %u\n",
         kparams->n_rows, kparams->dim, max_period, kparams->split_cols ? "cols" : "rows", kparams->n_hvx, kparams->chunk_rows);

    work_queue_run(octx->ctx->hvx_queue, timestep_thread, &tctx, kparams->n_hvx);

    return HTP_STATUS_OK;
}
