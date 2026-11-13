#include <math.h>
#include <string.h>

#include "hvx-exp.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "ssm-scan-ops.h"

struct htp_ssm_scan_context {
    struct htp_ops_context *                  octx;
    const struct htp_ssm_scan_kernel_params * kparams;
    struct htp_ssm_scan_vtcm_layout           layout;
    uint32_t                                  unit_start;
    uint32_t                                  nunits;
    uint32_t                                  units_per_hvx;
};

struct htp_ssm_scan_unit {
    uint32_t q;
    uint32_t group;
    uint32_t head;
    uint32_t row0;
    uint32_t nrows;
};

static inline HVX_Vector ssm_scan_vec_softplus_f32(HVX_Vector vx) {
    const HVX_Vector     v20 = hvx_vec_splat_f32(20.0f);
    const HVX_Vector     v1  = hvx_vec_splat_f32(1.0f);
    const HVX_VectorPred gt  = Q6_Q_vcmp_gt_VsfVsf(vx, v20);
    const HVX_Vector     sp  = hvx_vec_log_f32(hvx_vec_add_f32_f32(v1, hvx_vec_exp_f32(vx)));
    return Q6_V_vmux_QVV(gt, vx, sp);
}

static inline struct htp_ssm_scan_unit ssm_scan_decode_unit(
    const struct htp_ssm_scan_kernel_params * kparams,
    uint32_t unit_index
) {
    struct htp_ssm_scan_unit unit;
    unit.q              = fastdiv(unit_index, &kparams->div_units_per_seq);
    uint32_t unit_seq   = fastmodulo(unit_index, kparams->units_per_seq, &kparams->div_units_per_seq);
    unit.group          = fastdiv(unit_seq, &kparams->div_group_units);
    uint32_t unit_group = fastmodulo(unit_seq, kparams->group_units, &kparams->div_group_units);
    uint32_t head_group = fastdiv(unit_group, &kparams->div_n_row_blocks);
    uint32_t row_block  = fastmodulo(unit_group, kparams->n_row_blocks, &kparams->div_n_row_blocks);
    unit.head           = unit.group * kparams->heads_per_group + head_group;
    unit.row0           = row_block * kparams->row_tile;
    unit.nrows          = MIN(kparams->row_tile, kparams->head_dim - unit.row0);
    return unit;
}

// Fast 4-row step for Mamba-2 (n_a == 1, d_state % VLEN_FP32 == 0).
// state_in and state_out may alias (in-place prefill update).
static inline void ssm_scan_step4(
    const float * state_in,
    float * state_out,
    size_t state_stride_elems,
    const float * restrict B,
    const float * restrict C,
    HVX_Vector vdA,
    HVX_Vector vx,
    uint32_t d_state,
    float * restrict y
) {
    HVX_Vector acc0    = Q6_V_vzero();
    HVX_Vector acc1    = Q6_V_vzero();
    HVX_Vector acc2    = Q6_V_vzero();
    HVX_Vector acc3    = Q6_V_vzero();
    HVX_Vector vx0     = hvx_vec_repl_f32(vx);
    HVX_Vector vx1     = hvx_vec_repl_f32(Q6_V_vror_VR(vx, sizeof(float)));
    HVX_Vector vx2     = hvx_vec_repl_f32(Q6_V_vror_VR(vx, 2 * sizeof(float)));
    HVX_Vector vx3     = hvx_vec_repl_f32(Q6_V_vror_VR(vx, 3 * sizeof(float)));
    const float * in0  = state_in;
    const float * in1  = in0 + state_stride_elems;
    const float * in2  = in1 + state_stride_elems;
    const float * in3  = in2 + state_stride_elems;
    float *       out0 = state_out;
    float *       out1 = out0 + state_stride_elems;
    float *       out2 = out1 + state_stride_elems;
    float *       out3 = out2 + state_stride_elems;

    for (uint32_t i = 0; i < d_state; i += VLEN_FP32) {
        HVX_Vector b  = hvx_vmem(B + i);
        HVX_Vector c  = hvx_vmem(C + i);
        HVX_Vector s0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(in0 + i), vdA), hvx_vec_mul_f32_f32(b, vx0));
        HVX_Vector s1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(in1 + i), vdA), hvx_vec_mul_f32_f32(b, vx1));
        HVX_Vector s2 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(in2 + i), vdA), hvx_vec_mul_f32_f32(b, vx2));
        HVX_Vector s3 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(in3 + i), vdA), hvx_vec_mul_f32_f32(b, vx3));
        hvx_vmem(out0 + i) = s0;
        hvx_vmem(out1 + i) = s1;
        hvx_vmem(out2 + i) = s2;
        hvx_vmem(out3 + i) = s3;
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(s0, c));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(s1, c));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(s2, c));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(s3, c));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(y, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
}

// Single-row step for tail rows (nrows % 4 != 0), unaligned d_state, or vector-A (n_a == d_state).
static void ssm_scan_step1(
    const float * state_in,
    float * state_out,
    const float * restrict B,
    const float * restrict C,
    const float * restrict A_row,
    uint32_t n_a,
    float dt_softplus,
    float dA_scalar,
    float x_dt,
    uint32_t d_state,
    float * restrict y
) {
    HVX_Vector acc        = Q6_V_vzero();
    HVX_Vector vdt        = hvx_vec_splat_f32(dt_softplus);
    HVX_Vector vdA_splat  = hvx_vec_splat_f32(dA_scalar);
    HVX_Vector vx0        = hvx_vec_splat_f32(x_dt);

    uint32_t i = 0;
    for (; i + VLEN_FP32 <= d_state; i += VLEN_FP32) {
        HVX_Vector vdA = n_a == 1 ? vdA_splat : hvx_vec_exp_f32(hvx_vec_mul_f32_f32(hvx_vmem(A_row + i), vdt));
        HVX_Vector b   = hvx_vmem(B + i);
        HVX_Vector c   = hvx_vmem(C + i);
        HVX_Vector s0  = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(state_in + i), vdA), hvx_vec_mul_f32_f32(b, vx0));
        hvx_vmem(state_out + i) = s0;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(s0, c));
    }

    if (i < d_state) {
        const HVX_VectorPred mask = Q6_Q_vsetq_R((d_state - i) * sizeof(float));
        const HVX_Vector     zero = Q6_V_vzero();
        HVX_Vector vdA = n_a == 1
            ? vdA_splat
            : hvx_vec_exp_f32(hvx_vec_mul_f32_f32(Q6_V_vmux_QVV(mask, hvx_vmem(A_row + i), zero), vdt));
        HVX_Vector b  = Q6_V_vmux_QVV(mask, hvx_vmem(B + i), zero);
        HVX_Vector c  = Q6_V_vmux_QVV(mask, hvx_vmem(C + i), zero);
        HVX_Vector p0 = Q6_V_vmux_QVV(mask, hvx_vmem(state_in + i), zero);
        HVX_Vector s0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(p0, vdA), hvx_vec_mul_f32_f32(b, vx0));
        hvx_vmem(state_out + i) = s0;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(s0, c));
    }

    hvx_vec_store_u(y, sizeof(float), hvx_vec_reduce_sum_f32(acc));
}

static inline void ssm_scan_decode_push_inputs(
    dma_queue * dma_q,
    const struct htp_ssm_scan_context * scctx,
    const struct htp_ssm_scan_unit * unit,
    void * state_spad,
    void * B_spad,
    void * C_spad,
    void * x_spad,
    void * dt_spad,
    void * A_spad
) {
    const struct htp_ops_context *            octx             = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams          = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout           = &scctx->layout;
    const struct htp_tensor *                 s                = octx->src[0];
    const struct htp_tensor *                 x                = octx->src[1];
    const struct htp_tensor *                 dt               = octx->src[2];
    const struct htp_tensor *                 A                = octx->src[3];
    const struct htp_tensor *                 B                = octx->src[4];
    const struct htp_tensor *                 C                = octx->src[5];
    const struct htp_tensor *                 ids              = octx->src[6];
    const int32_t                             id               = ((const int32_t *) (uintptr_t) ids->data)[unit->q];
    const size_t                              state_row_bytes  = (size_t) kparams->d_state * sizeof(float);
    const size_t                              state_row_stride = layout->state_row_stride;
    const size_t                              x_bytes          = (size_t) unit->nrows * sizeof(float);
    const size_t                              a_bytes          = (size_t) kparams->n_a * sizeof(float);

    const dma_addr_t state_in = s->data + (size_t) id * s->nb[3] + (size_t) unit->head * s->nb[2] + (size_t) unit->row0 * s->nb[1];
    const dma_addr_t B_in     = B->data + (size_t) unit->q * B->nb[3] + (size_t) unit->group * B->nb[1];
    const dma_addr_t C_in     = C->data + (size_t) unit->q * C->nb[3] + (size_t) unit->group * C->nb[1];
    const dma_addr_t x_in     = x->data + (size_t) unit->q * x->nb[3] + (size_t) unit->head * x->nb[1] + (size_t) unit->row0 * sizeof(float);
    const dma_addr_t dt_in    = dt->data + (size_t) unit->q * dt->nb[2] + (size_t) unit->head * sizeof(float);
    const dma_addr_t A_in     = A->data + (size_t) unit->head * A->nb[1];

    dma_queue_push(dma_q, dma_make_data(state_spad, state_in), state_row_stride, s->nb[1], state_row_bytes, unit->nrows);
    dma_queue_push(dma_q, dma_make_data(B_spad, B_in), state_row_stride, state_row_bytes, state_row_bytes, 1);
    dma_queue_push(dma_q, dma_make_data(C_spad, C_in), state_row_stride, state_row_bytes, state_row_bytes, 1);
    dma_queue_push(dma_q, dma_make_data(x_spad, x_in), x_bytes, x_bytes, x_bytes, 1);
    dma_queue_push(dma_q, dma_make_data(dt_spad, dt_in), sizeof(float), sizeof(float), sizeof(float), 1);
    dma_queue_push(dma_q, dma_make_data(A_spad, A_in), a_bytes, a_bytes, a_bytes, 1);
}

static void ssm_scan_thread_f32_decode(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_scan_context *             scctx      = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx       = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams    = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout     = &scctx->layout;
    const struct htp_tensor *                 s          = octx->src[0];
    const struct htp_tensor *                 dst        = octx->dst;
    const uint32_t                            unit_start = scctx->unit_start + scctx->units_per_hvx * ith;
    const uint32_t                            unit_end   = MIN(unit_start + scctx->units_per_hvx, scctx->unit_start + scctx->nunits);

    if (unit_start >= unit_end) {
        return;
    }

    dma_queue *               dma_q              = th->dma_q;
    struct htp_thread_trace * tr                 = th->trace;
    uint8_t *                 base               = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;
    const size_t              state_row_bytes    = (size_t) kparams->d_state * sizeof(float);
    const size_t              state_row_stride   = layout->state_row_stride;
    const size_t              state_stride_elems = state_row_stride / sizeof(float);

    for (uint32_t unit_index = unit_start, spad_idx = 0; unit_index < unit_end && spad_idx < 2; ++unit_index, ++spad_idx) {
        const struct htp_ssm_scan_unit u = ssm_scan_decode_unit(kparams, unit_index);

        dma_queue_push(dma_q, dma_make_data(dst->data, base + layout->off_y[spad_idx]), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(dst->data, base + layout->off_state_out[spad_idx]), 0, 0, 0, 0);

        ssm_scan_decode_push_inputs(dma_q, scctx, &u,
                                    base + layout->off_state[spad_idx],
                                    base + layout->off_B[spad_idx],
                                    base + layout->off_C[spad_idx],
                                    base + layout->off_x[spad_idx],
                                    base + layout->off_dt[spad_idx],
                                    base + layout->off_A[spad_idx]);
    }

    for (uint32_t unit_index = unit_start; unit_index < unit_end; ++unit_index) {
        float *       y_work    = (float *)       (uintptr_t) dma_queue_pop(dma_q).src;
        float *       state_out = (float *)       (uintptr_t) dma_queue_pop(dma_q).src;
        const float * state_in  = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;
        const float * B_work    = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;
        const float * C_work    = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;
        const float * x_work    = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;
        const float * dt_work   = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;
        const float * A_work    = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;

        const struct htp_ssm_scan_unit unit = ssm_scan_decode_unit(kparams, unit_index);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) unit_index);
        const HVX_Vector vdt_sp      = ssm_scan_vec_softplus_f32(hvx_vmem(dt_work));
        const float      dt_softplus = hvx_vec_get_f32(vdt_sp);
        const float      dA_scalar   = kparams->n_a == 1
            ? hvx_vec_get_f32(hvx_vec_exp_f32(hvx_vec_mul_f32_f32(vdt_sp, hvx_vmem(A_work))))
            : 0.0f;

        uint32_t row = 0;
        if (kparams->fast_step4) {
            const HVX_Vector vdt = hvx_vec_splat_f32(dt_softplus);
            const HVX_Vector vdA = hvx_vec_splat_f32(dA_scalar);
            HVX_Vector       vx  = Q6_V_vzero();
            for (; row + 4 <= unit.nrows; row += 4) {
                if ((row & (VLEN_FP32 - 1)) == 0) {
                    vx = hvx_vec_mul_f32_f32(hvx_vmem(x_work + row), vdt);
                }
                ssm_scan_step4(state_in + (size_t) row * state_stride_elems,
                               state_out + (size_t) row * state_stride_elems, state_stride_elems,
                               B_work, C_work, vdA,
                               Q6_V_vror_VR(vx, (row & (VLEN_FP32 - 1)) * sizeof(float)),
                               kparams->d_state, y_work + row);
            }
        }
        for (; row < unit.nrows; ++row) {
            ssm_scan_step1(state_in + (size_t) row * state_stride_elems,
                           state_out + (size_t) row * state_stride_elems,
                           B_work, C_work, A_work, kparams->n_a, dt_softplus, dA_scalar,
                           x_work[row] * dt_softplus,
                           kparams->d_state, y_work + row);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) unit_index);

        const size_t     y_output_bytes = (size_t) unit.nrows * sizeof(float);
        const dma_addr_t y_out_ddr      = dst->data + (size_t) unit.q * kparams->y_seq_stride +
                                          ((size_t) unit.head * kparams->head_dim + unit.row0) * sizeof(float);
        const dma_addr_t state_out_ddr  = dst->data + kparams->y_bytes + (size_t) unit.q * s->nb[3] +
                                          (size_t) unit.head * s->nb[2] + (size_t) unit.row0 * s->nb[1];

        dma_queue_push(dma_q, dma_make_data(y_out_ddr, y_work), y_output_bytes, layout->xy_row_stride, y_output_bytes, 1);
        dma_queue_push(dma_q, dma_make_data(state_out_ddr, state_out), s->nb[1], state_row_stride, state_row_bytes, unit.nrows);

        const uint32_t next_unit = unit_index + 2;
        if (next_unit < unit_end) {
            const struct htp_ssm_scan_unit next = ssm_scan_decode_unit(kparams, next_unit);
            ssm_scan_decode_push_inputs(dma_q, scctx, &next,
                                        (void *) state_in, (void *) B_work, (void *) C_work,
                                        (void *) x_work, (void *) dt_work, (void *) A_work);
        }
    }

    dma_queue_flush(dma_q);
    (void) nth;
}

static inline void ssm_scan_prefill_push_fetch(
    dma_queue * dma_q,
    const struct htp_ssm_scan_context * scctx,
    uint8_t * base,
    uint32_t unit_index,
    uint32_t u_idx,
    uint32_t c,
    uint32_t s_idx
) {
    const struct htp_ops_context *            octx             = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams          = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout           = &scctx->layout;
    const struct htp_tensor *                 s                = octx->src[0];
    const struct htp_tensor *                 x                = octx->src[1];
    const struct htp_tensor *                 dt               = octx->src[2];
    const struct htp_tensor *                 A                = octx->src[3];
    const struct htp_tensor *                 B                = octx->src[4];
    const struct htp_tensor *                 C                = octx->src[5];
    const struct htp_tensor *                 ids              = octx->src[6];
    const struct htp_ssm_scan_unit            unit             = ssm_scan_decode_unit(kparams, unit_index);
    const uint32_t                            t0               = c * kparams->token_tile;
    const uint32_t                            nt               = MIN(kparams->token_tile, kparams->n_tokens - t0);
    const size_t                              state_row_bytes  = (size_t) kparams->d_state * sizeof(float);
    const size_t                              state_row_stride = layout->state_row_stride;
    const size_t                              xy_stride        = layout->xy_row_stride;
    const size_t                              row_bytes        = (size_t) unit.nrows * sizeof(float);

    if (c == 0) {
        const int32_t    id       = ((const int32_t *) (uintptr_t) ids->data)[unit.q];
        const size_t     a_bytes  = (size_t) kparams->n_a * sizeof(float);
        const dma_addr_t state_in = s->data + (size_t) id * s->nb[3] + (size_t) unit.head * s->nb[2] + (size_t) unit.row0 * s->nb[1];
        const dma_addr_t A_in     = A->data + (size_t) unit.head * A->nb[1];

        dma_queue_push(dma_q, dma_make_data(base + layout->off_state[u_idx], state_in),
                       state_row_stride, s->nb[1], state_row_bytes, unit.nrows);
        dma_queue_push(dma_q, dma_make_data(base + layout->off_A[u_idx], A_in),
                       a_bytes, a_bytes, a_bytes, 1);
    } else {
        dma_queue_push(dma_q, dma_make_data(base + layout->off_state[u_idx], s->data), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(base + layout->off_A[u_idx], A->data), 0, 0, 0, 0);
    }

    const dma_addr_t B_in  = B->data + (size_t) unit.q * B->nb[3] + (size_t) unit.group * B->nb[1] + (size_t) t0 * B->nb[2];
    const dma_addr_t C_in  = C->data + (size_t) unit.q * C->nb[3] + (size_t) unit.group * C->nb[1] + (size_t) t0 * C->nb[2];
    const dma_addr_t x_in  = x->data + (size_t) unit.q * x->nb[3] + (size_t) unit.head * x->nb[1] + (size_t) unit.row0 * sizeof(float) + (size_t) t0 * x->nb[2];
    const dma_addr_t dt_in = dt->data + (size_t) unit.q * dt->nb[2] + (size_t) unit.head * sizeof(float) + (size_t) t0 * dt->nb[1];

    dma_queue_push(dma_q, dma_make_data(base + layout->off_B[s_idx], B_in), state_row_stride, B->nb[2], state_row_bytes, nt);
    dma_queue_push(dma_q, dma_make_data(base + layout->off_C[s_idx], C_in), state_row_stride, C->nb[2], state_row_bytes, nt);
    dma_queue_push(dma_q, dma_make_data(base + layout->off_x[s_idx], x_in), xy_stride, x->nb[2], row_bytes, nt);
    dma_queue_push(dma_q, dma_make_data(base + layout->off_dt[s_idx], dt_in), sizeof(float), dt->nb[1], sizeof(float), nt);
}

static void ssm_scan_thread_f32_prefill(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_scan_context *             scctx      = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx       = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams    = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout     = &scctx->layout;
    const struct htp_tensor *                 s          = octx->src[0];
    const struct htp_tensor *                 dst        = octx->dst;
    const uint32_t                            unit_start = scctx->unit_start + scctx->units_per_hvx * ith;
    const uint32_t                            unit_end   = MIN(unit_start + scctx->units_per_hvx, scctx->unit_start + scctx->nunits);
    if (unit_start >= unit_end) {
        return;
    }

    dma_queue *               dma_q              = th->dma_q;
    struct htp_thread_trace * tr                 = th->trace;
    uint8_t *                 base               = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;
    const uint32_t            token_tile         = kparams->token_tile;
    const uint32_t            n_tokens           = kparams->n_tokens;
    const uint32_t            nchunks            = kparams->n_chunks;
    const uint32_t            total_steps        = (unit_end - unit_start) * nchunks;
    const size_t              xy_stride          = layout->xy_row_stride;
    const size_t              state_row_bytes    = (size_t) kparams->d_state * sizeof(float);
    const size_t              state_row_stride   = layout->state_row_stride;
    const size_t              state_stride_elems = state_row_stride / sizeof(float);
    const size_t              y_stride           = kparams->y_token_stride;

    uint32_t pref_u     = unit_start;
    uint32_t pref_u_idx = 0;
    uint32_t pref_c     = 0;

    for (uint32_t s_idx = 0; s_idx < 2 && s_idx < total_steps; ++s_idx) {
        dma_queue_push(dma_q, dma_make_data(dst->data, base + layout->off_y[s_idx]), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(dst->data, base + layout->off_state_out[s_idx]), 0, 0, 0, 0);
        ssm_scan_prefill_push_fetch(dma_q, scctx, base, pref_u, pref_u_idx, pref_c, s_idx);
        if (++pref_c == nchunks) {
            pref_c      = 0;
            pref_u     += 1;
            pref_u_idx ^= 1;
        }
    }

    uint32_t                 cur_u          = unit_start;
    uint32_t                 cur_c          = 0;
    struct htp_ssm_scan_unit unit           = { 0 };
    HVX_Vector               vA             = Q6_V_vzero();
    dma_addr_t               state_out_base = 0;
    dma_addr_t               y_base         = 0;

    for (uint32_t step = 0; step < total_steps; ++step) {
        uint8_t *       y_work    = (uint8_t *)       (uintptr_t) dma_queue_pop(dma_q).src;
        float *         state_out = (float *)         (uintptr_t) dma_queue_pop(dma_q).src;
        float *         state_work = (float *)        (uintptr_t) dma_queue_pop(dma_q).dst;
        const float *   A_work    = (const float *)   (uintptr_t) dma_queue_pop(dma_q).dst;
        const float *   B_work    = (const float *)   (uintptr_t) dma_queue_pop(dma_q).dst;
        const float *   C_work    = (const float *)   (uintptr_t) dma_queue_pop(dma_q).dst;
        const uint8_t * x_work    = (const uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;
        const float *   dt_work   = (const float *)   (uintptr_t) dma_queue_pop(dma_q).dst;

        if (cur_c == 0) {
            unit           = ssm_scan_decode_unit(kparams, cur_u);
            vA             = kparams->n_a == 1 ? hvx_vec_splat_f32(hvx_vec_get_f32(hvx_vmem(A_work))) : Q6_V_vzero();
            state_out_base = dst->data + kparams->y_bytes + (size_t) unit.head * s->nb[2] + (size_t) unit.row0 * s->nb[1];
            y_base         = dst->data + (size_t) unit.q * kparams->y_seq_stride +
                             ((size_t) unit.head * kparams->head_dim + unit.row0) * sizeof(float);
        }

        const uint32_t t0         = cur_c * token_tile;
        const uint32_t nt         = MIN(token_tile, n_tokens - t0);
        const bool     last_chunk = (cur_c + 1 == nchunks);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) step);
        float __attribute__((aligned(128))) dt_buf[VLEN_FP32];
        float __attribute__((aligned(128))) dA_buf[VLEN_FP32];
        const HVX_Vector vdt_sp = ssm_scan_vec_softplus_f32(hvx_vmem(dt_work));
        hvx_vmem(dt_buf)        = vdt_sp;
        hvx_vmem(dA_buf)        = kparams->n_a == 1 ? hvx_vec_exp_f32(hvx_vec_mul_f32_f32(vdt_sp, vA)) : Q6_V_vzero();
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) step);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) step);
        float * state_dst = last_chunk ? state_out : state_work;
        for (uint32_t t = 0; t < nt; ++t) {
            const float * state_src = (last_chunk && t > 0) ? state_out : state_work;
            const float * B_t       = B_work + (size_t) t * state_stride_elems;
            const float * C_t       = C_work + (size_t) t * state_stride_elems;
            const float * x_t       = (const float *) (x_work + (size_t) t * xy_stride);
            float *       y_t       = (float *) (y_work + (size_t) t * xy_stride);
            uint32_t      row       = 0;
            if (kparams->fast_step4) {
                const HVX_Vector vdt = hvx_vec_splat_f32(dt_buf[t]);
                const HVX_Vector vdA = hvx_vec_splat_f32(dA_buf[t]);
                HVX_Vector       vx  = Q6_V_vzero();
                for (; row + 4 <= unit.nrows; row += 4) {
                    if ((row & (VLEN_FP32 - 1)) == 0) {
                        vx = hvx_vec_mul_f32_f32(hvx_vmem(x_t + row), vdt);
                    }
                    ssm_scan_step4(state_src + (size_t) row * state_stride_elems,
                                   state_dst + (size_t) row * state_stride_elems, state_stride_elems,
                                   B_t, C_t, vdA,
                                   Q6_V_vror_VR(vx, (row & (VLEN_FP32 - 1)) * sizeof(float)),
                                   kparams->d_state, y_t + row);
                }
            }
            for (; row < unit.nrows; ++row) {
                ssm_scan_step1(state_src + (size_t) row * state_stride_elems,
                               state_dst + (size_t) row * state_stride_elems,
                               B_t, C_t, A_work, kparams->n_a, dt_buf[t], dA_buf[t],
                               x_t[row] * dt_buf[t],
                               kparams->d_state, y_t + row);
            }
        }

        if (!last_chunk && kparams->K > 1) {
            const uint32_t slot = n_tokens - 1 - t0;
            if (slot < kparams->K) {
                const size_t total_elems = (size_t) unit.nrows * state_stride_elems;
                for (size_t i = 0; i < total_elems; i += VLEN_FP32) {
                    hvx_vmem(state_out + i) = hvx_vmem(state_work + i);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) step);

        const size_t row_bytes = (size_t) unit.nrows * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(y_base + (size_t) t0 * y_stride, y_work),
                       y_stride, xy_stride, row_bytes, nt);

        const uint32_t slot = n_tokens - 1 - t0;
        if (last_chunk || (kparams->K > 1 && slot < kparams->K)) {
            const size_t     seq_idx       = last_chunk ? (size_t) unit.q : ((size_t) slot * kparams->n_seqs + unit.q);
            const dma_addr_t state_out_ddr = state_out_base + seq_idx * s->nb[3];
            dma_queue_push(dma_q, dma_make_data(state_out_ddr, state_out),
                           s->nb[1], state_row_stride, state_row_bytes, unit.nrows);
        } else {
            dma_queue_push(dma_q, dma_make_data(state_out_base, state_out), 0, 0, 0, 0);
        }

        if (step + 2 < total_steps) {
            ssm_scan_prefill_push_fetch(dma_q, scctx, base, pref_u, pref_u_idx, pref_c, step & 1);
            if (++pref_c == nchunks) {
                pref_c      = 0;
                pref_u     += 1;
                pref_u_idx ^= 1;
            }
        }

        if (++cur_c == nchunks) {
            cur_c  = 0;
            cur_u += 1;
        }
    }

    dma_queue_flush(dma_q);
    (void) nth;
}

static int op_ssm_scan_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * s   = octx->src[0];
    const struct htp_tensor * ids = octx->src[6];
    const struct htp_tensor * dst = octx->dst;

    const struct htp_ssm_scan_kernel_params * kparams =
        (const struct htp_ssm_scan_kernel_params *) octx->kernel_params;

    if (htp_tensor_is_extended(ids)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t * ids_data = (const int32_t *) ids->data;
    for (uint32_t q = 0; q < kparams->n_seqs; ++q) {
        if (ids_data[q] < 0 || (uint32_t) ids_data[q] >= s->ne[3]) {
            return HTP_STATUS_INVAL_PARAMS;
        }
    }
    if (octx->flags & HTP_OPFLAGS_STUB) {
        return HTP_STATUS_OK;
    }

    uint32_t unit_start = 0;
    uint32_t nunits     = kparams->total_units;

    if (octx->ctx->mdev.count > 1) {
        const uint32_t units_per_chunk = htp_tensor_mdev_data_aligned(dst) ? 1 : 0;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            nunits, units_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        unit_start = range.start;
        nunits     = range.count;
    }

    if (nunits == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx         = kparams->n_hvx;
    const uint32_t units_per_hvx = octx->ctx->mdev.count > 1
        ? fastdiv(nunits + n_hvx - 1, &kparams->div_n_hvx)
        : kparams->units_per_hvx;

    struct htp_ssm_scan_context scctx = {
        .octx          = octx,
        .kparams       = kparams,
        .unit_start    = unit_start,
        .nunits        = nunits,
        .units_per_hvx = units_per_hvx,
    };

    htp_ssm_scan_vtcm_layout_build(&scctx.layout, kparams->d_state, kparams->row_tile, kparams->token_tile, n_hvx);

    octx->n_hvx                 = n_hvx;
    octx->src0_spad.data         = octx->ctx->vtcm_base;
    octx->src0_spad.size_per_hvx = kparams->vtcm_per_hvx;
    octx->src0_spad.size         = kparams->vtcm_size;

    switch (kparams->kernel_type) {
        case HTP_SSM_SCAN_KERNEL_DECODE:
            work_queue_run(octx->ctx->hvx_queue, ssm_scan_thread_f32_decode, &scctx, n_hvx);
            break;

        case HTP_SSM_SCAN_KERNEL_PREFILL:
            work_queue_run(octx->ctx->hvx_queue, ssm_scan_thread_f32_prefill, &scctx, n_hvx);
            break;

        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    return octx->status;
}

int op_ssm_scan(struct htp_ops_context * octx) {
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            return op_ssm_scan_f32(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
