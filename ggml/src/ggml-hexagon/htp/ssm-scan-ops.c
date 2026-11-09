#include <math.h>
#include <string.h>

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
    dma_addr_t                                y_bytes;
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

enum {
    SSM_SCAN_DECODE_PACKAGE_SIZE = 8,
    SSM_SCAN_CHUNK_PACKAGE_SIZE  = 5,
    SSM_SCAN_STATE_PACKAGE_SIZE  = 3,
    SSM_SCAN_PAIR_PACKAGE_SIZE   = 7,
};

struct ssm_scan_dma_desc {
    dma_data data;
    size_t   dst_stride;
    size_t   src_stride;
    size_t   row_size;
    size_t   nrows;
};

static inline float ssm_scan_softplus(float x) {
    return x > 20.0f ? x : logf(1.0f + expf(x));
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

static bool ssm_scan_dma_submit(
    struct htp_ops_context * octx,
    dma_queue * dma_q,
    const struct ssm_scan_dma_desc * desc,
    uint32_t ndesc
) {
    for (uint32_t i = 0; i < ndesc; ++i) {
        if (!dma_queue_push(dma_q, desc[i].data, desc[i].dst_stride, desc[i].src_stride, desc[i].row_size,
                            desc[i].nrows)) {
            dma_queue_flush(dma_q);
            htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
            return false;
        }
    }
    return true;
}

// state_in and state_out may alias (in-place prefill update).
static inline void ssm_scan_mamba2_vtcm_step4(
    const float * state_in,
    float * state_out,
    const float * restrict B,
    const float * restrict C,
    HVX_Vector vdA,
    HVX_Vector vx,
    uint32_t d_state,
    float * restrict y
) {
    HVX_Vector acc0   = Q6_V_vzero();
    HVX_Vector acc1   = Q6_V_vzero();
    HVX_Vector acc2   = Q6_V_vzero();
    HVX_Vector acc3   = Q6_V_vzero();
    HVX_Vector vx0    = hvx_vec_repl_f32(vx);
    HVX_Vector vx1    = hvx_vec_repl_f32(Q6_V_vror_VR(vx, sizeof(float)));
    HVX_Vector vx2    = hvx_vec_repl_f32(Q6_V_vror_VR(vx, 2 * sizeof(float)));
    HVX_Vector vx3    = hvx_vec_repl_f32(Q6_V_vror_VR(vx, 3 * sizeof(float)));
    const float * in0  = state_in;
    const float * in1  = in0 + d_state;
    const float * in2  = in1 + d_state;
    const float * in3  = in2 + d_state;
    float *       out0 = state_out;
    float *       out1 = out0 + d_state;
    float *       out2 = out1 + d_state;
    float *       out3 = out2 + d_state;

    for (uint32_t i = 0; i < d_state; i += VLEN_FP32) {
        HVX_Vector b = hvx_vmem(B + i);
        HVX_Vector c = hvx_vmem(C + i);
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

static bool ssm_scan_decode_submit_package(
    struct htp_ssm_scan_context * scctx,
    dma_queue * dma_q,
    uint8_t * base,
    uint32_t slot,
    bool have_previous,
    struct htp_ssm_scan_unit previous,
    bool have_next,
    struct htp_ssm_scan_unit next
) {
    struct htp_ops_context * octx = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams = scctx->kparams;
    const struct htp_tensor * s   = octx->src[0];
    const struct htp_tensor * x   = octx->src[1];
    const struct htp_tensor * dt  = octx->src[2];
    const struct htp_tensor * A   = octx->src[3];
    const struct htp_tensor * B   = octx->src[4];
    const struct htp_tensor * C   = octx->src[5];
    const struct htp_tensor * ids = octx->src[6];
    const struct htp_tensor * dst = octx->dst;
    const struct htp_ssm_scan_vtcm_layout * layout = &scctx->layout;
    uint8_t * state_work     = base + layout->off_state[slot];
    uint8_t * state_out_work = base + layout->off_state_out[slot];
    uint8_t * B_work         = base + layout->off_B[slot];
    uint8_t * C_work         = base + layout->off_C[slot];
    uint8_t * x_work         = base + layout->off_x[slot];
    uint8_t * dt_work        = base + layout->off_dt[slot];
    uint8_t * A_work         = base + layout->off_A[slot];
    uint8_t * y_work         = base + layout->off_y[slot];

    dma_addr_t y_out             = dst->data;
    dma_addr_t state_out         = dst->data;
    size_t     output_rows       = 0;
    size_t     state_output_rows = 0;
    size_t     y_output_bytes    = 0;
    if (have_previous) {
        y_out = dst->data + ((size_t) previous.q * kparams->n_head * kparams->head_dim +
                            (size_t) previous.head * kparams->head_dim + previous.row0) * sizeof(float);
        state_out = dst->data + scctx->y_bytes + (size_t) previous.q * s->nb[3] +
                    (size_t) previous.head * s->nb[2] + (size_t) previous.row0 * s->nb[1];
        output_rows = 1;
        state_output_rows = previous.nrows;
        y_output_bytes = (size_t) previous.nrows * sizeof(float);
    }

    dma_addr_t state_in         = s->data;
    dma_addr_t B_in             = B->data;
    dma_addr_t C_in             = C->data;
    dma_addr_t x_in             = x->data;
    dma_addr_t dt_in            = dt->data;
    dma_addr_t A_in             = A->data;
    size_t     input_rows       = 0;
    size_t     state_input_rows = 0;
    size_t     x_bytes          = 0;
    if (have_next) {
        const int32_t id = ((const int32_t *) (uintptr_t) ids->data)[next.q];
        state_in = s->data + (size_t) id * s->nb[3] + (size_t) next.head * s->nb[2] + (size_t) next.row0 * s->nb[1];
        B_in = B->data + (size_t) next.q * B->nb[3] + (size_t) next.group * B->nb[1];
        C_in = C->data + (size_t) next.q * C->nb[3] + (size_t) next.group * C->nb[1];
        x_in = x->data + (size_t) next.q * x->nb[3] + (size_t) next.head * x->nb[1] + (size_t) next.row0 * sizeof(float);
        dt_in = dt->data + (size_t) next.q * dt->nb[2] + (size_t) next.head * sizeof(float);
        A_in = A->data + (size_t) next.head * A->nb[1];
        input_rows = 1;
        state_input_rows = next.nrows;
        x_bytes = (size_t) next.nrows * sizeof(float);
    }

    struct ssm_scan_dma_desc desc[SSM_SCAN_DECODE_PACKAGE_SIZE] = {
        { dma_make_data(y_out, y_work), y_output_bytes, layout->xy_row_stride, y_output_bytes, output_rows },
        { dma_make_data(state_out, state_out_work), s->nb[1], kparams->d_state * sizeof(float),
          kparams->d_state * sizeof(float), state_output_rows },
        { dma_make_data(state_work, state_in), kparams->d_state * sizeof(float), s->nb[1],
          kparams->d_state * sizeof(float), state_input_rows },
        { dma_make_data(B_work, B_in), kparams->d_state * sizeof(float), kparams->d_state * sizeof(float),
          kparams->d_state * sizeof(float), input_rows },
        { dma_make_data(C_work, C_in), kparams->d_state * sizeof(float), kparams->d_state * sizeof(float),
          kparams->d_state * sizeof(float), input_rows },
        { dma_make_data(x_work, x_in), x_bytes, x_bytes, x_bytes, input_rows },
        { dma_make_data(dt_work, dt_in), sizeof(float), sizeof(float), sizeof(float), input_rows },
        { dma_make_data(A_work, A_in), sizeof(float), sizeof(float), sizeof(float), input_rows },
    };

    return ssm_scan_dma_submit(octx, dma_q, desc, SSM_SCAN_DECODE_PACKAGE_SIZE);
}

static void ssm_scan_thread_mamba2_vtcm(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_scan_context *             scctx           = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx            = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams         = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout          = &scctx->layout;
    const uint32_t                            unit_start      = scctx->unit_start + scctx->units_per_hvx * ith;
    const uint32_t                            unit_end        = MIN(unit_start + scctx->units_per_hvx, scctx->unit_start + scctx->nunits);

    if (unit_start >= unit_end) {
        return;
    }

    dma_queue * dma_q = th->dma_q;
    uint8_t *   base  = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;
    struct htp_ssm_scan_unit slot_unit[2];

    uint32_t prime = MIN(2, unit_end - unit_start);
    for (uint32_t i = 0; i < prime; ++i) {
        uint32_t slot   = i;
        slot_unit[slot] = ssm_scan_decode_unit(kparams, unit_start + i);
        struct htp_ssm_scan_unit dummy      = { 0 };
        bool ok = ssm_scan_decode_submit_package(
            scctx, dma_q, base, slot, false, dummy, true, slot_unit[slot]);
        if (!ok) {
            return;
        }
    }

    struct htp_thread_trace * tr = th->trace;
    for (uint32_t unit_index = unit_start; unit_index < unit_end; ++unit_index) {
        uint32_t slot = (unit_index - unit_start) & 1;
        for (uint32_t i = 0; i < SSM_SCAN_DECODE_PACKAGE_SIZE; ++i) {
            dma_queue_pop(dma_q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) unit_index);
        struct htp_ssm_scan_unit unit = slot_unit[slot];
        const float * state_in  = (const float *) (base + layout->off_state[slot]);
        float *       state_out = (float *) (base + layout->off_state_out[slot]);
        float * B_work  = (float *) (base + layout->off_B[slot]);
        float * C_work  = (float *) (base + layout->off_C[slot]);
        float * x_work  = (float *) (base + layout->off_x[slot]);
        float * dt_work = (float *) (base + layout->off_dt[slot]);
        float * A_work  = (float *) (base + layout->off_A[slot]);
        float * y_work  = (float *) (base + layout->off_y[slot]);
        float dt_value      = hvx_vec_get_f32(hvx_vmem(dt_work));
        float A_value       = hvx_vec_get_f32(hvx_vmem(A_work));
        float dt_softplus   = ssm_scan_softplus(dt_value);
        HVX_Vector vdt = hvx_vec_splat_f32(dt_softplus);
        HVX_Vector vdA = hvx_vec_splat_f32(expf(dt_softplus * A_value));

        HVX_Vector vx = Q6_V_vzero();
        for (uint32_t row = 0; row < unit.nrows; row += 4) {
            if ((row & (VLEN_FP32 - 1)) == 0) {
                vx = hvx_vec_mul_f32_f32(hvx_vmem(x_work + row), vdt);
            }
            ssm_scan_mamba2_vtcm_step4(state_in + (size_t) row * kparams->d_state,
                state_out + (size_t) row * kparams->d_state, B_work, C_work, vdA, Q6_V_vror_VR(vx, (row & (VLEN_FP32 - 1)) * sizeof(float)),
                kparams->d_state, y_work + row);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) unit_index);

        uint32_t future_index = unit_index + 2;
        bool have_future = future_index < unit_end;
        struct htp_ssm_scan_unit future = { 0 };

        if (have_future) {
            future = ssm_scan_decode_unit(kparams, future_index);
            slot_unit[slot] = future;
        }

        bool ok = ssm_scan_decode_submit_package(
            scctx, dma_q, base, slot, true, unit, have_future, future);
        if (!ok) {
            return;
        }
    }

    // Trailing packages hold the final y and state write-backs and zero-row input dummies.
    for (uint32_t i = 0; i < prime * SSM_SCAN_DECODE_PACKAGE_SIZE; ++i) {
        dma_queue_pop(dma_q);
    }

    (void) nth;
}

static bool ssm_scan_prefill_submit_chunk(
    struct htp_ssm_scan_context * scctx,
    dma_queue * dma_q,
    uint8_t * base,
    uint32_t slot,
    const struct htp_ssm_scan_unit * unit,
    bool have_previous,
    uint32_t previous_t0,
    uint32_t previous_nt,
    bool have_next,
    uint32_t next_t0,
    uint32_t next_nt
) {
    struct htp_ops_context *                  octx      = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams   = scctx->kparams;
    const struct htp_tensor *                 x         = octx->src[1];
    const struct htp_tensor *                 dt        = octx->src[2];
    const struct htp_tensor *                 B         = octx->src[4];
    const struct htp_tensor *                 C         = octx->src[5];
    const struct htp_tensor *                 dst       = octx->dst;
    const struct htp_ssm_scan_vtcm_layout *   layout    = &scctx->layout;
    uint8_t *                                 B_work    = base + layout->off_B[slot];
    uint8_t *                                 C_work    = base + layout->off_C[slot];
    uint8_t *                                 x_work    = base + layout->off_x[slot];
    uint8_t *                                 dt_work   = base + layout->off_dt[slot];
    uint8_t *                                 y_work    = base + layout->off_y[slot];
    const size_t                              row_bytes = (size_t) unit->nrows * sizeof(float);
    const size_t                              bc_bytes  = (size_t) kparams->d_state * sizeof(float);

    dma_addr_t y_out  = dst->data;
    size_t     y_rows = 0;
    if (have_previous) {
        y_out = dst->data + ((((size_t) unit->q * kparams->n_tokens + previous_t0) * kparams->n_head + unit->head) *
                             kparams->head_dim + unit->row0) * sizeof(float);
        y_rows = previous_nt;
    }

    dma_addr_t B_in       = B->data;
    dma_addr_t C_in       = C->data;
    dma_addr_t x_in       = x->data;
    dma_addr_t dt_in      = dt->data;
    size_t     input_rows = 0;
    if (have_next) {
        B_in = B->data + (size_t) unit->q * B->nb[3] + (size_t) next_t0 * B->nb[2] + (size_t) unit->group * B->nb[1];
        C_in = C->data + (size_t) unit->q * C->nb[3] + (size_t) next_t0 * C->nb[2] + (size_t) unit->group * C->nb[1];
        x_in = x->data + (size_t) unit->q * x->nb[3] + (size_t) next_t0 * x->nb[2] + (size_t) unit->head * x->nb[1] +
               (size_t) unit->row0 * sizeof(float);
        dt_in = dt->data + (size_t) unit->q * dt->nb[2] + (size_t) next_t0 * dt->nb[1] + (size_t) unit->head * sizeof(float);
        input_rows = next_nt;
    }

    struct ssm_scan_dma_desc desc[SSM_SCAN_CHUNK_PACKAGE_SIZE] = {
        { dma_make_data(y_out, y_work), (size_t) kparams->n_head * kparams->head_dim * sizeof(float),
          layout->xy_row_stride, row_bytes, y_rows },
        { dma_make_data(B_work, B_in), bc_bytes, B->nb[2], bc_bytes, input_rows },
        { dma_make_data(C_work, C_in), bc_bytes, C->nb[2], bc_bytes, input_rows },
        { dma_make_data(x_work, x_in), layout->xy_row_stride, x->nb[2], row_bytes, input_rows },
        { dma_make_data(dt_work, dt_in), sizeof(float), dt->nb[1], sizeof(float), input_rows },
    };
    return ssm_scan_dma_submit(octx, dma_q, desc, SSM_SCAN_CHUNK_PACKAGE_SIZE);
}

// Two adjacent row blocks of one head share B, C and dt. The first member uses the x/y tiles, the second x1/y1.
static inline bool ssm_scan_prefill_pair_at(
    const struct htp_ssm_scan_kernel_params * kparams,
    uint32_t unit_index,
    uint32_t unit_end,
    struct htp_ssm_scan_unit * first,
    struct htp_ssm_scan_unit * second
) {
    *first = ssm_scan_decode_unit(kparams, unit_index);
    *second = *first;
    if (kparams->row_tile > VLEN_FP32 || unit_index + 1 >= unit_end) {
        return false;
    }
    *second = ssm_scan_decode_unit(kparams, unit_index + 1);

    return first->q == second->q && first->group == second->group && first->head == second->head;
}

static bool ssm_scan_prefill_submit_pair_chunk(
    struct htp_ssm_scan_context * scctx,
    dma_queue * dma_q,
    uint8_t * base,
    uint32_t slot,
    const struct htp_ssm_scan_unit * u0,
    const struct htp_ssm_scan_unit * u1,
    bool have_previous,
    uint32_t previous_t0,
    uint32_t previous_nt,
    bool have_next,
    uint32_t next_t0,
    uint32_t next_nt
) {
    struct htp_ops_context *                  octx       = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams    = scctx->kparams;
    const struct htp_tensor *                 x          = octx->src[1];
    const struct htp_tensor *                 dt         = octx->src[2];
    const struct htp_tensor *                 B          = octx->src[4];
    const struct htp_tensor *                 C          = octx->src[5];
    const struct htp_tensor *                 dst        = octx->dst;
    const struct htp_ssm_scan_vtcm_layout *   layout     = &scctx->layout;
    uint8_t *                                 B_work     = base + layout->off_B[slot];
    uint8_t *                                 C_work     = base + layout->off_C[slot];
    uint8_t *                                 dt_work    = base + layout->off_dt[slot];
    uint8_t *                                 x0_work    = base + layout->off_x[slot];
    uint8_t *                                 y0_work    = base + layout->off_y[slot];
    uint8_t *                                 x1_work    = base + layout->off_x1[slot];
    uint8_t *                                 y1_work    = base + layout->off_y1[slot];
    const size_t                              row_bytes0 = (size_t) u0->nrows * sizeof(float);
    const size_t                              row_bytes1 = (size_t) u1->nrows * sizeof(float);
    const size_t                              bc_bytes   = (size_t) kparams->d_state * sizeof(float);

    dma_addr_t y_out0 = dst->data;
    dma_addr_t y_out1 = dst->data;
    size_t     y_rows = 0;
    if (have_previous) {
        const size_t y_base =
            (((size_t) u0->q * kparams->n_tokens + previous_t0) * kparams->n_head + u0->head) * kparams->head_dim;
        y_out0 = dst->data + (y_base + u0->row0) * sizeof(float);
        y_out1 = dst->data + (y_base + u1->row0) * sizeof(float);
        y_rows = previous_nt;
    }

    dma_addr_t B_in       = B->data;
    dma_addr_t C_in       = C->data;
    dma_addr_t x_in0      = x->data;
    dma_addr_t x_in1      = x->data;
    dma_addr_t dt_in      = dt->data;
    size_t     input_rows = 0;
    if (have_next) {
        B_in = B->data + (size_t) u0->q * B->nb[3] + (size_t) next_t0 * B->nb[2] + (size_t) u0->group * B->nb[1];
        C_in = C->data + (size_t) u0->q * C->nb[3] + (size_t) next_t0 * C->nb[2] + (size_t) u0->group * C->nb[1];
        const size_t x_base = (size_t) u0->q * x->nb[3] + (size_t) next_t0 * x->nb[2] + (size_t) u0->head * x->nb[1];
        x_in0 = x->data + x_base + (size_t) u0->row0 * sizeof(float);
        x_in1 = x->data + x_base + (size_t) u1->row0 * sizeof(float);
        dt_in = dt->data + (size_t) u0->q * dt->nb[2] + (size_t) next_t0 * dt->nb[1] + (size_t) u0->head * sizeof(float);
        input_rows = next_nt;
    }

    const size_t y_stride = (size_t) kparams->n_head * kparams->head_dim * sizeof(float);
    struct ssm_scan_dma_desc desc[SSM_SCAN_PAIR_PACKAGE_SIZE] = {
        { dma_make_data(y_out0, y0_work), y_stride, layout->xy_row_stride, row_bytes0, y_rows },
        { dma_make_data(y_out1, y1_work), y_stride, layout->xy_row_stride, row_bytes1, y_rows },
        { dma_make_data(B_work, B_in), bc_bytes, B->nb[2], bc_bytes, input_rows },
        { dma_make_data(C_work, C_in), bc_bytes, C->nb[2], bc_bytes, input_rows },
        { dma_make_data(dt_work, dt_in), sizeof(float), dt->nb[1], sizeof(float), input_rows },
        { dma_make_data(x0_work, x_in0), layout->xy_row_stride, x->nb[2], row_bytes0, input_rows },
        { dma_make_data(x1_work, x_in1), layout->xy_row_stride, x->nb[2], row_bytes1, input_rows },
    };

    return ssm_scan_dma_submit(octx, dma_q, desc, SSM_SCAN_PAIR_PACKAGE_SIZE);
}

static bool ssm_scan_prefill_submit_state(
    struct htp_ssm_scan_context * scctx,
    dma_queue * dma_q,
    uint8_t * base,
    bool have_previous,
    const struct htp_ssm_scan_unit * previous,
    uint32_t previous_slot,
    bool have_next,
    const struct htp_ssm_scan_unit * next,
    uint32_t next_slot
) {
    struct htp_ops_context *                  octx            = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams         = scctx->kparams;
    const struct htp_tensor *                 s               = octx->src[0];
    const struct htp_tensor *                 A               = octx->src[3];
    const struct htp_tensor *                 ids             = octx->src[6];
    const struct htp_tensor *                 dst             = octx->dst;
    const struct htp_ssm_scan_vtcm_layout *   layout          = &scctx->layout;
    const size_t                              state_row_bytes = (size_t) kparams->d_state * sizeof(float);

    dma_addr_t state_out      = dst->data;
    size_t     state_out_rows = 0;
    if (have_previous) {
        state_out = dst->data + scctx->y_bytes + (size_t) previous->q * s->nb[3] +
                    (size_t) previous->head * s->nb[2] + (size_t) previous->row0 * s->nb[1];
        state_out_rows = previous->nrows;
    }

    dma_addr_t state_in      = s->data;
    dma_addr_t A_in          = A->data;
    size_t     state_in_rows = 0;
    size_t     A_rows        = 0;
    if (have_next) {
        const int32_t id = ((const int32_t *) (uintptr_t) ids->data)[next->q];
        state_in = s->data + (size_t) id * s->nb[3] + (size_t) next->head * s->nb[2] + (size_t) next->row0 * s->nb[1];
        A_in = A->data + (size_t) next->head * A->nb[1];
        state_in_rows = next->nrows;
        A_rows = 1;
    }

    struct ssm_scan_dma_desc desc[SSM_SCAN_STATE_PACKAGE_SIZE] = {
        { dma_make_data(state_out, base + layout->off_state[previous_slot]), s->nb[1], state_row_bytes,
          state_row_bytes, state_out_rows },
        { dma_make_data(base + layout->off_state[next_slot], state_in), state_row_bytes, s->nb[1],
          state_row_bytes, state_in_rows },
        { dma_make_data(base + layout->off_A[next_slot], A_in), sizeof(float), sizeof(float), sizeof(float), A_rows },
    };

    return ssm_scan_dma_submit(octx, dma_q, desc, SSM_SCAN_STATE_PACKAGE_SIZE);
}

static void ssm_scan_thread_mamba2_vtcm_prefill(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_scan_context *             scctx           = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx            = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams         = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout          = &scctx->layout;
    const uint32_t                            unit_start      = scctx->unit_start + scctx->units_per_hvx * ith;
    const uint32_t                            unit_end        = MIN(unit_start + scctx->units_per_hvx, scctx->unit_start + scctx->nunits);
    if (unit_start >= unit_end) {
        return;
    }

    dma_queue *               dma_q      = th->dma_q;
    struct htp_thread_trace * tr         = th->trace;
    uint8_t *                 base       = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;
    const uint32_t            token_tile = kparams->token_tile;
    const uint32_t            n_tokens   = kparams->n_tokens;
    const uint32_t            nchunks    = (n_tokens + token_tile - 1) / token_tile;
    const uint32_t            prime      = MIN(2, nchunks);
    const size_t              xy_stride  = layout->xy_row_stride;

    struct htp_ssm_scan_unit    unit;
    struct htp_ssm_scan_unit    second;
    bool                        ok      = true;
    bool                        loaded  = false;
    uint32_t                    slot0   = 0;

    for (uint32_t unit_index = unit_start; unit_index < unit_end; ) {
        const bool     pair               = ssm_scan_prefill_pair_at(kparams, unit_index, unit_end, &unit, &second);
        const uint32_t nunits             = pair ? 2 : 1;
        const uint32_t chunk_package_size = pair ? SSM_SCAN_PAIR_PACKAGE_SIZE : SSM_SCAN_CHUNK_PACKAGE_SIZE;

        if (!loaded) {
            slot0   = 0;
            ok = ssm_scan_prefill_submit_state(scctx, dma_q, base, false, &unit, 0, true, &unit, 0);
            if (pair && ok) {
                ok = ssm_scan_prefill_submit_state(scctx, dma_q, base, false, &second, 1, true, &second, 1);
            }
            if (!ok) {
                return;
            }
            for (uint32_t i = 0; i < nunits * SSM_SCAN_STATE_PACKAGE_SIZE; ++i) {
                dma_queue_pop(dma_q);
            }
        }

        float *     state_work  = (float *) (base + layout->off_state[slot0]);
        float *     state_work1 = (float *) (base + layout->off_state[1]);
        const float A_value     = hvx_vec_get_f32(hvx_vmem(base + layout->off_A[slot0]));

        for (uint32_t c = 0; c < prime; ++c) {
            const uint32_t t0 = c * token_tile;
            const uint32_t nt = MIN(token_tile, n_tokens - t0);
            ok = pair ?
                ssm_scan_prefill_submit_pair_chunk(scctx, dma_q, base, c, &unit, &second, false, 0, 0, true, t0, nt) :
                ssm_scan_prefill_submit_chunk(scctx, dma_q, base, c, &unit, false, 0, 0, true, t0, nt);
            if (!ok) {
                return;
            }
        }

        for (uint32_t c = 0; c < nchunks; ++c) {
            const uint32_t slot = c & 1;
            const uint32_t t0   = c * token_tile;
            const uint32_t nt   = MIN(token_tile, n_tokens - t0);
            for (uint32_t i = 0; i < chunk_package_size; ++i) {
                dma_queue_pop(dma_q);
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
            const float *   B_work  = (const float *) (base + layout->off_B[slot]);
            const float *   C_work  = (const float *) (base + layout->off_C[slot]);
            const uint8_t * x_work  = base + layout->off_x[slot];
            const float *   dt_work = (const float *) (base + layout->off_dt[slot]);
            uint8_t *       y_work  = base + layout->off_y[slot];

            // Copy dt out of VTCM with one HVX load, then do the scalar softplus and decay on the stack.
            float __attribute__((aligned(128))) dt_buf[VLEN_FP32];
            float __attribute__((aligned(128))) dA_buf[VLEN_FP32];
            hvx_vmem(dt_buf) = hvx_vmem(dt_work);
            for (uint32_t t = 0; t < nt; ++t) {
                const float dt_softplus = ssm_scan_softplus(dt_buf[t]);
                dt_buf[t]               = dt_softplus;
                dA_buf[t]               = expf(dt_softplus * A_value);
            }
            const HVX_Vector vdt_all = hvx_vmem(dt_buf);
            const HVX_Vector vdA_all = hvx_vmem(dA_buf);

            if (pair) {
                const uint8_t * x1_work = base + layout->off_x1[slot];
                uint8_t *       y1_work = base + layout->off_y1[slot];
                for (uint32_t t = 0; t < nt; ++t) {
                    const HVX_Vector vdt = hvx_vec_repl_f32(Q6_V_vror_VR(vdt_all, t * sizeof(float)));
                    const HVX_Vector vdA = hvx_vec_repl_f32(Q6_V_vror_VR(vdA_all, t * sizeof(float)));
                    const float *    B_t = B_work + (size_t) t * kparams->d_state;
                    const float *    C_t = C_work + (size_t) t * kparams->d_state;
                    const float *    x0  = (const float *) (x_work + (size_t) t * xy_stride);
                    const float *    x1  = (const float *) (x1_work + (size_t) t * xy_stride);
                    float *          y0  = (float *) (y_work + (size_t) t * xy_stride);
                    float *          y1  = (float *) (y1_work + (size_t) t * xy_stride);
                    // row_tile <= VLEN_FP32 for pairs: one vector holds all x rows of a member.
                    const HVX_Vector vx0 = hvx_vec_mul_f32_f32(hvx_vmem(x0), vdt);
                    const HVX_Vector vx1 = hvx_vec_mul_f32_f32(hvx_vmem(x1), vdt);
                    for (uint32_t row = 0; row < unit.nrows; row += 4) {
                        float * st = state_work + (size_t) row * kparams->d_state;
                        ssm_scan_mamba2_vtcm_step4(st, st, B_t, C_t, vdA, Q6_V_vror_VR(vx0, row * sizeof(float)),
                                                   kparams->d_state, y0 + row);
                    }
                    for (uint32_t row = 0; row < second.nrows; row += 4) {
                        float * st = state_work1 + (size_t) row * kparams->d_state;
                        ssm_scan_mamba2_vtcm_step4(st, st, B_t, C_t, vdA, Q6_V_vror_VR(vx1, row * sizeof(float)),
                                                   kparams->d_state, y1 + row);
                    }
                }
            } else {
                for (uint32_t t = 0; t < nt; ++t) {
                    const HVX_Vector vdt = hvx_vec_repl_f32(Q6_V_vror_VR(vdt_all, t * sizeof(float)));
                    const HVX_Vector vdA = hvx_vec_repl_f32(Q6_V_vror_VR(vdA_all, t * sizeof(float)));
                    const float *    B_t = B_work + (size_t) t * kparams->d_state;
                    const float *    C_t = C_work + (size_t) t * kparams->d_state;
                    const float *    x_t = (const float *) (x_work + (size_t) t * xy_stride);
                    float *          y_t = (float *) (y_work + (size_t) t * xy_stride);
                    HVX_Vector       vx  = Q6_V_vzero();
                    for (uint32_t row = 0; row < unit.nrows; row += 4) {
                        if ((row & (VLEN_FP32 - 1)) == 0) {
                            vx = hvx_vec_mul_f32_f32(hvx_vmem(x_t + row), vdt);
                        }
                        float * st = state_work + (size_t) row * kparams->d_state;
                        ssm_scan_mamba2_vtcm_step4(st, st, B_t, C_t, vdA,
                                                   Q6_V_vror_VR(vx, (row & (VLEN_FP32 - 1)) * sizeof(float)),
                                                   kparams->d_state, y_t + row);
                    }
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

            const uint32_t future      = c + 2;
            const bool     have_future = future < nchunks;
            const uint32_t future_t0   = have_future ? future * token_tile : 0;
            const uint32_t future_nt   = have_future ? MIN(token_tile, n_tokens - future_t0) : 0;
            ok = pair ?
                ssm_scan_prefill_submit_pair_chunk(scctx, dma_q, base, slot, &unit, &second, true, t0, nt,
                                                   have_future, future_t0, future_nt) :
                ssm_scan_prefill_submit_chunk(scctx, dma_q, base, slot, &unit, true, t0, nt,
                                              have_future, future_t0, future_nt);
            if (!ok) {
                return;
            }
        }

        // Trailing packages hold the final y write-backs and zero-row input dummies.
        for (uint32_t i = 0; i < prime * chunk_package_size; ++i) {
            dma_queue_pop(dma_q);
        }

        const uint32_t next_index = unit_index + nunits;
        struct htp_ssm_scan_unit next;
        struct htp_ssm_scan_unit next_second;
        const bool have_next = next_index < unit_end;
        const bool next_pair = have_next && ssm_scan_prefill_pair_at(kparams, next_index, unit_end, &next, &next_second);
        if (!pair && have_next && !next_pair) {
            // Single to single: overlap the write-back with the next state fetch in the other slot.
            ok = ssm_scan_prefill_submit_state(scctx, dma_q, base, true, &unit, slot0, true, &next, slot0 ^ 1);
            if (!ok) {
                return;
            }
            for (uint32_t i = 0; i < SSM_SCAN_STATE_PACKAGE_SIZE; ++i) {
                dma_queue_pop(dma_q);
            }
            slot0 ^= 1;
            loaded = true;
        } else {
            ok = ssm_scan_prefill_submit_state(scctx, dma_q, base, true, &unit, slot0, false, &unit, slot0);
            if (pair && ok) {
                ok = ssm_scan_prefill_submit_state(scctx, dma_q, base, true, &second, 1, false, &second, 1);
            }
            if (!ok) {
                return;
            }
            for (uint32_t i = 0; i < nunits * SSM_SCAN_STATE_PACKAGE_SIZE; ++i) {
                dma_queue_pop(dma_q);
            }
            loaded = false;
        }
        unit_index = next_index;
    }

    if (!dma_queue_empty(dma_q)) {
        dma_queue_flush(dma_q);
        htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
    }

    (void) nth;
}

// next and prev may alias: the state is updated in place after the first token.
static inline float ssm_scan_mamba2_hvx(float * next, const float * prev,
        const float * restrict B, const float * restrict C, float dA, float x_dt, uint32_t d_state) {
    HVX_Vector       acc   = Q6_V_vzero();
    const HVX_Vector vdA   = hvx_vec_splat_f32(dA);
    const HVX_Vector vx_dt = hvx_vec_splat_f32(x_dt);
    uint32_t         i     = 0;

    for (; i + VLEN_FP32 <= d_state; i += VLEN_FP32) {
        const HVX_Vector state      = hvx_vmemu(prev + i);
        const HVX_Vector next_state = hvx_vec_add_f32_f32(
            hvx_vec_mul_f32_f32(state, vdA), hvx_vec_mul_f32_f32(hvx_vmemu(B + i), vx_dt));
        hvx_vmemu(next + i)         = next_state;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(next_state, hvx_vmemu(C + i)));
    }

    float sum;
    hvx_vec_store_u(&sum, sizeof(sum), hvx_vec_reduce_sum_f32(acc));
    for (; i < d_state; ++i) {
        const float state = prev[i] * dA + B[i] * x_dt;
        sum += state * C[i];
        next[i] = state;
    }

    return sum;
}

static inline void ssm_scan_mamba2_hvx4(float * next0, float * next1,
        float * next2, float * next3, const float * prev0,
        const float * prev1, const float * prev2, const float * prev3,
        const float * restrict B, const float * restrict C, float dA, const float * x_dt,
        uint32_t d_state, float * restrict sums) {
    HVX_Vector       acc0 = Q6_V_vzero();
    HVX_Vector       acc1 = Q6_V_vzero();
    HVX_Vector       acc2 = Q6_V_vzero();
    HVX_Vector       acc3 = Q6_V_vzero();
    const HVX_Vector vdA  = hvx_vec_splat_f32(dA);
    const HVX_Vector vx0  = hvx_vec_splat_f32(x_dt[0]);
    const HVX_Vector vx1  = hvx_vec_splat_f32(x_dt[1]);
    const HVX_Vector vx2  = hvx_vec_splat_f32(x_dt[2]);
    const HVX_Vector vx3  = hvx_vec_splat_f32(x_dt[3]);

    for (uint32_t i = 0; i < d_state; i += VLEN_FP32) {
        const HVX_Vector b = hvx_vmemu(B + i);
        const HVX_Vector c = hvx_vmemu(C + i);
        const HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(prev0 + i), vdA), hvx_vec_mul_f32_f32(b, vx0));
        const HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(prev1 + i), vdA), hvx_vec_mul_f32_f32(b, vx1));
        const HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(prev2 + i), vdA), hvx_vec_mul_f32_f32(b, vx2));
        const HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(prev3 + i), vdA), hvx_vec_mul_f32_f32(b, vx3));
        hvx_vmemu(next0 + i) = out0;
        hvx_vmemu(next1 + i) = out1;
        hvx_vmemu(next2 + i) = out2;
        hvx_vmemu(next3 + i) = out3;
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, c));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, c));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, c));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, c));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4*sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
}

static void ssm_scan_thread_f32(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_scan_context * scctx = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context * octx = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams = scctx->kparams;
    const struct htp_tensor * s   = octx->src[0];
    const struct htp_tensor * x   = octx->src[1];
    const struct htp_tensor * dt  = octx->src[2];
    const struct htp_tensor * A   = octx->src[3];
    const struct htp_tensor * B   = octx->src[4];
    const struct htp_tensor * C   = octx->src[5];
    const struct htp_tensor * ids = octx->src[6];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t d_state  = kparams->d_state;
    const uint32_t head_dim = kparams->head_dim;
    const uint32_t n_head   = kparams->n_head;
    const uint32_t n_tokens = kparams->n_tokens;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t K        = kparams->K;
    const uint32_t h0       = scctx->unit_start + scctx->units_per_hvx * ith;
    const uint32_t h1       = MIN(h0 + scctx->units_per_hvx, scctx->unit_start + scctx->nunits);

    if (h0 >= h1) {
        return;
    }

    const uint8_t * s_data   = (const uint8_t *) s->data;
    const uint8_t * x_data   = (const uint8_t *) x->data;
    const uint8_t * dt_data  = (const uint8_t *) dt->data;
    const uint8_t * A_data   = (const uint8_t *) A->data;
    const uint8_t * B_data   = (const uint8_t *) B->data;
    const uint8_t * C_data   = (const uint8_t *) C->data;
    const int32_t * ids_data = (const int32_t *) ids->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;
    uint8_t *       states   = dst_data + scctx->y_bytes;
    htp_trace_event_start(th->trace, HTP_TRACE_EVT_HVX_COMP, (uint16_t) h0);
    for (uint32_t q = 0; q < n_seqs; ++q) {
        const int32_t id = ids_data[q];

        const uint8_t * s_in  = s_data + (size_t) id * s->nb[3];
        uint8_t *       s_out = states + (size_t) q * s->nb[3];

        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float * x_t  = (const float *) (x_data + (size_t) t * x->nb[2] + (size_t) q * x->nb[3]);
            const float * dt_t = (const float *) (dt_data + (size_t) t * dt->nb[1] + (size_t) q * dt->nb[2]);
            const float * B_t  = (const float *) (B_data + (size_t) t * B->nb[2] + (size_t) q * B->nb[3]);
            const float * C_t  = (const float *) (C_data + (size_t) t * C->nb[2] + (size_t) q * C->nb[3]);
            float *       y_t  = (float *) dst_data + ((size_t) q * n_tokens + t) * n_head * head_dim;

            for (uint32_t h = h0; h < h1; ++h) {
                const uint32_t group       = fastdiv(h, &kparams->div_heads_per_group);
                const float    dt_softplus = ssm_scan_softplus(dt_t[h]);
                const float *  A_h         = (const float *) A_data + (size_t) h * A->nb[1] / sizeof(float);
                const float    dA          = A->ne[0] == 1 ? expf(dt_softplus * A_h[0]) : 0.0f;
                const float *  B_h         = B_t + (size_t) group * B->nb[1] / sizeof(float);
                const float *  C_h         = C_t + (size_t) group * C->nb[1] / sizeof(float);

                uint32_t d = 0;
                if (A->ne[0] == 1 && d_state % VLEN_FP32 == 0) {
                    for (; d + 4 <= head_dim; d += 4) {
                        const uint32_t hd     = d + h * head_dim;
                        const float *  prev0  = (const float *) (s_in + (size_t) h * s->nb[2] + (size_t) d * s->nb[1]);
                        float *        next0  = (float *) (s_out + (size_t) h * s->nb[2] + (size_t) d * s->nb[1]);
                        const float    x_dt[] = { x_t[hd] * dt_softplus, x_t[hd + 1] * dt_softplus,
                                                  x_t[hd + 2] * dt_softplus, x_t[hd + 3] * dt_softplus };
                        float          sums[4];
                        ssm_scan_mamba2_hvx4(next0, next0 + d_state, next0 + 2 * d_state, next0 + 3 * d_state, prev0,
                                             prev0 + d_state, prev0 + 2 * d_state, prev0 + 3 * d_state, B_h, C_h, dA,
                                             x_dt, d_state, sums);
                        y_t[hd]     = sums[0];
                        y_t[hd + 1] = sums[1];
                        y_t[hd + 2] = sums[2];
                        y_t[hd + 3] = sums[3];
                    }
                }
                for (; d < head_dim; ++d) {
                    const uint32_t hd   = d + h * head_dim;
                    const float    x_dt = x_t[hd] * dt_softplus;
                    const float *  prev = (const float *) (s_in + (size_t) h * s->nb[2] + (size_t) d * s->nb[1]);
                    float *        next = (float *) (s_out + (size_t) h * s->nb[2] + (size_t) d * s->nb[1]);
                    float          sum  = 0.0f;
                    if (A->ne[0] == 1 && d_state >= VLEN_FP32) {
                        sum = ssm_scan_mamba2_hvx(next, prev, B_h, C_h, dA, x_dt, d_state);
                    } else {
                        for (uint32_t i = 0; i < d_state; ++i) {
                            const float decay = A->ne[0] == 1 ? dA : expf(dt_softplus * A_h[i]);
                            const float state = prev[i] * decay + B_h[i] * x_dt;
                            sum += state * C_h[i];
                            next[i] = state;
                        }
                    }

                    y_t[hd] = sum;
                }
            }

            const uint32_t slot = n_tokens - 1 - t;
            if (K > 1 && slot > 0 && slot < K) {
                uint8_t * snapshot = states + ((size_t) slot*n_seqs + q)*s->nb[3];
                for (uint32_t h = h0; h < h1; ++h) {
                    memcpy(snapshot + (size_t) h*s->nb[2], s_out + (size_t) h*s->nb[2], s->nb[2]);
                }
            }

            s_in = s_out;
        }
    }
    htp_trace_event_stop(th->trace, HTP_TRACE_EVT_HVX_COMP, (uint16_t) h0);

    (void) nth;
}

static int op_ssm_scan_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * s   = octx->src[0];
    const struct htp_tensor * ids = octx->src[6];
    const struct htp_tensor * dst = octx->dst;

    const struct htp_ssm_scan_kernel_params * kparams =
        (const struct htp_ssm_scan_kernel_params *) octx->kernel_params;

    const bool use_vtcm = kparams->kernel_type == HTP_SSM_SCAN_KERNEL_MAMBA2_VTCM;

    // ids is read through a scalar pointer; the fallback uses direct pointers for all tensors.
    if (htp_tensor_is_extended(ids)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!use_vtcm) {
        for (int i = 0; i < 7; i++) {
            if (htp_tensor_is_extended(octx->src[i])) {
                return HTP_STATUS_NO_SUPPORT;
            }
        }
        if (htp_tensor_is_extended(dst)) {
            return HTP_STATUS_NO_SUPPORT;
        }
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

    // VTCM path: units are (seq, head, row block). Fallback: units are heads.
    uint32_t unit_start = 0;
    uint32_t nunits     = use_vtcm ? kparams->total_units : kparams->n_head;

    if (octx->ctx->mdev.count > 1) {
        uint32_t units_per_chunk = 0;
        if (htp_tensor_mdev_data_aligned(dst)) {
            if (use_vtcm) {
                // y is written by DMA, state in row blocks of 128-byte multiples.
                units_per_chunk = 1;
            } else if ((kparams->n_head * kparams->head_dim * sizeof(float)) % HTP_TENSOR_MDEV_LINE_SIZE == 0) {
                // y and state are written through the cache: split on heads that start a 128-byte line.
                const uint32_t head_bytes = kparams->head_dim * sizeof(float);
                units_per_chunk = HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(head_bytes, HTP_TENSOR_MDEV_LINE_SIZE);
            }
        }
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            nunits, units_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        unit_start = range.start;
        nunits     = range.count;
    }

    if (nunits == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, nunits))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct htp_ssm_scan_context scctx = {
        .octx          = octx,
        .kparams       = kparams,
        .y_bytes       = (size_t) kparams->head_dim * kparams->n_head * kparams->n_tokens * kparams->n_seqs * sizeof(float),
        .unit_start    = unit_start,
        .nunits        = nunits,
        .units_per_hvx = fastdiv(nunits + octx->n_hvx - 1, &octx->n_hvx_div),
    };

    if (use_vtcm) {
        htp_ssm_scan_vtcm_layout_build(&scctx.layout, kparams->d_state, kparams->row_tile, kparams->token_tile,
                                       kparams->n_tokens, kparams->n_hvx);

        octx->src0_spad.data            = octx->ctx->vtcm_base;
        octx->src0_spad.size_per_hvx    = kparams->vtcm_per_hvx;
        octx->src0_spad.size            = kparams->vtcm_size;

        work_queue_func_t worker = kparams->n_tokens == 1 ?
            ssm_scan_thread_mamba2_vtcm : ssm_scan_thread_mamba2_vtcm_prefill;
        work_queue_run(octx->ctx->hvx_queue, worker, &scctx, octx->n_hvx);
    } else {
        work_queue_run(octx->ctx->hvx_queue, ssm_scan_thread_f32, &scctx, octx->n_hvx);
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
