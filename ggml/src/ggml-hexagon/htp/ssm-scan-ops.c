#include <math.h>
#include <string.h>

#include "hvx-exp.h"
#include "hvx-utils.h"
#include "hmx-fa-kernels.h"
#include "hmx-queue.h"
#include "hmx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "ssm-scan-ops.h"

struct htp_ssm_scan_hmx_thread_state {
    uint32_t                 unit_start;
    uint32_t                 unit_end;
    uint32_t                 total_steps;
    uint32_t                 pref_u;
    uint32_t                 pref_u_idx;
    uint32_t                 pref_c;
    uint32_t                 cur_u;
    uint32_t                 cur_c;
    uint32_t                 unit_q;
    uint32_t                 unit_group;
    uint32_t                 unit_head;
    uint32_t                 unit_row0;
    uint32_t                 unit_nrows;
    dma_addr_t               state_out_base;
    dma_addr_t               y_base;
    uint8_t *                y_work;
    float *                  state_out;
    float *                  state_work;
    const float *            A_work;
    float *                  B_work;
    float *                  C_work;
    uint8_t *                x_work;
    const float *            dt_work;
};

struct htp_ssm_scan_context {
    struct htp_ops_context *                  octx;
    const struct htp_ssm_scan_kernel_params * kparams;
    struct htp_ssm_scan_vtcm_layout           layout;
    uint32_t                                  unit_start;
    uint32_t                                  nunits;
    uint32_t                                  units_per_hvx;
    uint32_t                                  cur_step;
    struct htp_ssm_scan_hmx_thread_state      thr[HTP_MAX_NHVX];
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
        HVX_Vector s0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(in0 + i), vdA), Q6_Vqf32_vmpy_VsfVsf(b, vx0)));
        HVX_Vector s1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(in1 + i), vdA), Q6_Vqf32_vmpy_VsfVsf(b, vx1)));
        HVX_Vector s2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(in2 + i), vdA), Q6_Vqf32_vmpy_VsfVsf(b, vx2)));
        HVX_Vector s3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(in3 + i), vdA), Q6_Vqf32_vmpy_VsfVsf(b, vx3)));
        hvx_vmem(out0 + i) = s0;
        hvx_vmem(out1 + i) = s1;
        hvx_vmem(out2 + i) = s2;
        hvx_vmem(out3 + i) = s3;
        acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(s0, c));
        acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(s1, c));
        acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(s2, c));
        acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(s3, c));
    }

    HVX_Vector_x4 acc = { .v = {
        Q6_Vsf_equals_Vqf32(acc0),
        Q6_Vsf_equals_Vqf32(acc1),
        Q6_Vsf_equals_Vqf32(acc2),
        Q6_Vsf_equals_Vqf32(acc3),
    } };
    hvx_vec_store_u(y, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
}

// Prefill 4-row step across nt tokens. For d_state == 128, keeps all 4 state rows
// (16 HVX vectors) in registers across the token chunk to avoid per-token VTCM traffic.
// Assumes x_work has already been scaled by softplus(dt) per token.
static __attribute__((noinline)) void ssm_scan_prefill_step4(
    const float * state_in,
    float * state_out,
    size_t state_stride_elems,
    const float * restrict B_work,
    const float * restrict C_work,
    const uint8_t * restrict x_work,
    uint8_t * restrict y_work,
    size_t xy_stride,
    const float * restrict dA_buf,
    uint32_t row,
    uint32_t nt,
    uint32_t d_state
) {
    const float * in0  = state_in  + (size_t) (row + 0) * state_stride_elems;
    const float * in1  = state_in  + (size_t) (row + 1) * state_stride_elems;
    const float * in2  = state_in  + (size_t) (row + 2) * state_stride_elems;
    const float * in3  = state_in  + (size_t) (row + 3) * state_stride_elems;
    float *       out0 = state_out + (size_t) (row + 0) * state_stride_elems;
    float *       out1 = state_out + (size_t) (row + 1) * state_stride_elems;
    float *       out2 = state_out + (size_t) (row + 2) * state_stride_elems;
    float *       out3 = state_out + (size_t) (row + 3) * state_stride_elems;
    const uint32_t row_vec = row & ~(VLEN_FP32 - 1);
    const uint32_t row_rot = (row & (VLEN_FP32 - 1)) * sizeof(float);

    if (d_state == 4 * VLEN_FP32) {
        HVX_Vector s0_0 = hvx_vmem(in0 + 0 * VLEN_FP32);
        HVX_Vector s0_1 = hvx_vmem(in0 + 1 * VLEN_FP32);
        HVX_Vector s0_2 = hvx_vmem(in0 + 2 * VLEN_FP32);
        HVX_Vector s0_3 = hvx_vmem(in0 + 3 * VLEN_FP32);

        HVX_Vector s1_0 = hvx_vmem(in1 + 0 * VLEN_FP32);
        HVX_Vector s1_1 = hvx_vmem(in1 + 1 * VLEN_FP32);
        HVX_Vector s1_2 = hvx_vmem(in1 + 2 * VLEN_FP32);
        HVX_Vector s1_3 = hvx_vmem(in1 + 3 * VLEN_FP32);

        HVX_Vector s2_0 = hvx_vmem(in2 + 0 * VLEN_FP32);
        HVX_Vector s2_1 = hvx_vmem(in2 + 1 * VLEN_FP32);
        HVX_Vector s2_2 = hvx_vmem(in2 + 2 * VLEN_FP32);
        HVX_Vector s2_3 = hvx_vmem(in2 + 3 * VLEN_FP32);

        HVX_Vector s3_0 = hvx_vmem(in3 + 0 * VLEN_FP32);
        HVX_Vector s3_1 = hvx_vmem(in3 + 1 * VLEN_FP32);
        HVX_Vector s3_2 = hvx_vmem(in3 + 2 * VLEN_FP32);
        HVX_Vector s3_3 = hvx_vmem(in3 + 3 * VLEN_FP32);

        for (uint32_t t = 0; t < nt; ++t) {
            const float * B_t = B_work + (size_t) t * state_stride_elems;
            const float * C_t = C_work + (size_t) t * state_stride_elems;
            const float * x_t = (const float *) (x_work + (size_t) t * xy_stride);
            float *       y_t = (float *)       (y_work + (size_t) t * xy_stride) + row;

            const HVX_Vector vdA = hvx_vec_splat_f32(dA_buf[t]);
            const HVX_Vector vx  = Q6_V_vror_VR(hvx_vmem(x_t + row_vec), row_rot);
            const HVX_Vector vx0 = hvx_vec_repl_f32(vx);
            const HVX_Vector vx1 = hvx_vec_repl_f32(Q6_V_vror_VR(vx, sizeof(float)));
            const HVX_Vector vx2 = hvx_vec_repl_f32(Q6_V_vror_VR(vx, 2 * sizeof(float)));
            const HVX_Vector vx3 = hvx_vec_repl_f32(Q6_V_vror_VR(vx, 3 * sizeof(float)));

            HVX_Vector b0   = hvx_vmem(B_t + 0 * VLEN_FP32);
            HVX_Vector c0   = hvx_vmem(C_t + 0 * VLEN_FP32);
            s0_0            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0_0, vdA), Q6_Vqf32_vmpy_VsfVsf(b0, vx0)));
            HVX_Vector acc0 = Q6_Vqf32_vmpy_VsfVsf(s0_0, c0);
            s1_0            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1_0, vdA), Q6_Vqf32_vmpy_VsfVsf(b0, vx1)));
            HVX_Vector acc1 = Q6_Vqf32_vmpy_VsfVsf(s1_0, c0);
            s2_0            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s2_0, vdA), Q6_Vqf32_vmpy_VsfVsf(b0, vx2)));
            HVX_Vector acc2 = Q6_Vqf32_vmpy_VsfVsf(s2_0, c0);
            s3_0            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s3_0, vdA), Q6_Vqf32_vmpy_VsfVsf(b0, vx3)));
            HVX_Vector acc3 = Q6_Vqf32_vmpy_VsfVsf(s3_0, c0);

            HVX_Vector b1 = hvx_vmem(B_t + 1 * VLEN_FP32);
            HVX_Vector c1 = hvx_vmem(C_t + 1 * VLEN_FP32);
            s0_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0_1, vdA), Q6_Vqf32_vmpy_VsfVsf(b1, vx0)));
            acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(s0_1, c1));
            s1_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1_1, vdA), Q6_Vqf32_vmpy_VsfVsf(b1, vx1)));
            acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(s1_1, c1));
            s2_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s2_1, vdA), Q6_Vqf32_vmpy_VsfVsf(b1, vx2)));
            acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(s2_1, c1));
            s3_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s3_1, vdA), Q6_Vqf32_vmpy_VsfVsf(b1, vx3)));
            acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(s3_1, c1));

            HVX_Vector b2 = hvx_vmem(B_t + 2 * VLEN_FP32);
            HVX_Vector c2 = hvx_vmem(C_t + 2 * VLEN_FP32);
            s0_2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0_2, vdA), Q6_Vqf32_vmpy_VsfVsf(b2, vx0)));
            acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(s0_2, c2));
            s1_2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1_2, vdA), Q6_Vqf32_vmpy_VsfVsf(b2, vx1)));
            acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(s1_2, c2));
            s2_2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s2_2, vdA), Q6_Vqf32_vmpy_VsfVsf(b2, vx2)));
            acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(s2_2, c2));
            s3_2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s3_2, vdA), Q6_Vqf32_vmpy_VsfVsf(b2, vx3)));
            acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(s3_2, c2));

            HVX_Vector b3 = hvx_vmem(B_t + 3 * VLEN_FP32);
            HVX_Vector c3 = hvx_vmem(C_t + 3 * VLEN_FP32);
            s0_3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0_3, vdA), Q6_Vqf32_vmpy_VsfVsf(b3, vx0)));
            acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(s0_3, c3));
            s1_3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1_3, vdA), Q6_Vqf32_vmpy_VsfVsf(b3, vx1)));
            acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(s1_3, c3));
            s2_3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s2_3, vdA), Q6_Vqf32_vmpy_VsfVsf(b3, vx2)));
            acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(s2_3, c3));
            s3_3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s3_3, vdA), Q6_Vqf32_vmpy_VsfVsf(b3, vx3)));
            acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(s3_3, c3));

            HVX_Vector_x4 acc = { .v = {
                Q6_Vsf_equals_Vqf32(acc0),
                Q6_Vsf_equals_Vqf32(acc1),
                Q6_Vsf_equals_Vqf32(acc2),
                Q6_Vsf_equals_Vqf32(acc3),
            } };
            hvx_vec_store_u(y_t, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
        }

        hvx_vmem(out0 + 0 * VLEN_FP32) = s0_0;
        hvx_vmem(out0 + 1 * VLEN_FP32) = s0_1;
        hvx_vmem(out0 + 2 * VLEN_FP32) = s0_2;
        hvx_vmem(out0 + 3 * VLEN_FP32) = s0_3;

        hvx_vmem(out1 + 0 * VLEN_FP32) = s1_0;
        hvx_vmem(out1 + 1 * VLEN_FP32) = s1_1;
        hvx_vmem(out1 + 2 * VLEN_FP32) = s1_2;
        hvx_vmem(out1 + 3 * VLEN_FP32) = s1_3;

        hvx_vmem(out2 + 0 * VLEN_FP32) = s2_0;
        hvx_vmem(out2 + 1 * VLEN_FP32) = s2_1;
        hvx_vmem(out2 + 2 * VLEN_FP32) = s2_2;
        hvx_vmem(out2 + 3 * VLEN_FP32) = s2_3;

        hvx_vmem(out3 + 0 * VLEN_FP32) = s3_0;
        hvx_vmem(out3 + 1 * VLEN_FP32) = s3_1;
        hvx_vmem(out3 + 2 * VLEN_FP32) = s3_2;
        hvx_vmem(out3 + 3 * VLEN_FP32) = s3_3;
        return;
    }

    for (uint32_t t = 0; t < nt; ++t) {
        const float * state_src = (t == 0) ? in0 : out0;
        const float * B_t       = B_work + (size_t) t * state_stride_elems;
        const float * C_t       = C_work + (size_t) t * state_stride_elems;
        const float * x_t       = (const float *) (x_work + (size_t) t * xy_stride);
        float *       y_t       = (float *)       (y_work + (size_t) t * xy_stride);
        const HVX_Vector vdA    = hvx_vec_splat_f32(dA_buf[t]);
        const HVX_Vector vx     = hvx_vmem(x_t + row_vec);
        ssm_scan_step4(state_src, out0, state_stride_elems, B_t, C_t, vdA,
                       Q6_V_vror_VR(vx, row_rot), d_state, y_t + row);
    }
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
    HVX_Vector vx0,
    uint32_t d_state,
    float * restrict y
) {
    HVX_Vector acc        = Q6_V_vzero();
    HVX_Vector vdt        = hvx_vec_splat_f32(dt_softplus);
    HVX_Vector vdA_splat  = hvx_vec_splat_f32(dA_scalar);

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

        uint32_t         row = 0;
        const HVX_Vector vdt = hvx_vec_splat_f32(dt_softplus);
        if (kparams->fast_step4) {
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
            const HVX_Vector vx  = hvx_vec_mul_f32_f32(hvx_vmem(x_work + (row & ~(VLEN_FP32 - 1))), vdt);
            const HVX_Vector vx0 = hvx_vec_repl_f32(Q6_V_vror_VR(vx, (row & (VLEN_FP32 - 1)) * sizeof(float)));
            ssm_scan_step1(state_in + (size_t) row * state_stride_elems,
                           state_out + (size_t) row * state_stride_elems,
                           B_work, C_work, A_work, kparams->n_a, dt_softplus, dA_scalar,
                           vx0, kparams->d_state, y_work + row);
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
        uint8_t *     x_work    = (uint8_t *)     (uintptr_t) dma_queue_pop(dma_q).dst;
        const float * dt_work   = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;

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
        for (uint32_t t = 0; t < nt; ++t) {
            float *          x_t = (float *) (x_work + (size_t) t * xy_stride);
            const HVX_Vector vdt = hvx_vec_splat_f32(dt_buf[t]);
            for (uint32_t r = 0; r < unit.nrows; r += VLEN_FP32) {
                hvx_vmem(x_t + r) = hvx_vec_mul_f32_f32(hvx_vmem(x_t + r), vdt);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) step);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) step);
        float *  state_dst = last_chunk ? state_out : state_work;
        uint32_t row       = 0;
        if (kparams->fast_step4) {
            for (; row + 4 <= unit.nrows; row += 4) {
                ssm_scan_prefill_step4(state_work, state_dst, state_stride_elems,
                                       B_work, C_work, x_work, y_work, xy_stride,
                                       dA_buf, row, nt, kparams->d_state);
            }
        }
        if (row < unit.nrows) {
            const uint32_t row0 = row;
            for (uint32_t t = 0; t < nt; ++t) {
                const float * state_src = (last_chunk && t > 0) ? state_out : state_work;
                const float * B_t       = B_work + (size_t) t * state_stride_elems;
                const float * C_t       = C_work + (size_t) t * state_stride_elems;
                const float * x_t       = (const float *) (x_work + (size_t) t * xy_stride);
                float *       y_t       = (float *) (y_work + (size_t) t * xy_stride);
                for (uint32_t r = row0; r < unit.nrows; ++r) {
                    const HVX_Vector vx  = hvx_vmem(x_t + (r & ~(VLEN_FP32 - 1)));
                    const HVX_Vector vx0 = hvx_vec_repl_f32(Q6_V_vror_VR(vx, (r & (VLEN_FP32 - 1)) * sizeof(float)));
                    ssm_scan_step1(state_src + (size_t) r * state_stride_elems,
                                   state_dst + (size_t) r * state_stride_elems,
                                   B_t, C_t, A_work, kparams->n_a, dt_buf[t], dA_buf[t],
                                   vx0, kparams->d_state, y_t + r);
                }
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

// Use qf32 vadd with zero so exact 0.0f converts to exact 0.0f16 on v81.
static inline HVX_Vector ssm_scan_f32_to_f16_shuff(HVX_Vector v0, HVX_Vector v1) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector q0   = Q6_Vqf32_vadd_VsfVsf(v0, zero);
    const HVX_Vector q1   = Q6_Vqf32_vadd_VsfVsf(v1, zero);
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
}

static inline HVX_Vector ssm_scan_f32_to_f16(HVX_Vector v0, HVX_Vector v1) {
    return Q6_Vh_vdeal_Vh(ssm_scan_f32_to_f16_shuff(v0, v1));
}

static inline void ssm_scan_vshuff_w(HVX_Vector * a, HVX_Vector * b) {
    const HVX_VectorPair p = Q6_W_vshuff_VVR(*b, *a, -4);
    *a = Q6_V_lo_W(p);
    *b = Q6_V_hi_W(p);
}

#define SSM_SCAN_XPOSE_STAGE(v, m)                        \
    _Pragma("unroll") for (uint32_t i = 0; i < 16; i++) { \
        if (!(i & (m))) {                                 \
            ssm_scan_vshuff_w(&(v)[i], &(v)[i | (m)]);    \
        }                                                 \
    }

// In-register 16x32 word transpose for packing a 32x32 transposed HMX tile.
static inline void ssm_scan_xpose_w16(HVX_Vector * restrict v) {
    SSM_SCAN_XPOSE_STAGE(v, 8);
    SSM_SCAN_XPOSE_STAGE(v, 4);
    SSM_SCAN_XPOSE_STAGE(v, 2);
    SSM_SCAN_XPOSE_STAGE(v, 1);
    SSM_SCAN_XPOSE_STAGE(v, 8);
}

static inline uint32_t ssm_scan_xpose_slot(uint32_t v) {
    return ((v & 1U) << 3) | (v >> 1);
}

static inline HVX_Vector ssm_scan_vec_iota_i32(void) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = Q6_V_vsplat_R(1);
    HVX_Vector       v    = one;
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 64));
    return Q6_Vw_vsub_VwVw(v, one);
}

static inline HVX_Vector ssm_scan_prefix_sum_f32(HVX_Vector v, HVX_Vector carry_in) {
    const HVX_Vector zero = Q6_V_vzero();
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 64));
    return hvx_vec_add_f32_f32(v, carry_in);
}

static inline HVX_Vector ssm_scan_splat_last_f32(HVX_Vector v) {
    return hvx_vec_repl_f32(Q6_V_vror_VR(v, 31 * sizeof(float)));
}

static inline HVX_Vector ssm_scan_clamp_le_zero_f32(HVX_Vector v) {
    const HVX_Vector     zero = Q6_V_vzero();
    const HVX_VectorPred gt   = Q6_Q_vcmp_gt_VsfVsf(v, zero);
    return Q6_V_vmux_QVV(gt, zero, v);
}

static __attribute__((noinline)) HVX_Vector ssm_scan_hmx_softplus_vec(const float * restrict dt_ptr) {
    return ssm_scan_vec_softplus_f32(hvx_vmem(dt_ptr));
}

static __attribute__((noinline)) HVX_Vector ssm_scan_hmx_exp_clamped(HVX_Vector v) {
    return hvx_vec_exp_f32(ssm_scan_clamp_le_zero_f32(v));
}

static __attribute__((noinline)) void ssm_scan_hmx_prep_decay(
    const float * restrict dt_work,
    const float * restrict A_work,
    float * restrict B_work,
    float * restrict C_work,
    uint8_t * restrict x_work,
    float * restrict p_vtcm,
    uint32_t nt,
    size_t state_stride_elems,
    size_t xy_stride
) {
    const HVX_Vector vzero = Q6_V_vzero();
    if (nt < HTP_SSM_SCAN_HMX_TOKEN_TILE) {
        const size_t xy_vecs = xy_stride / sizeof(HVX_Vector);
        for (uint32_t t = nt; t < HTP_SSM_SCAN_HMX_TOKEN_TILE; ++t) {
            float *      B_t = B_work + (size_t) t * state_stride_elems;
            float *      C_t = C_work + (size_t) t * state_stride_elems;
            HVX_Vector * x_t = (HVX_Vector *) (x_work + (size_t) t * xy_stride);
            for (size_t j = 0; j < state_stride_elems; j += VLEN_FP32) {
                hvx_vmem(B_t + j) = vzero;
                hvx_vmem(C_t + j) = vzero;
            }
            for (size_t j = 0; j < xy_vecs; ++j) {
                x_t[j] = vzero;
            }
        }
    }

    HVX_Vector v_sp0 = ssm_scan_hmx_softplus_vec(dt_work + 0U);
    HVX_Vector v_sp1 = ssm_scan_hmx_softplus_vec(dt_work + 32U);
    if (nt < HTP_SSM_SCAN_HMX_TOKEN_TILE) {
        const HVX_Vector v_idx0 = ssm_scan_vec_iota_i32();
        const HVX_Vector v_idx1 = Q6_Vw_vadd_VwVw(v_idx0, Q6_V_vsplat_R(32));
        const HVX_Vector v_last = Q6_V_vsplat_R((int32_t) nt - 1);
        v_sp0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(v_idx0, v_last), vzero, v_sp0);
        v_sp1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(v_idx1, v_last), vzero, v_sp1);
    }
    hvx_vmem(p_vtcm + 0U)  = v_sp0;
    hvx_vmem(p_vtcm + 32U) = v_sp1;

    const HVX_Vector v_a   = hvx_vec_repl_f32(hvx_vmem(A_work));
    const HVX_Vector v_g0  = hvx_vec_mul_f32_f32(v_sp0, v_a);
    const HVX_Vector v_g1  = hvx_vec_mul_f32_f32(v_sp1, v_a);
    const HVX_Vector v_p0  = ssm_scan_prefix_sum_f32(v_g0, vzero);
    const HVX_Vector v_p1  = ssm_scan_prefix_sum_f32(v_g1, ssm_scan_splat_last_f32(v_p0));
    const HVX_Vector v_p63 = ssm_scan_splat_last_f32(v_p1);

    hvx_vmem(p_vtcm + 64U)  = v_g0;
    hvx_vmem(p_vtcm + 96U)  = v_g1;
    hvx_vmem(p_vtcm + 128U) = v_p0;
    hvx_vmem(p_vtcm + 160U) = v_p1;
    hvx_vmem(p_vtcm + 192U) = hvx_vec_sub_f32_f32(v_p63, v_p0);
    hvx_vmem(p_vtcm + 224U) = hvx_vec_sub_f32_f32(v_p63, v_p1);

    for (uint32_t i = 0; i < 6U; ++i) {
        hvx_vmem(p_vtcm + 64U + (size_t) i * 32U) = ssm_scan_hmx_exp_clamped(hvx_vmem(p_vtcm + 64U + (size_t) i * 32U));
    }

    hvx_vmem(p_vtcm + 192U) = hvx_vec_mul_f32_f32(hvx_vmem(p_vtcm + 192U), hvx_vmem(p_vtcm + 0U));
    hvx_vmem(p_vtcm + 224U) = hvx_vec_mul_f32_f32(hvx_vmem(p_vtcm + 224U), hvx_vmem(p_vtcm + 32U));
}

static __attribute__((noinline)) void ssm_scan_hmx_pack_c_and_b(
    __fp16 * restrict c_row_tiles,
    __fp16 * restrict b_col_tiles,
    const float * restrict C_work,
    const float * restrict B_work,
    uint32_t d_state,
    size_t state_stride_elems
) {
    const uint32_t n_ds_tiles = d_state / 32U;

    for (uint32_t r0 = 0; r0 < 2U; ++r0) {
        for (uint32_t c0 = 0; c0 < n_ds_tiles; ++c0) {
            const uint32_t tile_rc       = r0 * n_ds_tiles + c0;
            const uint32_t tile_cr       = c0 * 2U + r0;
            HVX_Vector * restrict tc     = (HVX_Vector *) (c_row_tiles + (size_t) tile_rc * HMX_FP16_TILE_N_ELMS);
            HVX_Vector * restrict tb_col = (HVX_Vector *) (b_col_tiles + (size_t) tile_cr * HMX_FP16_TILE_N_ELMS);

            const float * pc = C_work + (size_t) r0 * 32U * state_stride_elems + (size_t) c0 * 32U;
            const float * pb = B_work + (size_t) r0 * 32U * state_stride_elems + (size_t) c0 * 32U;

            for (uint32_t k = 0; k < 16U; ++k) {
                const HVX_Vector vc0 = hvx_vmem(pc + (size_t) (2U * k + 0U) * state_stride_elems);
                const HVX_Vector vc1 = hvx_vmem(pc + (size_t) (2U * k + 1U) * state_stride_elems);
                tc[k]                = ssm_scan_f32_to_f16_shuff(vc0, vc1);

                const HVX_Vector vb0 = hvx_vmem(pb + (size_t) (2U * k + 0U) * state_stride_elems);
                const HVX_Vector vb1 = hvx_vmem(pb + (size_t) (2U * k + 1U) * state_stride_elems);
                tb_col[k]            = ssm_scan_f32_to_f16_shuff(vb0, vb1);
            }
        }
    }
}

static __attribute__((noinline)) void ssm_scan_hmx_pack_state_t(
    __fp16 * restrict s_t_col_tiles,
    const float * restrict state_work,
    uint32_t head_dim,
    uint32_t d_state,
    size_t state_stride_elems
) {
    const uint32_t n_d_tiles  = head_dim / 32U;
    const uint32_t n_ds_tiles = d_state / 32U;

    for (uint32_t r0 = 0; r0 < n_d_tiles; ++r0) {
        for (uint32_t c0 = 0; c0 < n_ds_tiles; ++c0) {
            HVX_Vector * restrict ts = (HVX_Vector *) (s_t_col_tiles + (size_t) (r0 * n_ds_tiles + c0) * HMX_FP16_TILE_N_ELMS);
            const float * ps         = state_work + (size_t) r0 * 32U * state_stride_elems + (size_t) c0 * 32U;
            HVX_Vector    ys[16];

            #pragma unroll
            for (uint32_t k = 0; k < 16U; ++k) {
                const HVX_Vector vs0 = hvx_vmem(ps + (size_t) (2U * k + 0U) * state_stride_elems);
                const HVX_Vector vs1 = hvx_vmem(ps + (size_t) (2U * k + 1U) * state_stride_elems);
                ys[k]                = ssm_scan_f32_to_f16(vs0, vs1);
            }

            ssm_scan_xpose_w16(ys);
            #pragma unroll
            for (uint32_t v = 0; v < 16U; ++v) {
                ts[v] = ys[ssm_scan_xpose_slot(v)];
            }
        }
    }
}

static __attribute__((noinline)) void ssm_scan_hmx_build_L(
    float * restrict L_f32,
    const float * restrict p_vtcm
) {
    const HVX_Vector vzero = Q6_V_vzero();
    const HVX_Vector vone  = Q6_V_vmux_QVV(Q6_Q_vsetq_R(4), hvx_vec_splat_f32(1.0f), vzero);

    HVX_Vector v_row0 = vzero;
    HVX_Vector v_hot  = vone;
    HVX_Vector v_arot = hvx_vmem(p_vtcm + 64U);

    for (uint32_t t = 0; t < 32U; ++t) {
        const HVX_Vector va = hvx_vec_repl_f32(v_arot);
        v_arot              = Q6_V_vror_VR(v_arot, sizeof(float));
        v_row0              = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(v_row0, va), v_hot);
        v_hot               = Q6_V_vlalign_VVR(v_hot, vzero, sizeof(float));
        hvx_vmem(L_f32 + (size_t) t * 64U + 0U)  = v_row0;
        hvx_vmem(L_f32 + (size_t) t * 64U + 32U) = vzero;
    }

    HVX_Vector v_row1 = vzero;
    v_hot             = vone;
    v_arot            = hvx_vmem(p_vtcm + 96U);

    for (uint32_t t = 32U; t < 64U; ++t) {
        const HVX_Vector va = hvx_vec_repl_f32(v_arot);
        v_arot              = Q6_V_vror_VR(v_arot, sizeof(float));
        v_row0              = hvx_vec_mul_f32_f32(v_row0, va);
        v_row1              = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(v_row1, va), v_hot);
        v_hot               = Q6_V_vlalign_VVR(v_hot, vzero, sizeof(float));
        hvx_vmem(L_f32 + (size_t) t * 64U + 0U)  = v_row0;
        hvx_vmem(L_f32 + (size_t) t * 64U + 32U) = v_row1;
    }
}

static __attribute__((noinline)) void ssm_scan_hmx_pack_x(
    __fp16 * restrict x_tilde_col_tiles,
    __fp16 * restrict x_hat_t_row_tiles,
    uint8_t * restrict x_work,
    const float * restrict p_vtcm,
    uint32_t head_dim,
    size_t xy_stride
) {
    const uint32_t n_d_tiles = head_dim / 32U;

    for (uint32_t r0 = 0; r0 < 2U; ++r0) {
        const HVX_Vector v_dt_base = hvx_vmem(p_vtcm + (size_t) r0 * 32U);
        const HVX_Vector v_sc_base = hvx_vmem(p_vtcm + 192U + (size_t) r0 * 32U);
        for (uint32_t c0 = 0; c0 < n_d_tiles; ++c0) {
            const uint32_t tile_cr       = c0 * 2U + r0;
            HVX_Vector * restrict tx_col = (HVX_Vector *) (x_tilde_col_tiles + (size_t) tile_cr * HMX_FP16_TILE_N_ELMS);
            uint8_t *             px     = x_work + (size_t) r0 * 32U * xy_stride + (size_t) c0 * 32U * sizeof(float);
            HVX_Vector            vdt    = v_dt_base;
            HVX_Vector            vsc    = v_sc_base;

            for (uint32_t k = 0; k < 16U; ++k) {
                const HVX_Vector vd0 = hvx_vec_repl_f32(vdt);
                vdt                  = Q6_V_vror_VR(vdt, sizeof(float));
                const HVX_Vector vd1 = hvx_vec_repl_f32(vdt);
                vdt                  = Q6_V_vror_VR(vdt, sizeof(float));

                const HVX_Vector vs0 = hvx_vec_repl_f32(vsc);
                vsc                  = Q6_V_vror_VR(vsc, sizeof(float));
                const HVX_Vector vs1 = hvx_vec_repl_f32(vsc);
                vsc                  = Q6_V_vror_VR(vsc, sizeof(float));

                uint8_t *        p0  = px + (size_t) (2U * k + 0U) * xy_stride;
                uint8_t *        p1  = px + (size_t) (2U * k + 1U) * xy_stride;
                const HVX_Vector vx0 = hvx_vmem(p0);
                const HVX_Vector vx1 = hvx_vmem(p1);
                tx_col[k]            = ssm_scan_f32_to_f16_shuff(hvx_vec_mul_f32_f32(vx0, vd0),
                                                                 hvx_vec_mul_f32_f32(vx1, vd1));
                hvx_vmem(p0)         = hvx_vec_mul_f32_f32(vx0, vs0);
                hvx_vmem(p1)         = hvx_vec_mul_f32_f32(vx1, vs1);
            }
        }
    }

    for (uint32_t r0 = 0; r0 < 2U; ++r0) {
        for (uint32_t c0 = 0; c0 < n_d_tiles; ++c0) {
            const uint32_t tile_cr       = c0 * 2U + r0;
            HVX_Vector * restrict tx_hat = (HVX_Vector *) (x_hat_t_row_tiles + (size_t) tile_cr * HMX_FP16_TILE_N_ELMS);
            const uint8_t *       px     = x_work + (size_t) r0 * 32U * xy_stride + (size_t) c0 * 32U * sizeof(float);
            HVX_Vector            yx[16];

            #pragma unroll
            for (uint32_t k = 0; k < 16U; ++k) {
                const HVX_Vector vx0 = hvx_vmem(px + (size_t) (2U * k + 0U) * xy_stride);
                const HVX_Vector vx1 = hvx_vmem(px + (size_t) (2U * k + 1U) * xy_stride);
                yx[k]                = ssm_scan_f32_to_f16(vx0, vx1);
            }

            ssm_scan_xpose_w16(yx);
            #pragma unroll
            for (uint32_t v = 0; v < 16U; ++v) {
                tx_hat[v] = yx[ssm_scan_xpose_slot(v)];
            }
        }
    }
}

static __attribute__((noinline)) void ssm_scan_hmx_build_m_and_unpack_y(
    __fp16 * restrict q_m_tiles,
    const float * restrict L_f32,
    const __fp16 * restrict y_inter_tiles,
    uint8_t * restrict y_work,
    const float * restrict p_vtcm,
    uint32_t head_dim,
    size_t xy_stride
) {
    const HVX_Vector vzero = Q6_V_vzero();
    for (uint32_t r0 = 0; r0 < 2U; ++r0) {
        for (uint32_t c0 = 0; c0 < 2U; ++c0) {
            HVX_Vector * restrict qm = (HVX_Vector *) (q_m_tiles + (size_t) (r0 * 2U + c0) * HMX_FP16_TILE_N_ELMS);
            if (r0 == 0U && c0 == 1U) {
                for (uint32_t k = 0; k < 16U; ++k) {
                    qm[k] = vzero;
                }
                continue;
            }
            for (uint32_t k = 0; k < 16U; ++k) {
                const uint32_t       t0 = r0 * 32U + 2U * k + 0U;
                const uint32_t       t1 = r0 * 32U + 2U * k + 1U;
                const HVX_VectorPair pq = hvx_vec_f16_to_f32_shuff(qm[k]);
                const HVX_Vector     l0 = hvx_vmem(L_f32 + (size_t) t0 * 64U + (size_t) c0 * 32U);
                const HVX_Vector     l1 = hvx_vmem(L_f32 + (size_t) t1 * 64U + (size_t) c0 * 32U);
                const HVX_Vector     m0 = hvx_vec_mul_f32_f32(Q6_V_lo_W(pq), l0);
                const HVX_Vector     m1 = hvx_vec_mul_f32_f32(Q6_V_hi_W(pq), l1);
                qm[k]                   = ssm_scan_f32_to_f16_shuff(m0, m1);
            }
        }
    }

    const uint32_t n_d_tiles = head_dim / 32U;
    for (uint32_t r0 = 0; r0 < 2U; ++r0) {
        const HVX_Vector v_lam_base = hvx_vmem(p_vtcm + 128U + (size_t) r0 * 32U);
        for (uint32_t c0 = 0; c0 < n_d_tiles; ++c0) {
            const HVX_Vector * restrict yt = (const HVX_Vector *) (y_inter_tiles + (size_t) (r0 * n_d_tiles + c0) * HMX_FP16_TILE_N_ELMS);
            HVX_Vector vlam                = v_lam_base;
            for (uint32_t k = 0; k < 16U; ++k) {
                const HVX_Vector     vl0 = hvx_vec_repl_f32(vlam);
                vlam                     = Q6_V_vror_VR(vlam, sizeof(float));
                const HVX_Vector     vl1 = hvx_vec_repl_f32(vlam);
                vlam                     = Q6_V_vror_VR(vlam, sizeof(float));
                const uint32_t       t0  = r0 * 32U + 2U * k + 0U;
                const uint32_t       t1  = r0 * 32U + 2U * k + 1U;
                const HVX_VectorPair py  = hvx_vec_f16_to_f32_shuff(yt[k]);
                float *              p0  = (float *) (y_work + (size_t) t0 * xy_stride) + (size_t) c0 * 32U;
                float *              p1  = (float *) (y_work + (size_t) t1 * xy_stride) + (size_t) c0 * 32U;
                hvx_vmem(p0)             = hvx_vec_mul_f32_f32(Q6_V_lo_W(py), vl0);
                hvx_vmem(p1)             = hvx_vec_mul_f32_f32(Q6_V_hi_W(py), vl1);
            }
        }
    }
}

static __attribute__((noinline)) void ssm_scan_hmx_finalize_chunk(
    uint8_t * restrict y_work,
    const __fp16 * restrict y_intra_tiles,
    float * restrict state_dst,
    const float * restrict state_src,
    const __fp16 * restrict ds_tiles,
    const float * restrict p_vtcm,
    uint32_t head_dim,
    uint32_t d_state,
    size_t xy_stride,
    size_t state_stride_elems
) {
    const uint32_t n_d_tiles  = head_dim / 32U;
    const uint32_t n_ds_tiles = d_state / 32U;

    for (uint32_t r0 = 0; r0 < 2U; ++r0) {
        for (uint32_t c0 = 0; c0 < n_d_tiles; ++c0) {
            const HVX_Vector * restrict yt = (const HVX_Vector *) (y_intra_tiles + (size_t) (r0 * n_d_tiles + c0) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t k = 0; k < 16U; ++k) {
                const uint32_t       t0 = r0 * 32U + 2U * k + 0U;
                const uint32_t       t1 = r0 * 32U + 2U * k + 1U;
                const HVX_VectorPair py = hvx_vec_f16_to_f32_shuff(yt[k]);
                float *              p0 = (float *) (y_work + (size_t) t0 * xy_stride) + (size_t) c0 * 32U;
                float *              p1 = (float *) (y_work + (size_t) t1 * xy_stride) + (size_t) c0 * 32U;
                hvx_vmem(p0)            = hvx_vec_add_f32_f32(hvx_vmem(p0), Q6_V_lo_W(py));
                hvx_vmem(p1)            = hvx_vec_add_f32_f32(hvx_vmem(p1), Q6_V_hi_W(py));
            }
        }
    }

    const HVX_Vector v_decay = ssm_scan_splat_last_f32(hvx_vmem(p_vtcm + 160));
    for (uint32_t r0 = 0; r0 < n_d_tiles; ++r0) {
        for (uint32_t c0 = 0; c0 < n_ds_tiles; ++c0) {
            const HVX_Vector * restrict dt = (const HVX_Vector *) (ds_tiles + (size_t) (r0 * n_ds_tiles + c0) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t k = 0; k < 16U; ++k) {
                const uint32_t       row0 = r0 * 32U + 2U * k + 0U;
                const uint32_t       row1 = r0 * 32U + 2U * k + 1U;
                const HVX_VectorPair pds  = hvx_vec_f16_to_f32_shuff(dt[k]);
                const float *        sin0 = state_src + (size_t) row0 * state_stride_elems + (size_t) c0 * 32U;
                const float *        sin1 = state_src + (size_t) row1 * state_stride_elems + (size_t) c0 * 32U;
                float *              out0 = state_dst + (size_t) row0 * state_stride_elems + (size_t) c0 * 32U;
                float *              out1 = state_dst + (size_t) row1 * state_stride_elems + (size_t) c0 * 32U;
                hvx_vmem(out0)            = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(sin0), v_decay), Q6_V_lo_W(pds));
                hvx_vmem(out1)            = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(sin1), v_decay), Q6_V_hi_W(pds));
            }
        }
    }
}

static inline void ssm_scan_hmx_gemm(
    const __fp16 * row_tiles,
    const __fp16 * col_tiles,
    __fp16 * out_tiles,
    uint32_t n_row_tiles,
    uint32_t n_col_tiles,
    uint32_t n_dot_tiles
) {
    for (uint32_t r = 0; r < n_row_tiles; ++r) {
        for (uint32_t c = 0; c < n_col_tiles; ++c) {
            const __fp16 * r_ptr = row_tiles + (size_t) r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
            const __fp16 * c_ptr = col_tiles + (size_t) c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
            __fp16 *       o_ptr = out_tiles + (size_t) (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
            hmx_fa_qk_dot_tile(r_ptr, c_ptr, o_ptr, n_dot_tiles);
        }
    }
}

static void ssm_scan_hmx_burst1_worker(const htp_thread_t * th, void * data) {
    (void) th;
    struct htp_ssm_scan_context *             scctx      = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx       = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams    = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout     = &scctx->layout;
    const uint32_t                            step       = scctx->cur_step;
    const uint32_t                            n_hvx      = octx->n_hvx;
    const uint32_t                            n_ds_tiles = kparams->d_state / 32U;
    const uint32_t                            n_d_tiles  = kparams->head_dim / 32U;

    for (uint32_t ith = 0; ith < n_hvx; ++ith) {
        if (step >= scctx->thr[ith].total_steps) {
            continue;
        }
        uint8_t *      base      = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;
        const __fp16 * scales    = (const __fp16 *) (base + layout->off_hmx_scales);
        const __fp16 * c_row     = (const __fp16 *) (base + layout->off_hmx_c_row);
        const __fp16 * b_t_col   = (const __fp16 *) (base + layout->off_hmx_b_t_col);
        const __fp16 * s_t_col   = (const __fp16 *) (base + layout->off_hmx_s_t_col);
        __fp16 *       q_tiles   = (__fp16 *)       (base + layout->off_hmx_q_m);
        __fp16 *       y_inter   = (__fp16 *)       (base + layout->off_hmx_y_tiles);

        asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) scales));
        ssm_scan_hmx_gemm(c_row, b_t_col, q_tiles, 2U, 2U,        n_ds_tiles);
        ssm_scan_hmx_gemm(c_row, s_t_col, y_inter, 2U, n_d_tiles, n_ds_tiles);
    }
}

static void ssm_scan_hmx_gemm4_worker(const htp_thread_t * th, void * data) {
    (void) th;
    struct htp_ssm_scan_context *             scctx      = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx       = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams    = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout     = &scctx->layout;
    const uint32_t                            step       = scctx->cur_step;
    const uint32_t                            n_hvx      = octx->n_hvx;
    const uint32_t                            n_ds_tiles = kparams->d_state / 32U;
    const uint32_t                            n_d_tiles  = kparams->head_dim / 32U;

    for (uint32_t ith = 0; ith < n_hvx; ++ith) {
        if (step >= scctx->thr[ith].total_steps) {
            continue;
        }
        uint8_t *      base    = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;
        const __fp16 * scales  = (const __fp16 *) (base + layout->off_hmx_scales);
        const __fp16 * x_hat_t = (const __fp16 *) (base + layout->off_hmx_x_hat_t_row);
        const __fp16 * b_col   = (const __fp16 *) (base + layout->off_hmx_b_col);
        __fp16 *       ds_out  = (__fp16 *)       (base + layout->off_hmx_s_t_col);

        asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) scales));
        ssm_scan_hmx_gemm(x_hat_t, b_col, ds_out, n_d_tiles, n_ds_tiles, 2U);
    }
}

static void ssm_scan_hmx_gemm3_worker(const htp_thread_t * th, void * data) {
    (void) th;
    struct htp_ssm_scan_context *             scctx     = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx      = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams   = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout    = &scctx->layout;
    const uint32_t                            step      = scctx->cur_step;
    const uint32_t                            n_hvx     = octx->n_hvx;
    const uint32_t                            n_d_tiles = kparams->head_dim / 32U;

    for (uint32_t ith = 0; ith < n_hvx; ++ith) {
        if (step >= scctx->thr[ith].total_steps) {
            continue;
        }
        uint8_t *      base    = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;
        const __fp16 * scales  = (const __fp16 *) (base + layout->off_hmx_scales);
        const __fp16 * m_row   = (const __fp16 *) (base + layout->off_hmx_q_m);
        const __fp16 * x_tilde = (const __fp16 *) (base + layout->off_hmx_x_tilde_col);
        __fp16 *       y_intra = (__fp16 *)       (base + layout->off_hmx_y_tiles);

        asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) scales));
        ssm_scan_hmx_gemm(m_row, x_tilde, y_intra, 2U, n_d_tiles, 2U);
    }
}

static void ssm_scan_hmx_prologue_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_ssm_scan_context *             scctx   = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout  = &scctx->layout;
    const struct htp_tensor *                 dst     = octx->dst;
    struct htp_ssm_scan_hmx_thread_state *    st      = &scctx->thr[ith];

    st->unit_start  = scctx->unit_start + scctx->units_per_hvx * ith;
    st->unit_end    = MIN(st->unit_start + scctx->units_per_hvx, scctx->unit_start + scctx->nunits);
    st->total_steps = (st->unit_start < st->unit_end) ? (st->unit_end - st->unit_start) * kparams->n_chunks : 0U;
    st->pref_u      = st->unit_start;
    st->pref_u_idx  = 0U;
    st->pref_c      = 0U;
    st->cur_u       = st->unit_start;
    st->cur_c       = 0U;

    if (st->total_steps == 0U) {
        return;
    }

    dma_queue * dma_q = th->dma_q;
    uint8_t *   base  = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;

    hmx_init_column_scales(base + layout->off_hmx_scales, Q6_V_vsplat_R(0x3c00));

    for (uint32_t s_idx = 0; s_idx < 2U && s_idx < st->total_steps; ++s_idx) {
        dma_queue_push(dma_q, dma_make_data(dst->data, base + layout->off_y[s_idx]), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(dst->data, base + layout->off_state_out[s_idx]), 0, 0, 0, 0);
        ssm_scan_prefill_push_fetch(dma_q, scctx, base, st->pref_u, st->pref_u_idx, st->pref_c, s_idx);
        if (++st->pref_c == kparams->n_chunks) {
            st->pref_c      = 0U;
            st->pref_u     += 1U;
            st->pref_u_idx ^= 1U;
        }
    }
}

static void ssm_scan_hmx_phase1a_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_ssm_scan_context *             scctx   = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout  = &scctx->layout;
    const struct htp_tensor *                 s       = octx->src[0];
    const struct htp_tensor *                 dst     = octx->dst;
    struct htp_ssm_scan_hmx_thread_state *    st      = &scctx->thr[ith];
    const uint32_t                            step    = scctx->cur_step;

    if (step >= st->total_steps) {
        return;
    }

    dma_queue *               dma_q = th->dma_q;
    struct htp_thread_trace * tr    = th->trace;
    uint8_t *                 base  = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;

    st->y_work     = (uint8_t *)     (uintptr_t) dma_queue_pop(dma_q).src;
    st->state_out  = (float *)       (uintptr_t) dma_queue_pop(dma_q).src;
    st->state_work = (float *)       (uintptr_t) dma_queue_pop(dma_q).dst;
    st->A_work     = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;
    st->B_work     = (float *)       (uintptr_t) dma_queue_pop(dma_q).dst;
    st->C_work     = (float *)       (uintptr_t) dma_queue_pop(dma_q).dst;
    st->x_work     = (uint8_t *)     (uintptr_t) dma_queue_pop(dma_q).dst;
    st->dt_work    = (const float *) (uintptr_t) dma_queue_pop(dma_q).dst;

    if (st->cur_c == 0U) {
        const struct htp_ssm_scan_unit unit = ssm_scan_decode_unit(kparams, st->cur_u);
        st->unit_q         = unit.q;
        st->unit_group     = unit.group;
        st->unit_head      = unit.head;
        st->unit_row0      = unit.row0;
        st->unit_nrows     = unit.nrows;
        st->state_out_base = dst->data + kparams->y_bytes + (size_t) unit.head * s->nb[2] + (size_t) unit.row0 * s->nb[1];
        st->y_base         = dst->data + (size_t) unit.q * kparams->y_seq_stride +
                             ((size_t) unit.head * kparams->head_dim + unit.row0) * sizeof(float);
    }

    const uint32_t t0                 = st->cur_c * kparams->token_tile;
    const uint32_t nt                 = MIN(kparams->token_tile, kparams->n_tokens - t0);
    const size_t   state_stride_elems = layout->state_row_stride / sizeof(float);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) step);

    ssm_scan_hmx_prep_decay(
        st->dt_work, st->A_work, st->B_work, st->C_work, st->x_work,
        (float *) (base + layout->off_hmx_p),
        nt, state_stride_elems, layout->xy_row_stride
    );

    ssm_scan_hmx_pack_c_and_b(
        (__fp16 *) (base + layout->off_hmx_c_row),
        (__fp16 *) (base + layout->off_hmx_b_col),
        st->C_work, st->B_work,
        kparams->d_state, state_stride_elems
    );

    ssm_scan_hmx_pack_state_t(
        (__fp16 *) (base + layout->off_hmx_b_t_col),
        st->B_work, HTP_SSM_SCAN_HMX_TOKEN_TILE, kparams->d_state, state_stride_elems
    );

    ssm_scan_hmx_pack_state_t(
        (__fp16 *) (base + layout->off_hmx_s_t_col),
        st->state_work, kparams->head_dim, kparams->d_state, state_stride_elems
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) step);
}

static void ssm_scan_hmx_phase1b_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_ssm_scan_context *             scctx   = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout  = &scctx->layout;
    struct htp_ssm_scan_hmx_thread_state *    st      = &scctx->thr[ith];
    const uint32_t                            step    = scctx->cur_step;

    if (step >= st->total_steps) {
        return;
    }

    struct htp_thread_trace * tr   = th->trace;
    uint8_t *                 base = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) step);

    ssm_scan_hmx_build_L(
        (float *) (base + layout->off_hmx_L),
        (const float *) (base + layout->off_hmx_p)
    );

    ssm_scan_hmx_pack_x(
        (__fp16 *) (base + layout->off_hmx_x_tilde_col),
        (__fp16 *) (base + layout->off_hmx_x_hat_t_row),
        st->x_work, (const float *) (base + layout->off_hmx_p),
        kparams->head_dim, layout->xy_row_stride
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) step);
}

static void ssm_scan_hmx_phase2_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_ssm_scan_context *             scctx   = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout  = &scctx->layout;
    struct htp_ssm_scan_hmx_thread_state *    st      = &scctx->thr[ith];
    const uint32_t                            step    = scctx->cur_step;

    if (step >= st->total_steps) {
        return;
    }

    struct htp_thread_trace * tr   = th->trace;
    uint8_t *                 base = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) step);

    ssm_scan_hmx_build_m_and_unpack_y(
        (__fp16 *) (base + layout->off_hmx_q_m),
        (const float *) (base + layout->off_hmx_L),
        (const __fp16 *) (base + layout->off_hmx_y_tiles),
        st->y_work, (const float *) (base + layout->off_hmx_p),
        kparams->head_dim, layout->xy_row_stride
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) step);
}

static void ssm_scan_hmx_phase3_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_ssm_scan_context *             scctx   = (struct htp_ssm_scan_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams = scctx->kparams;
    const struct htp_ssm_scan_vtcm_layout *   layout  = &scctx->layout;
    const struct htp_tensor *                 s       = octx->src[0];
    struct htp_ssm_scan_hmx_thread_state *    st      = &scctx->thr[ith];
    const uint32_t                            step    = scctx->cur_step;

    if (step >= st->total_steps) {
        return;
    }

    dma_queue *               dma_q              = th->dma_q;
    struct htp_thread_trace * tr                 = th->trace;
    uint8_t *                 base               = octx->ctx->vtcm_base + (size_t) kparams->vtcm_per_hvx * ith;
    const uint32_t            t0                 = st->cur_c * kparams->token_tile;
    const uint32_t            nt                 = MIN(kparams->token_tile, kparams->n_tokens - t0);
    const bool                last_chunk         = (st->cur_c + 1U == kparams->n_chunks);
    float *                   state_dst          = last_chunk ? st->state_out : st->state_work;
    const size_t              state_row_bytes    = (size_t) kparams->d_state * sizeof(float);
    const size_t              state_row_stride   = layout->state_row_stride;
    const size_t              state_stride_elems = state_row_stride / sizeof(float);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) step);

    ssm_scan_hmx_finalize_chunk(
        st->y_work,
        (const __fp16 *) (base + layout->off_hmx_y_tiles),
        state_dst, st->state_work,
        (const __fp16 *) (base + layout->off_hmx_s_t_col),
        (const float *) (base + layout->off_hmx_p),
        kparams->head_dim, kparams->d_state,
        layout->xy_row_stride, state_stride_elems
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) step);

    const size_t row_bytes = (size_t) st->unit_nrows * sizeof(float);
    dma_queue_push(dma_q, dma_make_data(st->y_base + (size_t) t0 * kparams->y_token_stride, st->y_work),
                   kparams->y_token_stride, layout->xy_row_stride, row_bytes, nt);

    if (last_chunk) {
        const dma_addr_t state_out_ddr = st->state_out_base + (size_t) st->unit_q * s->nb[3];
        dma_queue_push(dma_q, dma_make_data(state_out_ddr, st->state_out),
                       s->nb[1], state_row_stride, state_row_bytes, st->unit_nrows);
    } else {
        dma_queue_push(dma_q, dma_make_data(st->state_out_base, st->state_out), 0, 0, 0, 0);
    }

    if (step + 2U < st->total_steps) {
        ssm_scan_prefill_push_fetch(dma_q, scctx, base, st->pref_u, st->pref_u_idx, st->pref_c, step & 1U);
        if (++st->pref_c == kparams->n_chunks) {
            st->pref_c      = 0U;
            st->pref_u     += 1U;
            st->pref_u_idx ^= 1U;
        }
    }

    if (++st->cur_c == kparams->n_chunks) {
        st->cur_c  = 0U;
        st->cur_u += 1U;
    }
}

static void ssm_scan_hmx_epilogue_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    (void) ith;
    (void) data;
    dma_queue_flush(th->dma_q);
}

static void ssm_scan_run_hmx_chunked(struct htp_ssm_scan_context * scctx) {
    struct htp_ops_context *                  octx       = scctx->octx;
    const struct htp_ssm_scan_kernel_params * kparams    = scctx->kparams;
    work_queue_t                              hvx_q      = octx->ctx->hvx_queue;
    hmx_queue_t                               hmx_q      = octx->ctx->hmx_queue;
    const uint32_t                            n_hvx      = octx->n_hvx;
    const uint32_t                            max_steps  = scctx->units_per_hvx * kparams->n_chunks;

    work_queue_run(hvx_q, ssm_scan_hmx_prologue_worker, scctx, n_hvx);

    for (uint32_t step = 0; step < max_steps; ++step) {
        scctx->cur_step = step;

        work_queue_run(hvx_q, ssm_scan_hmx_phase1a_worker, scctx, n_hvx);

        hmx_queue_push(hmx_q, hmx_queue_make_desc(ssm_scan_hmx_burst1_worker, scctx));
        work_queue_run(hvx_q, ssm_scan_hmx_phase1b_worker, scctx, n_hvx);
        hmx_queue_pop(hmx_q);

        hmx_queue_push(hmx_q, hmx_queue_make_desc(ssm_scan_hmx_gemm4_worker, scctx));
        work_queue_run(hvx_q, ssm_scan_hmx_phase2_worker, scctx, n_hvx);
        hmx_queue_pop(hmx_q);

        hmx_queue_push(hmx_q, hmx_queue_make_desc(ssm_scan_hmx_gemm3_worker, scctx));
        hmx_queue_pop(hmx_q);

        work_queue_run(hvx_q, ssm_scan_hmx_phase3_worker, scctx, n_hvx);
    }

    work_queue_run(hvx_q, ssm_scan_hmx_epilogue_worker, scctx, n_hvx);
}

static int op_ssm_scan_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * s   = octx->src[0];
    const struct htp_tensor * ids = octx->src[6];

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
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            nunits, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        unit_start = range.start;
        nunits     = range.count;
    }

    if (nunits == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx         = MIN((uint32_t) kparams->n_hvx, nunits);
    const uint32_t units_per_hvx = octx->ctx->mdev.count > 1
        ? (nunits < kparams->n_hvx ? 1U : fastdiv(nunits + n_hvx - 1, &kparams->div_n_hvx))
        : kparams->units_per_hvx;

    struct htp_ssm_scan_context scctx = {
        .octx          = octx,
        .kparams       = kparams,
        .unit_start    = unit_start,
        .nunits        = nunits,
        .units_per_hvx = units_per_hvx,
    };

    htp_ssm_scan_vtcm_layout_build(
        &scctx.layout, kparams->d_state, kparams->row_tile, kparams->token_tile, kparams->n_hvx,
        kparams->kernel_type == HTP_SSM_SCAN_KERNEL_HMX
    );

    octx->n_hvx                 = n_hvx;
    octx->src0_spad.data         = octx->ctx->vtcm_base;
    octx->src0_spad.size_per_hvx = kparams->vtcm_per_hvx;
    octx->src0_spad.size         = kparams->vtcm_size;

    switch (kparams->kernel_type) {
        case HTP_SSM_SCAN_KERNEL_HVX_DECODE:
            work_queue_run(octx->ctx->hvx_queue, ssm_scan_thread_f32_decode, &scctx, n_hvx);
            break;

        case HTP_SSM_SCAN_KERNEL_HVX_PREFILL:
            work_queue_run(octx->ctx->hvx_queue, ssm_scan_thread_f32_prefill, &scctx, n_hvx);
            break;

        case HTP_SSM_SCAN_KERNEL_HMX:
            ssm_scan_run_hmx_chunked(&scctx);
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
