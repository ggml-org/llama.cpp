#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

// Dot product of FP32 and FP16 vectors, accumulating to float
static void hvx_dot_f32_f16_uu(float * restrict r, const void * restrict y, const void * restrict x, unsigned int n, float s) {
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y; // fp32
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vsplat_R(0);
    HVX_Vector       rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf));

        // Load x (fp16)
        HVX_Vector x_hf = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(x_hf), y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf));

        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector      x_hf = Q6_V_vand_QV(bmask, vx[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(x_hf), y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(rsum), hvx_vec_splat_fp32(s));
    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(rsum));

    hvx_vec_store_u(r, 4, rsum);
}

// Dot product of two F16 vectors, accumulating to float
static void hvx_dot_f16_f16_uu(float * restrict r, const void * restrict x, const void * restrict y, unsigned int n, float s) {
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x; // fp16
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vsplat_R(0);
    HVX_Vector       rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf = vy[i];
        HVX_Vector x_hf = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        HVX_Vector y_hf = vy[i];

        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector      x_hf = Q6_V_vand_QV(bmask, vx[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(x_hf), y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(rsum), hvx_vec_splat_fp32(s));
    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

// MAD: y (F32) += x (F16) * v (float)
static void hvx_mad_f32_f16_au(float * restrict y, const void * restrict x, int n, float s) {
    const HVX_UVector * restrict ptr_x = (const HVX_UVector *) x;
    HVX_Vector * restrict ptr_y = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector S = hvx_vec_splat_fp16(s);

    uint32_t i = 0;
    for (i = 0; i < nvec; ++i) {
        // Multiply x * s -> pair of F32 vectors
        HVX_VectorPair xs_p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(ptr_x[i]), S);
        ptr_y[i*2]   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xs_p), ptr_y[i*2]));
        ptr_y[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xs_p), ptr_y[i*2+1]));
    }

    if (nloe) {
        HVX_VectorPair xs_p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(ptr_x[i]), S);

        HVX_Vector xs = Q6_V_lo_W(xs_p);
        i = 2 * i; // index for ptr_y

        if (nloe >= 32) {
            ptr_y[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(xs, ptr_y[i]));
            nloe -= 32; ++i; xs = Q6_V_hi_W(xs_p);
        }

        if (nloe) {
            HVX_Vector xy = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(xs, ptr_y[i]));
            hvx_vec_store_u(&ptr_y[i], nloe * 4, xy);
        }
    }
}

static void flash_attn_ext_f16_thread(struct htp_ops_context * octx, int ith, int nth) {
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    const struct htp_tensor * mask  = (octx->src3.data) ? &octx->src3 : NULL;
    const struct htp_tensor * sinks = (octx->src4.data) ? &octx->src4 : NULL;
    struct htp_tensor * dst = &octx->dst;

    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];

    const uint32_t nek0 = k->ne[0];
    const uint32_t nek1 = k->ne[1];
    const uint32_t nek2 = k->ne[2];
    const uint32_t nek3 = k->ne[3];

    const uint32_t nev0 = v->ne[0];
    const uint32_t nev1 = v->ne[1];
    const uint32_t nev2 = v->ne[2];
    const uint32_t nev3 = v->ne[3];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbq3 = q->nb[3];

    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbk3 = k->nb[3];

    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];
    const uint32_t nbv3 = v->nb[3];

    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) octx->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) octx->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) octx->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // total rows in q
    const uint32_t nr = neq1*neq2*neq3;

    const uint32_t dr = (nr + nth - 1) / nth;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    // Use scratchpad for the accumulator
    // VKQ32: DV * sizeof(float)

    uint8_t * spad = octx->src0_spad.data + octx->src0_spad.size_per_thread * ith;
    float * VKQ32 = (float *) spad;

    const uint32_t n_head = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &octx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &octx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &octx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &octx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &octx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &octx->broadcast_rv2);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        // Clear accumulator
        memset(VKQ32, 0, DV * sizeof(float));

        const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);

        const __fp16 * mp_base = NULL;
        if (mask) {
            // Mask offset calculation from ggml-cpu:
            // mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]
            // We need broadcasting indices.
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &octx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &octx->src3_div3);
            mp_base = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        uint32_t ic = 0;

        // Process in blocks of 32 (VLEN_FP32)
        for (; ic + VLEN_FP32 <= nek1; ic += VLEN_FP32) {
            // 1. Compute scores
            float __attribute__((aligned(VLEN))) scores_arr[VLEN_FP32];
            for (int j = 0; j < VLEN_FP32; ++j) {
                const uint32_t cur_ic = ic + j;
                const uint8_t * k_ptr = (const uint8_t *) k->data + (cur_ic*nbk1 + ik2*nbk2 + ik3*nbk3);
                if (q->type == HTP_TYPE_F32) {
                    hvx_dot_f32_f16_uu(&scores_arr[j], q_row_ptr, k_ptr, DK, scale);
                } else {
                    hvx_dot_f16_f16_uu(&scores_arr[j], q_row_ptr, k_ptr, DK, scale);
                }
            }

            HVX_Vector scores = *(HVX_Vector *) scores_arr;

            // 2. Softcap
            if (logit_softcap != 0.0f) {
                // scores = logit_softcap * tanh(scores)
                // Note: hvx_vec_tanh_fp32 uses sigmoid approximation
                scores = hvx_vec_tanh_fp32(scores);
                scores = Q6_Vqf32_vmpy_VsfVsf(scores, hvx_vec_splat_fp32(logit_softcap));
                scores = Q6_Vsf_equals_Vqf32(scores);
            }

            // 3. Mask
            if (mask) {
                // Load 32 masks. mask is F16.
                const __fp16 * mp = mp_base + ic;
                HVX_Vector m_vals_fp16 = *(const HVX_UVector *) mp;

                HVX_Vector one_fp16 = Q6_Vh_vsplat_R(0x3c00);
                HVX_VectorPair m_vals_fp32_pair = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(m_vals_fp16), one_fp16);

                HVX_Vector m_vals_fp32 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(m_vals_fp32_pair));

                // scores += slope * mask
                HVX_Vector slope_vec = hvx_vec_splat_fp32(slope);
                HVX_Vector add_val = Q6_Vqf32_vmpy_VsfVsf(m_vals_fp32, slope_vec);
                scores = Q6_Vqf32_vadd_VsfVsf(scores, Q6_Vsf_equals_Vqf32(add_val));
                scores = Q6_Vsf_equals_Vqf32(scores);
            }

            // 4. Online Softmax Update
            // Find max in scores
            HVX_Vector v_max = hvx_vec_reduce_max_fp32(scores);
            float m_block = hvx_vec_get_fp32(v_max);

            float M_old = M;
            float M_new = (m_block > M) ? m_block : M;
            M = M_new;

            float ms = expf(M_old - M_new);

            // Update global accumulator
            hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, ms);
            S = S * ms;

            // Compute exps: P = exp(scores - M_new)
            HVX_Vector M_new_vec = hvx_vec_splat_fp32(M_new);
            HVX_Vector scores_shifted = Q6_Vqf32_vsub_VsfVsf(scores, M_new_vec);
            HVX_Vector P = hvx_vec_exp_fp32(Q6_Vsf_equals_Vqf32(scores_shifted));

            // Add to S
            HVX_Vector p_sum_vec = hvx_vec_fp32_reduce_sum(P);
            float p_sum = hvx_vec_get_fp32(p_sum_vec);
            S += p_sum;

            // 5. Accumulate V: VKQ32 += Sum(V[ic+j] * P[j])
            float __attribute__((aligned(VLEN))) p_arr[VLEN_FP32];
            *(HVX_Vector*)p_arr = P;

            // Optimized V accumulation
            // We iterate DV in chunks of 32 (VLEN_FP32) because we can load 32 floats from VKQ32.
            // But V is FP16. V[ic+j] is a row.
            // We want to add P[j] * V[ic+j] to VKQ32.
            // We can just iterate j=0..31
            for (int j = 0; j < VLEN_FP32; ++j) {
                const uint32_t cur_ic = ic + j;
                const uint8_t * v_ptr = (const uint8_t *) v->data + (cur_ic*nbv1 + iv2*nbv2 + iv3*nbv3);
                // P[j] is p_arr[j]
                hvx_mad_f32_f16_au(VKQ32, v_ptr, DV, p_arr[j]);
            }
        }

        // Leftover
        for (; ic < nek1; ++ic) {
            float s_val;

            const uint8_t * k_ptr = (const uint8_t *) k->data + (ic*nbk1 + ik2*nbk2 + ik3*nbk3);

            if (q->type == HTP_TYPE_F32) {
                hvx_dot_f32_f16_uu(&s_val, q_row_ptr, k_ptr, DK, scale);
            } else {
                hvx_dot_f16_f16_uu(&s_val, q_row_ptr, k_ptr, DK, scale);
            }

            if (logit_softcap != 0.0f) {
                s_val = logit_softcap * tanhf(s_val);
            }

            if (mask) {
                const float m_val = mp_base[ic];
                s_val += slope * m_val;
            }

            const float Mold = M;
            float ms = 1.0f;
            float vs = 1.0f;

            if (s_val > M) {
                M = s_val;
                ms = expf(Mold - M);
                hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, ms);
            } else {
                vs = expf(s_val - M);
            }

            const uint8_t * v_ptr = (const uint8_t *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3);

            // Accumulate V: VKQ32 += v_ptr * vs
            hvx_mad_f32_f16_au(VKQ32, v_ptr, DV, vs);

            S = S * ms + vs;
        }

        // sinks
        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, ms);
            } else {
                vs = expf(s - M);
            }

            S = S * ms + vs;
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, S_inv);

        // Store result
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // dst is permuted
        uint8_t * dst_ptr = (uint8_t *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1) * nb1;

        if (dst->type == HTP_TYPE_F32) {
            hvx_copy_fp32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        } else if (dst->type == HTP_TYPE_F16) {
            __fp16 * d = (__fp16 *) dst_ptr;
            for (int i = 0; i < DV; ++i) {
                d[i] = (__fp16)VKQ32[i];
            }
        }
    }
}

static void htp_flash_attn_ext_job(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;
    flash_attn_ext_f16_thread(octx, i, n);
}

int op_flash_attn_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    const struct htp_tensor * mask = (octx->src3.type != HTP_TYPE_COUNT) ? &octx->src3 : NULL;
    struct htp_tensor * dst = &octx->dst;

    // Check support
    if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        k->type != HTP_TYPE_F16 ||
        v->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    octx->src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    octx->src0_div1  = init_fastdiv_values(q->ne[1]);

    octx->broadcast_rk2 = init_fastdiv_values(q->ne[2]/k->ne[2]);
    octx->broadcast_rk3 = init_fastdiv_values(q->ne[3]/k->ne[3]);
    octx->broadcast_rv2 = init_fastdiv_values(q->ne[2]/v->ne[2]);
    octx->broadcast_rv3 = init_fastdiv_values(q->ne[3]/v->ne[3]);

    if (mask) {
        octx->src3_div2 = init_fastdiv_values(mask->ne[2]);
        octx->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    size_t spad_size_vkq = htp_round_up(octx->src2.ne[0] * sizeof(float), 128); // VKQ32

    octx->src0_spad.size_per_thread = spad_size_vkq;
    octx->src0_spad.size =  octx->src0_spad.size_per_thread * octx->n_threads;

    if (octx->ctx->vtcm_size < octx->src0_spad.size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        worker_pool_run_func(octx->ctx->worker_pool, htp_flash_attn_ext_job, octx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}
