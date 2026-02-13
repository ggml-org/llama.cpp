#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <assert.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

// Dot product of two F16 vectors, accumulating to float
static inline void hvx_dot_f16_f16_aa(float * restrict r, const void * restrict x, const void * restrict y, unsigned int n, float s) {
    const HVX_Vector * restrict vx = (const HVX_Vector * restrict) x; // fp16
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf = vy[i];
        HVX_Vector x_hf = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf), Q6_V_hi_W(xy_qf)), rsum));
    }

    if (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, vx[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf), Q6_V_hi_W(xy_qf)), rsum));
    }

    rsum = Q6_Vqf32_vmpy_VsfVsf(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32(rsum));
    hvx_vec_store_u(r, 4, Q6_Vsf_equals_Vqf32(rsum));
}

static inline void hvx_dot_f16_f16_aa_rx2(float * restrict r,
                                          const void * restrict y,
                                          const void * restrict x0,
                                          const void * restrict x1,
                                          unsigned int n,
                                          float        s) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) x0;  // fp16
    const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) x1;  // fp16
    const HVX_Vector * restrict vy  = (const HVX_Vector * restrict) y;   // fp16

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_Vector rsum0 = Q6_V_vsplat_R(0);
    HVX_Vector rsum1 = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf  = vy[i];
        HVX_Vector x0_hf = vx0[i];
        HVX_Vector x1_hf = vx1[i];

        HVX_VectorPair xy0_qf = Q6_Wqf32_vmpy_VhfVhf(x0_hf, y_hf);
        HVX_VectorPair xy1_qf = Q6_Wqf32_vmpy_VhfVhf(x1_hf, y_hf);

        rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy0_qf), Q6_V_hi_W(xy0_qf)), rsum0));
        rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy1_qf), Q6_V_hi_W(xy1_qf)), rsum1));
    }

    if (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_Vector x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_Vector y_hf  = Q6_V_vand_QV(bmask, vy[i]);

        HVX_VectorPair xy0_qf = Q6_Wqf32_vmpy_VhfVhf(x0_hf, y_hf);
        HVX_VectorPair xy1_qf = Q6_Wqf32_vmpy_VhfVhf(x1_hf, y_hf);

        rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy0_qf), Q6_V_hi_W(xy0_qf)), rsum0));
        rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy1_qf), Q6_V_hi_W(xy1_qf)), rsum1));
    }

    HVX_Vector rsum = Q6_Vqf32_vmpy_VsfVsf(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32x2(rsum0, rsum1));
    hvx_vec_store_u(r, 8, Q6_Vsf_equals_Vqf32(rsum));
}

// MAD: y (F32) += x (F16) * s (F32)
static inline void hvx_mad_f32_f16_aa(float * restrict y, const void * restrict x, int n, float s) {
    const HVX_Vector * restrict ptr_x = (const HVX_Vector *) x;
    HVX_Vector * restrict ptr_y = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector S = hvx_vec_splat_f16(s);

    uint32_t i = 0;
    #pragma unroll(4)
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
            hvx_vec_store_a(&ptr_y[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F32) + x1 (F16) * s1 (F32)
static inline void hvx_mad_f32_f16_aa_rx2(float * restrict y,
                                          const void * restrict x0,
                                          const void * restrict x1,
                                          float s0,
                                          float s1,
                                          int   n) {
    const HVX_Vector * restrict ptr_x0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict ptr_x1 = (const HVX_Vector *) x1;
    HVX_Vector * restrict ptr_y        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(s0);
    HVX_Vector S1 = hvx_vec_splat_f16(s1);

    uint32_t i = 0;
    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        // Multiply x * s -> pair of F32 vectors
        HVX_VectorPair xs0_p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(ptr_x0[i]), S0);
        HVX_VectorPair xs1_p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(ptr_x1[i]), S1);

        HVX_Vector xs_p_lo = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xs0_p), Q6_V_lo_W(xs1_p));
        HVX_Vector xs_p_hi = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(xs0_p), Q6_V_hi_W(xs1_p));

        ptr_y[i * 2]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(xs_p_lo, ptr_y[i * 2]));
        ptr_y[i * 2 + 1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(xs_p_hi, ptr_y[i * 2 + 1]));
    }

    if (nloe) {
        HVX_VectorPair xs0_p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(ptr_x0[i]), S0);
        HVX_VectorPair xs1_p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(ptr_x1[i]), S1);

        HVX_Vector xs_p_lo = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xs0_p), Q6_V_lo_W(xs1_p));
        HVX_Vector xs      = xs_p_lo;
        i = 2 * i;  // index for ptr_y

        if (nloe >= 32) {
            ptr_y[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(xs, ptr_y[i]));
            nloe -= 32; ++i;
            xs = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(xs0_p), Q6_V_hi_W(xs1_p));
        }

        if (nloe) {
            HVX_Vector xy = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(xs, ptr_y[i]));
            hvx_vec_store_a(&ptr_y[i], nloe * 4, xy);
        }
    }
}

#define FLASH_ATTN_BLOCK_SIZE 128

struct htp_fa_context {
    const struct htp_ops_context * octx;

    struct fastdiv_values src0_div21;
    struct fastdiv_values src0_div1;

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t n_head_log2;
    float m0;
    float m1;

    uint32_t n_blocks;

    size_t size_q_row_padded;
    size_t size_k_row_padded;
    size_t size_v_row_padded;

    size_t size_k_block;
    size_t size_v_block;
    size_t size_m_block;

    bool is_q_fp32;
};

static inline void hvx_scale_vec_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, HVX_Vector vs) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);

    const HVX_Vector * restrict vsrc = (const HVX_Vector * restrict) src;
    HVX_Vector * restrict vdst       = (HVX_Vector * restrict) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #pragma unroll(4)
    for (; i < nvec; ++i) {
        vdst[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs));
    }
    if (nloe) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), Q6_Vsf_equals_Vqf32(v));
    }
}

static void flash_attn_ext_f16_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    const struct htp_tensor * mask  = (octx->src3.data) ? &octx->src3 : NULL;
    const struct htp_tensor * sinks = (octx->src4.data) ? &octx->src4 : NULL;
    const struct htp_tensor * dst = &octx->dst;

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

    // total rows in q
    const uint32_t nr = neq1*neq2*neq3;

    const uint32_t dr = (nr + nth - 1) / nth;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    dma_queue * dma = octx->ctx->dma[ith];

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    const size_t size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    const size_t size_k_row = DK * sizeof(__fp16);
    const size_t size_v_row = DV * sizeof(__fp16);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = octx->src0_spad.data + octx->src0_spad.size_per_thread * ith;
    uint8_t * spad_k = octx->src1_spad.data + octx->src1_spad.size_per_thread * ith;
    uint8_t * spad_v = octx->src2_spad.data + octx->src2_spad.size_per_thread * ith;
    uint8_t * spad_m = octx->src3_spad.data + octx->src3_spad.size_per_thread * ith;
    uint8_t * spad_a = octx->dst_spad.data  + octx->dst_spad.size_per_thread  * ith;

    const HVX_Vector logit_cap = hvx_vec_splat_f32(factx->logit_softcap);

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &factx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &factx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &factx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &factx->broadcast_rv2);

        // Fetch Q row
        const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);
        dma_queue_push(dma, dma_make_ptr(spad_q, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);

        const uint32_t h = iq2; // head index
        const float slope = (factx->max_bias > 0.0f) ? (h < factx->n_head_log2 ? powf(factx->m0, h + 1) : powf(factx->m1, 2*(h - factx->n_head_log2) + 1)) : 1.0f;

        HVX_Vector S_vec = hvx_vec_splat_f32(0.0f);
        HVX_Vector M_vec = hvx_vec_splat_f32(-INFINITY);

        // Clear accumulator
        hvx_splat_f32_a(spad_a, 0, DV);
        float * VKQ32 = (float *) spad_a;

        const __fp16 * mp_base = NULL;
        if (mask) {
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &factx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &factx->src3_div3);
            mp_base = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        // Prefetch first two blocks
        for (uint32_t ib = 0; ib < MIN(factx->n_blocks, 2); ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // K
            const uint8_t * k_src = (const uint8_t *) k->data + (ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
            uint8_t * k_dst = spad_k + (ib % 2) * factx->size_k_block;
            dma_queue_push(dma, dma_make_ptr(k_dst, k_src), factx->size_k_row_padded, nbk1, size_k_row, current_block_size);

            // V
            const uint8_t * v_src = (const uint8_t *) v->data + (ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
            uint8_t * v_dst = spad_v + (ib % 2) * factx->size_v_block;
            dma_queue_push(dma, dma_make_ptr(v_dst, v_src), factx->size_v_row_padded, nbv1, size_v_row, current_block_size);

            // Mask
            if (mask) {
                const uint8_t * m_src = (const uint8_t *) (mp_base + ic_start);
                uint8_t * m_dst = spad_m + (ib % 2) * factx->size_m_block;
                // Mask is 1D contiguous for this row
                dma_queue_push(dma, dma_make_ptr(m_dst, m_src), current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
            }
        }

        uint8_t * q_ptr_vtcm = dma_queue_pop(dma).dst;
        if (factx->is_q_fp32) {
            hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);  // inplace convert f32 to f16
        }

        const HVX_Vector slope_vec = hvx_vec_splat_f16(slope);
        for (uint32_t ib = 0; ib < factx->n_blocks; ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_base = dma_queue_pop(dma).dst; // K
            uint8_t * v_base = dma_queue_pop(dma).dst; // V
            __fp16  * m_base = mask ? dma_queue_pop(dma).dst : NULL; // M

            // Inner loop processing the block from VTCM
            uint32_t ic = 0;

            // Process in blocks of 32 (VLEN_FP32)
            static_assert(FLASH_ATTN_BLOCK_SIZE / VLEN_FP32 <= 4, "FLASH_ATTN_BLOCK_SIZE changed, fix HVX_Vector_x4 usage");
            HVX_Vector_x4 scores_x4;
            HVX_Vector v_max = hvx_vec_splat_f32(-INFINITY);
            for (uint32_t iv = 0; ic + VLEN_FP32 <= current_block_size; ic += VLEN_FP32, ++iv) {
                // 1. Compute scores
                float __attribute__((aligned(VLEN))) scores_arr[VLEN_FP32];
                for (uint32_t j = 0; j < VLEN_FP32; j += 2) {
                    const uint32_t cur_ic = ic + j;
                    const uint8_t * k_ptr = k_base + cur_ic * factx->size_k_row_padded;
                    hvx_dot_f16_f16_aa_rx2(&scores_arr[j], q_ptr_vtcm, k_ptr, k_ptr + factx->size_k_row_padded, DK, factx->scale);
                }

                HVX_Vector scores = *(HVX_Vector *) scores_arr;

                // 2. Softcap
                if (factx->logit_softcap != 0.0f) {
                    scores = hvx_vec_tanh_f32(scores);
                    scores = Q6_Vqf32_vmpy_VsfVsf(scores, logit_cap);
                    scores = Q6_Vsf_equals_Vqf32(scores);
                }

                // 3. Mask
                if (mask) {
                    const __fp16 * mp = m_base + ic;
                    HVX_Vector m_vals_f16 = *(const HVX_UVector *) mp;
                    HVX_VectorPair m_vals_f32_pair = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(m_vals_f16), slope_vec);
                    HVX_Vector add_val = Q6_V_lo_W(m_vals_f32_pair);
                    scores = Q6_Vqf32_vadd_Vqf32Vsf(add_val, scores);
                    scores = Q6_Vsf_equals_Vqf32(scores);
                }

                scores_x4.v[iv] = scores;
                v_max = hvx_vec_reduce_max2_f32(scores, v_max); // All lanes have block max
            }

            {
                // 4. Online Softmax Update
                HVX_Vector M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec);
                HVX_Vector diff_vec  = Q6_Vqf32_vsub_VsfVsf(M_vec, M_new_vec);
                HVX_Vector ms_vec    = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(diff_vec));
                M_vec = M_new_vec;

                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                HVX_Vector p_sum_vec = hvx_vec_splat_f32(0.0f);
                for (uint32_t ic2 = 0, iv = 0; ic2 + VLEN_FP32 <= current_block_size; ic2 += VLEN_FP32, ++iv) {
                    HVX_Vector scores = scores_x4.v[iv];
                    HVX_Vector scores_shifted = Q6_Vqf32_vsub_VsfVsf(scores, M_vec);
                    HVX_Vector P = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(scores_shifted));

                    p_sum_vec = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p_sum_vec, P));

                    // 5. Accumulate V
                    float __attribute__((aligned(VLEN))) p_arr[VLEN_FP32];
                    *(HVX_Vector *) p_arr = P;

                    for (uint32_t j = 0; j < VLEN_FP32; j += 2) {
                        const uint32_t  cur_ic = ic2 + j;
                        const uint8_t * v_ptr  = v_base + cur_ic * factx->size_v_row_padded;
                        hvx_mad_f32_f16_aa_rx2(VKQ32, v_ptr, v_ptr + factx->size_v_row_padded, p_arr[j], p_arr[j + 1], DV);
                    }
                }

                p_sum_vec = hvx_vec_reduce_sum_f32(p_sum_vec);
                S_vec = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(S_vec, ms_vec)), p_sum_vec));
            }

            // Sync scalars for leftover/next block if needed
            float M = hvx_vec_get_f32(M_vec);
            float S = hvx_vec_get_f32(S_vec);

            // Leftover
            for (; ic < current_block_size; ++ic) {
                float s_val;
                const uint8_t * k_ptr = k_base + ic * factx->size_k_row_padded;
                hvx_dot_f16_f16_aa(&s_val, q_ptr_vtcm, k_ptr, DK, factx->scale);
                if (factx->logit_softcap != 0.0f) {
                    s_val = factx->logit_softcap * tanhf(s_val);
                }

                if (mask) {
                    const float m_val = m_base[ic];
                    s_val += slope * m_val;
                }

                const float Mold = M;
                float vs = 1.0f;

                if (s_val > M) {
                    M = s_val;
                    HVX_Vector diff_vec = hvx_vec_splat_f32(Mold - M);
                    HVX_Vector ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                    float ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } else {
                    HVX_Vector diff_vec = hvx_vec_splat_f32(s_val - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }

                const uint8_t * v_ptr = v_base + ic * factx->size_v_row_padded;

                hvx_mad_f32_f16_aa(VKQ32, v_ptr, DV, vs);
            }
            M_vec = hvx_vec_splat_f32(M);
            S_vec = hvx_vec_splat_f32(S);

            // Issue DMA for next+1 block (if exists)
            if (ib + 2 < factx->n_blocks) {
                const uint32_t next_ib = ib + 2;
                const uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                // K
                const uint8_t * k_src = (const uint8_t *) k->data + (next_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
                dma_queue_push(dma, dma_make_ptr(k_base, k_src), factx->size_k_row_padded, nbk1, size_k_row, next_block_size);

                // V
                const uint8_t * v_src = (const uint8_t *) v->data + (next_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
                dma_queue_push(dma, dma_make_ptr(v_base, v_src), factx->size_v_row_padded, nbv1, size_v_row, next_block_size);

                // Mask
                if (mask) {
                    const uint8_t * m_src = (const uint8_t *) (mp_base + next_ic_start);
                    dma_queue_push(dma, dma_make_ptr(m_base, m_src), next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                }
            }
        }

        // sinks
        float M = hvx_vec_get_f32(M_vec);
        float S = hvx_vec_get_f32(S_vec);

        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float vs = 1.0f;

            if (s > M) {
                HVX_Vector diff_vec = hvx_vec_splat_f32(M - s);
                HVX_Vector ms_vec   = hvx_vec_exp_f32(diff_vec);
                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                float ms = hvx_vec_get_f32(ms_vec);
                S = S * ms + vs;
            } else {
                HVX_Vector diff_vec = hvx_vec_splat_f32(s - M);
                vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                S += vs;
            }
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, S_inv);

        // Store result
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // dst is permuted
        uint8_t * dst_ptr = (uint8_t *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1) * nb1;

        if (dst->type == HTP_TYPE_F32) {
            hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        } else if (dst->type == HTP_TYPE_F16) {
            hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        }
    }
}

int op_flash_attn_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    const struct htp_tensor * mask = (octx->src3.data) ? &octx->src3 : NULL;
    const struct htp_tensor * dst = &octx->dst;

    // Check support
    if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) || k->type != HTP_TYPE_F16 || v->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_fa_context factx;
    factx.octx = octx;

    factx.src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    factx.src0_div1  = init_fastdiv_values(q->ne[1]);

    factx.broadcast_rk2 = init_fastdiv_values(q->ne[2]/k->ne[2]);
    factx.broadcast_rk3 = init_fastdiv_values(q->ne[3]/k->ne[3]);
    factx.broadcast_rv2 = init_fastdiv_values(q->ne[2]/v->ne[2]);
    factx.broadcast_rv3 = init_fastdiv_values(q->ne[3]/v->ne[3]);

    if (mask) {
        factx.src3_div2 = init_fastdiv_values(mask->ne[2]);
        factx.src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.size_q_row_padded = hex_round_up(q->ne[0] * (factx.is_q_fp32 ? 4 : 2), 128);
    factx.size_k_row_padded = hex_round_up(k->ne[0] * sizeof(__fp16), 128);
    factx.size_v_row_padded = hex_round_up(v->ne[0] * sizeof(__fp16), 128);

    size_t size_q_block = factx.size_q_row_padded * 1; // single row for now
    factx.size_k_block = factx.size_k_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_v_block = factx.size_v_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_m_block = hex_round_up(FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16), 128);

    factx.n_blocks = (k->ne[1] + FLASH_ATTN_BLOCK_SIZE - 1) / FLASH_ATTN_BLOCK_SIZE;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) octx->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) octx->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) octx->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    factx.scale = scale;
    factx.max_bias = max_bias;
    factx.logit_softcap = logit_softcap;

    uint32_t n_head = q->ne[2];
    factx.n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    factx.m0 = powf(2.0f, -(max_bias       ) / factx.n_head_log2);
    factx.m1 = powf(2.0f, -(max_bias / 2.0f) / factx.n_head_log2);

    size_t size_vkq_acc = hex_round_up(v->ne[0] * sizeof(float), 128); // VKQ32

    octx->src0_spad.size_per_thread = size_q_block * 1;
    octx->src1_spad.size_per_thread = factx.size_k_block * 2;
    octx->src2_spad.size_per_thread = factx.size_v_block * 2;
    octx->src3_spad.size_per_thread = mask ? factx.size_m_block * 2 : 0;
    octx->dst_spad.size_per_thread  = size_vkq_acc;

    octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
    octx->src1_spad.size = octx->src1_spad.size_per_thread * octx->n_threads;
    octx->src2_spad.size = octx->src2_spad.size_per_thread * octx->n_threads;
    octx->src3_spad.size = octx->src3_spad.size_per_thread * octx->n_threads;
    octx->dst_spad.size  = octx->dst_spad.size_per_thread  * octx->n_threads;

    size_t total_spad = octx->src0_spad.size + octx->src1_spad.size + octx->src2_spad.size + octx->src3_spad.size + octx->dst_spad.size;

    if (octx->ctx->vtcm_size < total_spad) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->src2_spad.data = octx->src1_spad.data + octx->src1_spad.size;
    octx->src3_spad.data = octx->src2_spad.data + octx->src2_spad.size;
    octx->dst_spad.data  = octx->src3_spad.data + octx->src3_spad.size;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        worker_pool_run_func(octx->ctx->worker_pool, flash_attn_ext_f16_thread, &factx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}
