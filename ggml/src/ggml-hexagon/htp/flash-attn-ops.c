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

static inline HVX_Vector hvx_load_f32_to_f16(const HVX_Vector * restrict src, const HVX_Vector zero) {
    HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(src[0], zero);  // 32 elements
    HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(src[1], zero);  // 32 elements
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf)));
}

// Dot product of FP32 and FP16 vectors, accumulating to float
static inline void hvx_dot_f32_f16_aa(float * restrict r, const void * restrict y, const void * restrict x, unsigned int n, float s) {
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y; // fp32
    const HVX_Vector * restrict vx = (const HVX_Vector * restrict) x; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vsplat_R(0);
    HVX_Vector       rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y_hf  = hvx_load_f32_to_f16(&vy[i*2], zero);

        // Load x (fp16)
        HVX_Vector x_hf  = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf), Q6_V_hi_W(xy_qf)), rsum));
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y_hf  = hvx_load_f32_to_f16(&vy[i*2], zero);

        // Load x (fp16)
        HVX_Vector x_hf  = vx[i];

        // Zero-out unused elements
        // Note that we need to clear both x and y because they may contain NANs
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        x_hf = Q6_V_vand_QV(bmask, x_hf);
        y_hf = Q6_V_vand_QV(bmask, y_hf);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf), Q6_V_hi_W(xy_qf)), rsum));
    }

    rsum = Q6_Vqf32_vmpy_VsfVsf(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32(rsum));
    hvx_vec_store_u(r, 4, Q6_Vsf_equals_Vqf32(rsum));
}

// Dot product of FP32 and FP16 vectors, accumulating to float
static inline void hvx_dot_f32_f16_aa_rx2(float * restrict r,
                                          const void * restrict y,
                                          const void * restrict x0,
                                          const void * restrict x1,
                                          unsigned int n,
                                          float        s) {
    const HVX_Vector * restrict vy  = (const HVX_Vector * restrict) y;   // fp32
    const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) x0;  // fp16
    const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) x1;  // fp16

    uint32_t nvec = n / VLEN_FP16;                                       // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;                                       // leftover elements

    const HVX_Vector zero  = Q6_V_vsplat_R(0);
    HVX_Vector       rsum0 = Q6_V_vsplat_R(0);
    HVX_Vector       rsum1 = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y_hf  = hvx_load_f32_to_f16(&vy[i*2], zero);
        // Load x (fp16)
        HVX_Vector x0_hf = vx0[i];
        HVX_Vector x1_hf = vx1[i];

        HVX_VectorPair xy0_qf = Q6_Wqf32_vmpy_VhfVhf(x0_hf, y_hf);
        HVX_VectorPair xy1_qf = Q6_Wqf32_vmpy_VhfVhf(x1_hf, y_hf);

        rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy0_qf), Q6_V_hi_W(xy0_qf)), rsum0));
        rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy1_qf), Q6_V_hi_W(xy1_qf)), rsum1));
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y_hf  = hvx_load_f32_to_f16(&vy[i*2], zero);

        // Load x (fp16)
        HVX_Vector x0_hf = vx0[i];
        HVX_Vector x1_hf = vx1[i];

        // Zero-out unused elements
        // Note that we need to clear both x and y because they may contain NANs
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        x0_hf                = Q6_V_vand_QV(bmask, x0_hf);
        x1_hf                = Q6_V_vand_QV(bmask, x1_hf);
        y_hf                 = Q6_V_vand_QV(bmask, y_hf);

        HVX_VectorPair xy0_qf = Q6_Wqf32_vmpy_VhfVhf(x0_hf, y_hf);
        HVX_VectorPair xy1_qf = Q6_Wqf32_vmpy_VhfVhf(x1_hf, y_hf);

        rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy0_qf), Q6_V_hi_W(xy0_qf)), rsum0));
        rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy1_qf), Q6_V_hi_W(xy1_qf)), rsum1));
    }

    HVX_Vector rsum = Q6_Vqf32_vmpy_VsfVsf(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32x2(rsum0, rsum1));
    hvx_vec_store_u(r, 8, Q6_Vsf_equals_Vqf32(rsum));
}

// Dot product of two F16 vectors, accumulating to float
static inline void hvx_dot_f16_f16_aa(float * restrict r, const void * restrict x, const void * restrict y, unsigned int n, float s) {
    const HVX_Vector * restrict vx = (const HVX_Vector * restrict) x; // fp16
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vsplat_R(0);
    HVX_Vector       rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf = vy[i];
        HVX_Vector x_hf = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf), Q6_V_hi_W(xy_qf)), rsum));
    }

    if (nloe) {
        HVX_Vector y_hf = vy[i];

        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector      x_hf = Q6_V_vand_QV(bmask, vx[i]);

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

    uint32_t nvec = n / VLEN_FP16;                                       // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;                                       // leftover elements

    const HVX_Vector zero  = Q6_V_vsplat_R(0);
    HVX_Vector       rsum0 = Q6_V_vsplat_R(0);
    HVX_Vector       rsum1 = Q6_V_vsplat_R(0);

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
        HVX_Vector y_hf = vy[i];

        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_Vector     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);

        HVX_VectorPair xy0_qf = Q6_Wqf32_vmpy_VhfVhf(x0_hf, y_hf);
        HVX_VectorPair xy1_qf = Q6_Wqf32_vmpy_VhfVhf(x1_hf, y_hf);

        rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy0_qf), Q6_V_hi_W(xy0_qf)), rsum0));
        rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy1_qf), Q6_V_hi_W(xy1_qf)), rsum1));
    }

    HVX_Vector rsum = Q6_Vqf32_vmpy_VsfVsf(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32x2(rsum0, rsum1));
    hvx_vec_store_u(r, 8, Q6_Vsf_equals_Vqf32(rsum));
}

// MAD: y (F32) += x (F16) * s (float)
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

#define FLASH_ATTN_BLOCK_SIZE 128

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

    dma_queue * dma = octx->ctx->dma[ith];

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    const size_t size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    const size_t size_q_row_padded = hex_round_up(size_q_row, 128);

    const size_t size_k_row = DK * sizeof(__fp16);
    const size_t size_v_row = DV * sizeof(__fp16);
    const size_t size_m_row = FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16); // Treat block as one row for mask

    const size_t size_k_row_padded = hex_round_up(size_k_row, 128);
    const size_t size_v_row_padded = hex_round_up(size_v_row, 128);

    const size_t size_k_block = size_k_row_padded * FLASH_ATTN_BLOCK_SIZE;
    const size_t size_v_block = size_v_row_padded * FLASH_ATTN_BLOCK_SIZE;
    const size_t size_m_block = hex_round_up(FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16), 128);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = octx->src0_spad.data + octx->src0_spad.size_per_thread * ith;
    uint8_t * spad_k = octx->src1_spad.data + octx->src1_spad.size_per_thread * ith;
    uint8_t * spad_v = octx->src2_spad.data + octx->src2_spad.size_per_thread * ith;
    uint8_t * spad_m = octx->src3_spad.data + octx->src3_spad.size_per_thread * ith;
    uint8_t * spad_a = octx->dst_spad.data  + octx->dst_spad.size_per_thread  * ith;

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

        // Fetch Q row
        const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);
        dma_queue_push(dma, dma_make_ptr(spad_q, q_row_ptr), size_q_row_padded, nbq1, size_q_row, 1);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        // Clear accumulator
        hvx_splat_f32_a(spad_a, 0, DV);
        float * VKQ32 = (float *) spad_a;

        const __fp16 * mp_base = NULL;
        if (mask) {
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &octx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &octx->src3_div3);
            mp_base = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        const uint32_t n_blocks = (nek1 + FLASH_ATTN_BLOCK_SIZE - 1) / FLASH_ATTN_BLOCK_SIZE;

        // Prefetch first two blocks
        for (uint32_t ib = 0; ib < MIN(n_blocks, 2); ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // K
            const uint8_t * k_src = (const uint8_t *) k->data + (ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
            uint8_t * k_dst = spad_k + (ib % 2) * size_k_block;
            dma_queue_push(dma, dma_make_ptr(k_dst, k_src), size_k_row_padded, nbk1, size_k_row, current_block_size);

            // V
            const uint8_t * v_src = (const uint8_t *) v->data + (ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
            uint8_t * v_dst = spad_v + (ib % 2) * size_v_block;
            dma_queue_push(dma, dma_make_ptr(v_dst, v_src), size_v_row_padded, nbv1, size_v_row, current_block_size);

            // Mask
            if (mask) {
                const uint8_t * m_src = (const uint8_t *) (mp_base + ic_start);
                uint8_t * m_dst = spad_m + (ib % 2) * size_m_block;
                // Mask is 1D contiguous for this row
                dma_queue_push(dma, dma_make_ptr(m_dst, m_src), current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
            }
        }

        const uint8_t * q_ptr_vtcm = dma_queue_pop(dma).dst;

        for (uint32_t ib = 0; ib < n_blocks; ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_base = dma_queue_pop(dma).dst; // K
            uint8_t * v_base = dma_queue_pop(dma).dst; // V
            __fp16  * m_base = mask ? dma_queue_pop(dma).dst : NULL; // M

            // Inner loop processing the block from VTCM
            uint32_t ic = 0;

            const bool is_q_fp32 = (q->type == HTP_TYPE_F32);

            // Process in blocks of 32 (VLEN_FP32)
            static_assert(FLASH_ATTN_BLOCK_SIZE / VLEN_FP32 <= 4, "FLASH_ATTN_BLOCK_SIZE changed, fix HVX_Vector_x4 usage");
            HVX_Vector_x4 scores_x4;
            HVX_Vector v_max = hvx_vec_splat_f32(-INFINITY);
            for (uint32_t iv = 0; ic + VLEN_FP32 <= current_block_size; ic += VLEN_FP32, ++iv) {
                // 1. Compute scores
                float __attribute__((aligned(VLEN))) scores_arr[VLEN_FP32];
                for (int j = 0; j < VLEN_FP32; j += 2) {
                    const uint32_t cur_ic = ic + j;
                    const uint8_t * k_ptr = k_base + cur_ic * size_k_row_padded;
                    if (is_q_fp32) {
                        hvx_dot_f32_f16_aa_rx2(&scores_arr[j], q_ptr_vtcm, k_ptr, k_ptr + size_k_row_padded, DK, scale);
                    } else {
                        hvx_dot_f16_f16_aa_rx2(&scores_arr[j], q_ptr_vtcm, k_ptr, k_ptr + size_k_row_padded, DK, scale);
                    }
                }

                HVX_Vector scores = *(HVX_Vector *) scores_arr;

                // 2. Softcap
                if (logit_softcap != 0.0f) {
                    scores = hvx_vec_tanh_f32(scores);
                    scores = Q6_Vqf32_vmpy_VsfVsf(scores, hvx_vec_splat_f32(logit_softcap));
                    scores = Q6_Vsf_equals_Vqf32(scores);
                }

                // 3. Mask
                if (mask) {
                    const __fp16 * mp = m_base + ic;
                    HVX_Vector m_vals_f16 = *(const HVX_UVector *) mp;

                    HVX_Vector one_f16 = Q6_Vh_vsplat_R(0x3c00);
                    HVX_VectorPair m_vals_f32_pair = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(m_vals_f16), one_f16);

                    HVX_Vector m_vals_f32 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(m_vals_f32_pair));

                    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);
                    HVX_Vector add_val = Q6_Vqf32_vmpy_VsfVsf(m_vals_f32, slope_vec);
                    scores = Q6_Vqf32_vadd_VsfVsf(scores, Q6_Vsf_equals_Vqf32(add_val));
                    scores = Q6_Vsf_equals_Vqf32(scores);
                }

                scores_x4.v[iv] = scores;
                v_max = Q6_Vsf_vmax_VsfVsf(scores, v_max);
            }

            {
                // 4. Online Softmax Update
                v_max = hvx_vec_reduce_max_f32(v_max);
                float m_block = hvx_vec_get_f32(v_max);
                float M_old = M;
                float M_new = (m_block > M) ? m_block : M;
                M = M_new;

                const float ms = expf(M_old - M_new);
                hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms);

                HVX_Vector M_new_vec = hvx_vec_splat_f32(M_new);
                HVX_Vector p_sum_vec = hvx_vec_splat_f32(0.0f);
                for (uint32_t ic2 = 0, iv = 0; ic2 + VLEN_FP32 <= current_block_size; ic2 += VLEN_FP32, ++iv) {
                    HVX_Vector scores = scores_x4.v[iv];
                    HVX_Vector scores_shifted = Q6_Vqf32_vsub_VsfVsf(scores, M_new_vec);
                    HVX_Vector P = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(scores_shifted));

                    p_sum_vec = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p_sum_vec, P));

                    // 5. Accumulate V
                    float __attribute__((aligned(VLEN))) p_arr[VLEN_FP32];
                    *(HVX_Vector*)p_arr = P;

                    for (int j = 0; j < VLEN_FP32; ++j) {
                        const uint32_t cur_ic = ic2 + j;
                        const uint8_t * v_ptr = v_base + cur_ic * size_v_row_padded;
                        hvx_mad_f32_f16_aa(VKQ32, v_ptr, DV, p_arr[j]);
                    }
                }

                p_sum_vec = hvx_vec_reduce_sum_f32(p_sum_vec);
                S = S * ms + hvx_vec_get_f32(p_sum_vec);
            }

            // Leftover
            for (; ic < current_block_size; ++ic) {
                float s_val;
                const uint8_t * k_ptr = k_base + ic * size_k_row_padded;

                if (is_q_fp32) {
                    hvx_dot_f32_f16_aa(&s_val, q_ptr_vtcm, k_ptr, DK, scale);
                } else {
                    hvx_dot_f16_f16_aa(&s_val, q_ptr_vtcm, k_ptr, DK, scale);
                }

                if (logit_softcap != 0.0f) {
                    s_val = logit_softcap * tanhf(s_val);
                }

                if (mask) {
                    const float m_val = m_base[ic];
                    s_val += slope * m_val;
                }

                const float Mold = M;
                float ms = 1.0f;
                float vs = 1.0f;

                if (s_val > M) {
                    M = s_val;
                    ms = expf(Mold - M);
                    hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms);
                } else {
                    vs = expf(s_val - M);
                }

                const uint8_t * v_ptr = v_base + ic * size_v_row_padded;

                hvx_mad_f32_f16_aa(VKQ32, v_ptr, DV, vs);

                S = S * ms + vs;
            }

            // Issue DMA for next+1 block (if exists)
            if (ib + 2 < n_blocks) {
                const uint32_t next_ib = ib + 2;
                const uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                // K
                const uint8_t * k_src = (const uint8_t *) k->data + (next_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
                dma_queue_push(dma, dma_make_ptr(k_base, k_src), size_k_row_padded, nbk1, size_k_row, next_block_size);

                // V
                const uint8_t * v_src = (const uint8_t *) v->data + (next_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
                dma_queue_push(dma, dma_make_ptr(v_base, v_src), size_v_row_padded, nbv1, size_v_row, next_block_size);

                // Mask
                if (mask) {
                    const uint8_t * m_src = (const uint8_t *) (mp_base + next_ic_start);
                    dma_queue_push(dma, dma_make_ptr(m_base, m_src), next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                }
            }
        }

        // sinks
        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms);
            } else {
                vs = expf(s - M);
            }

            S = S * ms + vs;
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

    size_t size_q_row_padded = hex_round_up(q->ne[0] * (q->type == HTP_TYPE_F32 ? 4 : 2), 128);
    size_t size_k_row_padded = hex_round_up(k->ne[0] * sizeof(__fp16), 128);
    size_t size_v_row_padded = hex_round_up(v->ne[0] * sizeof(__fp16), 128);

    size_t size_q_block = size_q_row_padded * 1; // single row for now
    size_t size_k_block = size_k_row_padded * FLASH_ATTN_BLOCK_SIZE;
    size_t size_v_block = size_v_row_padded * FLASH_ATTN_BLOCK_SIZE;
    size_t size_m_block = hex_round_up(FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16), 128);

    size_t size_vkq_acc = hex_round_up(v->ne[0] * sizeof(float), 128); // VKQ32

    octx->src0_spad.size_per_thread = size_q_block * 1;
    octx->src1_spad.size_per_thread = size_k_block * 2;
    octx->src2_spad.size_per_thread = size_v_block * 2;
    octx->src3_spad.size_per_thread = mask ? size_m_block * 2 : 0;
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
        worker_pool_run_func(octx->ctx->worker_pool, htp_flash_attn_ext_job, octx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}
